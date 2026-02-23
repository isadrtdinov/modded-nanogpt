import os
import math
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from torch import Tensor, nn
from kernels import get_kernel
from triton_kernels import FusedLinearReLUSquareFunction, FusedSoftcappedCrossEntropy
from hyperparams import Hyperparameters

try:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
except:
    device = torch.device("cuda", 0)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model
def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinearT(nn.Module):
    """
    Linear layer with transposed weight storage (in_features, out_features) which
    addresses the slow kernel that was used for gradient accumulation. @chrisjmccormick
    """
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.weight) # @Grad62304977 and others

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out = torch.ops.nanogpt.mm_t(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return x @ self.weight.type_as(x)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len, paired=False):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.paired = paired
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1, factor2 = (
            self.factor1[None, : x_BTHD.size(-3), None, :],
            self.factor2[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = angular_freq.repeat_interleave(2)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//2)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=device)
        if not self.paired:
            theta = torch.outer(t, angular_freq)
            self.factor1 = nn.Buffer(
                theta.cos().to(torch.bfloat16), persistent=False
            )
            self.factor2 = nn.Buffer(
                theta.sin().to(torch.bfloat16), persistent=False
            )
        else:
            t_even = 2 * t
            t_odd = 2 * t + 1
            theta1 = torch.outer(t_even, angular_freq)
            theta2 = torch.outer(t_odd, angular_freq)
            self.factor1 = nn.Buffer(
                torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
            self.factor2 = nn.Buffer(
                torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        if not self.paired:
            theta = torch.outer(t, self.angular_freq)
            self.factor1.copy_(theta.cos())
            self.factor2.copy_(theta.sin())
        else:
            t_even = 2 * t
            t_odd = 2 * t + 1
            theta1 = torch.outer(t_even, self.angular_freq)
            theta2 = torch.outer(t_odd, self.angular_freq)
            self.factor1.copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
            self.factor2.copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    yarn: Yarn
    key_offset: bool
    attn_gate_w: torch.Tensor
    ve_gate_w: torch.Tensor
    train_max_seq_len: torch.Tensor

flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, max_seq_len: int, paired: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        self.paired = paired
        self.val_max_seq_len = max_seq_len
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        # Weights are stored in parameter banks and passed via forward()

    def forward(self, x: Tensor, attn_args: AttnArgs, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        yarn = attn_args.yarn
        ve, sa_lambdas, key_offset = attn_args.ve, attn_args.sa_lambdas, attn_args.key_offset
        seqlens, bm_size = attn_args.seqlens, attn_args.bm_size
        # sparse gated attention to enable context based no-op by @classiclarryd
        # only include gates on layers with value embeds used on forward pass
        attn_gate_w, ve_gate_w = attn_args.attn_gate_w, attn_args.ve_gate_w
        train_max_seq_len = attn_args.train_max_seq_len

        q, k, v = F.linear(x, sa_lambdas[0] * qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        max_len = train_max_seq_len if self.training else self.val_max_seq_len

        q, k = norm(q), norm(k) # QK norm @Grad62304977

        if not self.paired:
            q, k = yarn.rotary(q), yarn.rotary(k)

            if key_offset:
                # shift keys forward for the stationary head dims. Enables 1-layer induction.
                k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]

            if ve is not None:
                # gate pattern g(x[:6] + ve[:6]) by @photomz
                ve_gate_out = 2 * torch.sigmoid(F.linear(torch.cat([x[..., :6], ve[None, ..., :6]], dim=-1), ve_gate_w)).view(B, T, self.num_heads, 1)
                v = v + ve_gate_out * ve.view_as(v) # @ KoszarskyB & @Grad62304977

        else:
            # Paired heads: adjacent heads' queries attend to each other's keys.
            # Two copies of the input stream are interleaved to achieve this, which:
            # - doubles the length of each sequence
            # - halves the effective window size
            q = q.view(B, T, self.num_heads // 2, self.head_dim * 2)
            k = k.view(B, T, self.num_heads // 2, self.head_dim * 2)
            v = v.reshape(B, T * 2, self.num_heads // 2, self.head_dim)

            q, k = yarn.rotary(q), yarn.rotary(k)

            q = q.view(B, T * 2, self.num_heads // 2, self.head_dim)
            k = k.view(B, T * 2, self.num_heads // 2, self.head_dim)

            if ve is not None:
                ve_gate_out = 2 * torch.sigmoid(F.linear(x[..., :12], ve_gate_w)).view(B, T * 2, self.num_heads // 2, 1)
                v = v + ve_gate_out * ve.view_as(v)

            seqlens = 2 * seqlens
            max_len = 2 * max_len

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
                                                        max_seqlen_q=max_len, max_seqlen_k=max_len,
                                                        causal=True, softmax_scale=yarn.attn_scale, window_size=(bm_size, 0))
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
        return y

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Weights are stored in parameter banks and passed via forward()

    def forward(self, x: Tensor, c_fc: Tensor, c_proj: Tensor):
        # relu(x)^2:
        # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        # Fused triton kernel for relu(x @ W1.T)^2 @ W2.T
        return FusedLinearReLUSquareFunction.apply(x, c_fc, c_proj)

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, max_seq_len: int, has_attn: bool, has_mlp: bool, use_paired_head: bool):
        super().__init__()
        # skip attention of blocks.6 (the 7th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, head_dim, num_heads, max_seq_len=max_seq_len, paired=use_paired_head) if has_attn else None
        # skip MLP blocks for first MLP layer by @EmelyanenkoK
        self.mlp = MLP() if has_mlp else None

    def forward(self, x: Tensor, attn_args: AttnArgs, qkvo_w: Tensor = None, c_fc: Tensor = None, c_proj: Tensor = None):
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args, qkvo_w)
        if self.mlp is not None:
            x = x + self.mlp(norm(x), c_fc, c_proj)
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


@dataclass
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor
    ws_short: int
    ws_long: int
    train_max_seq_len: int


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int,
                 max_seq_len: int, args: Hyperparameters):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)

        self.smear_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)
        self.smear_gate.weight.label = 'smear_gate'

        self.skip_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.skip_gate.weight)
        self.skip_gate.weight.label = 'skip_gate'

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        # spherical gaussian init by @photomz
        self.value_embeds = nn.Parameter(0.01 * torch.randn(5 * self.vocab_size, model_dim, dtype=torch.bfloat16))
        self.value_embeds.label = 'value_embed'

        # parameter banks for attention and value embedding gate weights
        self.attn_gate_bank = nn.Parameter(torch.zeros(10, num_heads, 12))  # 10 layers
        self.attn_gate_bank.label = 'attn_gate_bank'
        self.ve_gate_bank = nn.Parameter(torch.zeros(5, num_heads, 12))  # 5 unique gates
        self.ve_gate_bank.label = 've_gate_bank'

        # -----------------------------------
        # Parameter banks for sharded optimization, by @chrisjmccormick

        # Identify which layers have attention/MLP
        # Attention is skipped in layer 6 by @YouJiacheng
        self.attn_layer_indices = [i for i in range(num_layers) if i != 6]
        # All layers have MLP (At 11 layers--dropped first layer @EmelyanenkoK)
        self.mlp_layer_indices = list(range(num_layers))

        hdim = num_heads * head_dim
        mlp_hdim = 4 * model_dim

        # Create index mappings: layer_idx -> bank_idx
        self.layer_to_attn_idx = {layer_idx: bank_idx for bank_idx, layer_idx in enumerate(self.attn_layer_indices)}
        self.layer_to_mlp_idx = {layer_idx: bank_idx for bank_idx, layer_idx in enumerate(self.mlp_layer_indices)}

        # Attention bank: stores QKVO weights for all attention layers
        # merged QKVO weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        # Simplified layout by @chrisjmccormick
        # Shape: (num_attn_layers, 4*model_dim, hdim) = (10, 3072, 768)
        # Reshape for sharding: (40, 768, 768) for even distribution across 8 GPUs
        self.attn_bank = nn.Parameter(torch.empty(len(self.attn_layer_indices), 4 * model_dim, hdim))
        self.attn_bank.label = 'attn'
        self.attn_bank.reshape = (len(self.attn_layer_indices) * 4, hdim, hdim)  # (40, 768, 768)

        # MLP bank: stores c_fc and c_proj for all MLP layers
        # Shape: (num_mlp_layers + padding, 2, mlp_hdim, model_dim) = (12, 2, 3072, 768)
        # We add 1 padding layer (index 11) to get 12*2=24 matrices for even distribution across 8 GPUs
        # Reshape for sharding: (24, 3072, 768)
        num_mlp_with_padding = len(self.mlp_layer_indices) + 1  # 11 + 1 = 12
        self.mlp_bank = nn.Parameter(torch.empty(num_mlp_with_padding, 2, mlp_hdim, model_dim))
        self.mlp_bank.label = 'mlp'
        self.mlp_bank.reshape = (num_mlp_with_padding * 2, mlp_hdim, model_dim)  # (24, 3072, 768)

        # improved init scale by @YouJiacheng and @srashedll
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.attn_bank.uniform_(-bound, bound)
            self.mlp_bank[:, 0, :, :].uniform_(-bound, bound)  # c_fc
            self.mlp_bank[:, 1, :, :].zero_()  # c_proj - zero init suggested by @Grad62304977

        # Create blocks with has_attn/has_mlp flags
        self.paired_head_layers = [0, 2, 5, 9]
        self.blocks = nn.ModuleList([
            Block(model_dim, head_dim, num_heads,
                  max_seq_len=max_seq_len,
                  has_attn=(i in self.layer_to_attn_idx),
                  has_mlp=(i in self.layer_to_mlp_idx),
                  use_paired_head=(i in self.paired_head_layers))
            for i in range(num_layers)
        ])
        self.yarn = Yarn(head_dim, max_seq_len)
        self.yarn_paired_head = Yarn(head_dim, max_seq_len, paired=True)
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        # Transposed weight storage for faster gradient accumulation
        self.lm_head = CastedLinearT(model_dim, self.vocab_size, use_fp8=use_fp8, x_s=100 / 448, w_s=1.6 / 448,
                                     grad_s=args.grad_scale * 0.75 / 448)

        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)
        self.lm_head.weight.label = 'lm_head'

        self.embed = nn.Embedding(self.vocab_size, model_dim)
        self.embed.weight.label = 'embed'
        with torch.no_grad():
            self.embed.weight.copy_(self.lm_head.weight.T)

        self.bigram_embed = nn.Embedding(args.bigram_vocab_size, model_dim)
        self.bigram_embed.weight.label = 'bigram_embed'
        nn.init.zeros_(self.bigram_embed.weight)

        # x0_lambdas separated out for different optimizer treatment (no beta smoothing)
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        self.x0_lambdas.label = 'x0_lambdas'

        pad = (-num_layers * 3 - 3) % args.world_size  # updated: 3*num_layers instead of 4*
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    1.1 * torch.ones(num_layers),
                    # resid lambdas. 1.1 init such that layer i weight is i^(num_layers-i).
                    *[torch.tensor([0.5, 1.0]) for _ in range(num_layers)],  # SA lambdas
                    0.1 * torch.ones(num_layers),  # bigram lambdas
                    torch.zeros(1),  # smear_lambda
                    0.5 * torch.ones(1),  # backout_lambda
                    -1.5 * torch.ones(1),  # skip_lambda -> σ(-1.5) ≈ 0.18
                    torch.ones(pad),
                ]
            )
        )
        self.scalars.label = 'scalars'

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, bigram_input_seq: Tensor,
                schedule_cfg: ForwardScheduleConfig):
        assert input_seq.ndim == 1

        # unpack schedule_cfg
        mtp_weights, train_max_seq_len = schedule_cfg.mtp_weights, schedule_cfg.train_max_seq_len
        ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long

        # set configs
        skip_connections = []
        skip_in = [3]  # long attention window on layer 3
        skip_out = [6]  # no attn op on layer 6
        x_backout = None
        backout_layer = 7

        # set lambdas
        resid_lambdas = self.scalars[: 1 * self.num_layers]
        x0_lambdas = self.x0_lambdas
        sa_lambdas = self.scalars[1 * self.num_layers: 3 * self.num_layers].view(-1, 2)
        bigram_lambdas = self.scalars[3 * self.num_layers: 4 * self.num_layers]
        smear_lambda = self.scalars[4 * self.num_layers]
        backout_lambda = self.scalars[4 * self.num_layers + 1]
        skip_lambda = self.scalars[4 * self.num_layers + 2]

        # set block masks and key shift
        bm_sizes = [ws_short, ws_short, ws_short, ws_long, ws_short, ws_short, None, ws_short, ws_short, ws_short,
                    ws_long]
        assert len(bm_sizes) == self.num_layers
        key_offset = [b == ws_long for b in bm_sizes]  # apply partial key offset to long windows

        # Embedding lookup - embed is synced from lm_head during tied phase by optimizer
        x = self.embed(input_seq)

        x0_bigram = self.bigram_embed(bigram_input_seq)[None]

        # Value embeddings - always computed (not precomputed)
        ve = self.value_embeds.view(5, self.vocab_size, -1)[:, input_seq]
        # Shifted .01 ... 234 structure on token value embeddings by @photomz
        ve = [None, ve[0], ve[1]] + [None] * (self.num_layers - 6) + [ve[2], ve[3], ve[4]]
        assert len(ve) == self.num_layers

        # smear token embed forward 1 position @classiclarryd
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        # unbind gate banks to avoid select_backwards kernel
        ag = [w.bfloat16() for w in self.attn_gate_bank.unbind(0)]
        veg = [w.bfloat16() for w in self.ve_gate_bank.unbind(0)]
        attn_gates = ag[:6] + [None] + ag[6:]
        ve_gates = [None, veg[0], veg[1]] + [None] * (self.num_layers - 6) + [veg[2], veg[3], veg[4]]
        assert len(attn_gates) == self.num_layers
        assert len(ve_gates) == self.num_layers

        # unbind weight banks to avoid select_backwards kernel
        attn_weights = self.attn_bank.unbind(0)  # tuple of [4*dim, hdim] tensors
        mlp_fcs = self.mlp_bank[:, 0, :, :].unbind(0)  # tuple of [mlp_hdim, dim] tensors
        mlp_projs = self.mlp_bank[:, 1, :, :].unbind(0)  # tuple of [mlp_hdim, dim] tensors

        for i in range(self.num_layers):
            yarn = self.yarn_paired_head if i in self.paired_head_layers else self.yarn
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=seqlens,
                bm_size=bm_sizes[i],
                yarn=yarn,
                key_offset=key_offset[i],
                attn_gate_w=attn_gates[i],
                ve_gate_w=ve_gates[i],
                train_max_seq_len=train_max_seq_len
            )
            if i in skip_out:
                skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(
                    self.skip_gate(x0[..., :self.skip_gate.weight.size(-1)]))
                x = x + skip_gate_out * skip_connections.pop()
            if i == 0:
                x = (resid_lambdas[0] + x0_lambdas[0]) * x + bigram_lambdas[0] * x0_bigram
            else:
                x = resid_lambdas[i] * x + x0_lambdas[i] * x0 + bigram_lambdas[i] * x0_bigram

            # Get weights for this layer from banks
            qkvo_w = attn_weights[self.layer_to_attn_idx[i]] if i in self.layer_to_attn_idx else None
            c_fc = mlp_fcs[self.layer_to_mlp_idx[i]] if i in self.layer_to_mlp_idx else None
            c_proj = mlp_projs[self.layer_to_mlp_idx[i]] if i in self.layer_to_mlp_idx else None

            x = self.blocks[i](x, attn_args, qkvo_w, c_fc, c_proj)
            if i in skip_in:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        # back out contributions from first 7 layers that are only required for downstream context and not direct prediction
        x -= backout_lambda * x_backout
        x = norm(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15
        # @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1). @classiclarryd updated to 23*sigmoid((logits+5)/7.5)
        if self.training:
            losses = FusedSoftcappedCrossEntropy.apply(x.view(-1, x.size(-1)), target_seq, mtp_weights,
                                                       self.lm_head.weight, self.lm_head.x_s, self.lm_head.w_s,
                                                       self.lm_head.grad_s)
            loss = losses.sum()
        else:
            logits = self.lm_head(x)
            logits = 23 * torch.sigmoid((logits + 5) / 7.5)
            logits_for_loss = logits.float()
            loss = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)), target_seq, reduction="mean")
        return loss
