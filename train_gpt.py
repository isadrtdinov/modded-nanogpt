import os
import sys

# Read the current file and the kernels file code ASAP, for logging
with open(sys.argv[0], 'r') as f:
    code = f.read()
with open(os.path.join(os.path.dirname(sys.argv[0]), 'triton_kernels.py'), 'r') as f:
    code += f"\n\n{'-'*40}\n# triton_kernels.py\n{'-'*40}\n\n"
    code += f.read()

import copy
import glob
import threading
import time
from dataclasses import dataclass
from itertools import accumulate, pairwise
from pathlib import Path
import gc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import triton
import numpy as np

torch.empty(
    1, device=f"cuda:{os.environ['LOCAL_RANK']}", requires_grad=True
).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
from torch import Tensor, nn

from hyperparams import Hyperparameters
from model import next_multiple_of_n, ForwardScheduleConfig, GPT
from optim import _sparse_comms_active, sparse_comms_start, sparse_comms_share_indexes, NorMuonAndAdam

dynamo.config.recompile_limit = 64

# -----------------------------------------------------------------------------
# Distributed training setup
args = Hyperparameters()
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
args.world_size = world_size
assert 8 % world_size == 0, "world_size must be a divisor of 8"
args.grad_accum_steps = 8 // world_size
args.grad_scale = 1 / args.grad_accum_steps # consistent grad magnitudes between different num_devices
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256

class Shard:
    def __init__(self, tokens: Tensor, world_size: int = 1):
        self.tokens = tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0

        # Partial index now, full index async
        self.bos_idx = (tokens[:6_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._loader_thread = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        # Switch to full index as soon as async scan completes
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file: Path, world_size: int = 1):
        """Returns getter function for async shard loading"""
        result = {}
        ready = threading.Event()
        def load():
            tokens = _load_data_shard(file)
            result['shard'] = Shard(tokens, world_size)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result['shard']
        return get

def get_bigram_hash(x):
    """
    Computes bigram hash for each position using [prev_token, curr_token].
    Multiply by arbitary large ints to get even spread over int32 range.
    Position 0 is mapped to the reserved index (vocab_size - 1).
    BOS_tokens within the batch will hash based on last token of prior doc. Masking this ran slower and showed no improvement.
    """
    rand_int_1 = 36313
    rand_int_2 = 27191
    mod = args.bigram_vocab_size-1
    x = x.to(torch.int32)
    out = torch.empty_like(x, pin_memory=True)
    out.copy_(x)
    out[0] = mod
    out[1:] = torch.bitwise_xor(rand_int_1 * out[1:], rand_int_2 * out[:-1]) % mod
    return out

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        shard = Shard(tokens, world_size)
        next_shard_getter = Shard.load_async(next(file_iter), world_size)
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)  # median doc length is ~400

        if align_to_bos:
            try:
                seq_starts, seq_ends = shard.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                shard = next_shard_getter()
                tokens = shard.tokens
                try:
                    next_shard_getter = Shard.load_async(next(file_iter), world_size)
                except StopIteration:
                    next_shard_getter = None  # no more shards to preload
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        _bigram_inputs = get_bigram_hash(_inputs)

        new_params = yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
            _bigram_inputs.to(device="cuda", non_blocking=True),
            _bigram_inputs.numpy(),
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len

# -----------------------------------------------------------------------------
# Training Management

@dataclass
class TrainingStage:
    lr_mul: float
    batch_size: int
    window_sizes: tuple[int, int]  # (short, long) in block units
    mtp_weights_start: list[float]
    mtp_weights_end: list[float]
    train_max_seq_len: int
    duration: float = None

class TrainingSchedule:
    """
    Training schedule initialized via TRAINING_STAGES
        1. Multi Token Prediction schedule of [1, 0.5, 0.25->0] -> [1, 0.5->0] -> [1] @varunneal
        2. Sliding Attention window schedule of [1,3] -> [3,7] -> [5,11] -> [6,13]
        3. YaRN updates to RoPE on window changes
        4. Split embed and lm head at 2/3 of training
        5. Batch size schedule of 8 -> 16 -> 24
        6. Post training extension of long windows from 13 to 20
        7. Seq len updates from 896 to 2048 at 1/3 of training
    """

    def __init__(self, stages: list[TrainingStage], scheduled_iterations: int, extension_iterations: int,
                 cooldown_frac: float = 0.5, split_embed_stage: int = 2, ws_post_yarn_ext: int = 20):
        self.stages = stages
        self.scheduled_iterations = scheduled_iterations
        self.cooldown_frac = cooldown_frac
        # increase final validation ws, used for YaRN extension and short window size @classiclarryd
        self.ws_post_yarn_ext = ws_post_yarn_ext

        self.total_steps = self.scheduled_iterations + extension_iterations

        # Build stage boundaries (last is extension stage)
        ends = [0] + [round(c * scheduled_iterations) for c in accumulate(s.duration for s in stages[:-1])] + [self.total_steps]
        assert self.scheduled_iterations == ends[-2]
        self.boundaries = list(pairwise(ends))

        # Split embed at specified stage (ensure odd step for Adam)
        self.split_step = self.boundaries[split_embed_stage][0] | 1

        # Precompute MTP weights for all steps
        self.mtp_weights = []
        for step in range(self.total_steps + 1):
            stage, t = self.lookup(step)
            w = [a + (b - a) * t for a, b in zip(stage.mtp_weights_start, stage.mtp_weights_end)]
            self.mtp_weights.append(torch.tensor(w, device=device))

    def lookup(self, step: int) -> tuple[TrainingStage, float]:
        # Returns stage and % of the way through that stage
        for i, (start, end) in enumerate(self.boundaries):
            if step < end:
                t = (step - start) / (end - start)
                return self.stages[i], t
        return self.stages[-1], 1.0

    def get_lr(self, step: int) -> float:
        # learning rate schedule: tied to batch size schedule, with cooldown at the end
        stage, _ = self.lookup(step)
        lr = stage.lr_mul
        cd_start = int(self.scheduled_iterations * (1 - self.cooldown_frac))
        if step >= cd_start:
            t = min(1.0, (step - cd_start) / (self.scheduled_iterations - cd_start))
            lr = lr * (1 - t) + 0.15 * t
        return lr

# window_sizes are in units of `block_size` tokens (defined in TrainingManager)
TRAINING_STAGES = [
    TrainingStage(duration=1/3, train_max_seq_len=896, batch_size=8 * 2048 * 8, window_sizes=(1, 3), lr_mul=1.0,
                  mtp_weights_start=[1.0, 0.5, 0.25], mtp_weights_end=[1.0, 0.5, 0.0]),
    TrainingStage(duration=1/3, train_max_seq_len=2048, batch_size=16 * 2048 * 8, window_sizes=(3, 7), lr_mul=1.52,  # (16/8)**0.6
                  mtp_weights_start=[1.0, 0.5], mtp_weights_end=[1.0, 0.0]),
    TrainingStage(duration=1/3, train_max_seq_len=2048, batch_size=24 * 2048 * 8, window_sizes=(5, 11), lr_mul=1.73,  # (24/8)**0.5
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0]),
    # extension stage
    TrainingStage(train_max_seq_len=2048, batch_size=24 * 2048 * 8, window_sizes=(6, 13), lr_mul=1.0,  # lr_mul is not used
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0]),
]

training_schedule = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations, cooldown_frac=0.55)

def get_muon_momentum(step: int, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    # warmup phase: linearly increase momentum from min to max
    # cooldown phase: linearly decrease momentum from max to min
    momentum_cd_start = training_schedule.total_steps - muon_cooldown_steps
    if step < muon_warmup_steps:
        frac = step / muon_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / muon_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:
        momentum = momentum_max
    return momentum

class TrainingManager():
    """
    Manages the NorMuonAndAdam for all parameters with explicit ordering.
        1. Scalars are given higher momentum terms to smooth learning @ChrisJMcCormick
        2. Adam optimizers are only stepped on odd steps @classiclarryd
        3. Explicit scatter_order and work_order for communication scheduling (no backward hooks)
        4. Muon has a linear momentum warmup and cooldown schedule
        5. Learning rates follow a linear decay schedule
        6. Embed is tied to lm_head until split step (2/3 of training), then untied @classiclarryd
    """
    def __init__(self, model):
        self.model = model
        self.block_size = 128

        # - Ordering dictates when to launch reduce/reduce_scatter operations
        # - "sharded" parameters use reduce_scatter/all_gather and "replicated" ones use all_reduce
        # - lr_mul and wd_mul are per-parameter learning rate and weight decay multipliers
        # self.param_table = {
        #     "attn":           {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
        #     "mlp":            {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
        #     "scalars":        {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 5.0,  "wd_mul": 0.0},
        #     "smear_gate":     {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.01, "wd_mul": 0.0},
        #     "skip_gate":      {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.05, "wd_mul": 0.0},
        #     "attn_gate_bank": {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99]},
        #     "ve_gate_bank":   {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99]},
        #     "x0_lambdas":     {"optim": "adam",    "comms": "replicated", "adam_betas": [0.65, 0.95], "lr_mul": 5.0,  "wd_mul": 0.0},
        #     "bigram_embed":   {"optim": "adam",    "comms": "sharded_sparse",    "adam_betas": [0.75, 0.95], "lr_mul": 75.,  "wd_mul": 5.0},
        #     "lm_head":        {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
        #     "value_embed":    {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.75, 0.95], "lr_mul": 75.,  "wd_mul": 5.0},
        #     "embed":          {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
        # }
        self.param_table = {
            "attn":           {"optim": "adam", "comms": "sharded",    "adam_betas": [0.9,  0.95]},
            "mlp":            {"optim": "adam", "comms": "sharded",    "adam_betas": [0.9,  0.95]},
            "scalars":        {"optim": "adam", "comms": "replicated", "adam_betas": [0.9,  0.95]},
            "smear_gate":     {"optim": "adam", "comms": "replicated", "adam_betas": [0.9,  0.95]},
            "skip_gate":      {"optim": "adam",  "comms": "replicated", "adam_betas": [0.9,  0.95]},
            "attn_gate_bank": {"optim": "adam",  "comms": "replicated", "adam_betas": [0.9,  0.95]},
            "ve_gate_bank":   {"optim": "adam",  "comms": "replicated", "adam_betas": [0.9,  0.95]},
            "x0_lambdas":     {"optim": "adam",  "comms": "replicated", "adam_betas": [0.9, 0.95]},
            "bigram_embed":   {"optim": "adam",  "comms": "sharded_sparse",    "adam_betas": [0.9, 0.95]},
            "lm_head":        {"optim": "adam",  "comms": "sharded",    "adam_betas": [0.9,  0.95]},
            "value_embed":    {"optim": "adam",  "comms": "sharded",    "adam_betas": [0.9, 0.95]},
            "embed":          {"optim": "adam",  "comms": "sharded",    "adam_betas": [0.9,  0.95]},
        }

        # - Process smaller/faster params first while large reduces complete
        # - lm_head must complete before embed sync (when tied)
        self.work_order = [
            "scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank", "x0_lambdas",  # Small, fast
            "lm_head",
            "bigram_embed",  # Medium
            "value_embed",
            "embed",   # lm_head must complete before embed sync (when tied)
            "attn", "mlp",        # Large, polar express - process last to maximize overlap
        ]

        adam_defaults = dict(
            lr=0.008,
            eps=1e-10,
            weight_decay=0.005,
        )

        normuon_defaults = dict(
            lr=0.023,
            momentum=0.95,
            beta2=0.95,
            weight_decay=1.2,
        )

        self.optimizer = NorMuonAndAdam(
            model.named_parameters(),
            dist=dist,
            param_table=self.param_table,
            scatter_order=list(self.param_table.keys()),  # Dict order defines scatter priority
            work_order=self.work_order,
            adam_defaults=adam_defaults,
            normuon_defaults=normuon_defaults,
        )

        # Split embed from lm_head at 2/3 of training (on an odd step so Adam updates)
        self.split_step = training_schedule.split_step

        self.reset()

    def apply_final_ws_ext(self):
        self.ws_long = training_schedule.ws_post_yarn_ext

    def get_forward_args(self):
        return ForwardScheduleConfig(
            mtp_weights = self.mtp_weights,
            ws_short = self.ws_short * self.block_size,
            ws_long = self.ws_long * self.block_size,
            train_max_seq_len = self.train_max_seq_len
        )

    def _is_adam_step(self, step: int):
        """Adam params are only updated on odd steps."""
        return step % 2 == 1

    def get_transition_steps(self):
        return [start for start, _ in training_schedule.boundaries[1:]]

    def advance_schedule(self, step: int):
        stage, _ = training_schedule.lookup(step)
        self.ws_short, new_ws_long = stage.window_sizes
        if new_ws_long != self.ws_long:
            self.model.yarn.apply(self.ws_long * self.block_size, new_ws_long * self.block_size)
            self.model.yarn_paired_head.apply(self.ws_long * self.block_size, new_ws_long * self.block_size)

        new_batch_size = stage.batch_size
        new_train_max_seq_len = stage.train_max_seq_len
        if new_batch_size != self.batch_size or new_train_max_seq_len != self.train_max_seq_len:
            self.train_loader_send_args = (new_batch_size, new_train_max_seq_len, args.grad_accum_steps)
            self.batch_size = new_batch_size
            self.train_max_seq_len = new_train_max_seq_len
        else:
            self.train_loader_send_args = None

        self.ws_long = new_ws_long
        self.mtp_weights = training_schedule.mtp_weights[step]

    def step_optimizers(self, step: int):
        step_lr = training_schedule.get_lr(step)
        muon_momentum = get_muon_momentum(step)
        do_adam = self._is_adam_step(step)

        # Update learning rates and momentum for all params
        for param, p_cfg in self.optimizer.param_cfgs.items():
            p_cfg.lr = p_cfg.initial_lr * step_lr
            if p_cfg.optim == "normuon":
                p_cfg.momentum = muon_momentum

        # Step optimizer with do_adam flag
        self.optimizer.step(do_adam=do_adam)

        # At split step: copy lm_head optimizer state to embed and mark as split
        if step == self.split_step:
            self.optimizer.copy_lm_state_to_embed()

    def reset(self, state=None):
        if state is not None:
            self.optimizer.load_state_dict(state)

        # Reset NorMuon momentum buffers and split_embed state
        self.optimizer.reset()

        stage, _ = training_schedule.lookup(0)
        self.ws_short, self.ws_long = stage.window_sizes
        self.batch_size = stage.batch_size
        self.train_max_seq_len = stage.train_max_seq_len
        self.model.yarn.reset()
        self.model.yarn_paired_head.reset()
        if _sparse_comms_active(world_size):
            self.row_update_mask = np.zeros(args.bigram_vocab_size, dtype=np.uint8)
            self.sparse_counts_state = None
            # buffer we use for fast GPU uploads of send indexes
            self.send_idxes_buffer = torch.empty(args.bigram_vocab_size, dtype=torch.int32, pin_memory=True)


    def get_state(self):
        return copy.deepcopy(self.optimizer.state_dict())

    def sparse_index_update(self, step, bigram_indexes):
        if not _sparse_comms_active(world_size):
            return

        self.row_update_mask[bigram_indexes] = 1

        if self._is_adam_step(step):
            with torch.no_grad():
                bigram_idx_np = np.flatnonzero(self.row_update_mask).astype(np.int32)
                send_idxes, send_counts, recv_counts, recv_counts_fut = sparse_comms_start(
                    bigram_idx_np, args.bigram_vocab_size, rank, world_size, self.send_idxes_buffer,
                    dist, device
                )
                self.sparse_counts_state = (send_idxes, send_counts, recv_counts, recv_counts_fut)

    def sparse_index_share(self, step):
        if not _sparse_comms_active(world_size) or not self._is_adam_step(step):
            return

        send_idxes, send_counts, recv_counts, recv_counts_fut = self.sparse_counts_state
        self.sparse_counts_state = None

        recv_counts_fut.wait()
        recv_idxes, sparse_state, idxes_fut = sparse_comms_share_indexes(send_idxes, send_counts, recv_counts, dist, device)
        self.optimizer._reduce_futures[model.bigram_embed.weight] = [idxes_fut, recv_idxes]
        self.optimizer._sparse_async_data[model.bigram_embed.weight] = sparse_state

        self.row_update_mask.fill(0)


        

# -----------------------------------------------------------------------------
# int main

# begin logging
logfile = None
if master_process:
    run_id = args.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running Triton version {triton.__version__}")

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=11,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=args.val_batch_size // (args.grad_accum_steps * world_size),
    args=args
).cuda()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.weight.data = m.weight.data.bfloat16()
model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()
model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()
model.attn_bank.data = model.attn_bank.data.bfloat16()
model.mlp_bank.data = model.mlp_bank.data.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)
training_manager = TrainingManager(model)


########################################
#            Warmup kernels            #
########################################
print0("Compiling model and warming up kernels (~7 minutes on first execution)", console=True)
# Warmup the training kernels, then re-initialize the state so we aren't cheating
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizer=training_manager.get_state()) # save the initial state
train_loader = distributed_data_generator(args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=args.grad_accum_steps)
val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=args.grad_accum_steps, align_to_bos=False)

transition_steps = training_manager.get_transition_steps()
# first and last pair of steps in each transition
warmup_steps = sorted({0, 1 } | set(s + offset for s in transition_steps for offset in [-2, -1, 0, 1] if s + offset >= 0))
print0(f"Sampling steps {warmup_steps} for warmup", console=True)
for step in warmup_steps:
    training_manager.advance_schedule(step)
    model.eval()
    with torch.no_grad():
        inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
        model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args())
    model.train()
    for idx in range(args.grad_accum_steps):
        send_args = training_manager.train_loader_send_args
        inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(send_args)
        training_manager.sparse_index_update(step, bigram_cpu)
        loss = model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()) * args.grad_scale
        training_manager.sparse_index_share(step)
        loss.backward()
        del loss
    training_manager.step_optimizers(step)
print0("Resetting Model", console=True)
model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
training_manager.reset(initial_state["optimizer"])
del val_loader, train_loader, initial_state
model.train()

########################################
#        Training and validation       #
########################################
train_loader = distributed_data_generator(args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=args.grad_accum_steps)

gc.collect()

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = training_schedule.total_steps
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    training_manager.advance_schedule(step)
    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            training_manager.apply_final_ws_ext()
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = args.grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=args.grad_accum_steps, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
                val_loss += model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args())
        val_loss /= val_steps
        del val_loader
        dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step or step % args.checkpoint_freq == 0:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizer=training_manager.get_state())
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        if last_step:
            break

    # --------------- TRAINING SECTION -----------------
    for idx in range(args.grad_accum_steps):
        inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(training_manager.train_loader_send_args)
        training_manager.sparse_index_update(step, bigram_cpu)
        loss = model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()) * args.grad_scale
        training_manager.sparse_index_share(step)
        loss.backward()
        del loss
    training_manager.step_optimizers(step)

    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
