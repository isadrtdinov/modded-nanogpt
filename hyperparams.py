import os
import uuid
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    # data
    data_path = os.environ.get("DATA_PATH", ".")
    train_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_train_*.bin") # input .bin to train on
    val_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_val_*.bin") # input .bin to eval validation loss on
    val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # batch sizes
    val_batch_size: int = 4 * 64 * 1024 * 8
    # schedule
    num_scheduled_iterations: int = 1490  # number of steps to complete lr and ws schedule
    num_extension_iterations: int = 40  # number of steps to continue training at final lr and ws
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 250  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint: bool = True
    checkpoint_freq: int = 500  # how often to save checkpoints
    # bigram hash embedding
    bigram_vocab_size: int = 50304 * 5
    grad_scale: float = 1.0  # in train_gpt.py grad_scale is adjusted to world size
    world_size: int = None  # world_size is overwritten in train_gpt.py
    grad_accum_steps: int = 1  # grad_accum_steps is overwritten in train_gpt.py
