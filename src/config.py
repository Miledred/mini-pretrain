from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
    # Raw text files (kept for inspection)
    train_file: str = str(PROJECT_ROOT / "data" / "train.txt")
    val_file: str = str(PROJECT_ROOT / "data" / "val.txt")

    # Tokenized binary files for real training
    train_tokens_file: str = str(PROJECT_ROOT / "data" / "train.bin")
    val_tokens_file: str = str(PROJECT_ROOT / "data" / "val.bin")
    data_meta_file: str = str(PROJECT_ROOT / "data" / "data_meta.json")

    tokenizer_name: str = "gpt2"

    # Options:
    # - "my_adamw": hand-written AdamW baseline
    # - "torch_adamw": PyTorch AdamW sanity check
    # - "my_muon": Muon for hidden 2D matrix weights + AdamW for embeddings/bias/norm
    optimizer_name: str = "my_muon"

    track_update_stats: bool = True
    update_stats_interval: int = 100

    # Model
    block_size: int = 256
    batch_size: int = 8

    train_stride: int = 256
    val_stride: int = 256

    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512

    # AdamW/default optimizer hyperparameters.
    # For my_muon, learning_rate is used as the Muon base learning rate.
    learning_rate: float = 1.5e-4
    min_lr: float = 1.5e-5
    weight_decay: float = 0.01

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # Muon hyperparameters. Used only when optimizer_name == "my_muon".
    # Muon updates are applied only to selected 2D hidden weight matrices.
    # Other parameters still use AdamW so that embeddings, bias, and LayerNorm remain stable.
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_eps: float = 1e-7
    muon_use_bfloat16: bool = True

    # Optional global multiplier for Muon updates. Keep 1.0 for the first smoke test.
    muon_update_scale: float = 1.0

    # Shape scaling for orthogonalized matrix updates.
    # "sqrt_out_in" is a common practical choice: scale by sqrt(max(1, rows / cols)).
    # "none" disables shape scaling for a more conservative ablation.
    muon_update_scale_mode: str = "sqrt_out_in"

    # Non-Muon parameters are trained by AdamW. If None, use learning_rate.
    aux_adamw_lr: float | None = None

    # Reproducibility
    seed: int = 42

    # Training length
    max_steps: int = 50000
    eval_interval: int = 500

    # Scheduler
    use_lr_scheduler: bool = True
    warmup_steps: int = 4000

    # Stability
    grad_clip_norm: float | None = 1.0

    # DataLoader
    num_workers: int = 0
    pin_memory: bool = True
    train_num_samples: int | None = None

    device: str = "cuda"

    # Experiment naming
    exp_name: str = "bigmodel"

    # Logs
    log_dir: str = "logs"

    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 5000
    save_best_only: bool = False

    resume_from: str | None = None