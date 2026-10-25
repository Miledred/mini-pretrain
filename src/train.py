from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter


# Allow direct execution with:
# python .\src\train.py
if __package__ is None or __package__ == "":
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


from src.config import TrainConfig
from src.dataset import MemMapTokenBlockDataset, load_data_meta
from src.model_factory import build_model
from src.optim_factory import build_optimizer
from src.tokenizer import build_tokenizer
from src.trainer import Trainer
from src.utils import set_seed


def str2bool(value):
    """
    Convert command-line boolean strings to Python bool values.

    Supported values:
        true, false
        1, 0
        yes, no
        y, n
        on, off
    """
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in {"true", "1", "yes", "y", "on"}:
        return True

    if value in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}"
    )


def parse_args():
    """
    Parse command-line arguments.

    Every argument defaults to None. A None value means that the
    corresponding value from config.py will be used. A non-None
    command-line value overrides config.py.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train mini_pretrain with command-line configuration."
        )
    )

    # ============================================================
    # Model architecture
    # ============================================================
    parser.add_argument(
        "--n-layer",
        type=int,
        default=None,
        help="Number of Transformer layers.",
    )

    parser.add_argument(
        "--n-head",
        type=int,
        default=None,
        help="Number of attention heads.",
    )

    parser.add_argument(
        "--n-embd",
        type=int,
        default=None,
        help="Embedding dimension.",
    )

    # ============================================================
    # Dataset and batch configuration
    # ============================================================
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size.",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Sequence length.",
    )

    parser.add_argument(
        "--train-stride",
        type=int,
        default=None,
        help="Stride used to construct training windows.",
    )

    parser.add_argument(
        "--val-stride",
        type=int,
        default=None,
        help="Stride used to construct validation windows.",
    )

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help=(
            "Number of sampled training examples. "
            "Defaults to max_steps * batch_size."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader worker processes.",
    )

    parser.add_argument(
        "--pin-memory",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Whether DataLoader should use pinned memory.",
    )

    # ============================================================
    # Common optimizer options
    # ============================================================
    parser.add_argument(
        "--optimizer-name",
        type=str,
        default=None,
        help=(
            "Optimizer name, such as my_adamw, "
            "torch_adamw, or my_muon."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Peak learning rate. When Muon is used, this is "
            "the learning rate for Muon matrix parameters."
        ),
    )

    parser.add_argument(
        "--aux-adamw-lr",
        type=float,
        default=None,
        help=(
            "AdamW learning rate for auxiliary parameters "
            "when Muon is used."
        ),
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help="Minimum learning rate after cosine decay.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay coefficient.",
    )

    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=None,
        help="AdamW beta1 value.",
    )

    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=None,
        help="AdamW beta2 value.",
    )

    parser.add_argument(
        "--adam-eps",
        type=float,
        default=None,
        help="AdamW epsilon value.",
    )

    # ============================================================
    # Muon-specific options
    # ============================================================
    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=None,
        help="Muon momentum value.",
    )

    parser.add_argument(
        "--muon-nesterov",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Whether Muon should use Nesterov momentum.",
    )

    parser.add_argument(
        "--muon-ns-steps",
        type=int,
        default=None,
        help="Number of Newton-Schulz iterations.",
    )

    parser.add_argument(
        "--muon-update-scale",
        type=float,
        default=None,
        help="Base scaling factor applied to Muon updates.",
    )

    parser.add_argument(
        "--muon-update-scale-mode",
        type=str,
        default=None,
        help=(
            "Muon update scaling mode, such as sqrt_out_in."
        ),
    )

    # ============================================================
    # Training process
    # ============================================================
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of training steps.",
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Validation interval in training steps.",
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of linear warmup steps.",
    )

    parser.add_argument(
        "--use-lr-scheduler",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Whether to use warmup followed by cosine decay.",
    )

    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Gradient clipping norm.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed used for model initialization "
            "and training-data sampling."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device, such as cuda or cpu.",
    )

    # ============================================================
    # Update statistics
    # ============================================================
    parser.add_argument(
        "--track-update-stats",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Whether to calculate parameter-update statistics.",
    )

    parser.add_argument(
        "--update-stats-interval",
        type=int,
        default=None,
        help="Interval for expensive update-stat calculations.",
    )

    # ============================================================
    # Logging and checkpoint configuration
    # ============================================================
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name appended to the run directory.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard and CSV output directory.",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Checkpoint output directory.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Checkpoint saving interval.",
    )

    parser.add_argument(
        "--save-best-only",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Whether to save only the best checkpoint.",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint path used to resume training.",
    )

    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    """
    Override config.py values with command-line values.
    """
    names = [
        # Model
        "n_layer",
        "n_head",
        "n_embd",

        # Data
        "batch_size",
        "block_size",
        "train_stride",
        "val_stride",
        "train_num_samples",
        "num_workers",
        "pin_memory",

        # Optimizer
        "optimizer_name",
        "learning_rate",
        "aux_adamw_lr",
        "min_lr",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",

        # Muon
        "muon_momentum",
        "muon_nesterov",
        "muon_ns_steps",
        "muon_update_scale",
        "muon_update_scale_mode",

        # Training
        "max_steps",
        "eval_interval",
        "warmup_steps",
        "use_lr_scheduler",
        "grad_clip_norm",
        "seed",
        "device",

        # Statistics
        "track_update_stats",
        "update_stats_interval",

        # Output
        "exp_name",
        "log_dir",
        "save_dir",
        "save_every",
        "save_best_only",
        "resume_from",
    ]

    for name in names:
        value = getattr(args, name)

        if value is not None:
            setattr(cfg, name, value)

    # Keep compatibility with an older config.py that does not
    # contain the newer fields.
    if not hasattr(cfg, "seed"):
        cfg.seed = 42

    if not hasattr(cfg, "aux_adamw_lr"):
        cfg.aux_adamw_lr = cfg.learning_rate

    if not hasattr(cfg, "muon_momentum"):
        cfg.muon_momentum = 0.95

    if not hasattr(cfg, "muon_nesterov"):
        cfg.muon_nesterov = True

    if not hasattr(cfg, "muon_ns_steps"):
        cfg.muon_ns_steps = 5

    if not hasattr(cfg, "muon_update_scale"):
        cfg.muon_update_scale = 1.0

    if not hasattr(cfg, "muon_update_scale_mode"):
        cfg.muon_update_scale_mode = "sqrt_out_in"


def validate_config(cfg):
    """
    Validate important configuration values before training starts.
    """
    if cfg.n_layer <= 0:
        raise ValueError("n_layer must be positive.")

    if cfg.n_head <= 0:
        raise ValueError("n_head must be positive.")

    if cfg.n_embd <= 0:
        raise ValueError("n_embd must be positive.")

    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError(
            f"n_embd={cfg.n_embd} must be divisible "
            f"by n_head={cfg.n_head}."
        )

    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if cfg.block_size <= 0:
        raise ValueError("block_size must be positive.")

    if cfg.train_stride <= 0:
        raise ValueError("train_stride must be positive.")

    if cfg.val_stride <= 0:
        raise ValueError("val_stride must be positive.")

    if cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if cfg.min_lr < 0:
        raise ValueError("min_lr cannot be negative.")

    if cfg.min_lr > cfg.learning_rate:
        raise ValueError(
            "min_lr cannot be larger than learning_rate."
        )

    if cfg.max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    if cfg.eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")

    if cfg.save_every <= 0:
        raise ValueError("save_every must be positive.")

    if cfg.warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")

    if cfg.warmup_steps > cfg.max_steps:
        raise ValueError(
            "warmup_steps cannot be larger than max_steps."
        )


def build_run_name(cfg):
    """
    Build the log and checkpoint directory name.
    """
    return (
        f"{cfg.optimizer_name}"
        f"_L{cfg.n_layer}"
        f"_H{cfg.n_head}"
        f"_D{cfg.n_embd}"
        f"_bs{cfg.batch_size}"
        f"_blk{cfg.block_size}"
        f"_ts{cfg.train_stride}"
        f"_{cfg.exp_name}"
    )


def build_scheduler(optimizer, cfg):
    """
    Build a linear-warmup plus cosine-decay scheduler.

    LambdaLR applies the same multiplicative factor to every
    optimizer parameter group. Therefore, Muon matrix parameters
    and auxiliary AdamW parameters preserve the ratio between
    their initial learning rates.
    """
    if not cfg.use_lr_scheduler:
        return None

    min_lr_ratio = cfg.min_lr / cfg.learning_rate

    def lr_lambda(current_step):
        # Linear warmup phase.
        if current_step < cfg.warmup_steps:
            return float(current_step + 1) / float(
                max(1, cfg.warmup_steps)
            )

        # Cosine-decay phase.
        progress = (
            current_step - cfg.warmup_steps
        ) / float(
            max(
                1,
                cfg.max_steps - cfg.warmup_steps,
            )
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return (
            min_lr_ratio
            + (1.0 - min_lr_ratio) * cosine
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    return scheduler


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    step,
    best_val_loss,
    cfg,
):
    """
    Save model, optimizer, scheduler, step, and configuration.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "step": step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg.__dict__,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = (
            scheduler.state_dict()
        )

    torch.save(
        checkpoint,
        path,
    )


def move_to_device(value, device):
    """
    Recursively move optimizer-state values to the target device.

    This supports both AdamW state tensors and Muon matrix states.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device)

    if isinstance(value, dict):
        return {
            key: move_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            move_to_device(item, device)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            move_to_device(item, device)
            for item in value
        )

    return value


def load_checkpoint(
    path,
    model,
    optimizer,
    device,
    scheduler=None,
):
    """
    Restore training state from a checkpoint.
    """
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # Move optimizer-state tensors to the active device.
    for parameter, state in optimizer.state.items():
        optimizer.state[parameter] = move_to_device(
            state,
            device,
        )

    step = checkpoint.get(
        "step",
        0,
    )

    best_val_loss = checkpoint.get(
        "best_val_loss",
        float("inf"),
    )

    if scheduler is not None:
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )
        else:
            # Support older checkpoints that did not save the
            # scheduler state.
            for _ in range(step):
                scheduler.step()

    return step, best_val_loss


def open_csv_logger(
    csv_path,
    append,
):
    """
    Create or resume metrics.csv.
    """
    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mode = "a" if append else "w"

    csv_file = open(
        csv_path,
        mode,
        newline="",
        encoding="utf-8",
    )

    fieldnames = [
        "step",
        "tokens_seen",
        "train_loss",
        "val_loss",
        "val_ppl",
        "grad_norm",
        "param_norm",
        "update_norm",
        "num_updated_params",
        "update_ratio",
        "lr",
        "step_time_sec",
        "tokens_per_sec",
        "max_memory_mb",
    ]

    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    if (
        not append
        or csv_path.stat().st_size == 0
    ):
        csv_writer.writeheader()
        csv_file.flush()

    return csv_file, csv_writer


def print_configuration(
    cfg,
    run_name,
):
    """
    Print important experiment settings before training starts.
    """
    print("=" * 72)
    print("Experiment configuration")
    print(f"Random seed:       {cfg.seed}")
    print(f"Run name:          {run_name}")
    print(f"Optimizer:         {cfg.optimizer_name}")
    print(
        f"Model:             "
        f"L{cfg.n_layer}, "
        f"H{cfg.n_head}, "
        f"D{cfg.n_embd}"
    )
    print(f"Batch size:        {cfg.batch_size}")
    print(f"Block size:        {cfg.block_size}")
    print(f"Learning rate:     {cfg.learning_rate}")
    print(f"Minimum LR:        {cfg.min_lr}")
    print(f"Max steps:         {cfg.max_steps}")
    print(f"Warmup steps:      {cfg.warmup_steps}")
    print(f"Eval interval:     {cfg.eval_interval}")

    if cfg.optimizer_name == "my_muon":
        print(
            f"Aux AdamW LR:      "
            f"{cfg.aux_adamw_lr}"
        )
        print(
            f"Muon momentum:     "
            f"{cfg.muon_momentum}"
        )
        print(
            f"Muon Nesterov:     "
            f"{cfg.muon_nesterov}"
        )
        print(
            f"Muon NS steps:     "
            f"{cfg.muon_ns_steps}"
        )
        print(
            f"Muon update scale: "
            f"{cfg.muon_update_scale}"
        )
        print(
            f"Muon scale mode:   "
            f"{cfg.muon_update_scale_mode}"
        )

    print("=" * 72)


def main():
    # ============================================================
    # 1. Load config.py and apply command-line overrides.
    # ============================================================
    cfg = TrainConfig()

    args = parse_args()

    apply_cli_overrides(
        cfg,
        args,
    )

    validate_config(cfg)

    # ============================================================
    # 2. Set the random seed.
    #
    # This must be done before model initialization and before
    # RandomSampler and DataLoader objects are created.
    # ============================================================
    set_seed(cfg.seed)

    run_name = build_run_name(cfg)

    print_configuration(
        cfg,
        run_name,
    )

    # ============================================================
    # 3. Build tokenizer and datasets.
    # ============================================================
    tokenizer = build_tokenizer(
        cfg.tokenizer_name
    )

    data_meta = load_data_meta(
        cfg.data_meta_file
    )

    if (
        data_meta["tokenizer_name"]
        != cfg.tokenizer_name
    ):
        raise ValueError(
            f"Tokenizer mismatch: "
            f"config uses {cfg.tokenizer_name}, "
            f"but data was built with "
            f"{data_meta['tokenizer_name']}"
        )

    train_dataset = MemMapTokenBlockDataset(
        token_file=cfg.train_tokens_file,
        dtype_name=data_meta["train"]["dtype"],
        block_size=cfg.block_size,
        stride=cfg.train_stride,
    )

    val_dataset = MemMapTokenBlockDataset(
        token_file=cfg.val_tokens_file,
        dtype_name=data_meta["val"]["dtype"],
        block_size=cfg.block_size,
        stride=cfg.val_stride,
    )

    train_num_samples = cfg.train_num_samples

    if train_num_samples is None:
        train_num_samples = (
            cfg.max_steps
            * cfg.batch_size
        )

    if len(train_dataset) == 0:
        raise ValueError(
            "Train dataset is empty. "
            "Check train.bin, block_size, "
            "and train_stride."
        )

    if len(val_dataset) == 0:
        print(
            "Warning: validation dataset is empty."
        )

    train_sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=train_num_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(
            torch.cuda.is_available()
            and cfg.pin_memory
        ),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(
            torch.cuda.is_available()
            and cfg.pin_memory
        ),
        drop_last=False,
    )

    # ============================================================
    # 4. Build model, optimizer, scheduler, and trainer.
    # ============================================================
    model = build_model(
        cfg,
        tokenizer,
    )

    optimizer = build_optimizer(
        model,
        cfg,
    )

    scheduler = build_scheduler(
        optimizer,
        cfg,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        scheduler=scheduler,
    )

    num_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Number of parameters: {num_params}"
    )
    print(
        f"Train windows: {len(train_dataset)}"
    )
    print(
        f"Val windows: {len(val_dataset)}"
    )
    print(
        f"Train tokens: "
        f"{data_meta['train']['num_tokens']}"
    )
    print(
        f"Val tokens: "
        f"{data_meta['val']['num_tokens']}"
    )
    print(
        f"Train token file: "
        f"{cfg.train_tokens_file}"
    )
    print(
        f"Val token file:   "
        f"{cfg.val_tokens_file}"
    )

    # ============================================================
    # 5. Create experiment directories.
    # ============================================================
    run_dir = (
        Path(cfg.log_dir)
        / run_name
    )

    checkpoint_dir = (
        Path(cfg.save_dir)
        / run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the full resolved configuration, including the seed.
    with open(
        run_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            cfg.__dict__,
            config_file,
            indent=2,
            default=str,
        )

    writer = SummaryWriter(
        log_dir=str(run_dir)
    )

    csv_path = (
        run_dir
        / "metrics.csv"
    )

    append_csv = (
        cfg.resume_from is not None
        and csv_path.exists()
    )

    csv_file, csv_writer = open_csv_logger(
        csv_path,
        append=append_csv,
    )

    step = 0

    best_val_loss = float("inf")

    # ============================================================
    # 6. Restore a checkpoint when requested.
    # ============================================================
    if cfg.resume_from is not None:
        resume_path = Path(
            cfg.resume_from
        )

        print(
            f"Resuming from checkpoint: "
            f"{resume_path}"
        )

        step, best_val_loss = load_checkpoint(
            resume_path,
            trainer.model,
            trainer.optimizer,
            trainer.device,
            trainer.scheduler,
        )

        # Keep the Trainer's internal step synchronized with the
        # restored outer training step.
        trainer.global_step = step

        print(
            f"Resumed at step={step}, "
            f"best_val_loss="
            f"{best_val_loss:.4f}"
        )

    print("Start training...")

    # ============================================================
    # 7. Main training loop.
    # ============================================================
    try:
        while step < cfg.max_steps:
            for batch in train_loader:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                start_time = time.perf_counter()

                metrics = trainer.train_step(
                    batch
                )

                step_time = (
                    time.perf_counter()
                    - start_time
                )

                step += 1

                tokens_this_step = (
                    cfg.batch_size
                    * cfg.block_size
                )

                tokens_seen = (
                    step
                    * tokens_this_step
                )

                if torch.cuda.is_available():
                    max_memory_mb = (
                        torch.cuda.max_memory_allocated()
                        / 1024
                        / 1024
                    )
                else:
                    max_memory_mb = 0.0

                tokens_per_sec = (
                    tokens_this_step
                    / max(
                        step_time,
                        1e-12,
                    )
                )

                val_loss = None
                val_ppl = None
                improved = False

                # =================================================
                # Run validation.
                # =================================================
                if (
                    step
                    % cfg.eval_interval
                    == 0
                ):
                    val_loss = trainer.evaluate()

                    if val_loss is not None:
                        val_ppl = math.exp(
                            min(
                                val_loss,
                                20.0,
                            )
                        )

                        if (
                            val_loss
                            < best_val_loss
                        ):
                            best_val_loss = val_loss

                            improved = True

                            save_checkpoint(
                                checkpoint_dir
                                / "best.pt",
                                trainer.model,
                                trainer.optimizer,
                                trainer.scheduler,
                                step,
                                best_val_loss,
                                cfg,
                            )

                    save_checkpoint(
                        checkpoint_dir
                        / "latest.pt",
                        trainer.model,
                        trainer.optimizer,
                        trainer.scheduler,
                        step,
                        best_val_loss,
                        cfg,
                    )

                # =================================================
                # Save periodic checkpoints.
                # =================================================
                if (
                    step
                    % cfg.save_every
                    == 0
                ):
                    if not cfg.save_best_only:
                        save_checkpoint(
                            checkpoint_dir
                            / f"step_{step}.pt",
                            trainer.model,
                            trainer.optimizer,
                            trainer.scheduler,
                            step,
                            best_val_loss,
                            cfg,
                        )

                    save_checkpoint(
                        checkpoint_dir
                        / "latest.pt",
                        trainer.model,
                        trainer.optimizer,
                        trainer.scheduler,
                        step,
                        best_val_loss,
                        cfg,
                    )

                # =================================================
                # Print training metrics.
                # =================================================
                message = (
                    f"step={step}, "
                    f"train_loss="
                    f"{metrics['loss']:.4f}, "
                    f"grad_norm="
                    f"{metrics['grad_norm']:.4f}, "
                    f"param_norm="
                    f"{metrics['param_norm']:.4f}, "
                    f"update_norm="
                    f"{metrics['update_norm']:.4f}, "
                    f"updated_params="
                    f"{metrics['num_updated_params']}, "
                    f"update_ratio="
                    f"{metrics['update_ratio']:.6f}, "
                    f"lr="
                    f"{metrics['lr']:.6e}, "
                    f"step_time="
                    f"{step_time:.4f}s, "
                    f"tok_s="
                    f"{tokens_per_sec:.2f}, "
                    f"max_mem_mb="
                    f"{max_memory_mb:.2f}"
                )

                if val_loss is not None:
                    message += (
                        f", val_loss="
                        f"{val_loss:.4f}, "
                        f"val_ppl="
                        f"{val_ppl:.2f}"
                    )

                    if improved:
                        message += " [best]"

                print(message)

                # =================================================
                # Write TensorBoard metrics.
                # =================================================
                writer.add_scalar(
                    "train/loss",
                    metrics["loss"],
                    step,
                )

                writer.add_scalar(
                    "train/grad_norm",
                    metrics["grad_norm"],
                    step,
                )

                writer.add_scalar(
                    "train/param_norm",
                    metrics["param_norm"],
                    step,
                )

                writer.add_scalar(
                    "train/update_norm",
                    metrics["update_norm"],
                    step,
                )

                writer.add_scalar(
                    "train/update_ratio",
                    metrics["update_ratio"],
                    step,
                )

                writer.add_scalar(
                    "train/lr",
                    metrics["lr"],
                    step,
                )

                writer.add_scalar(
                    "system/step_time_sec",
                    step_time,
                    step,
                )

                writer.add_scalar(
                    "system/tokens_per_sec",
                    tokens_per_sec,
                    step,
                )

                writer.add_scalar(
                    "system/max_memory_mb",
                    max_memory_mb,
                    step,
                )

                if val_loss is not None:
                    writer.add_scalar(
                        "val/loss",
                        val_loss,
                        step,
                    )

                    writer.add_scalar(
                        "val/ppl",
                        val_ppl,
                        step,
                    )

                    writer.add_scalar(
                        "val/best_loss",
                        best_val_loss,
                        step,
                    )

                writer.flush()

                # =================================================
                # Write metrics.csv.
                # =================================================
                row = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": metrics["loss"],
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "grad_norm": metrics["grad_norm"],
                    "param_norm": metrics["param_norm"],
                    "update_norm": metrics["update_norm"],
                    "num_updated_params": (
                        metrics["num_updated_params"]
                    ),
                    "update_ratio": (
                        metrics["update_ratio"]
                    ),
                    "lr": metrics["lr"],
                    "step_time_sec": step_time,
                    "tokens_per_sec": tokens_per_sec,
                    "max_memory_mb": max_memory_mb,
                }

                csv_writer.writerow(row)
                csv_file.flush()

                if step >= cfg.max_steps:
                    break

        # Save the final state after the training loop exits.
        save_checkpoint(
            checkpoint_dir
            / "latest.pt",
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            step,
            best_val_loss,
            cfg,
        )

    finally:
        writer.close()
        csv_file.close()

    print("Training finished.")
    print(
        f"Run dir:           "
        f"{run_dir}"
    )
    print(
        f"Checkpoint dir:    "
        f"{checkpoint_dir}"
    )
    print(
        f"Latest checkpoint: "
        f"{checkpoint_dir / 'latest.pt'}"
    )
    print(
        f"Best checkpoint:   "
        f"{checkpoint_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()