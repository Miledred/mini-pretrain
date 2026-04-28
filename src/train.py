import os
import sys
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.config import TrainConfig
from src.tokenizer import build_tokenizer
from src.dataset import load_data_meta, MemMapTokenBlockDataset
from src.model_factory import build_model
from src.optim_factory import build_optimizer
from src.trainer import Trainer
from src.utils import set_seed


def build_run_name(cfg: TrainConfig) -> str:
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


def build_scheduler(optimizer, cfg: TrainConfig):
    if not cfg.use_lr_scheduler:
        return None

    min_lr_ratio = cfg.min_lr / cfg.learning_rate

    def lr_lambda(current_step: int):
        if current_step < cfg.warmup_steps:
            return float(current_step + 1) / float(max(1, cfg.warmup_steps))

        progress = (current_step - cfg.warmup_steps) / float(
            max(1, cfg.max_steps - cfg.warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    step: int,
    best_val_loss: float,
    cfg: TrainConfig,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg.__dict__,
    }

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(ckpt, path)


def load_checkpoint(path: str | Path, model, optimizer, device, scheduler=None):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

    step = ckpt.get("step", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    if scheduler is not None:
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(step):
                scheduler.step()

    return step, best_val_loss


def open_csv_logger(csv_path: Path, append: bool):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    csv_file = open(csv_path, mode, newline="", encoding="utf-8")

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

    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    if (not append) or csv_path.stat().st_size == 0:
        csv_writer.writeheader()
        csv_file.flush()

    return csv_file, csv_writer


def main():
    cfg = TrainConfig()

    set_seed(42)

    tokenizer = build_tokenizer(cfg.tokenizer_name)
    data_meta = load_data_meta(cfg.data_meta_file)

    if data_meta["tokenizer_name"] != cfg.tokenizer_name:
        raise ValueError(
            f"Tokenizer mismatch: config uses {cfg.tokenizer_name}, "
            f"but data was built with {data_meta['tokenizer_name']}"
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
        train_num_samples = cfg.max_steps * cfg.batch_size

    if len(train_dataset) == 0:
        raise ValueError(
            "Train dataset is empty. Check train.bin, block_size, and train_stride."
        )
    if len(val_dataset) == 0:
        print("Warning: validation dataset is empty.")

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
        pin_memory=torch.cuda.is_available() and cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available() and cfg.pin_memory,
        drop_last=False,
    )

    model = build_model(cfg, tokenizer)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        scheduler=scheduler,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    print(f"Train windows: {len(train_dataset)}")
    print(f"Val windows: {len(val_dataset)}")
    print(f"Train tokens: {data_meta['train']['num_tokens']}")
    print(f"Val tokens: {data_meta['val']['num_tokens']}")
    print(f"Train token file: {cfg.train_tokens_file}")
    print(f"Val token file:   {cfg.val_tokens_file}")

    run_name = build_run_name(cfg)

    run_dir = Path(cfg.log_dir) / run_name
    ckpt_dir = Path(cfg.save_dir) / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    writer = SummaryWriter(log_dir=str(run_dir))

    csv_path = run_dir / "metrics.csv"
    append_csv = cfg.resume_from is not None and csv_path.exists()
    csv_file, csv_writer = open_csv_logger(csv_path, append=append_csv)

    step = 0
    best_val_loss = float("inf")

    if cfg.resume_from is not None:
        resume_path = Path(cfg.resume_from)
        print(f"Resuming from checkpoint: {resume_path}")
        step, best_val_loss = load_checkpoint(
            resume_path,
            trainer.model,
            trainer.optimizer,
            trainer.device,
            trainer.scheduler,
        )
        print(f"Resumed at step={step}, best_val_loss={best_val_loss:.4f}")

    print("Start training...")

    while step < cfg.max_steps:
        for batch in train_loader:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            metrics = trainer.train_step(batch)
            step_time = time.perf_counter() - t0

            step += 1

            tokens_this_step = cfg.batch_size * cfg.block_size
            tokens_seen = step * tokens_this_step

            max_memory_mb = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else 0.0
            )
            tokens_per_sec = tokens_this_step / max(step_time, 1e-12)

            val_loss = None
            val_ppl = None
            improved = False

            if step % cfg.eval_interval == 0:
                val_loss = trainer.evaluate()
                if val_loss is not None:
                    val_ppl = math.exp(min(val_loss, 20.0))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        improved = True
                        save_checkpoint(
                            ckpt_dir / "best.pt",
                            trainer.model,
                            trainer.optimizer,
                            trainer.scheduler,
                            step,
                            best_val_loss,
                            cfg,
                        )

                save_checkpoint(
                    ckpt_dir / "latest.pt",
                    trainer.model,
                    trainer.optimizer,
                    trainer.scheduler,
                    step,
                    best_val_loss,
                    cfg,
                )

            if step % cfg.save_every == 0:
                if not cfg.save_best_only:
                    save_checkpoint(
                        ckpt_dir / f"step_{step}.pt",
                        trainer.model,
                        trainer.optimizer,
                        trainer.scheduler,
                        step,
                        best_val_loss,
                        cfg,
                    )

                save_checkpoint(
                    ckpt_dir / "latest.pt",
                    trainer.model,
                    trainer.optimizer,
                    trainer.scheduler,
                    step,
                    best_val_loss,
                    cfg,
                )

            msg = (
                f"step={step}, "
                f"train_loss={metrics['loss']:.4f}, "
                f"grad_norm={metrics['grad_norm']:.4f}, "
                f"param_norm={metrics['param_norm']:.4f}, "
                f"update_norm={metrics['update_norm']:.4f}, "
                f"updated_params={metrics['num_updated_params']}, "
                f"update_ratio={metrics['update_ratio']:.6f}, "
                f"lr={metrics['lr']:.6e}, "
                f"step_time={step_time:.4f}s, "
                f"tok_s={tokens_per_sec:.2f}, "
                f"max_mem_mb={max_memory_mb:.2f}"
            )

            if val_loss is not None:
                msg += f", val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}"
                if improved:
                    msg += " [best]"

            print(msg)

            writer.add_scalar("train/loss", metrics["loss"], step)
            writer.add_scalar("train/grad_norm", metrics["grad_norm"], step)
            writer.add_scalar("train/param_norm", metrics["param_norm"], step)
            writer.add_scalar("train/update_norm", metrics["update_norm"], step)
            writer.add_scalar("train/update_ratio", metrics["update_ratio"], step)
            writer.add_scalar("train/lr", metrics["lr"], step)

            writer.add_scalar("system/step_time_sec", step_time, step)
            writer.add_scalar("system/tokens_per_sec", tokens_per_sec, step)
            writer.add_scalar("system/max_memory_mb", max_memory_mb, step)

            if val_loss is not None:
                writer.add_scalar("val/loss", val_loss, step)
                writer.add_scalar("val/ppl", val_ppl, step)
                writer.add_scalar("val/best_loss", best_val_loss, step)

            writer.flush()

            row = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": metrics["loss"],
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "grad_norm": metrics["grad_norm"],
                "param_norm": metrics["param_norm"],
                "update_norm": metrics["update_norm"],
                "num_updated_params": metrics["num_updated_params"],
                "update_ratio": metrics["update_ratio"],
                "lr": metrics["lr"],
                "step_time_sec": step_time,
                "tokens_per_sec": tokens_per_sec,
                "max_memory_mb": max_memory_mb,
            }
            csv_writer.writerow(row)
            csv_file.flush()

            if step >= cfg.max_steps:
                break

    save_checkpoint(
        ckpt_dir / "latest.pt",
        trainer.model,
        trainer.optimizer,
        trainer.scheduler,
        step,
        best_val_loss,
        cfg,
    )

    writer.close()
    csv_file.close()

    print("Training finished.")
    print(f"Run dir:           {run_dir}")
    print(f"Checkpoint dir:    {ckpt_dir}")
    print(f"Latest checkpoint: {ckpt_dir / 'latest.pt'}")
    print(f"Best checkpoint:   {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()