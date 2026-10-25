#!/usr/bin/env python3
"""
Phase-1 baseline audit for mini_pretrain.

Purpose
-------
Scan existing logs/checkpoints, summarize an AdamW baseline run, and generate
plots that can be used directly in a supervisor update or technical report.

Typical usage
-------------
From the project root:
    python tools/phase1_baseline_audit.py --project-root .

Or, if this file is copied to the project root:
    python phase1_baseline_audit.py --project-root .

Outputs
-------
phase1_outputs/
├── run_inventory.csv
├── phase1_baseline_report.md
└── <run_name>/
    ├── train_loss.png
    ├── val_loss.png
    ├── val_ppl.png
    ├── lr.png
    ├── grad_norm.png
    ├── update_norm.png
    ├── update_ratio.png
    ├── tokens_per_sec.png
    └── max_memory_mb.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional


METRIC_COLUMNS = [
    "train_loss",
    "val_loss",
    "val_ppl",
    "grad_norm",
    "param_norm",
    "update_norm",
    "update_ratio",
    "lr",
    "step_time_sec",
    "tokens_per_sec",
    "max_memory_mb",
]

PLOT_COLUMNS = [
    "train_loss",
    "val_loss",
    "val_ppl",
    "lr",
    "grad_norm",
    "update_norm",
    "update_ratio",
    "tokens_per_sec",
    "max_memory_mb",
]


@dataclass
class RunSummary:
    run_name: str
    metrics_path: str
    config_path: str
    checkpoint_dir: str
    num_rows: int
    final_step: Optional[int]
    max_step: Optional[int]
    final_train_loss: Optional[float]
    final_val_loss: Optional[float]
    final_val_ppl: Optional[float]
    best_val_loss: Optional[float]
    best_val_step: Optional[int]
    best_val_ppl: Optional[float]
    min_train_loss: Optional[float]
    avg_tokens_per_sec_last_100: Optional[float]
    max_memory_mb: Optional[float]
    best_checkpoint_exists: bool
    latest_checkpoint_exists: bool
    periodic_checkpoint_count: int
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=".", help="Project root containing logs/ and checkpoints/.")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Logs directory relative to project root.")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Checkpoint directory relative to project root.")
    parser.add_argument("--out-dir", type=str, default="phase1_outputs", help="Output directory relative to project root.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name to audit, e.g. my_adamw_L8_H8_D512_bs8_blk256_ts256_bigmodel.")
    parser.add_argument("--max-points", type=int, default=2000, help="Maximum plotted points per curve, after uniform downsampling.")
    return parser.parse_args()


def as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def as_int(value: object) -> Optional[int]:
    val = as_float(value)
    if val is None:
        return None
    return int(val)


def load_metrics(metrics_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if key == "step" or key == "tokens_seen" or key == "num_updated_params":
                    parsed[key] = as_int(value)
                elif key in METRIC_COLUMNS:
                    parsed[key] = as_float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def non_null_pairs(rows: Iterable[dict[str, object]], col: str) -> list[tuple[int, float]]:
    pairs: list[tuple[int, float]] = []
    for row in rows:
        step = row.get("step")
        value = row.get(col)
        if isinstance(step, int) and isinstance(value, float):
            pairs.append((step, value))
    return pairs


def last_non_null(rows: Iterable[dict[str, object]], col: str) -> Optional[float]:
    for row in reversed(list(rows)):
        value = row.get(col)
        if isinstance(value, float):
            return value
    return None


def min_non_null(rows: Iterable[dict[str, object]], col: str) -> Optional[float]:
    values = [row[col] for row in rows if isinstance(row.get(col), float)]
    return min(values) if values else None


def best_val(rows: list[dict[str, object]]) -> tuple[Optional[float], Optional[int], Optional[float]]:
    candidates = []
    for row in rows:
        step = row.get("step")
        val_loss = row.get("val_loss")
        val_ppl = row.get("val_ppl")
        if isinstance(step, int) and isinstance(val_loss, float):
            candidates.append((val_loss, step, val_ppl if isinstance(val_ppl, float) else None))
    if not candidates:
        return None, None, None
    val_loss, step, val_ppl = min(candidates, key=lambda x: x[0])
    return val_loss, step, val_ppl


def mean_last(values: list[float], n: int = 100) -> Optional[float]:
    clean = [v for v in values if isinstance(v, float)]
    if not clean:
        return None
    return statistics.mean(clean[-n:])


def discover_metrics(project_root: Path, logs_dir: Path, run_name: Optional[str]) -> list[Path]:
    if run_name:
        candidate = logs_dir / run_name / "metrics.csv"
        return [candidate] if candidate.exists() else []
    return sorted(logs_dir.glob("**/metrics.csv"))


def uniform_downsample(pairs: list[tuple[int, float]], max_points: int) -> list[tuple[int, float]]:
    if max_points <= 0 or len(pairs) <= max_points:
        return pairs
    stride = math.ceil(len(pairs) / max_points)
    return pairs[::stride]


def plot_metric(pairs: list[tuple[int, float]], out_path: Path, title: str, ylabel: str, max_points: int) -> None:
    if not pairs:
        return
    import matplotlib.pyplot as plt

    pairs = uniform_downsample(pairs, max_points=max_points)
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def summarize_run(metrics_path: Path, project_root: Path, checkpoints_dir: Path, out_dir: Path, max_points: int) -> RunSummary:
    run_name = metrics_path.parent.name
    rows = load_metrics(metrics_path)

    ckpt_dir = checkpoints_dir / run_name
    config_path = metrics_path.parent / "config.json"

    steps = [row.get("step") for row in rows if isinstance(row.get("step"), int)]
    final_step = steps[-1] if steps else None
    max_step = max(steps) if steps else None
    best_loss, best_step, best_ppl = best_val(rows)
    tps_values = [row["tokens_per_sec"] for row in rows if isinstance(row.get("tokens_per_sec"), float)]
    mem_values = [row["max_memory_mb"] for row in rows if isinstance(row.get("max_memory_mb"), float)]

    # Plot all available core curves.
    run_out = out_dir / run_name
    for col in PLOT_COLUMNS:
        pairs = non_null_pairs(rows, col)
        plot_metric(
            pairs,
            out_path=run_out / f"{col}.png",
            title=f"{run_name}: {col}",
            ylabel=col,
            max_points=max_points,
        )

    periodic_count = len(list(ckpt_dir.glob("step_*.pt"))) if ckpt_dir.exists() else 0
    best_exists = (ckpt_dir / "best.pt").exists()
    latest_exists = (ckpt_dir / "latest.pt").exists()

    status_parts = []
    if final_step is None:
        status_parts.append("no_step")
    elif final_step < 50000:
        status_parts.append("possibly_incomplete")
    else:
        status_parts.append("complete_or_long")
    if best_loss is None:
        status_parts.append("no_validation_rows")
    if not best_exists:
        status_parts.append("missing_best_ckpt")
    if not latest_exists:
        status_parts.append("missing_latest_ckpt")

    return RunSummary(
        run_name=run_name,
        metrics_path=str(metrics_path.relative_to(project_root) if metrics_path.is_relative_to(project_root) else metrics_path),
        config_path=str(config_path.relative_to(project_root) if config_path.exists() and config_path.is_relative_to(project_root) else config_path),
        checkpoint_dir=str(ckpt_dir.relative_to(project_root) if ckpt_dir.is_relative_to(project_root) else ckpt_dir),
        num_rows=len(rows),
        final_step=final_step,
        max_step=max_step,
        final_train_loss=last_non_null(rows, "train_loss"),
        final_val_loss=last_non_null(rows, "val_loss"),
        final_val_ppl=last_non_null(rows, "val_ppl"),
        best_val_loss=best_loss,
        best_val_step=best_step,
        best_val_ppl=best_ppl,
        min_train_loss=min_non_null(rows, "train_loss"),
        avg_tokens_per_sec_last_100=mean_last(tps_values, 100),
        max_memory_mb=max(mem_values) if mem_values else None,
        best_checkpoint_exists=best_exists,
        latest_checkpoint_exists=latest_exists,
        periodic_checkpoint_count=periodic_count,
        status=";".join(status_parts),
    )


def write_inventory(summaries: list[RunSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(asdict(summaries[0]).keys()) if summaries else [
            "run_name", "metrics_path", "status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def fmt(value: object, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def write_report(summaries: list[RunSummary], out_path: Path, project_root: Path, logs_dir: Path, checkpoints_dir: Path) -> None:
    lines: list[str] = []
    lines.append("# Phase 1 Baseline Audit Report")
    lines.append("")
    lines.append("## 1. Audit scope")
    lines.append("")
    lines.append(f"- Project root: `{project_root}`")
    lines.append(f"- Logs directory: `{logs_dir}`")
    lines.append(f"- Checkpoint directory: `{checkpoints_dir}`")
    lines.append("")

    if not summaries:
        lines.append("## 2. Current finding")
        lines.append("")
        lines.append("No `metrics.csv` was found under the logs directory. The source code is ready for Phase 1 analysis, but the actual training logs/checkpoints must be copied back into the project root or generated by running training.")
        lines.append("")
        lines.append("Expected locations:")
        lines.append("")
        lines.append("```text")
        lines.append("logs/<run_name>/metrics.csv")
        lines.append("logs/<run_name>/config.json")
        lines.append("checkpoints/<run_name>/best.pt")
        lines.append("checkpoints/<run_name>/latest.pt")
        lines.append("```")
        lines.append("")
        lines.append("Immediate command after logs are available:")
        lines.append("")
        lines.append("```bash")
        lines.append("python phase1_baseline_audit.py --project-root .")
        lines.append("```")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## 2. Run inventory")
    lines.append("")
    lines.append("| Run | Final step | Best val loss | Best step | Final train loss | Final val ppl | Avg tok/s last100 | Max memory MB | Checkpoints | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for s in summaries:
        ckpt = f"best={s.best_checkpoint_exists}, latest={s.latest_checkpoint_exists}, periodic={s.periodic_checkpoint_count}"
        lines.append(
            f"| `{s.run_name}` | {fmt(s.final_step, 0)} | {fmt(s.best_val_loss)} | {fmt(s.best_val_step, 0)} | "
            f"{fmt(s.final_train_loss)} | {fmt(s.final_val_ppl)} | {fmt(s.avg_tokens_per_sec_last_100, 2)} | "
            f"{fmt(s.max_memory_mb, 2)} | {ckpt} | {s.status} |"
        )
    lines.append("")

    lines.append("## 3. Interpretation checklist")
    lines.append("")
    lines.append("For each AdamW baseline run, inspect the generated PNG files and answer these four questions:")
    lines.append("")
    lines.append("1. Does `train_loss` decrease smoothly, or does it show spikes/instability?")
    lines.append("2. Does `val_loss` keep decreasing, plateau, or rebound after a certain step?")
    lines.append("3. Is `update_ratio` stable, vanishing, or suddenly increasing?")
    lines.append("4. Is throughput stable enough for fair comparison with Muon later?")
    lines.append("")

    lines.append("## 4. Files generated")
    lines.append("")
    lines.append("- `run_inventory.csv`: machine-readable run summary.")
    lines.append("- `<run_name>/*.png`: single-metric curves for direct insertion into slides.")
    lines.append("")

    lines.append("## 5. Baseline status wording for supervisor update")
    lines.append("")
    lines.append("> I have reorganized the AdamW baseline results. The current training pipeline records train/validation loss, perplexity, learning rate, gradient norm, update norm, update ratio, throughput, memory usage, and checkpoints. The next step is to inspect whether the validation curve plateaus or rebounds, then use the same logging protocol for Muon so that the comparison is fair.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    logs_dir = (project_root / args.logs_dir).resolve()
    checkpoints_dir = (project_root / args.checkpoints_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_paths = discover_metrics(project_root, logs_dir, args.run_name)
    summaries = [
        summarize_run(path, project_root, checkpoints_dir, out_dir, max_points=args.max_points)
        for path in metrics_paths
    ]

    write_inventory(summaries, out_dir / "run_inventory.csv")
    write_report(summaries, out_dir / "phase1_baseline_report.md", project_root, logs_dir, checkpoints_dir)

    print(f"Found {len(summaries)} run(s).")
    print(f"Inventory: {out_dir / 'run_inventory.csv'}")
    print(f"Report:    {out_dir / 'phase1_baseline_report.md'}")
    if summaries:
        print("Generated plots under:")
        for summary in summaries:
            print(f"  - {out_dir / summary.run_name}")


if __name__ == "__main__":
    main()
