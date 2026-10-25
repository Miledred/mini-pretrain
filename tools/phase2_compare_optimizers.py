#!/usr/bin/env python3
r"""
Generate unified optimizer-comparison figures and a machine-readable summary.

Typical usage from the mini_pretrain project root (PowerShell):

python .\tools\phase2_compare_optimizers.py `
  --project-root . `
  --run "AdamW L8=my_adamw_L8_H8_D512_bs8_blk256_ts256_bigmodel" `
  --run "Muon 1.5e-4=my_muon_L8_H8_D512_bs8_blk256_ts256_muon_l8_lr1p5e4_50k" `
  --run "Muon 7.5e-4=my_muon_L8_H8_D512_bs8_blk256_ts256_muon_l8_lr7p5e4_50k"

Outputs:
phase2_comparison_outputs/
├── comparison_summary.csv
├── comparison_report.md
├── train_loss_comparison.png
├── val_loss_comparison.png
├── val_ppl_comparison.png
├── grad_norm_comparison.png
├── update_norm_comparison.png
├── update_ratio_comparison.png
├── lr_comparison.png
├── tokens_per_sec_comparison.png
└── max_memory_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


NUMERIC_COLUMNS = {
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
}

DEFAULT_RUNS = [
    ("AdamW L8", "my_adamw_L8_H8_D512_bs8_blk256_ts256_bigmodel"),
    ("Muon 1.5e-4", "my_muon_L8_H8_D512_bs8_blk256_ts256_muon_l8_lr1p5e4_50k"),
    ("Muon 7.5e-4", "my_muon_L8_H8_D512_bs8_blk256_ts256_muon_l8_lr7p5e4_50k"),
]


@dataclass
class RunData:
    label: str
    run_name: str
    run_dir: Path
    rows: list[dict[str, float | int | None]]
    config: dict


@dataclass
class RunSummary:
    label: str
    run_name: str
    optimizer_name: str
    peak_learning_rate: Optional[float]
    min_lr: Optional[float]
    warmup_steps: Optional[int]
    max_steps: Optional[int]
    final_step: Optional[int]
    best_val_loss: Optional[float]
    best_val_step: Optional[int]
    best_val_ppl: Optional[float]
    final_val_loss: Optional[float]
    final_val_ppl: Optional[float]
    final_train_loss: Optional[float]
    mean_train_loss_last_200: Optional[float]
    final_grad_norm: Optional[float]
    peak_update_norm: Optional[float]
    final_update_norm: Optional[float]
    peak_update_ratio: Optional[float]
    final_update_ratio: Optional[float]
    median_tokens_per_sec: Optional[float]
    p10_tokens_per_sec: Optional[float]
    p90_tokens_per_sec: Optional[float]
    max_memory_mb: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare mini_pretrain optimizer runs")
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root containing logs/.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Logs directory relative to project root.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="phase2_comparison_outputs",
        help="Output directory relative to project root.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        metavar="LABEL=RUN_NAME",
        help=(
            "Run to compare. May be supplied multiple times, for example "
            '"AdamW L8=my_adamw_...".'
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=200,
        help="Rolling window for noisy step-level metrics.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2500,
        help="Maximum points plotted per line after downsampling.",
    )
    return parser.parse_args()


def parse_run_specs(specs: Optional[list[str]]) -> list[tuple[str, str]]:
    if not specs:
        return DEFAULT_RUNS

    parsed: list[tuple[str, str]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --run value {spec!r}. Expected LABEL=RUN_NAME."
            )
        label, run_name = spec.split("=", 1)
        label = label.strip()
        run_name = run_name.strip()
        if not label or not run_name:
            raise ValueError(
                f"Invalid --run value {spec!r}. Both label and run name are required."
            )
        parsed.append((label, run_name))
    return parsed


def parse_number(value: object) -> float | int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def load_csv(path: Path) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "step" not in reader.fieldnames:
            raise ValueError(f"metrics.csv has no step column: {path}")

        for raw in reader:
            row: dict[str, float | int | None] = {}
            for key, value in raw.items():
                if key in NUMERIC_COLUMNS:
                    row[key] = parse_number(value)
            rows.append(row)
    return rows


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def load_runs(
    logs_dir: Path,
    run_specs: list[tuple[str, str]],
) -> list[RunData]:
    runs: list[RunData] = []
    missing: list[str] = []

    for label, run_name in run_specs:
        run_dir = logs_dir / run_name
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue

        rows = load_csv(metrics_path)
        if not rows:
            raise ValueError(f"No rows found in {metrics_path}")

        runs.append(
            RunData(
                label=label,
                run_name=run_name,
                run_dir=run_dir,
                rows=rows,
                config=load_json(run_dir / "config.json"),
            )
        )

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The following metrics files were not found:\n"
            f"{formatted}\n"
            "Check the run directory names with: Get-ChildItem .\\logs"
        )

    if len(runs) < 2:
        raise ValueError("At least two runs are required for comparison.")

    return runs


def numeric_values(rows: Iterable[dict], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def last_value(rows: list[dict], column: str) -> Optional[float]:
    for row in reversed(rows):
        value = row.get(column)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def rolling_stat(
    values: list[float],
    window: int,
    function: Callable[[list[float]], float],
) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values.copy()

    output: list[float] = []
    buffer: list[float] = []
    for value in values:
        buffer.append(value)
        if len(buffer) > window:
            buffer.pop(0)
        output.append(float(function(buffer)))
    return output


def metric_pairs(run: RunData, column: str) -> list[tuple[int, float]]:
    pairs: list[tuple[int, float]] = []
    for row in run.rows:
        step = row.get("step")
        value = row.get(column)
        if isinstance(step, (int, float)) and isinstance(value, (int, float)):
            pairs.append((int(step), float(value)))
    return pairs


def downsample(
    x: list[int],
    y: list[float],
    max_points: int,
) -> tuple[list[int], list[float]]:
    if max_points <= 0 or len(x) <= max_points:
        return x, y
    stride = math.ceil(len(x) / max_points)
    return x[::stride], y[::stride]


def plot_lines(
    runs: list[RunData],
    column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    max_points: int,
    smooth_window: int = 1,
    smooth_function: Callable[[list[float]], float] = statistics.mean,
) -> None:
    import matplotlib.pyplot as plt

    plotted = False
    plt.figure(figsize=(9, 5.2))

    for run in runs:
        pairs = metric_pairs(run, column)
        if not pairs:
            continue

        x = [step for step, _ in pairs]
        y = [value for _, value in pairs]
        if smooth_window > 1:
            y = rolling_stat(y, smooth_window, smooth_function)
        x, y = downsample(x, y, max_points)

        plt.plot(x, y, linewidth=1.7, label=run.label)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_run(run: RunData) -> RunSummary:
    validation_rows = [
        row
        for row in run.rows
        if isinstance(row.get("val_loss"), (int, float))
    ]

    best_row: Optional[dict] = None
    if validation_rows:
        best_row = min(validation_rows, key=lambda row: float(row["val_loss"]))

    train_losses = numeric_values(run.rows, "train_loss")
    update_norms = numeric_values(run.rows, "update_norm")
    update_ratios = numeric_values(run.rows, "update_ratio")
    throughputs = numeric_values(run.rows[100:], "tokens_per_sec")
    memories = numeric_values(run.rows, "max_memory_mb")

    final_step = last_value(run.rows, "step")

    return RunSummary(
        label=run.label,
        run_name=run.run_name,
        optimizer_name=str(run.config.get("optimizer_name", "unknown")),
        peak_learning_rate=_optional_float(run.config.get("learning_rate")),
        min_lr=_optional_float(run.config.get("min_lr")),
        warmup_steps=_optional_int(run.config.get("warmup_steps")),
        max_steps=_optional_int(run.config.get("max_steps")),
        final_step=int(final_step) if final_step is not None else None,
        best_val_loss=(
            float(best_row["val_loss"]) if best_row is not None else None
        ),
        best_val_step=(
            int(best_row["step"])
            if best_row is not None and isinstance(best_row.get("step"), (int, float))
            else None
        ),
        best_val_ppl=(
            float(best_row["val_ppl"])
            if best_row is not None and isinstance(best_row.get("val_ppl"), (int, float))
            else None
        ),
        final_val_loss=last_value(run.rows, "val_loss"),
        final_val_ppl=last_value(run.rows, "val_ppl"),
        final_train_loss=last_value(run.rows, "train_loss"),
        mean_train_loss_last_200=(
            statistics.mean(train_losses[-200:]) if train_losses else None
        ),
        final_grad_norm=last_value(run.rows, "grad_norm"),
        peak_update_norm=max(update_norms) if update_norms else None,
        final_update_norm=last_value(run.rows, "update_norm"),
        peak_update_ratio=max(update_ratios) if update_ratios else None,
        final_update_ratio=last_value(run.rows, "update_ratio"),
        median_tokens_per_sec=(
            statistics.median(throughputs) if throughputs else None
        ),
        p10_tokens_per_sec=percentile(throughputs, 0.10),
        p90_tokens_per_sec=percentile(throughputs, 0.90),
        max_memory_mb=max(memories) if memories else None,
    )


def _optional_float(value: object) -> Optional[float]:
    parsed = parse_number(value)
    return float(parsed) if isinstance(parsed, (int, float)) else None


def _optional_int(value: object) -> Optional[int]:
    parsed = parse_number(value)
    return int(parsed) if isinstance(parsed, (int, float)) else None


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_summary_csv(path: Path, summaries: list[RunSummary]) -> None:
    fieldnames = list(RunSummary.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def relative_improvement(
    baseline: Optional[float],
    candidate: Optional[float],
) -> Optional[float]:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def write_report(path: Path, summaries: list[RunSummary]) -> None:
    baseline = summaries[0]
    lines = [
        "# Phase 2 Optimizer Comparison Report",
        "",
        "## 1. Run summary",
        "",
        "| Run | Optimizer | Peak LR | Best val loss | Best step | Best val PPL | Final train loss | Mean train loss (last 200) | Median tok/s | Max memory MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.label,
                    summary.optimizer_name,
                    fmt(summary.peak_learning_rate, 7),
                    fmt(summary.best_val_loss, 4),
                    str(summary.best_val_step or ""),
                    fmt(summary.best_val_ppl, 2),
                    fmt(summary.final_train_loss, 4),
                    fmt(summary.mean_train_loss_last_200, 4),
                    fmt(summary.median_tokens_per_sec, 2),
                    fmt(summary.max_memory_mb, 2),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 2. Relative comparison against the first run",
            "",
            f"Baseline: `{baseline.label}`",
            "",
            "| Candidate | Val-loss reduction | PPL reduction | Final-train-loss reduction | Throughput change | Memory change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for candidate in summaries[1:]:
        val_gain = relative_improvement(
            baseline.best_val_loss, candidate.best_val_loss
        )
        ppl_gain = relative_improvement(
            baseline.best_val_ppl, candidate.best_val_ppl
        )
        train_gain = relative_improvement(
            baseline.final_train_loss, candidate.final_train_loss
        )
        throughput_change = None
        if (
            baseline.median_tokens_per_sec is not None
            and candidate.median_tokens_per_sec is not None
            and baseline.median_tokens_per_sec != 0
        ):
            throughput_change = (
                candidate.median_tokens_per_sec - baseline.median_tokens_per_sec
            ) / baseline.median_tokens_per_sec * 100.0
        memory_change = None
        if (
            baseline.max_memory_mb is not None
            and candidate.max_memory_mb is not None
            and baseline.max_memory_mb != 0
        ):
            memory_change = (
                candidate.max_memory_mb - baseline.max_memory_mb
            ) / baseline.max_memory_mb * 100.0

        lines.append(
            "| "
            + " | ".join(
                [
                    candidate.label,
                    _percent_or_blank(val_gain),
                    _percent_or_blank(ppl_gain),
                    _percent_or_blank(train_gain),
                    _signed_percent_or_blank(throughput_change),
                    _signed_percent_or_blank(memory_change),
                ]
            )
            + " |"
        )

    best = min(
        (summary for summary in summaries if summary.best_val_loss is not None),
        key=lambda summary: float(summary.best_val_loss),
    )

    lines.extend(
        [
            "",
            "## 3. Automatic observations",
            "",
            f"- Lowest validation loss: **{best.label}**, `{fmt(best.best_val_loss, 4)}` at step `{best.best_val_step}`.",
            f"- Its validation perplexity at the best checkpoint is `{fmt(best.best_val_ppl, 2)}`.",
            "- Training-loss plots use a rolling mean; validation metrics use the original evaluation points.",
            "- Throughput is summarized with the median after the first 100 steps to reduce sensitivity to transient system slowdowns.",
            "- Different optimizer-specific learning rates must be reported explicitly; do not describe the comparison as using identical hyperparameters.",
            "",
            "## 4. Generated figures",
            "",
            "- `train_loss_comparison.png`",
            "- `val_loss_comparison.png`",
            "- `val_ppl_comparison.png`",
            "- `grad_norm_comparison.png`",
            "- `update_norm_comparison.png`",
            "- `update_ratio_comparison.png`",
            "- `lr_comparison.png`",
            "- `tokens_per_sec_comparison.png`",
            "- `max_memory_comparison.png`",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _percent_or_blank(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.2f}%"


def _signed_percent_or_blank(value: Optional[float]) -> str:
    return "" if value is None else f"{value:+.2f}%"


def plot_memory_bar(
    summaries: list[RunSummary],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = [summary.label for summary in summaries]
    values = [summary.max_memory_mb or 0.0 for summary in summaries]

    plt.figure(figsize=(8.5, 5.0))
    bars = plt.bar(labels, values)
    plt.ylabel("Peak allocated memory (MB)")
    plt.title("Peak GPU Memory Comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=12, ha="right")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.0f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    logs_dir = project_root / args.logs_dir
    out_dir = project_root / args.out_dir

    run_specs = parse_run_specs(args.run)
    runs = load_runs(logs_dir, run_specs)
    summaries = [summarize_run(run) for run in runs]

    # No smoothing on validation metrics or learning-rate schedule.
    plot_lines(
        runs,
        "train_loss",
        out_dir / "train_loss_comparison.png",
        "Training Loss Comparison",
        "Cross-entropy loss",
        args.max_points,
        smooth_window=args.smooth_window,
        smooth_function=statistics.mean,
    )
    plot_lines(
        runs,
        "val_loss",
        out_dir / "val_loss_comparison.png",
        "Validation Loss Comparison",
        "Cross-entropy loss",
        args.max_points,
    )
    plot_lines(
        runs,
        "val_ppl",
        out_dir / "val_ppl_comparison.png",
        "Validation Perplexity Comparison",
        "Perplexity",
        args.max_points,
    )
    plot_lines(
        runs,
        "grad_norm",
        out_dir / "grad_norm_comparison.png",
        "Gradient Norm Comparison",
        "Gradient norm",
        args.max_points,
        smooth_window=args.smooth_window,
        smooth_function=statistics.mean,
    )
    plot_lines(
        runs,
        "update_norm",
        out_dir / "update_norm_comparison.png",
        "Parameter Update Norm Comparison",
        "Update norm",
        args.max_points,
    )
    plot_lines(
        runs,
        "update_ratio",
        out_dir / "update_ratio_comparison.png",
        "Relative Update Ratio Comparison",
        "Update norm / parameter norm",
        args.max_points,
    )
    plot_lines(
        runs,
        "lr",
        out_dir / "lr_comparison.png",
        "Learning-Rate Schedule Comparison",
        "Learning rate",
        args.max_points,
    )
    plot_lines(
        runs,
        "tokens_per_sec",
        out_dir / "tokens_per_sec_comparison.png",
        "Training Throughput Comparison",
        "Tokens per second (rolling median)",
        args.max_points,
        smooth_window=args.smooth_window,
        smooth_function=statistics.median,
    )
    plot_memory_bar(summaries, out_dir / "max_memory_comparison.png")

    write_summary_csv(out_dir / "comparison_summary.csv", summaries)
    write_report(out_dir / "comparison_report.md", summaries)

    print("Comparison complete.")
    print(f"Output directory: {out_dir}")
    print(f"Summary CSV:      {out_dir / 'comparison_summary.csv'}")
    print(f"Report:           {out_dir / 'comparison_report.md'}")
    for summary in summaries:
        print(
            f"- {summary.label}: best_val_loss={fmt(summary.best_val_loss, 4)}, "
            f"best_val_ppl={fmt(summary.best_val_ppl, 2)}, "
            f"median_tok_s={fmt(summary.median_tokens_per_sec, 2)}"
        )


if __name__ == "__main__":
    main()
