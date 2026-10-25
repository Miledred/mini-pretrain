import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def safe_float(value):
    if value is None:
        return None

    value = str(value).strip()
    if value == "" or value.lower() == "none":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def read_metrics(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def collect_runs(log_dir: Path):
    runs = {}

    for metrics_path in sorted(log_dir.glob("*/metrics.csv")):
        run_name = metrics_path.parent.name
        rows = read_metrics(metrics_path)

        if len(rows) == 0:
            continue

        runs[run_name] = rows

    return runs


def plot_metric(runs, metric_name: str, out_dir: Path):
    plt.figure(figsize=(10, 6))

    plotted = False

    for run_name, rows in runs.items():
        xs = []
        ys = []

        for row in rows:
            step = safe_float(row.get("step"))
            value = safe_float(row.get(metric_name))

            if step is None or value is None:
                continue

            xs.append(step)
            ys.append(value)

        if len(xs) == 0:
            continue

        plt.plot(xs, ys, label=run_name)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"{metric_name}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--out-dir", type=str, default="analysis/plots")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(log_dir)

    if len(runs) == 0:
        print(f"No metrics.csv files found under {log_dir}")
        return

    metric_names = [
        "train_loss",
        "val_loss",
        "val_ppl",
        "grad_norm",
        "param_norm",
        "update_norm",
        "update_ratio",
        "lr",
        "tokens_per_sec",
        "step_time_sec",
        "max_memory_mb",
        "tokens_seen",
    ]

    for metric_name in metric_names:
        plot_metric(runs, metric_name, out_dir)


if __name__ == "__main__":
    main()