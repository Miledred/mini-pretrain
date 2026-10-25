from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAG_TO_COLUMN = {
    "train/loss": "train_loss",
    "val/loss": "val_loss",
    "val/ppl": "val_ppl",
    "train/grad_norm": "grad_norm",
    "train/param_norm": "param_norm",
    "train/update_norm": "update_norm",
    "train/update_ratio": "update_ratio",
    "train/lr": "lr",
    "system/step_time_sec": "step_time_sec",
    "system/tokens_per_sec": "tokens_per_sec",
    "system/max_memory_mb": "max_memory_mb",
}


FIELDNAMES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover metrics.csv from one TensorBoard event file."
    )
    parser.add_argument(
        "--event-file",
        required=True,
        help="Path to the selected events.out.tfevents file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output metrics CSV path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num-updated-params",
        type=int,
        default=100,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    event_file = Path(args.event_file).resolve()
    output_path = Path(args.output).resolve()

    if not event_file.exists():
        raise FileNotFoundError(f"Event file not found: {event_file}")

    accumulator = EventAccumulator(
        str(event_file),
        size_guidance={
            "scalars": 0,
        },
    )
    accumulator.Reload()

    scalar_tags = set(accumulator.Tags().get("scalars", []))

    print("Available scalar tags:")
    for tag in sorted(scalar_tags):
        print(f"  {tag}")

    rows_by_step: dict[int, dict[str, float | int | str]] = {}

    for tensorboard_tag, csv_column in TAG_TO_COLUMN.items():
        if tensorboard_tag not in scalar_tags:
            print(f"Warning: tag not found: {tensorboard_tag}")
            continue

        for event in accumulator.Scalars(tensorboard_tag):
            step = int(event.step)

            if step not in rows_by_step:
                rows_by_step[step] = {
                    field: "" for field in FIELDNAMES
                }
                rows_by_step[step]["step"] = step
                rows_by_step[step]["tokens_seen"] = (
                    step * args.batch_size * args.block_size
                )
                rows_by_step[step]["num_updated_params"] = (
                    args.num_updated_params
                )

            # 同一步同一指标若重复，以事件文件中最后读到的值为准。
            rows_by_step[step][csv_column] = float(event.value)

    if not rows_by_step:
        raise RuntimeError(
            "No scalar records were recovered from the selected event file."
        )

    rows = [
        rows_by_step[step]
        for step in sorted(rows_by_step)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    val_rows = [
        row
        for row in rows
        if isinstance(row.get("val_loss"), float)
    ]

    print()
    print(f"Recovered rows: {len(rows)}")
    print(f"First step: {rows[0]['step']}")
    print(f"Final step: {rows[-1]['step']}")

    if val_rows:
        best_row = min(
            val_rows,
            key=lambda row: float(row["val_loss"]),
        )
        print(f"Validation records: {len(val_rows)}")
        print(f"Best val loss: {best_row['val_loss']:.6f}")
        print(f"Best val step: {best_row['step']}")
        print(f"Best val PPL: {best_row['val_ppl']}")
    else:
        print("Warning: no validation records were recovered.")

    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()