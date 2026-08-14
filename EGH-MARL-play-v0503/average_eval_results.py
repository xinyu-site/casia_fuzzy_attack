#!/usr/bin/env python3
"""Average evaluation results that share the same configuration."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ignore the timestamp column, group rows by all configuration "
            "columns, and average the last two numeric columns."
        )
    )
    parser.add_argument("input", type=Path, help="input CSV-like txt file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file (default: <input_stem>_averaged.txt)",
    )
    return parser.parse_args()


def average_results(input_path: Path, output_path: Path) -> None:
    # key: all columns except timestamp, average return, and win rate
    totals: dict[tuple[str, ...], list[float | int]] = defaultdict(
        lambda: [0.0, 0.0, 0]
    )

    with input_path.open("r", encoding="utf-8", newline="") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) < 4:
                raise ValueError(
                    f"Line {line_number}: expected timestamp, configuration, "
                    "average return, and win rate"
                )

            row = [field.strip() for field in row]
            config = tuple(row[1:-2])
            try:
                average_return = float(row[-2])
                win_rate = float(row[-1])
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number}: the last two columns must be numbers"
                ) from error

            totals[config][0] += average_return
            totals[config][1] += win_rate
            totals[config][2] += 1

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            ["algorithm", "attack", "param1", "param2", "avg_return", "avg_win_rate", "count"]
        )
        for config, (return_sum, win_rate_sum, count) in totals.items():
            writer.writerow(
                [*config, f"{return_sum / count:.4f}", f"{win_rate_sum / count:.4f}", count]
            )


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input.with_name(
        f"{args.input.stem}_averaged.txt"
    )
    average_results(args.input, output_path)
    print(f"Wrote averaged results to {output_path}")


if __name__ == "__main__":
    main()
