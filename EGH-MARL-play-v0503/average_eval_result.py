#!/usr/bin/env python3
"""Insert average rows for repeated evaluation configurations."""

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


DEFAULT_PATTERN = "eval_result_smacv2_"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read eval_result*.txt files, group rows by the same config, "
            "and write new files with one average row after each group."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input result file. If not provided, auto-find files starting with eval_result_smac_",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file. Default: <input_stem>_avg<input_suffix>",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Include original rows plus average rows in output.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for average reward. Default: 4",
    )
    return parser.parse_args()


def read_rows(input_path):
    rows = []
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not item.strip() for item in row):
                continue
            row = [item.strip() for item in row]
            if len(row) < 3:
                raise ValueError(f"Line {line_no} has too few columns: {row}")
            try:
                reward = float(row[-1])
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_no} reward column is not a number: {row[-1]}"
                ) from exc
            rows.append((row, reward))
    return rows


def group_rows(rows):
    groups = OrderedDict()
    for row, reward in rows:
        config_key = tuple(row[1:-1])
        if config_key not in groups:
            groups[config_key] = {"rows": [], "rewards": []}
        groups[config_key]["rows"].append(row)
        groups[config_key]["rewards"].append(reward)
    return groups


def build_output_rows(groups, summary_only, precision):
    output_rows = []
    for config_key, group in groups.items():
        rewards = group["rewards"]
        avg_reward = sum(rewards) / len(rewards)
        avg_row = ["AVG", *config_key, f"{avg_reward:.{precision}f}"]
        output_rows.append(avg_row)
    return output_rows


def find_input_files(pattern):
    current_dir = Path.cwd()
    input_files = sorted(current_dir.glob(f"{pattern}*"))
    input_files = [f for f in input_files if "avg" not in f.name.lower()]
    if not input_files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}*")
    return input_files

def process_file(input_path, output_path, summary_only, precision):
    rows = read_rows(input_path)
    groups = group_rows(rows)
    output_rows = build_output_rows(groups, summary_only, precision)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(output_rows)

    print(f"Read {len(rows)} rows from {input_path}")
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    print(f"Grouped into {len(groups)} configs")
    return len(rows), len(output_rows), len(groups)

def main():
    args = parse_args()
    
    if args.input:
        input_paths = [Path(args.input)]
        if not input_paths[0].exists():
            raise FileNotFoundError(f"Input file not found: {input_paths[0]}")
    else:
        input_paths = find_input_files(DEFAULT_PATTERN)
        print(f"Found {len(input_paths)} files matching pattern '{DEFAULT_PATTERN}*'")

    total_input_rows = 0
    total_output_rows = 0
    total_groups = 0

    for input_path in input_paths:
        print(f"\nProcessing: {input_path.name}")
        output_path = (
            Path(args.output)
            if args.output
            else input_path.with_name(f"{input_path.stem}_avg{input_path.suffix}")
        )
        input_rows, output_rows, groups = process_file(
            input_path, output_path, not args.include_original, args.precision
        )
        total_input_rows += input_rows
        total_output_rows += output_rows
        total_groups += groups

    print(f"\n{'='*50}")
    print(f"Total: Processed {len(input_paths)} file(s)")
    print(f"Total input rows: {total_input_rows}")
    print(f"Total output rows: {total_output_rows}")
    print(f"Total groups: {total_groups}")


if __name__ == "__main__":
    main()