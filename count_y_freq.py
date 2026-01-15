#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count occurrences of y (first CSV column) and x (second CSV column) "
            "and plot frequency of K[y]/K[x] values."
        )
    )
    parser.add_argument("csv_path", help="Input CSV file path.")
    parser.add_argument(
        "-o",
        "--output",
        default="y_count_freq.png",
        help="Output plot path (default: y_count_freq.png).",
    )
    parser.add_argument(
        "--x-output",
        default="x_count_freq.png",
        help="Output plot path for x (default: x_count_freq.png).",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Skip the first row as a header.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)

    y_counts = Counter()
    x_counts = Counter()
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader):
            if not row:
                continue
            if idx == 0 and args.has_header:
                continue
            y_val = row[0].strip()
            if idx == 0 and not args.has_header and y_val.lower() in {"y", "value", "label"}:
                continue
            if not y_val:
                continue
            y_counts[y_val] += 1
            if len(row) > 1:
                x_val = row[1].strip()
                if x_val:
                    x_counts[x_val] += 1

    if not y_counts and not x_counts:
        print("No data rows found. Check the CSV path or header handling.")
        return

    if y_counts:
        y_freq = Counter(y_counts.values())
        xs = sorted(y_freq.keys())
        ys = [y_freq[x] for x in xs]

        plt.figure(figsize=(8, 4))
        plt.bar(xs, ys)
        plt.xlabel("K[y] (count per y)")
        plt.ylabel("Frequency of K[y]")
        plt.title("Frequency of K[y] Values")
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")

    if x_counts:
        x_freq = Counter(x_counts.values())
        xs = sorted(x_freq.keys())
        ys = [x_freq[x] for x in xs]

        plt.figure(figsize=(8, 4))
        plt.bar(xs, ys)
        plt.xlabel("K[x] (count per x)")
        plt.ylabel("Frequency of K[x]")
        plt.title("Frequency of K[x] Values")
        plt.tight_layout()
        plt.savefig(args.x_output, dpi=150)
        print(f"Saved plot to {args.x_output}")


if __name__ == "__main__":
    main()
