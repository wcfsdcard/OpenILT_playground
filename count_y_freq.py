#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count occurrences of y (first CSV column) and x (second CSV column) "
            "and plot frequency of K[y]/K[x] values."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Input CSV file path. You can also pass as: csv_path <path>.",
    )
    parser.add_argument(
        "--csv-path",
        dest="csv_path_flag",
        default=None,
        help="Input CSV file path (alternate flag).",
    )
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
    parser.add_argument(
        "--grid-output",
        default="xy_window_counts.png",
        help="Output plot path for the x/y window-count heatmap (default: xy_window_counts.png).",
    )
    parser.add_argument(
        "--full-size",
        type=int,
        default=2048,
        help="Full window size per axis (default: 2048).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Subwindow size per axis (default: 128).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path_value = args.csv_path_flag or args.csv_path
    if csv_path_value is None and len(sys.argv) >= 3 and sys.argv[1] == "csv_path":
        csv_path_value = sys.argv[2]
    if csv_path_value is None:
        print("Missing CSV path. Use: python count_y_freq.py <csv_path>")
        return
    csv_path = Path(csv_path_value)

    y_counts = Counter()
    x_counts = Counter()
    points = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader):
            if not row:
                continue
            if idx == 0 and args.has_header:
                continue
            y_val = row[0].strip()
            header_tokens = {"x", "y", "value", "label"}
            if idx == 0 and not args.has_header and y_val.lower() in header_tokens:
                continue
            if not y_val:
                continue
            y_counts[y_val] += 1
            if len(row) > 1:
                x_val = row[1].strip()
                if x_val:
                    x_counts[x_val] += 1
                try:
                    y_num = float(y_val)
                    x_num = float(x_val)
                except (TypeError, ValueError):
                    continue
                points.append((x_num, y_num))

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

    if args.full_size % args.window_size != 0:
        print("full-size must be divisible by window-size to build a uniform grid.")
        return

    if points:
        grid = args.full_size // args.window_size
        counts = np.zeros((grid, grid), dtype=int)
        for x_val, y_val in points:
            if 0 <= x_val < args.full_size and 0 <= y_val < args.full_size:
                gx = int(x_val) // args.window_size
                gy = int(y_val) // args.window_size
                counts[gy, gx] += 1

        plt.figure(figsize=(6, 5))
        plt.imshow(counts, origin="lower")
        plt.colorbar(label="Points per window")
        plt.xlabel("Window X index")
        plt.ylabel("Window Y index")
        plt.title("Counts per Subwindow")
        max_count = counts.max() if counts.size else 0
        for row in range(counts.shape[0]):
            for col in range(counts.shape[1]):
                value = counts[row, col]
                if value == 0:
                    continue
                color = "white" if value > max_count * 0.5 else "black"
                plt.text(col, row, str(value), ha="center", va="center", color=color, fontsize=8)
        plt.tight_layout()
        plt.savefig(args.grid_output, dpi=150)
        print(f"Saved plot to {args.grid_output}")


if __name__ == "__main__":
    main()
