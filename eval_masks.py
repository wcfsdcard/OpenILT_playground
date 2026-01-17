#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

import pyilt.evaluation as evaluation
import pylitho.exact as lithosim

MASK_RE = re.compile(r"^metalSet_pixelILT_cell(?P<idx>\d+)_s=(?P<s>\d+\.\d+)\.png$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate lithography metrics for metalSet masks against targets."
    )
    parser.add_argument(
        "masks",
        nargs="*",
        help="Optional list of mask files. If omitted, scans --mask-dir with --glob.",
    )
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Mask directory (default: <repo>/data/mask).",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Target directory (default: <repo>/data/target).",
    )
    parser.add_argument(
        "--glob",
        default="metalSet_pixelILT_cell*_s=*.png",
        help="Glob pattern used when masks are not provided.",
    )
    parser.add_argument(
        "--csv-dir",
        default=None,
        help="Output directory for per-cell CSV files (default: <repo>/data/eval).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale factor for evaluation (default: infer from mask/target sizes).",
    )
    parser.add_argument(
        "--shots",
        action="store_true",
        help="Also compute shot count.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N masks after sorting.",
    )
    return parser.parse_args()


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image.astype(np.float32) / 255.0


def infer_scale(mask_shape, target_shape):
    if mask_shape == target_shape:
        return 1
    if target_shape[0] % mask_shape[0] != 0 or target_shape[1] % mask_shape[1] != 0:
        return None
    sy = target_shape[0] // mask_shape[0]
    sx = target_shape[1] // mask_shape[1]
    return sy if sx == sy else None


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    mask_dir = Path(args.mask_dir) if args.mask_dir else repo_root / "data" / "mask"
    target_dir = Path(args.target_dir) if args.target_dir else repo_root / "data" / "target"
    csv_dir = Path(args.csv_dir) if args.csv_dir else repo_root / "data" / "eval"

    if args.masks:
        mask_paths = [Path(p) for p in args.masks]
    else:
        mask_paths = sorted(mask_dir.glob(args.glob))
        if args.limit:
            mask_paths = mask_paths[: args.limit]

    if not mask_paths:
        print("No masks found. Check --mask-dir or --glob.")
        return

    litho = lithosim.LithoSim(str(repo_root / "config" / "lithosimple.txt"))

    l2s = []
    pvbs = []
    epes = []
    shots = []
    per_cell = {}

    for mask_path in mask_paths:
        match = MASK_RE.match(mask_path.name)
        if not match:
            print(f"[Skip] Unrecognized mask name: {mask_path.name}")
            continue

        idx = int(match.group("idx"))
        s_val = float(match.group("s"))
        target_path = target_dir / f"metalSet_target_cell{idx}.png"
        if not target_path.exists():
            print(f"[Skip] Missing target: {target_path}")
            continue

        mask = load_grayscale(mask_path)
        target = load_grayscale(target_path)

        scale = args.scale
        if scale is None:
            scale = infer_scale(mask.shape, target.shape)

        if scale is None:
            mask = cv2.resize(
                mask, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            scale = 1
        elif scale == 1 and mask.shape != target.shape:
            mask = cv2.resize(
                mask, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        else:
            scaled_shape = (
                int(round(mask.shape[0] * scale)),
                int(round(mask.shape[1] * scale)),
            )
            if scaled_shape != target.shape:
                target = cv2.resize(
                    target,
                    (scaled_shape[1], scaled_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

        l2, pvb, epe, shot = evaluation.evaluate(
            mask, target, litho, scale=scale, shots=args.shots
        )
        shot_str = f"; Shot {shot:.0f}" if args.shots else ""
        print(
            f"[cell{idx} s={s_val:.2f}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}{shot_str}"
        )

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        if args.shots:
            shots.append(shot)
        per_cell.setdefault(idx, []).append((s_val, l2, pvb, epe))

    if per_cell:
        csv_dir.mkdir(parents=True, exist_ok=True)
        for idx, rows in per_cell.items():
            rows_sorted = sorted(rows, key=lambda item: item[0])
            csv_path = csv_dir / f"metalSet_eval_cell{idx}.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["s", "L2", "PVBand", "EPE"])
                for s_val, l2, pvb, epe in rows_sorted:
                    writer.writerow([f"{s_val:.2f}", f"{l2:.0f}", f"{pvb:.0f}", f"{epe:.0f}"])

    if l2s:
        shot_avg = f"; Shot {np.mean(shots):.1f}" if args.shots else ""
        print(
            f"[Mean]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; "
            f"EPE {np.mean(epes):.1f}{shot_avg}"
        )


if __name__ == "__main__":
    main()
