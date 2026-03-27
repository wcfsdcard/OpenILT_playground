#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import pyilt.evaluation as evaluation
import pylitho.exact as lithosim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate lithography metrics for ICCAD M1_test masks."
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Output CSV path (default: <repo>/data/eval/m1_iccad_eval.csv).",
    )
    return parser.parse_args()


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image.astype(np.float32) / 255.0


def testcase_paths(repo_root: Path, idx: int):
    testcase = f"M1_test{idx}"
    base_dir = repo_root / "data" / testcase
    target_path = base_dir / f"{testcase}.png"
    mask_path = base_dir / f"{testcase}.png.mask_retarget_1_morph_3.png"
    return testcase, target_path, mask_path


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    csv_path = Path(args.csv_path) if args.csv_path else repo_root / "data" / "eval" / "m1_iccad_eval.csv"

    litho = lithosim.LithoSim(str(repo_root / "config" / "lithosimple.txt"))

    rows = []
    l2s = []
    pvbs = []
    epes = []
    shots = []

    for idx in range(1, 11):
        testcase, target_path, mask_path = testcase_paths(repo_root, idx)

        if not target_path.exists():
            raise FileNotFoundError(f"Missing target image for {testcase}: {target_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask image for {testcase}: {mask_path}")

        target = load_grayscale(target_path)
        mask = load_grayscale(mask_path)

        if mask.shape != target.shape:
            mask = cv2.resize(
                mask, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        l2, pvb, epe, shot = evaluation.evaluate(mask, target, litho, scale=1, shots=False)

        print(
            f"[{testcase}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot {shot:.0f}"
        )

        rows.append((testcase, l2, pvb, epe, shot))
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        shots.append(shot)

    avg_l2 = float(np.mean(l2s))
    avg_pvb = float(np.mean(pvbs))
    avg_epe = float(np.mean(epes))
    avg_shot = float(np.mean(shots))

    print(
        f"[Average]: L2 {avg_l2:.0f}; PVBand {avg_pvb:.0f}; EPE {avg_epe:.1f}; Shot {avg_shot:.1f}"
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["testcase", "L2", "PVBand", "EPE", "Shots"])
        for testcase, l2, pvb, epe, shot in rows:
            writer.writerow(
                [testcase, f"{l2:.0f}", f"{pvb:.0f}", f"{epe:.0f}", f"{shot:.0f}"]
            )
        writer.writerow(
            [
                "Average",
                f"{avg_l2:.0f}",
                f"{avg_pvb:.0f}",
                f"{avg_epe:.1f}",
                f"{avg_shot:.1f}",
            ]
        )


if __name__ == "__main__":
    main()
