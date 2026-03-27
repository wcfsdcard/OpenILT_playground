#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import torch

from pycommon.settings import DEVICE, REALTYPE
import pyilt.evaluation as evaluation
import pylitho.exact as openilt_litho
import pylitho.tcc_eval as tcc_litho


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare OpenILT and external TCC lithography kernels on ICCAD masks."
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=list(range(1, 11)),
        help="Optional testcase indices to evaluate (default: 1..10).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Lithography config path (default: <repo>/config/lithosimple.txt).",
    )
    parser.add_argument(
        "--tcc-kernel-path",
        default=None,
        help="TCC kernel npy path (default: <repo>/external_reference/tcc/optKernel_bc.npy).",
    )
    parser.add_argument(
        "--tcc-scale-path",
        default=None,
        help="TCC scale npy path (default: <repo>/external_reference/tcc/optKernel_scale.npy).",
    )
    parser.add_argument(
        "--center-size",
        type=int,
        default=None,
        help="Optional centered crop size for metric computation.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional output CSV path for side-by-side metrics.",
    )
    return parser.parse_args()


def load_grayscale(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image.astype("float32") / 255.0


def testcase_paths(repo_root: Path, idx: int):
    testcase = f"M1_test{idx}"
    base_dir = repo_root / "data" / testcase
    target_path = base_dir / f"{testcase}.png"
    mask_path = base_dir / f"{testcase}.png.mask_retarget_1_morph_3.png"
    return testcase, target_path, mask_path


def center_crop(tensor, size):
    if size is None:
        return tensor
    height, width = tensor.shape[-2:]
    if size > height or size > width:
        raise ValueError(f"Crop size {size} exceeds tensor shape {tensor.shape[-2:]}")
    start_h = (height - size) // 2
    start_w = (width - size) // 2
    return tensor[..., start_h : start_h + size, start_w : start_w + size]


def evaluate_metrics(mask, target, litho, center_size=None, threshold=0.5):
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=REALTYPE, device=DEVICE)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=REALTYPE, device=DEVICE)

    with torch.no_grad():
        binary_mask = torch.zeros_like(mask)
        binary_mask[mask >= threshold] = 1.0
        printed_nom, printed_max, printed_min = litho(binary_mask)

        binary_nom = torch.zeros_like(printed_nom)
        binary_max = torch.zeros_like(printed_max)
        binary_min = torch.zeros_like(printed_min)
        binary_nom[printed_nom >= threshold] = 1.0
        binary_max[printed_max >= threshold] = 1.0
        binary_min[printed_min >= threshold] = 1.0

        target_eval = center_crop(target, center_size)
        nominal_eval = center_crop(binary_nom, center_size)
        max_eval = center_crop(binary_max, center_size)
        min_eval = center_crop(binary_min, center_size)

        l2 = torch.sum((nominal_eval - target_eval) ** 2).item()
        pvb = torch.sum(max_eval != min_eval).item()
        vposes, hposes = evaluation.boundaries(target_eval)
        epe_in, epe_out, _ = evaluation.epecheck(nominal_eval, target_eval, vposes, hposes)
        epe = epe_in + epe_out
    return l2, pvb, epe


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else repo_root / "config" / "lithosimple.txt"
    tcc_kernel_path = (
        Path(args.tcc_kernel_path)
        if args.tcc_kernel_path
        else repo_root / "external_reference" / "tcc" / "optKernel_bc.npy"
    )
    tcc_scale_path = (
        Path(args.tcc_scale_path)
        if args.tcc_scale_path
        else repo_root / "external_reference" / "tcc" / "optKernel_scale.npy"
    )

    openilt = openilt_litho.LithoSim(str(config_path)).to(DEVICE)
    tcc = tcc_litho.TCCLithoSim(
        str(config_path),
        kernel_path=str(tcc_kernel_path),
        scale_path=str(tcc_scale_path),
    ).to(DEVICE)

    rows = []
    for idx in args.indices:
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

        openilt_l2, openilt_pvb, openilt_epe = evaluate_metrics(
            mask, target, openilt, args.center_size
        )
        tcc_l2, tcc_pvb, tcc_epe = evaluate_metrics(mask, target, tcc, args.center_size)
        delta_l2 = tcc_l2 - openilt_l2
        delta_pvb = tcc_pvb - openilt_pvb
        delta_epe = tcc_epe - openilt_epe

        print(
            f"[{testcase}] OpenILT: L2 {openilt_l2:.0f}; PVBand {openilt_pvb:.0f}; EPE {openilt_epe:.0f} | "
            f"TCC: L2 {tcc_l2:.0f}; PVBand {tcc_pvb:.0f}; EPE {tcc_epe:.0f} | "
            f"Delta: L2 {delta_l2:+.0f}; PVBand {delta_pvb:+.0f}; EPE {delta_epe:+.0f}"
        )

        rows.append(
            (
                testcase,
                openilt_l2,
                openilt_pvb,
                openilt_epe,
                tcc_l2,
                tcc_pvb,
                tcc_epe,
                delta_l2,
                delta_pvb,
                delta_epe,
            )
        )

    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "testcase",
                    "openilt_l2",
                    "openilt_pvband",
                    "openilt_epe",
                    "tcc_l2",
                    "tcc_pvband",
                    "tcc_epe",
                    "delta_l2",
                    "delta_pvband",
                    "delta_epe",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row[0],
                        f"{row[1]:.0f}",
                        f"{row[2]:.0f}",
                        f"{row[3]:.0f}",
                        f"{row[4]:.0f}",
                        f"{row[5]:.0f}",
                        f"{row[6]:.0f}",
                        f"{row[7]:+.0f}",
                        f"{row[8]:+.0f}",
                        f"{row[9]:+.0f}",
                    ]
                )


if __name__ == "__main__":
    main()
