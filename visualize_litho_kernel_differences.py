#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize differences between OpenILT and external TCC lithography kernels."
    )
    parser.add_argument(
        "--openilt-kernel-dir",
        default=None,
        help="OpenILT kernel directory (default: <repo>/kernel/kernels).",
    )
    parser.add_argument(
        "--openilt-scale-dir",
        default=None,
        help="OpenILT scale directory (default: <repo>/kernel/scales).",
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
        "--kernel-num",
        type=int,
        default=24,
        help="Number of kernels to compare from each family (default: 24).",
    )
    parser.add_argument(
        "--component",
        choices=["magnitude", "real", "imag", "phase"],
        default="magnitude",
        help="Scalar component to visualize per kernel layer (default: magnitude).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated figures (default: <repo>/data/eval/kernel_diff_visuals).",
    )
    return parser.parse_args()


def load_openilt_family(kernel_dir: Path, scale_dir: Path, family: str, kernel_num: int):
    kernels = torch.load(kernel_dir / f"{family}.pt", map_location="cpu").numpy().transpose(2, 0, 1)
    scales = torch.load(scale_dir / f"{family}.pt", map_location="cpu").numpy()
    return kernels[:kernel_num], scales[:kernel_num]


def load_tcc_family(kernel_path: Path, scale_path: Path, family: str, kernel_num: int):
    kernels = np.load(kernel_path)
    scales = np.load(scale_path)
    family_index = 0 if family == "focus" else 1
    return kernels[family_index, :kernel_num], scales[family_index, :kernel_num]


def component_layers(kernels, component: str):
    if component == "magnitude":
        return np.abs(kernels), "magma", "magnitude"
    if component == "real":
        return kernels.real, "coolwarm", "real"
    if component == "imag":
        return kernels.imag, "coolwarm", "imag"
    if component == "phase":
        return np.angle(kernels), "twilight", "phase"
    raise ValueError(f"Unsupported component: {component}")


def diff_layers(openilt_kernels, tcc_kernels, component: str):
    if component == "magnitude":
        return np.abs(openilt_kernels - tcc_kernels), "magma", "abs_diff"
    if component == "real":
        return openilt_kernels.real - tcc_kernels.real, "coolwarm", "real_diff"
    if component == "imag":
        return openilt_kernels.imag - tcc_kernels.imag, "coolwarm", "imag_diff"
    if component == "phase":
        return np.angle(openilt_kernels * np.conj(tcc_kernels)), "twilight", "phase_diff"
    raise ValueError(f"Unsupported component: {component}")


def color_limits(layers, component: str, is_diff: bool):
    if component == "phase":
        return -np.pi, np.pi
    if component == "magnitude" and not is_diff:
        return 0.0, float(layers.max())
    if component == "magnitude" and is_diff:
        return 0.0, max(float(layers.max()), 1e-12)
    vmax = max(float(np.abs(layers).max()), 1e-12)
    return -vmax, vmax


def save_layer_grid(output_path: Path, family: str, title: str, layers, cmap: str, vmin: float, vmax: float):
    n_layers = layers.shape[0]
    cols = 6
    rows = int(np.ceil(n_layers / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    last_im = None
    for idx in range(rows * cols):
        axis = axes[idx // cols, idx % cols]
        if idx >= n_layers:
            axis.axis("off")
            continue
        last_im = axis.imshow(layers[idx], cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(f"K{idx}")
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(f"{family} {title}", y=0.995)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scale_plot(output_path: Path, family: str, openilt_scales, tcc_scales):
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(openilt_scales, marker="o", label="OpenILT")
    axis.plot(tcc_scales, marker="x", label="TCC")
    axis.set_title(f"{family} scale vectors")
    axis.set_xlabel("Kernel index")
    axis.set_ylabel("Scale")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    openilt_kernel_dir = (
        Path(args.openilt_kernel_dir) if args.openilt_kernel_dir else repo_root / "kernel" / "kernels"
    )
    openilt_scale_dir = (
        Path(args.openilt_scale_dir) if args.openilt_scale_dir else repo_root / "kernel" / "scales"
    )
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
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / "data" / "eval" / "kernel_diff_visuals"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for family in ["focus", "defocus"]:
        openilt_kernels, openilt_scales = load_openilt_family(
            openilt_kernel_dir, openilt_scale_dir, family, args.kernel_num
        )
        tcc_kernels, tcc_scales = load_tcc_family(
            tcc_kernel_path, tcc_scale_path, family, args.kernel_num
        )

        openilt_layers, component_cmap, component_label = component_layers(
            openilt_kernels, args.component
        )
        tcc_layers, _, _ = component_layers(tcc_kernels, args.component)
        diff_layer_values, diff_cmap, diff_label = diff_layers(
            openilt_kernels, tcc_kernels, args.component
        )

        combined_layers = np.concatenate([openilt_layers, tcc_layers], axis=0)
        openilt_vmin, openilt_vmax = color_limits(combined_layers, args.component, is_diff=False)
        tcc_vmin, tcc_vmax = openilt_vmin, openilt_vmax
        diff_vmin, diff_vmax = color_limits(diff_layer_values, args.component, is_diff=True)

        save_layer_grid(
            output_dir / f"{family}_openilt_{component_label}_layers.png",
            family,
            f"OpenILT {component_label} layers",
            openilt_layers,
            component_cmap,
            openilt_vmin,
            openilt_vmax,
        )
        save_layer_grid(
            output_dir / f"{family}_tcc_{component_label}_layers.png",
            family,
            f"TCC {component_label} layers",
            tcc_layers,
            component_cmap,
            tcc_vmin,
            tcc_vmax,
        )
        save_layer_grid(
            output_dir / f"{family}_{diff_label}_layers.png",
            family,
            f"{diff_label} layers",
            diff_layer_values,
            diff_cmap,
            diff_vmin,
            diff_vmax,
        )
        save_scale_plot(output_dir / f"{family}_scales.png", family, openilt_scales, tcc_scales)

    print(f"Wrote kernel comparison figures to {output_dir}")


if __name__ == "__main__":
    main()
