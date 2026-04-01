import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(".")

from pycommon.settings import COMPLEXTYPE, DEVICE, REALTYPE
import pycommon.utils as common


class TCCLithoSim(nn.Module):
    """Forward-only lithography simulator backed by external TCC kernel files."""

    def __init__(
        self,
        config,
        kernel_path="./external_reference/tcc/optKernel_bc.npy",
        scale_path="./external_reference/tcc/optKernel_scale.npy",
        device=DEVICE,
    ):
        super(TCCLithoSim, self).__init__()
        if isinstance(config, dict):
            self._config = dict(config)
        elif isinstance(config, str):
            self._config = common.parseConfig(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)!r}")

        required = [
            "KernelNum",
            "TargetDensity",
            "PrintThresh",
            "PrintSteepness",
            "DoseMax",
            "DoseMin",
            "DoseNom",
        ]
        for key in required:
            assert key in self._config, f"[TCCLithoSim]: Cannot find the config {key}."

        self._config["KernelNum"] = int(self._config["KernelNum"])
        for key in ["TargetDensity", "PrintThresh", "PrintSteepness", "DoseMax", "DoseMin", "DoseNom"]:
            self._config[key] = float(self._config[key])

        kernel_path = Path(kernel_path)
        scale_path = Path(scale_path)
        kernel_head = np.load(kernel_path)
        kernel_scale = np.load(scale_path)
        kernel_num = self._config["KernelNum"]

        if kernel_head.ndim != 4 or kernel_head.shape[0] < 2:
            raise ValueError(f"Unexpected TCC kernel shape: {kernel_head.shape}")
        if kernel_scale.ndim != 2 or kernel_scale.shape[0] < 2:
            raise ValueError(f"Unexpected TCC scale shape: {kernel_scale.shape}")

        focus_kernel = torch.as_tensor(
            kernel_head[0, :kernel_num], dtype=COMPLEXTYPE, device=device
        )
        defocus_kernel = torch.as_tensor(
            kernel_head[1, :kernel_num], dtype=COMPLEXTYPE, device=device
        )
        focus_scale = torch.as_tensor(
            kernel_scale[0, :kernel_num], dtype=REALTYPE, device=device
        )
        defocus_scale = torch.as_tensor(
            kernel_scale[1, :kernel_num], dtype=REALTYPE, device=device
        )

        self.register_buffer("_focus_kernel", focus_kernel)
        self.register_buffer("_defocus_kernel", defocus_kernel)
        self.register_buffer("_focus_scale", focus_scale)
        self.register_buffer("_defocus_scale", defocus_scale)
        self._saved = None

    def _simulate_aerial(self, mask, dose, kernels, scales):
        added_batch = False
        if len(mask.shape) == 2:
            mask = mask[None, None, :, :]
            added_batch = True
        elif len(mask.shape) == 3:
            mask = mask[:, None, :, :]
        else:
            raise ValueError(f"[TCCLithoSim]: Invalid mask shape {mask.shape}")

        mask = mask.to(dtype=REALTYPE)
        bsz, _, height, width = mask.shape
        kernel_num, kernel_h, kernel_w = kernels.shape
        offset_h = height // 2 - kernel_h // 2
        offset_w = width // 2 - kernel_w // 2

        mask_fft = torch.fft.fftshift(torch.fft.fft2(mask * dose), dim=(-2, -1))
        mask_fft = torch.repeat_interleave(mask_fft, kernel_num, dim=1)
        fields = torch.zeros(
            (bsz, kernel_num, height, width), dtype=COMPLEXTYPE, device=mask.device
        )
        fields[
            :, :, offset_h : offset_h + kernel_h, offset_w : offset_w + kernel_w
        ] = (
            mask_fft[:, :, offset_h : offset_h + kernel_h, offset_w : offset_w + kernel_w]
            * kernels[None, :, :, :]
        )
        fields = torch.fft.ifft2(fields)
        intensity = torch.abs(fields) ** 2
        aerial = torch.sum(intensity * scales.view(1, kernel_num, 1, 1), dim=1)

        if added_batch:
            return aerial[0]
        return aerial

    def forward(self, mask):
        aerial_nom = self._simulate_aerial(
            mask, self._config["DoseNom"], self._focus_kernel, self._focus_scale
        )
        aerial_defocus_nom = self._simulate_aerial(
            mask, self._config["DoseNom"], self._defocus_kernel, self._defocus_scale
        )

        aerial_max = aerial_nom * (self._config["DoseMax"] / self._config["DoseNom"]) ** 2
        aerial_min = aerial_defocus_nom * (self._config["DoseMin"] / self._config["DoseNom"]) ** 2

        printed_nom = torch.sigmoid(
            self._config["PrintSteepness"] * (aerial_nom - self._config["TargetDensity"])
        )
        printed_max = torch.sigmoid(
            self._config["PrintSteepness"] * (aerial_max - self._config["TargetDensity"])
        )
        printed_min = torch.sigmoid(
            self._config["PrintSteepness"] * (aerial_min - self._config["TargetDensity"])
        )

        self._saved = aerial_nom, aerial_max, aerial_min
        return printed_nom, printed_max, printed_min
