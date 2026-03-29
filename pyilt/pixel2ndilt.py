import sys
sys.path.append(".")
import argparse
import time
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from pycommon.settings import *
import pycommon.utils as common
import pycommon.glp as glp
# import pylitho.simple as lithosim
# import pylitho.exact as lithosim
import pylitho.exact2 as exact2_lithosim
import pylitho.tcc_eval as tcc_lithosim

import pyilt.initializer as initializer
import pyilt.evaluation as evaluation

UPDATEHESS = 16

class NewCfg: 
    def __init__(self, config): 
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = common.parseConfig(config)
        required = ["Iterations", "TargetDensity", "SigmoidSteepness", "SigmoidOffset", "WeightPVBand", 
                    "WeightNom", "WeightMin", "WeightMax", "StepSize", "WeightCurv", "WeightArea", "ScaleTanh", 
                    "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY", "ThreshArea", "ThreshRange"]
        for key in required: 
            assert key in self._config, f"[Pixel2ndILT]: Cannot find the config {key}."
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY", "ThreshArea", "ThreshRange"]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "SigmoidSteepness", "SigmoidOffset", "WeightPVBand", 
                       "WeightNom", "WeightMin", "WeightMax", "StepSize", "WeightCurv", "WeightArea", "ScaleTanh"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
    
    def __getitem__(self, key): 
        return self._config[key]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Pixel2ndILT with either the default OpenILT kernels or external TCC kernels."
    )
    parser.add_argument(
        "--litho-kernel-source",
        choices=["openilt", "tcc"],
        default="openilt",
        help="Lithography kernel source to use during optimization and evaluation (default: openilt).",
    )
    parser.add_argument(
        "--litho-config",
        default="./config/lithosimple.txt",
        help="Lithography simulator config path (default: ./config/lithosimple.txt).",
    )
    parser.add_argument(
        "--tcc-kernel-path",
        default=None,
        help="Optional TCC kernel npy path. Defaults to <repo>/external_references/tcc/optKernel_bc.npy or <repo>/external_reference/tcc/optKernel_bc.npy.",
    )
    parser.add_argument(
        "--tcc-scale-path",
        default=None,
        help="Optional TCC scale npy path. Defaults to <repo>/external_references/tcc/optKernel_scale.npy or <repo>/external_reference/tcc/optKernel_scale.npy.",
    )
    return parser.parse_args()


def resolve_repo_root():
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(repo_root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def resolve_tcc_dir(repo_root):
    candidates = [
        repo_root / "external_references" / "tcc",
        repo_root / "external_reference" / "tcc",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a TCC directory at "
        f"{candidates[0]} or {candidates[1]}."
    )


def build_lithosim(args, repo_root):
    config_path = resolve_repo_path(repo_root, args.litho_config)
    if args.litho_kernel_source == "tcc":
        tcc_dir = resolve_tcc_dir(repo_root)
        kernel_path = (
            resolve_repo_path(repo_root, args.tcc_kernel_path)
            if args.tcc_kernel_path
            else tcc_dir / "optKernel_bc.npy"
        )
        scale_path = (
            resolve_repo_path(repo_root, args.tcc_scale_path)
            if args.tcc_scale_path
            else tcc_dir / "optKernel_scale.npy"
        )
        return tcc_lithosim.TCCLithoSim(
            str(config_path), kernel_path=str(kernel_path), scale_path=str(scale_path)
        )
    return exact2_lithosim.LithoSim(str(config_path))


class NewILT: 
    def __init__(self, config=None, lithosim=None, device=DEVICE, multigpu=False): 
        super(NewILT, self).__init__()
        if config is None:
            config = NewCfg("./config/pixelilt512.txt")
        self._config = config
        self._device = device
        # Lithosim
        if lithosim is None:
            lithosim = exact2_lithosim.LithoSim("./config/lithosimple.txt")
        self._lithosim = lithosim.to(DEVICE)
        if multigpu: 
            self._lithosim = nn.DataParallel(self._lithosim)
        # Filter
        self._filter = torch.zeros([self._config["TileSizeX"], self._config["TileSizeY"]], dtype=REALTYPE, device=self._device)
        self._filter[self._config["OffsetX"]:self._config["OffsetX"]+self._config["ILTSizeX"], \
                     self._config["OffsetY"]:self._config["OffsetY"]+self._config["ILTSizeY"]] = 1
        # self._filter = torch.ones([self._config["TileSizeX"], self._config["TileSizeY"]], dtype=REALTYPE, device=self._device)
    
    def solve(self, target, params, verbose=0): 
        # Initialize
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        if not isinstance(params, torch.Tensor): 
            params = torch.tensor(params, dtype=REALTYPE, device=self._device)
        params = params.clone().detach().requires_grad_(True)
        targetCPU = target.detach().cpu().numpy()
        dilatedCPU = cv2.dilate(targetCPU, np.ones((3,3)))
        eroded1CPU = cv2.erode(targetCPU, np.ones((3,3)))
        eroded2CPU = cv2.erode(eroded1CPU, np.ones((3,3)))
        dilated = torch.tensor(dilatedCPU, dtype=REALTYPE, device=DEVICE)
        eroded1 = torch.tensor(eroded1CPU, dtype=REALTYPE, device=DEVICE)
        eroded2 = torch.tensor(eroded2CPU, dtype=REALTYPE, device=DEVICE)

        # Pure second-order optimizer: Hessian-guided + sign() SGD (from test6.py)
        # sign() makes each step exactly lr, so scale lr for 200 iterations
        # lr = 0.12 * self._config["StepSize"]
        lr = 10 * self._config["StepSize"]
        opt = optim.SGD([params], lr=lr)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-4
        gradP = torch.zeros_like(params).requires_grad_(False)
        hessP = torch.zeros_like(params).requires_grad_(False)
        clip_gamma = 0.01

        # Optimization process
        lossBest, l2Best, pvbBest = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        lastMask = None
        lastNom = None
        for idx in range(self._config["Iterations"]): 
            if len(params.shape) == 2: 
                pooled = func.avg_pool2d(params[None, None, :, :], 7, stride=1, padding=3)[0, 0]
            else: 
                pooled = func.avg_pool2d(params.unsqueeze(1), 7, stride=1, padding=3)[:, 0]
            mask = torch.sigmoid(self._config["SigmoidSteepness"] * (pooled - self._config["SigmoidOffset"])) * self._filter
            printedNom, printedMax, printedMin = self._lithosim(mask)
            lossNom = func.mse_loss(printedNom, target, reduction="sum")
            lossMin = func.mse_loss(printedMin, target, reduction="sum")
            lossMax = func.mse_loss(printedMax, target, reduction="sum")
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            loss = self._config["WeightNom"] * lossNom + self._config["WeightMin"] * lossMin + self._config["WeightMax"] * lossMax + self._config["WeightPVBand"] * pvbloss

            # Curvature Penalty
            kernelCurv = torch.tensor([[-1.0/16, 5.0/16, -1.0/16], [5.0/16, -1.0, 5.0/16], [-1.0/16, 5.0/16, -1.0/16]], dtype=REALTYPE, device=DEVICE)
            curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
            losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
            loss += self._config["WeightCurv"] * losscurv

            # NILS
            aerialNom = self._lithosim._saved[0]
            outer = (dilated - target) * aerialNom
            bound = (target - eroded1) * aerialNom
            inner = (eroded1 - eroded2) * aerialNom
            lossnils = ((bound**2).sum()) / ((outer.sum() - inner.sum())**2 + 1e-3)
            loss += 1.5e2 * lossnils if lossnils.item() < 10 else 0.0

            # MEEF
            if lastMask is None or lastNom is None: 
                lossmeef = 0
                lastMask = mask.clone().detach()
                lastNom = printedNom.clone().detach()
            else: 
                lossmeef = ((printedNom - lastNom)**2).sum() / ((mask - lastMask)**2).sum()
                lastMask = mask.clone().detach()
                lastNom = printedNom.clone().detach()
            loss += 1.5e0 * (idx/self._config["Iterations"]) * lossmeef

            # Metric for best tracking
            l2 = func.mse_loss((printedNom > 0.5).to(REALTYPE), target, reduction="sum")
            pvb = func.mse_loss((printedMax > 0.5).to(REALTYPE), (printedMin > 0.5).to(REALTYPE), reduction="sum")
            l2 = l2.item()
            pvb = pvb.item()
            metric = l2 + pvb
            if verbose == 1: 
                print(f"[Iteration {idx}]: L2 = {l2:.0f}; PVBand: {pvb:.0f}; Loss={metric:.0f}/{lossBest:.0f}")

            if (bestParams is None) or (bestMask is None) or (metric < lossBest): 
                lossBest = metric
                l2Best = l2
                pvbBest = pvb
                bestParams = params.detach().clone()
                if len(params.shape) == 2: 
                    pooled = func.avg_pool2d(bestParams[None, None, :, :], 7, stride=1, padding=3)[0, 0]
                else: 
                    pooled = func.avg_pool2d(bestParams.unsqueeze(1), 7, stride=1, padding=3)[:, 0]
                bestMask = torch.sigmoid(self._config["SigmoidSteepness"] * (pooled - self._config["SigmoidOffset"])) * self._filter
                bestMask[bestMask > 0.5] = 1.0
                bestMask[bestMask <= 0.5] = 0.0

            # Hessian-guided second-order update with sign() gradient
            if idx % UPDATEHESS == 0: 
                # Compute Hessian diagonal via Hutchinson's method
                grad = torch.autograd.grad(loss, params, create_graph=True)[0]
                v = 2 * torch.randint_like(params, high=2) - 1
                hv = torch.autograd.grad(grad, params, grad_outputs=v, only_inputs=True, retain_graph=True)[0]
                param_size = hv.size()
                if len(param_size) <= 2: 
                    hut_trace = hv.abs()
                elif len(param_size) == 3: 
                    hut_trace = torch.mean(hv.abs(), dim=[1, 2], keepdim=True)
                elif len(param_size) == 4: 
                    hut_trace = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
                gradP.mul_(beta1).add_(grad.detach(), alpha=1-beta1)
                hessP.mul_(beta2).addcmul_(hut_trace, hut_trace, value=1-beta2)
            else: 
                grad = torch.autograd.grad(loss, params)[0]
                gradP.mul_(beta1).add_(grad.detach(), alpha=1-beta1)

            # Bias-corrected preconditioned gradient
            bias_correction1 = 1 - beta1 ** (idx + 1)
            bias_correction2 = 1 - beta2 ** (idx + 1)
            denom = (hessP.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            preconditioned = (gradP / bias_correction1 / denom).detach()
            # preconditioned = gradP.detach()
            # denom = (clip_gamma * denom).clamp_min(eps)
            # preconditioned = (gradP / bias_correction1 / denom).detach().clamp(max=1.0)

            # Sign SGD: apply sign() to preconditioned gradient
            # params.grad = preconditioned.sign()
            params.grad = preconditioned
            opt.step()
            opt.zero_grad()
        
        return l2Best, pvbBest, bestParams, bestMask


def serial(args): 
    SCALE = 4
    l2s = []
    pvbs = []
    epes = []
    shots = []
    nilses = []
    runtimes = []
    repo_root = resolve_repo_root()
    cfg   = NewCfg("./config/pixelilt512.txt")
    litho = build_lithosim(args, repo_root)
    print(f"[LithoSim] Using {args.litho_kernel_source} kernels.")
    solver = NewILT(cfg, litho)
    test = evaluation.Basic(litho, 0.5)
    epeCheck = evaluation.EPEChecker(litho, 0.5)
    shotCount = evaluation.ShotCounter(litho, 0.5)
    for idx in range(1, 11): 
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, params = initializer.PlainInit().run(design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        
        begin = time.time()
        l2, pvb, bestParams, bestMask = solver.solve(target, params)
        runtime = time.time() - begin
        
        ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        target, params = initializer.PlainInit().run(ref, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        l2, pvb = test.run(bestMask, target, scale=SCALE)
        epeIn, epeOut = epeCheck.run(bestMask, target, scale=SCALE)
        epe = epeIn + epeOut
        shot = -1 # shotCount.run(bestMask, shape=(512, 512))
        nils = evaluation.nils(bestMask, target, litho, scale=SCALE)
        mask, resist = test.sim(bestMask, target, scale=SCALE)
        cv2.imwrite(f"./tmp/Pixel2ndILT_target{idx}.png", (target * 255).detach().cpu().numpy(), (cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE))
        cv2.imwrite(f"./tmp/Pixel2ndILT_mask{idx}.png", (bestMask * 255).detach().cpu().numpy(), (cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE))
        cv2.imwrite(f"./tmp/Pixel2ndILT_resist{idx}.png", (resist * 255).detach().cpu().numpy(), (cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE))

        print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; NILS {nils:.1f}; SolveTime: {runtime:.2f}s")

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        shots.append(shot)
        nilses.append(nils)
        runtimes.append(runtime)
    
    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Shot {np.mean(shots):.1f}; NILS {np.mean(nilses):.1f}; SolveTime {np.mean(runtimes):.2f}s")


if __name__ == "__main__": 
    serial(parse_args())

'''
[Testcase 1]: L2 36271; PVBand 47102; EPE 3; Shot: -1; NILS 103.3; SolveTime: 4.35s
[Testcase 2]: L2 28450; PVBand 37383; EPE 0; Shot: -1; NILS 113.0; SolveTime: 2.54s
[Testcase 3]: L2 57921; PVBand 82784; EPE 11; Shot: -1; NILS 47.4; SolveTime: 2.29s
[Testcase 4]: L2 10321; PVBand 21326; EPE 0; Shot: -1; NILS 46.8; SolveTime: 2.52s
[Testcase 5]: L2 27853; PVBand 51444; EPE 0; Shot: -1; NILS 915.2; SolveTime: 2.53s
[Testcase 6]: L2 29307; PVBand 45445; EPE 0; Shot: -1; NILS 1280.2; SolveTime: 2.54s
[Testcase 7]: L2 15127; PVBand 38001; EPE 0; Shot: -1; NILS 667.0; SolveTime: 2.54s
[Testcase 8]: L2 11757; PVBand 18823; EPE 0; Shot: -1; NILS 155.6; SolveTime: 2.54s
[Testcase 9]: L2 33062; PVBand 58430; EPE 0; Shot: -1; NILS 1339.4; SolveTime: 2.53s
[Testcase 10]: L2 8330; PVBand 15657; EPE 0; Shot: -1; NILS 137.3; SolveTime: 2.53s
[Result]: L2 25840; PVBand 41640; EPE 1.4; Shot -1.0; NILS 480.5; SolveTime 2.69s
'''
