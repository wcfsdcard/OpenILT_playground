import os
import sys
sys.path.append(".")
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from pycommon.settings import *
import pycommon.utils as common
import pycommon.glp as glp
import pylitho.simple as lithosim
# import pylitho.exact as lithosim

import pyilt.initializer as initializer


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_convergence_plot(history, sample_name, output_dir="./plots"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)
    iterations = range(len(history["total"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, history["total"], label="total")
    ax.plot(iterations, history["u_l2"], label="u_l2")
    ax.plot(iterations, history["u_weighted_pvbl2"], label="u_weighted_pvbl2")
    ax.plot(iterations, history["u_weighted_pvbloss"], label="u_weighted_pvbloss")
    ax.plot(iterations, history["moreau_coupling"], label="moreau_coupling")
    if "u_weighted_curv" in history:
        ax.plot(iterations, history["u_weighted_curv"], label="u_weighted_curv")
    ax.set_title(f"{sample_name} moreau convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{sample_name}_moreau_convergence.png"))
    plt.close(fig)


class SimpleCfg:
    def __init__(self, config):
        # Read the config from file or a given dict
        if isinstance(config, dict):
            self._config = config
        elif isinstance(config, str):
            self._config = common.parseConfig(config)
        required = ["Iterations", "TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize",
                    "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in required:
            assert key in self._config, f"[SimpleILT]: Cannot find the config {key}."
        self._config.setdefault("MoreauLambda", 5.0)
        self._config.setdefault("MoreauBeta", 0.1)
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in intfields:
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize",
                       "MoreauLambda", "MoreauBeta"]
        for key in floatfields:
            self._config[key] = float(self._config[key])

    def __getitem__(self, key):
        return self._config[key]


class SimpleILT:
    def __init__(self, config=SimpleCfg("./config/simpleilt2048.txt"), lithosim=lithosim.LithoSim("./config/lithosimple.txt"), device=DEVICE, multigpu=False):
        super(SimpleILT, self).__init__()
        self._config = config
        self._device = device
        # Lithosim
        self._lithosim = lithosim.to(DEVICE)
        if multigpu:
            self._lithosim = nn.DataParallel(self._lithosim)
        # Filter
        self._filter = torch.zeros([self._config["TileSizeX"], self._config["TileSizeY"]], dtype=REALTYPE, device=self._device)
        self._filter[self._config["OffsetX"]:self._config["OffsetX"]+self._config["ILTSizeX"],
                     self._config["OffsetY"]:self._config["OffsetY"]+self._config["ILTSizeY"]] = 1

    def _mask_from_logits(self, logits):
        return torch.sigmoid(self._config["SigmoidSteepness"] * logits) * self._filter

    def _base_objective(self, mask, target, curv=None):
        printedNom, printedMax, printedMin = self._lithosim(mask)
        l2loss = func.mse_loss(printedNom, target, reduction="sum")
        pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
        pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
        pvband = torch.sum((printedMax >= self._config["TargetDensity"]) != (printedMin >= self._config["TargetDensity"]))
        weighted_pvbl2 = self._config["WeightPVBL2"] * pvbl2
        weighted_pvbloss = self._config["WeightPVBand"] * pvbloss
        weighted_curv = None
        base_total = l2loss + weighted_pvbl2 + weighted_pvbloss
        if curv is not None:
            kernelCurv = torch.tensor(
                [[-1.0 / 16, 5.0 / 16, -1.0 / 16], [5.0 / 16, -1.0, 5.0 / 16], [-1.0 / 16, 5.0 / 16, -1.0 / 16]],
                dtype=REALTYPE,
                device=DEVICE,
            )
            if mask.dim() == 2:
                mask4d = mask[None, None, :, :]
            else:
                mask4d = mask[:, None, :, :]
            curvature = func.conv2d(mask4d, kernelCurv[None, None, :, :])[:, 0]
            losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
            weighted_curv = curv * losscurv
            base_total += weighted_curv
        return {
            "l2loss": l2loss,
            "weighted_pvbl2": weighted_pvbl2,
            "weighted_pvbloss": weighted_pvbloss,
            "weighted_curv": weighted_curv,
            "pvband": pvband,
            "base_total": base_total,
        }

    def solve(self, target, params, curv=None, verbose=0, record_history=False):
        # Initialize
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=REALTYPE, device=self._device)
        moreau_lambda = self._config["MoreauLambda"]
        moreau_beta = self._config["MoreauBeta"]
        u = params.clone().detach().requires_grad_(True)
        z = params.clone().detach()

        # Optimizer
        # opt = optim.SGD([u], lr=self._config["StepSize"])
        opt = optim.Adam([u], lr=self._config["StepSize"])

        history = None
        if record_history:
            history = {
                "total": [],
                "u_l2": [],
                "u_weighted_pvbl2": [],
                "u_weighted_pvbloss": [],
                "moreau_coupling": [],
                "u_z_distance": [],
            }
            if curv is not None:
                history["u_weighted_curv"] = []

        # Optimization process
        best_loss = float("inf")
        best_params = None
        best_mask = None
        best_l2 = None
        best_pvb = None
        for idx in range(self._config["Iterations"]):
            mask_u = self._mask_from_logits(u)
            objective_u = self._base_objective(mask_u, target, curv=curv)
            moreau_coupling = 0.5 / moreau_lambda * torch.sum((u - z.detach()) ** 2)
            total_loss = objective_u["base_total"] + moreau_coupling
            if verbose == 1:
                print(f"[Iteration {idx}]: L2 = {objective_u['l2loss'].item():.0f}; PVBand: {objective_u['pvband'].item():.0f}")

            if history is not None:
                history["total"].append(total_loss.item())
                history["u_l2"].append(objective_u["l2loss"].item())
                history["u_weighted_pvbl2"].append(objective_u["weighted_pvbl2"].item())
                history["u_weighted_pvbloss"].append(objective_u["weighted_pvbloss"].item())
                history["moreau_coupling"].append(moreau_coupling.item())
                if objective_u["weighted_curv"] is not None:
                    history["u_weighted_curv"].append(objective_u["weighted_curv"].item())

            if objective_u["base_total"].item() < best_loss:
                best_loss = objective_u["base_total"].item()
                best_l2 = objective_u["l2loss"].item()
                best_pvb = objective_u["pvband"].item()
                best_params = u.detach().clone()
                best_mask = mask_u.detach().clone()

            opt.zero_grad()
            total_loss.backward()
            opt.step()
            with torch.no_grad():
                z = (1.0 - moreau_beta) * z + moreau_beta * u.detach()
                if history is not None:
                    history["u_z_distance"].append(torch.norm(u.detach() - z).item())

        if history is not None:
            return best_l2, best_pvb, best_params, best_mask, history
        return best_l2, best_pvb, best_params, best_mask


def parallel():
    import cv2
    import pyilt.evaluation as evaluation

    SCALE = 4
    l2s = []
    pvbs = []
    epes = []
    shots = []
    targetsAll = []
    paramsAll = []
    cfg = SimpleCfg("./config/simpleilt512.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = SimpleILT(cfg, litho, multigpu=True)
    test = evaluation.Basic(litho, 0.5)
    epeCheck = evaluation.EPEChecker(litho, 0.5)
    shotCount = evaluation.ShotCounter(litho, 0.5)
    for idx in range(1, 11):
        print(f"[SimpleILT]: Preparing testcase {idx}")
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, params = initializer.PixelInit().run(design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        targetsAll.append(torch.unsqueeze(target, 0))
        paramsAll.append(torch.unsqueeze(params, 0))
    count = torch.cuda.device_count()
    print(f"Using {count} GPUs")
    while count > 0 and len(targetsAll) % count != 0:
        targetsAll.append(targetsAll[-1])
        paramsAll.append(paramsAll[-1])
    print(f"Augmented to {len(targetsAll)} samples. ")
    targetsAll = torch.cat(targetsAll, 0)
    paramsAll = torch.cat(paramsAll, 0)

    begin = time.time()
    l2, pvb, bestParams, bestMask = solver.solve(targetsAll, paramsAll)
    runtime = time.time() - begin

    for idx in range(1, 11):
        mask = bestMask[idx-1]
        ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        target, params = initializer.PixelInit().run(ref, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        l2, pvb = test.run(mask, target, scale=SCALE)
        epeIn, epeOut = epeCheck.run(mask, target, scale=SCALE)
        epe = epeIn + epeOut
        shot = shotCount.run(mask, shape=(512, 512))
        cv2.imwrite(f"./tmp/MOSAIC_test{idx}.png", (mask * 255).detach().cpu().numpy())

        print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        shots.append(shot)

    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.0f}; Shot {np.mean(shots):.0f}; SolveTime {runtime:.2f}s")


def serial():
    import cv2
    import pyilt.evaluation as evaluation

    SCALE = 1
    l2s = []
    pvbs = []
    epes = []
    shots = []
    runtimes = []
    ensure_dir("./plots")
    cfg = SimpleCfg("./config/simpleilt2048.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = SimpleILT(cfg, litho)
    for idx in range(1, 11):
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, params = initializer.PixelInit().run(design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])

        begin = time.time()
        l2, pvb, bestParams, bestMask, history = solver.solve(target, params, curv=None, record_history=True)
        runtime = time.time() - begin
        sample_name = f"M1_test{idx}"
        save_convergence_plot(history, sample_name)

        ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        target, params = initializer.PixelInit().run(ref, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, scale=SCALE, shots=True)
        cv2.imwrite(f"./tmp/MOSAIC_moreau_test{idx}.png", (bestMask * 255).detach().cpu().numpy())

        print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; SolveTime: {runtime:.2f}s")

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        shots.append(shot)
        runtimes.append(runtime)

    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Shot {np.mean(shots):.1f}; SolveTime {np.mean(runtimes):.2f}s")


if __name__ == "__main__":
    serial()
    # parallel()
