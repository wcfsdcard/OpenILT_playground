import os
import sys
sys.path.append(".")
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from pycommon.settings import *
import pycommon.utils as common
import pycommon.glp as glp
# import pylitho.simple as lithosim
import pylitho.exact as lithosim

import pyilt.initializer as initializer
import pyilt.evaluation as evaluation


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_convergence_plot(history, sample_name, level_name, output_dir="./plots"):
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
    ax.plot(iterations, history["u_weighted_curv"], label="u_weighted_curv")
    ax.set_title(f"{sample_name} {level_name} moreau convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{sample_name}_{level_name}_moreau_convergence.png"))
    plt.close(fig)


class CurvILTCfg:
    def __init__(self, config):
        # Read the config from file or a given dict
        if isinstance(config, dict):
            self._config = config
        elif isinstance(config, str):
            self._config = common.parseConfig(config)
        required = ["Iterations", "TargetDensity", "SigmoidSteepness", "SigmoidOffset", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize",
                    "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in required:
            assert key in self._config, f"[CurvILT]: Cannot find the config {key}."
        self._config.setdefault("MoreauLambda", 5.0)
        self._config.setdefault("MoreauBeta", 0.1)
        self._config.setdefault("MoreauLambdaDecay", 1.0)
        self._config.setdefault("MoreauLambdaMin", self._config["MoreauLambda"])
        self._config.setdefault("MoreauRandomInitZ", 0)
        self._config.setdefault("MoreauRandomInitZStd", 1.0)
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY", "MoreauRandomInitZ"]
        for key in intfields:
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "SigmoidSteepness", "SigmoidOffset", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize",
                       "MoreauLambda", "MoreauBeta", "MoreauLambdaDecay", "MoreauLambdaMin", "MoreauRandomInitZStd"]
        for key in floatfields:
            self._config[key] = float(self._config[key])
        assert self._config["MoreauLambda"] > 0.0, "[CurvILT]: MoreauLambda must be positive."
        assert self._config["MoreauLambdaDecay"] > 0.0, "[CurvILT]: MoreauLambdaDecay must be positive."
        assert self._config["MoreauLambdaMin"] > 0.0, "[CurvILT]: MoreauLambdaMin must be positive."
        assert self._config["MoreauRandomInitZStd"] >= 0.0, "[CurvILT]: MoreauRandomInitZStd must be non-negative."

    def __getitem__(self, key):
        return self._config[key]


class CurvILT:
    def __init__(self, config, lithosim=lithosim.LithoSim("./config/lithosimple.txt"), device=DEVICE, multigpu=True):
        super(CurvILT, self).__init__()
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

    def _pooled_from_logits(self, logits):
        if len(logits.shape) == 2:
            return func.avg_pool2d(logits[None, None, :, :], 7, stride=1, padding=3)[0, 0]
        return func.avg_pool2d(logits.unsqueeze(1), 7, stride=1, padding=3)[:, 0]

    def _mask_from_logits(self, logits):
        pooled = self._pooled_from_logits(logits)
        return torch.sigmoid(self._config["SigmoidSteepness"] * (pooled - self._config["SigmoidOffset"])) * self._filter

    def _base_objective(self, mask, target):
        printedNom, printedMax, printedMin = self._lithosim(mask)
        l2loss = func.mse_loss(printedMax, target, reduction="sum")
        pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
        pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
        pvband = torch.sum((printedMax >= self._config["TargetDensity"]) != (printedMin >= self._config["TargetDensity"]))

        kernelCurv = torch.tensor([[-1.0 / 16, 5.0 / 16, -1.0 / 16], [5.0 / 16, -1.0, 5.0 / 16], [-1.0 / 16, 5.0 / 16, -1.0 / 16]], dtype=REALTYPE, device=DEVICE)
        if mask.dim() == 2:
            printed_nom_4d = printedNom[None, None, :, :]
            curvature = func.conv2d(printed_nom_4d, kernelCurv[None, None, :, :])[0, 0]
        else:
            printed_nom_4d = printedNom[:, None, :, :]
            curvature = func.conv2d(printed_nom_4d, kernelCurv[None, None, :, :])[:, 0]
        losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
        weighted_curv = 2e2 * losscurv

        weighted_pvbl2 = self._config["WeightPVBL2"] * pvbl2
        weighted_pvbloss = self._config["WeightPVBand"] * pvbloss
        base_total = l2loss + weighted_pvbl2 + weighted_pvbloss + weighted_curv
        return {
            "l2loss": l2loss,
            "weighted_pvbl2": weighted_pvbl2,
            "weighted_pvbloss": weighted_pvbloss,
            "pvband": pvband,
            "weighted_curv": weighted_curv,
            "base_total": base_total,
        }

    def solve(self, target, params, init_z=None, verbose=0, record_history=False):
        # Initialize
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=REALTYPE, device=self._device)
        moreau_lambda = self._config["MoreauLambda"]
        moreau_beta = self._config["MoreauBeta"]
        moreau_lambda_decay = self._config["MoreauLambdaDecay"]
        moreau_lambda_min = self._config["MoreauLambdaMin"]
        u = params.clone().detach().requires_grad_(True)
        if init_z is not None:
            if not isinstance(init_z, torch.Tensor):
                init_z = torch.tensor(init_z, dtype=REALTYPE, device=self._device)
            z = init_z.clone().detach()
        elif self._config["MoreauRandomInitZ"]:
            z = params.clone().detach() + self._config["MoreauRandomInitZStd"] * torch.randn_like(params)
        else:
            z = 0.5*torch.ones_like(params)
            # z = params.clone().detach()

        # Optimizer
        opt = optim.SGD([u], lr=self._config["StepSize"])
        # opt = optim.Adam([u], lr=self._config["StepSize"])

        history = None
        if record_history:
            history = {
                "total": [],
                "u_l2": [],
                "u_weighted_pvbl2": [],
                "u_weighted_pvbloss": [],
                "u_weighted_curv": [],
                "moreau_coupling": [],
                "moreau_lambda": [],
                "u_z_distance": [],
            }

        # Optimization process
        best_loss = float("inf")
        best_params = None
        best_mask = None
        best_l2 = None
        best_pvb = None
        for idx in range(self._config["Iterations"]):
            mask_u = self._mask_from_logits(u)
            objective_u = self._base_objective(mask_u, target)
            moreau_coupling = 0.5 / moreau_lambda * torch.sum((u - z.detach()) ** 2)
            total_loss = objective_u["base_total"] + moreau_coupling
            if verbose == 1:
                print(f"[Iteration {idx}]: L2 = {objective_u['l2loss'].item():.0f}; PVBand: {objective_u['pvband'].item():.0f}")

            if history is not None:
                history["total"].append(total_loss.item())
                history["u_l2"].append(objective_u["l2loss"].item())
                history["u_weighted_pvbl2"].append(objective_u["weighted_pvbl2"].item())
                history["u_weighted_pvbloss"].append(objective_u["weighted_pvbloss"].item())
                history["u_weighted_curv"].append(objective_u["weighted_curv"].item())
                history["moreau_coupling"].append(moreau_coupling.item())
                history["moreau_lambda"].append(moreau_lambda)

            if objective_u["base_total"].item() < best_loss:
                best_loss = objective_u["base_total"].item()
                best_l2 = objective_u["l2loss"].item()
                best_pvb = objective_u["pvband"].item()
                best_params = u.detach().clone()
                best_mask = self._mask_from_logits(best_params).detach().clone()
                best_mask[best_mask > 0.5] = 1.0
                best_mask[best_mask <= 0.5] = 0.0

            opt.zero_grad()
            total_loss.backward()
            opt.step()
            with torch.no_grad():
                z = (1.0 - moreau_beta) * z + moreau_beta * u.detach()
                if history is not None:
                    history["u_z_distance"].append(torch.norm(u.detach() - z).item())
            moreau_lambda = max(moreau_lambda_min, moreau_lambda * moreau_lambda_decay)

        final_z = z.detach().clone()
        if history is not None:
            return best_l2, best_pvb, best_params, best_mask, final_z, history
        return best_l2, best_pvb, best_params, best_mask, final_z


if __name__ == "__main__":
    ScaleLow = 8
    ScaleMid = 4
    ScaleHigh = 2
    l2s = []
    pvbs = []
    epes = []
    runtimes = []
    ensure_dir("./plots")
    ensure_dir("./tmp")
    cfgLow = CurvILTCfg("./config/curvilt256.txt")
    cfgMid = CurvILTCfg("./config/curvilt512.txt")
    cfgHigh = CurvILTCfg("./config/curvilt1024.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solverLow = CurvILT(cfgLow, litho)
    solverMid = CurvILT(cfgMid, litho)
    solverHigh = CurvILT(cfgHigh, litho)
    test = evaluation.Basic(litho, 0.5)
    epeCheck = evaluation.EPEChecker(litho, 0.5)
    for idx in range(1, 11):
        runtime = 0
        sample_name = f"CurvILTMoreau_test{idx}"
        # Reference
        ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref.center(cfgMid["TileSizeX"]*ScaleMid, cfgMid["TileSizeY"]*ScaleMid, cfgMid["OffsetX"]*ScaleMid, cfgMid["OffsetY"]*ScaleMid)
        # Low resolution
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=ScaleLow)
        design.center(cfgLow["TileSizeX"], cfgLow["TileSizeY"], cfgLow["OffsetX"], cfgLow["OffsetY"])
        target, params = initializer.PixelInit().run(design, cfgLow["TileSizeX"], cfgLow["TileSizeY"], cfgLow["OffsetX"], cfgLow["OffsetY"])
        begin = time.time()
        l2, pvb, bestParamsLow, bestMaskLow, finalZLow, historyLow = solverLow.solve(target, target, init_z=None, record_history=True)
        runtime += time.time() - begin
        save_convergence_plot(historyLow, sample_name, "low")
        # -> Evaluation
        target, params = initializer.PixelInit().run(ref, cfgLow["TileSizeX"]*ScaleLow, cfgLow["TileSizeY"]*ScaleLow, cfgLow["OffsetX"]*ScaleLow, cfgLow["OffsetY"]*ScaleLow)
        l2, pvb = test.run(bestMaskLow, target, scale=ScaleLow)
        epeIn, epeOut = epeCheck.run(bestMaskLow, target, scale=ScaleLow)
        epe = epeIn + epeOut
        logLow = f"L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}"
        # Mid resolution
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=ScaleMid)
        design.center(cfgMid["TileSizeX"], cfgMid["TileSizeY"], cfgMid["OffsetX"], cfgMid["OffsetY"])
        target, params = initializer.PixelInit().run(design, cfgMid["TileSizeX"], cfgMid["TileSizeY"], cfgMid["OffsetX"], cfgMid["OffsetY"])
        paramsMid = func.interpolate(bestParamsLow[None, None, :, :], scale_factor=2, mode="nearest")[0, 0]
        zMid = func.interpolate(finalZLow[None, None, :, :], scale_factor=2, mode="nearest")[0, 0]
        begin = time.time()
        l2, pvb, bestParamsMid, bestMaskMid, finalZMid, historyMid = solverMid.solve(target, paramsMid, init_z=zMid, record_history=True)
        runtime += time.time() - begin
        save_convergence_plot(historyMid, sample_name, "mid")
        # -> Evaluation
        target, params = initializer.PixelInit().run(ref, cfgMid["TileSizeX"]*ScaleMid, cfgMid["TileSizeY"]*ScaleMid, cfgMid["OffsetX"]*ScaleMid, cfgMid["OffsetY"]*ScaleMid)
        l2, pvb = test.run(bestMaskMid, target, scale=ScaleMid)
        epeIn, epeOut = epeCheck.run(bestMaskMid, target, scale=ScaleMid)
        epe = epeIn + epeOut
        logMid = f"L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}"
        # High resolution
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=ScaleHigh)
        design.center(cfgHigh["TileSizeX"], cfgHigh["TileSizeY"], cfgHigh["OffsetX"], cfgHigh["OffsetY"])
        target, params = initializer.PixelInit().run(design, cfgHigh["TileSizeX"], cfgHigh["TileSizeY"], cfgHigh["OffsetX"], cfgHigh["OffsetY"])
        paramsHigh = func.interpolate(bestParamsMid[None, None, :, :], scale_factor=2, mode="nearest")[0, 0]
        zHigh = func.interpolate(finalZMid[None, None, :, :], scale_factor=2, mode="nearest")[0, 0]
        begin = time.time()
        l2, pvb, bestParamsHigh, bestMaskHigh, finalZHigh, historyHigh = solverHigh.solve(target, paramsHigh, init_z=zHigh, record_history=True)
        runtime += time.time() - begin
        save_convergence_plot(historyHigh, sample_name, "high")
        # -> Evaluation
        target, params = initializer.PixelInit().run(ref, cfgHigh["TileSizeX"]*ScaleHigh, cfgHigh["TileSizeY"]*ScaleHigh, cfgHigh["OffsetX"]*ScaleHigh, cfgHigh["OffsetY"]*ScaleHigh)
        l2, pvb, epe, shot = evaluation.evaluate(bestMaskHigh, target, litho, scale=ScaleHigh, shots=True)
        logHigh = f"L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shots: {shot:.0f}"
        # Print Information
        print(f"[Testcase {idx}]: Low: {logLow} -> Mid: {logMid} -> High: {logHigh}; Runtime: {runtime:.2f}s")
        mask, resist = test.sim(bestMaskHigh, target, scale=ScaleHigh)
        cv2.imwrite(f"tmp/CurvILTMoreau_target{idx}.png", cv2.resize((target * 255).detach().cpu().numpy(), (2048, 2048)))
        cv2.imwrite(f"tmp/CurvILTMoreau_mask{idx}.png", cv2.resize((mask * 255).detach().cpu().numpy(), (2048, 2048)))
        cv2.imwrite(f"tmp/CurvILTMoreau_resist{idx}.png", cv2.resize((resist * 255).detach().cpu().numpy(), (2048, 2048)))
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        runtimes.append(runtime)

    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Runtime: {np.mean(runtimes):.2f}s")
