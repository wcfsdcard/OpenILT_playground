import sys
import json
import time
sys.path.append(".")

import torch
import torch.nn as nn

from pycommon.settings import *
import pycommon.utils as common 


class Kernel:
    def __init__(self, basedir="./kernel", defocus=False, conjuncture=False, combo=False, device=DEVICE):
        self._basedir = basedir
        self._defocus = defocus
        self._conjuncture = conjuncture
        self._combo = combo
        self._device = device

        self._kernels = torch.load(self._kernel_file(), map_location=device, weights_only=True).permute(2, 0, 1)
        self._scales = torch.load(self._scale_file(), map_location=device, weights_only=True)

        self._knx, self._kny = self._kernels.shape[:2]

    @property
    def kernels(self): 
        return self._kernels
        
    @property
    def scales(self): 
        return self._scales

    def _kernel_file(self):
        filename = ""
        if self._defocus:
            filename = "defocus" + filename
        else:
            filename = "focus" + filename
        if self._conjuncture:
            filename = "ct_" + filename
        if self._combo:
            filename = "combo_" + filename
        filename = self._basedir + "/kernels/" + filename + ".pt"
        return filename

    def _scale_file(self):
        filename = self._basedir + "/scales/"
        if self._combo:
            return filename + "combo.pt"
        else:
            if self._defocus:
                return filename + "defocus.pt"
            else:
                return filename + "focus.pt"

def _maskFloat(mask, dose):
    return (dose * mask).to(COMPLEXTYPE)

def _kernelMult(kernel, maskFFT, kernelNum):
    # kernel: [24, 35, 35]
    knx, kny = kernel.shape[-2:]
    knxh, knyh = knx // 2, kny // 2
    output = None
    if kernel.device != maskFFT.device: 
        kernel = kernel.to(maskFFT.device)
    if len(maskFFT.shape) == 3: 
        output = torch.zeros([kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :knxh+1, :knyh+1] = maskFFT[:, :knxh+1, :knyh+1] * kernel[:kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :knxh+1, -knyh:] = maskFFT[:, :knxh+1, -knyh:] * kernel[:kernelNum, -(knxh+1):, :knyh]
        output[:, -knxh:, :knyh+1] = maskFFT[:, -knxh:, :knyh+1] * kernel[:kernelNum, :knxh, -(knyh+1):]
        output[:, -knxh:, -knyh:] = maskFFT[:, -knxh:, -knyh:] * kernel[:kernelNum, :knxh, :knyh]
    else: 
        assert len(maskFFT.shape) == 4, f"[_kernelMult]: Invalid shape of maskFFT: {maskFFT.shape}"
        output = torch.zeros([maskFFT.shape[0], kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :, :knxh+1, :knyh+1] = maskFFT[:, :, :knxh+1, :knyh+1] * kernel[None, :kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :, :knxh+1, -knyh:]  = maskFFT[:, :, :knxh+1, -knyh:]  * kernel[None, :kernelNum, -(knxh+1):, :knyh]
        output[:, :, -knxh:, :knyh+1]  = maskFFT[:, :, -knxh:, :knyh+1]  * kernel[None, :kernelNum, :knxh, -(knyh+1):]
        output[:, :, -knxh:, -knyh:]   = maskFFT[:, :, -knxh:, -knyh:]   * kernel[None, :kernelNum, :knxh, :knyh]
    return output

def _computeImageMask(cmask, kernel, scale, kernelNum):
    # cmask: [2048, 2048], kernel: [24, 35, 35], scale: [24]
    if scale.device != cmask.device: 
        scale = scale.to(cmask.device)
    cmask = torch.unsqueeze(cmask, len(cmask.shape) - 2)
    cmask_fft = torch.fft.fft2(cmask, norm="forward")
    tmp = _kernelMult(kernel, cmask_fft, kernelNum)
    tmp = torch.fft.ifft2(tmp, norm="forward")
    return tmp
def _convMask(mask, dose, kernel, scale, kernelNum): 
    cmask = _maskFloat(mask, dose)
    image = _computeImageMask(cmask, kernel, scale, kernelNum)
    return image
def lithosim(mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad): 
    if len(mask.shape) == 4 and mask.shape[0] == 1: 
        pass
    tmp = _convMask(mask, dose, kernel, scale, kernelNum)
    if len(mask.shape) == 2: 
        scale = scale[:kernelNum].unsqueeze(1).unsqueeze(2)
        return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=0)
    else: 
        assert len(mask.shape) == 3, f"[_LithoSim.forward]: Invalid shape: {mask.shape}"
        scale = scale[:kernelNum].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=1)

def _fftMask(mask, dose): 
    cmask = _maskFloat(mask, dose)
    cmask = torch.unsqueeze(cmask, len(cmask.shape) - 2)
    cmask_fft = torch.fft.fft2(cmask, norm="forward")
    return cmask_fft
def _computeConv(cmask_fft, kernel, kernelNum): 
    tmp = _kernelMult(kernel, cmask_fft, kernelNum)
    tmp = torch.fft.ifft2(tmp, norm="forward")
    return tmp
def _lithosim(cmask_fft, kernel, scale, kernelNum): 
    if scale.device != cmask_fft.device: 
        scale = scale.to(cmask_fft.device)
    tmp = _computeConv(cmask_fft, kernel, kernelNum)
    if len(cmask_fft.shape) == 3: 
        scale = scale[:kernelNum].unsqueeze(1).unsqueeze(2)
        # return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=0)
        return torch.sum(scale * torch.view_as_real(tmp).square_().sum(dim=-1), dim=0)
    else: 
        assert len(cmask_fft.shape) == 4, f"[_LithoSim.forward]: Invalid shape: {cmask_fft.shape}"
        scale = scale[:kernelNum].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=1)
        return torch.sum(scale * torch.view_as_real(tmp).square_().sum(dim=-1), dim=1)

def _together(mask, dose, kernel1, scale1, kernel2, scale2, kernelNum): 
    if kernel1.device != mask.device: 
        kernel1 = kernel1
    if kernel2.device != mask.device: 
        kernel2 = kernel2.to(mask.device)
    if scale1.device != mask.device: 
        scale1 = scale1.to(mask.device)
    if scale2.device != mask.device: 
        scale2 = scale2.to(mask.device)
    cmask = _maskFloat(mask, dose)
    cmask = torch.unsqueeze(cmask, len(cmask.shape) - 2)
    maskFFT = torch.fft.fft2(cmask, norm="forward")
    extended = maskFFT[None, ...]
    
    kernel = torch.cat([kernel1[None, ...], kernel2[None, ...]])
    scale = torch.cat([scale1[None, ...], scale2[None, ...]])
    knx, kny = kernel.shape[-2:]
    knxh, knyh = knx // 2, kny // 2
    output = None
    if len(maskFFT.shape) == 3: 
        output = torch.zeros([2, kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :, :knxh+1, :knyh+1] = extended[..., :knxh+1, :knyh+1] * kernel[:, :kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :, :knxh+1, -knyh:] = extended[..., :knxh+1, -knyh:] * kernel[:, :kernelNum, -(knxh+1):, :knyh]
        output[:, :, -knxh:, :knyh+1] = extended[..., -knxh:, :knyh+1] * kernel[:, :kernelNum, :knxh, -(knyh+1):]
        output[:, :, -knxh:, -knyh:] = extended[..., -knxh:, -knyh:] * kernel[:, :kernelNum, :knxh, :knyh]
    else: 
        assert len(maskFFT.shape) == 4, f"[_kernelMult]: Invalid shape of maskFFT: {maskFFT.shape}"
        output = torch.zeros([2, maskFFT.shape[0], kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :, :, :knxh+1, :knyh+1] = extended[..., :knxh+1, :knyh+1] * kernel[None, :, :kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :, :, :knxh+1, -knyh:]  = extended[..., :knxh+1, -knyh:]  * kernel[None, :, :kernelNum, -(knxh+1):, :knyh]
        output[:, :, :, -knxh:, :knyh+1]  = extended[..., -knxh:, :knyh+1]  * kernel[None, :, :kernelNum, :knxh, -(knyh+1):]
        output[:, :, :, -knxh:, -knyh:]   = extended[..., -knxh:, -knyh:]   * kernel[None, :, :kernelNum, :knxh, :knyh]
    
    tmp = torch.fft.ifft2(output, norm="forward")
    if len(maskFFT.shape) == 3: 
        scale = scale[:, :kernelNum, None, None]
        result = torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=1)
        return result[0], result[1]
    else: 
        assert len(maskFFT.shape) == 4, f"[_LithoSim.forward]: Invalid shape: {maskFFT.shape}"
        scale = scale[:, None, :kernelNum, None, None]
        result = torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=2)
        return result[0], result[1]


class LithoSim(nn.Module): # Mask -> Aerial -> Printed
    def __init__(self, config): 
        super(LithoSim, self).__init__()
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = common.parseConfig(config)
        required = ["KernelDir", "KernelNum", "TargetDensity", "PrintThresh", "PrintSteepness", "DoseMax", "DoseMin", "DoseNom"]
        for key in required: 
            assert key in self._config, f"[LithoSim]: Cannot find the config {key}."
        intfields = ["KernelNum", ]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "PrintThresh", "PrintSteepness", "DoseMax", "DoseMin", "DoseNom"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
        # Read the kernels
        self._kernels = {"focus": Kernel(self._config["KernelDir"]), 
                         "defocus": Kernel(self._config["KernelDir"], defocus=True)}
        self._saved = None

    def forward(self, mask): 
        simmed1, simmed2 = _together(mask, self._config["DoseNom"], 
                                     self._kernels["focus"].kernels, self._kernels["focus"].scales, 
                                     self._kernels["defocus"].kernels, self._kernels["defocus"].scales, 
                                     self._config["KernelNum"])
        aerialNom = simmed1
        aerialMax = simmed1 * (self._config["DoseMax"]/self._config["DoseNom"])**2
        aerialMin = simmed2 * (self._config["DoseMin"]/self._config["DoseNom"])**2
        printedNom = torch.sigmoid(self._config["PrintSteepness"] * (aerialNom - self._config["TargetDensity"]))
        printedMax = torch.sigmoid(self._config["PrintSteepness"] * (aerialMax - self._config["TargetDensity"]))
        printedMin = torch.sigmoid(self._config["PrintSteepness"] * (aerialMin - self._config["TargetDensity"]))
        self._saved = aerialNom, aerialMax, aerialMin
        return printedNom, printedMax, printedMin

if __name__ == "__main__":
    import pycommon.glp as glp
    lithosim = LithoSim("./config/lithosimple.txt")
    image = glp.Design("./benchmark/ICCAD2013/M1_test1.glp").image()
    image = torch.tensor(image > 0.0, dtype=REALTYPE, device=DEVICE)
    printed = lithosim(image)
    
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(image.detach().cpu().numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(printed[0].detach().cpu().numpy())
    plt.show()