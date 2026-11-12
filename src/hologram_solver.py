import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import gzip
import os
import torchvision
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import glob
import yaml
import cv2
import torch_dct as dct
from torch import fft
from utils.img_utils import ODT_data, clear
from tqdm import tqdm
from nibabel.viewers import OrthoSlicer3D
from torch.nn.functional import conv2d

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device('cuda')


def holo_ref_finder(input, Bound):
    idx_arr = []
    batch_num = 2
    for i in range(batch_num):
        batch_size = input.shape[0]//batch_num
        start_idx = i * batch_size
        end_idx = (i+1)*batch_size if (i+1)*batch_size else None

        size_input = input[start_idx:end_idx].shape
        Spectrum = torch.fft.fft2(input[start_idx:end_idx]).abs()
        SpectrumLeft = Spectrum[:, 0:size_input[1] // 2, :]
        SpectrumLeft[:, 0:Bound[0], 0:Bound[1]] = 0
        ## adapt for PSB data
        ## True for  pectrumLeft[:, 0:Bound[0], size_input[2] - Bound[1] - 1:] = 0
        SpectrumLeft[:, 0:150, size_input[2] - 150 - 1:] =0
        SpectrumLeft[:, 0:3, :] = 0
        SpectrumLeft[:, :, 0:3] = 0
        SpectrumLeft[:, :, -3:] = 0
        SpecCum1 = torch.sum(SpectrumLeft, 2)
        SpecCum2 = torch.sum(SpectrumLeft, 1)
        index1 = torch.argmax(SpecCum1, dim=1)
        index2 = torch.argmax(SpecCum2, dim=1)
        index2 = index2 - size_input[2]
        idx_arr.append(torch.stack([index1, index2], 1))
    idx_arr = torch.concat(idx_arr, 0).to(torch.float32)
    return idx_arr



def holo_filter(input, kRef, Bound, size_Spec):
    out_arr = []
    batch_num = 2
    for i in range(batch_num):
        batch_size = input.shape[0]//batch_num
        start_idx = i * batch_size
        end_idx = (i+1)*batch_size if (i+1)*batch_size else None
        size_input = input[start_idx:end_idx].shape
        ax = torch.arange(0, size_input[1], device=input.device)
        ay = torch.arange(0, size_input[2], device=input.device)
        X, Y = torch.meshgrid(ax, ay)
        shiftInput = input[start_idx:end_idx] * torch.exp(2.0j * np.pi * (Y * kRef[1] / size_input[1] + X * kRef[0] / size_input[2])).unsqueeze(0)
        ax = torch.arange(-size_Spec[1] // 2, size_Spec[1] // 2, device=input.device)
        ay = torch.arange(-size_Spec[0] // 2, size_Spec[0] // 2, device=input.device)
        Sx, Sy = torch.meshgrid(ax, ay)
        Spectrum = fft.fftshift(fft.fft2(fft.fftshift(shiftInput)))
        start1 = (size_input[1] - size_Spec[0]) // 2
        end1 = (size_input[1] - size_Spec[0]) // 2 + size_Spec[0]
        start2 = (size_input[2] - size_Spec[1]) // 2
        end2 = (size_input[2] - size_Spec[1]) // 2+ size_Spec[1]
        output = Spectrum[:, start1:end1, start2:end2]
        output[:, ((Sy / Bound[1])**2 + (Sx / Bound[0])**2 > 1)] = 0
        output = fft.ifftshift(fft.ifft2(fft.ifftshift(output)))
        out_arr.append(output)
    out_arr = torch.concat(out_arr, 0)
    return out_arr



def wrap_to_Pi(psi):
    psiWrap = torch.remainder(psi, 2*np.pi)
    mask = torch.abs(psiWrap) > np.pi
    psiWrap[mask] -= 2*np.pi * torch.sign(psiWrap[mask])
    return psiWrap

def phase_unwrap(psi):
    '''
    single do PhaseUnwrap
    '''
    psi_size = psi.shape
    dx = torch.zeros((psi_size[0], psi_size[1], psi_size[2] + 1), device=psi.device)
    dy = torch.zeros((psi_size[0], psi_size[1] + 1, psi_size[2]), device=psi.device)
    dx[:, :, 1:-1] = wrap_to_Pi(torch.diff(psi, 1, 2))
    dy[:, 1:-1, :] = wrap_to_Pi(torch.diff(psi, 1, 1))
    rho = torch.diff(dx, 1, 2) + torch.diff(dy, 1, 1)
    dctRho = dct.dct_2d(rho, norm='ortho')
    ax = torch.arange(0, psi_size[1], device=psi.device)
    ay = torch.arange(0, psi_size[2], device=psi.device)
    X, Y = torch.meshgrid(ax, ay)
    temp = (torch.cos(np.pi * Y / psi_size[2]) + torch.cos(np.pi * X / psi_size[1]) - 2)
    dctPhi = dctRho / 2.0 / temp
    dctPhi[:, 0,0] = 0.
    phi = dct.idct_2d(dctPhi, norm='ortho')
    return phi

def least_square(x, y):
    length = x.shape[-1]
    return (length * torch.sum(x * y, axis = -1, keepdims=True) - torch.sum(x, axis = -1, keepdims=True) * torch.sum(y, axis = -1, keepdims=True)) / (length * torch.sum(x**2, axis = -1, keepdims=True) - torch.sum(x, axis = -1, keepdims=True)**2)

def holo_VISA(input, kScanIn, itertime = 5):
    size_input = input.shape[1:]
    ax = torch.arange(0, size_input[0], device=input.device)
    ay = torch.arange(0, size_input[1], device=input.device)
    X, Y = torch.meshgrid(ax, ay)
    X = X.unsqueeze(0)
    Y = Y.unsqueeze(0)
    for i in range(itertime):
        shiftInput = input * torch.exp(2.0j * np.pi * (Y * kScanIn[:, 1].unsqueeze(-1).unsqueeze(-1) / size_input[0] + X * kScanIn[:, 0].unsqueeze(-1).unsqueeze(-1) / size_input[1]))
        Phs = shiftInput.angle()
        Phs = phase_unwrap(Phs)
        x = torch.arange(0, size_input[0], 1, device=input.device).unsqueeze(0)
        y = torch.mean(Phs, axis = 2)
        alpha1 = least_square(x, y) * size_input[0]
        y = torch.mean(Phs, axis = 1)
        alpha2 = least_square(x, y) * size_input[0]
        kScanIn = kScanIn - torch.concatenate([alpha1, alpha2], -1) / 2 / np.pi
    Amp = shiftInput.abs()
    kScanOut = kScanIn
    return Amp, Phs  , kScanOut


def compute_wavefront(amp, phs, k):
    nx = amp.shape[1]
    ny = amp.shape[2]
    xs = torch.linspace(-nx//2, nx//2-1, nx, dtype=torch.float32, device=amp.device)
    ys = torch.linspace(-ny//2, ny//2-1, ny, dtype=torch.float32, device=amp.device)
    XS, YS = torch.meshgrid(xs, ys, indexing='ij')
    XS = XS.unsqueeze(0)
    YS = YS.unsqueeze(0)
    kx = -k[:, 0].unsqueeze(-1).unsqueeze(-1)
    ky = -k[:, 1].unsqueeze(-1).unsqueeze(-1)
    print(kx.shape)
    field = amp * torch.exp(1j * phs) * torch.exp(2j * np.pi * (XS * kx / nx + YS * ky / ny))
    return field



def phase_residual_correction_with_mask(phase_img, mask, order=1):
    height, width = phase_img.shape[1:]
    device = phase_img.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device).float(),
        torch.arange(width, device=device).float(),
        indexing='ij'
    )
    valid_indices = mask > 0
    valid_x = x[valid_indices]
    valid_y = y[valid_indices]
    valid_values = phase_img[:, valid_indices]
    valid_x = valid_x / width
    valid_y = valid_y / height

    # 构建设计矩阵
    design_matrix = []

    # 添加X的多项式项
    for i in range(1, order + 1):
        design_matrix.append(valid_x ** i)

    # 添加Y的多项式项
    for i in range(1, order + 1):
        design_matrix.append(valid_y ** i)
    # 添加常数项
    design_matrix.append(torch.ones_like(valid_x))

    # 将列表转换为设计矩阵
    design_matrix = torch.stack(design_matrix, dim=1)

    # 使用最小二乘法求解多项式系数 (A^T A)^-1 A^T b
    ATA = torch.matmul(design_matrix.T, design_matrix).unsqueeze(0).repeat(valid_values.shape[0], 1, 1)  # 批次化
    ATb = torch.einsum('ij,kj->ki', design_matrix.T, valid_values)
    # 增加少量正则化以确保可逆性
    epsilon = 1e-10
    ATA_reg = ATA + epsilon * torch.eye(ATA.shape[1], device=device).unsqueeze(0)
    coefficients = torch.linalg.solve(ATA_reg, ATb)
    # 构建背景模型
    background = torch.zeros_like(phase_img)
    # 应用X多项式项
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    for i in range(order):
        background += coefficients[:, i].unsqueeze(-1).unsqueeze(-1) * (x / width) ** (i + 1)
    # 应用Y多项式项
    for i in range(order):
        background += coefficients[:, order + i].unsqueeze(-1).unsqueeze(-1) * (y / height) ** (i + 1)
    # 应用常数项
    background += coefficients[:, -1].unsqueeze(-1).unsqueeze(-1)
    # 从原始相位中减去背景
    return background

def holo_process_with_mask(odt_data, mask, device, VISATimes=5):
    raw_Samp = odt_data.sp
    raw_Back = odt_data.bg
    size_Spec = odt_data.spec_size
    para_NStack = odt_data.angle_num
    kBoundPixel = [odt_data.km_bound_pixel, odt_data.km_bound_pixel]
    conjFlag = odt_data.conjugate_flag
    raw_Back = raw_Back.to(device)
    raw_Samp = raw_Samp.to(device)
    k_ScanSamp = holo_ref_finder(raw_Samp, kBoundPixel) * conjFlag
    k_ScanBack = holo_ref_finder(raw_Back, kBoundPixel) * conjFlag
    plt.scatter(k_ScanBack.cpu().numpy()[:, 0], k_ScanBack.cpu().numpy()[:, 1])
    plt.scatter(k_ScanSamp.cpu().numpy()[:, 0], k_ScanSamp.cpu().numpy()[:, 1])
    plt.show()
    k_RefSamp = torch.mean(k_ScanSamp, 0)
    k_RefBack = torch.mean(k_ScanBack, 0)
    k_ScanSamp = k_ScanSamp - k_RefSamp
    k_ScanBack = k_ScanBack - k_RefBack
    batch_num = para_NStack // odt_data.batch_size
    u_Samp_amp = []
    u_Samp_phs = []
    u_Back_amp = []
    u_Back_phs = []
    full_k_ScanSamp = []
    full_k_ScanBack = []
    print(raw_Samp.shape)
    print(k_ScanBack.shape)
    for i in tqdm(range(odt_data.batch_size)):
        start_idx = i * batch_num
        end_idx = (i + 1) * batch_num if (i + 1) * batch_num < para_NStack else None
        u_Samp = holo_filter(raw_Samp[start_idx:end_idx, ...], k_RefSamp, kBoundPixel, size_Spec)
        u_Back = holo_filter(raw_Back[start_idx:end_idx, ...], k_RefSamp, kBoundPixel, size_Spec)
        sub_u_Back_amp, sub_u_Back_phs, sub_k_ScanBack = holo_VISA(u_Back, k_ScanBack[start_idx:end_idx, :],
                                                                   VISATimes)

        sub_u_Samp_amp, sub_u_Samp_phs, sub_k_ScanSamp = holo_VISA(u_Samp, k_ScanSamp[start_idx:end_idx, :],
                                                                   VISATimes)

        residual_phs = sub_u_Samp_phs - sub_u_Back_phs
        background = phase_residual_correction_with_mask(residual_phs, mask)
        sub_u_Samp_phs = sub_u_Samp_phs - background
        sub_k_ScanSamp = sub_k_ScanBack

        u_Samp_amp.append(sub_u_Samp_amp)
        u_Samp_phs.append(sub_u_Samp_phs)
        full_k_ScanSamp.append(sub_k_ScanSamp)
        full_k_ScanBack.append(sub_k_ScanBack)
        u_Back_amp.append(sub_u_Back_amp)
        u_Back_phs.append(sub_u_Back_phs)
    u_Samp_amp = torch.concat(u_Samp_amp, 0)
    u_Samp_phs = torch.concat(u_Samp_phs, 0)
    u_Back_amp = torch.concat(u_Back_amp)
    u_Back_phs = torch.concat(u_Back_phs)
    k_ScanSamp = torch.concat(full_k_ScanSamp)
    k_ScanBack = torch.concat(full_k_ScanBack)
    temp = k_ScanSamp[1:-1, :] * odt_data.spec_pixel_size / odt_data.km
    temp = torch.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2)[-96:].mean().item()
    print(f'scan with theta {np.arcsin(temp) * 180 / np.pi}')
    meanAmp = u_Back_amp.mean()
    u_Samp_amp = u_Samp_amp / meanAmp
    u_Back_amp = u_Back_amp / meanAmp
    # ## remove ref
    # # plt.imshow(torch.exp(mean_log_back_amp).cpu().numpy(), cmap='gray')
    # # plt.show()
    # # plt.imshow(torch.exp(mean_back_phs).cpu().numpy(), cmap='gray')
    # # plt.show()
    #
    ## background correction

    odt_data.sp_amp = u_Samp_amp
    odt_data.sp_phs = u_Samp_phs
    odt_data.bg_amp = u_Back_amp
    odt_data.bg_phs = u_Back_phs
    odt_data.k_scan_sp = k_ScanSamp
    odt_data.k_scan_bg = k_ScanBack
    del odt_data.sp
    del odt_data.bg
    print(f'complex field shape {odt_data.sp_amp.shape}')
    print('sp phs mean', odt_data.sp_phs.mean())
    print('bg phs mean', odt_data.bg_phs.mean())
    u_out = compute_wavefront(odt_data.sp_amp, odt_data.sp_phs, k_ScanSamp)
    u_in = compute_wavefront(odt_data.bg_amp, odt_data.bg_phs,
                             k_ScanSamp)  # 必须要用同一个波矢，不然不匹配，k samp和k scan毕竟还是有细微差别的
    check_phase_correction = (u_out - u_in).abs().mean(axis=0)
    return odt_data, u_out, u_in, k_ScanSamp, k_ScanBack, check_phase_correction



if __name__ == '__main__':
    data_config = 'configs/rytov/data_config.yaml'
    physics_config = 'configs/rytov/physics_config.yaml'
    device = torch.device('cuda')
    cos7_1 = ODT_data(data_config, physics_config)
    cos7_1 = holo_process(cos7_1, device)
    plt.imshow(clear(cos7_1.sp_amp[1]))
    plt.colorbar()
    plt.savefig('test_plot/uSampAmp.png')
    plt.close('all')
    plt.imshow(clear(cos7_1.sp_phs[1]))
    plt.colorbar()
    plt.savefig('test_plot/uSampPhs.png')
    plt.close('all')
