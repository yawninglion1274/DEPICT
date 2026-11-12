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
from utils.img_utils import ODT_data, clear, plot_volume
from tqdm import tqdm
from nibabel.viewers import OrthoSlicer3D

def angular_spectrum_kernel(domain_size, spec_pixel_size, pixel_size, km):
    assert domain_size[1] % 2 == 0 and domain_size[2] % 2 == 0, "domain_size[1] and domain_size[2] must be even"
    kx = (torch.linspace((-domain_size[1] // 2+1), (domain_size[1] // 2 ), domain_size[1]) - 1) * spec_pixel_size
    ky = (torch.linspace((-domain_size[2] // 2+1), (domain_size[2] // 2 ), domain_size[2]) - 1) * spec_pixel_size
    [Ky, Kx] = torch.meshgrid(ky, kx, indexing='ij')
    K2 = Kx**2 + Ky**2
    Kz = torch.sqrt(-K2 + km**2 + 0j)
    Kz[-K2 + km**2<0]=0.
    kernel = torch.exp(1j * Kz * pixel_size[0])
    ol_correction = km / Kz
    ol_correction[-K2 + (km * 1.2/1.33)**2<0]=0.
    return (torch.fft.fftshift(kernel)), (torch.fft.fftshift(ol_correction))


def compute_pps_grad(u_inlet, u_outlet, delta_ri, ol_correction, as_kernel, k0, pixel_size, batch_size,device):
    grad = torch.zeros_like(delta_ri)
    norm = 0
    for i in (range(u_inlet.shape[0]//batch_size)):#之后这部分可以用torch.chunk替代
        start_idx = i * batch_size
        end_idx = (i + 1)*batch_size if (i + 1) * batch_size < u_inlet.shape[0] else None
        sub_u_in = u_inlet[start_idx:end_idx, ...]
        sub_u_out = u_outlet[start_idx:end_idx, ...]
        u = sub_u_in
        s = torch.zeros((batch_size,) + delta_ri.shape, dtype=torch.cfloat, device=device)
        for m in (range(delta_ri.shape[0])):
            u = torch.fft.ifft2(torch.fft.fft2((u), norm='ortho') * as_kernel, norm='ortho')#广播机制
            s[:, m, ...] = u.conj()
            ri_slice = delta_ri[m, ...].unsqueeze(0) * 1j * k0 * pixel_size[0]
            u_next =torch.fft.fft2(u, norm='ortho')
            ol_correction_cur = ol_correction.clone()
            for q in range(1, 4):
                u_next +=torch.fft.fft2(u * (ri_slice), norm='ortho') * (ol_correction_cur)
                ol_correction_cur *= ol_correction / (q + 1)
                ri_slice *=  ri_slice
            u = torch.fft.ifft2(u_next, norm='ortho')
        r = u - sub_u_out
        norm +=(r.abs()).mean().item()
        for m in reversed(range(delta_ri.shape[0])):
            ol_correction_cur = ol_correction.clone()
            ri_slice = delta_ri[m, ...].unsqueeze(0) * (-1j) * k0 * pixel_size[0]
            s_temp = 0
            r_temp = torch.fft.fft2(r, norm='ortho')
            for q in range(1, 4):
                s_temp += q * (ri_slice**(q-1)) * torch.fft.ifft2(r_temp * ol_correction_cur, norm='ortho')
                r += (ri_slice**(q)) * torch.fft.ifft2(r_temp * ol_correction_cur, norm='ortho')
                ol_correction_cur *= ol_correction / (q + 1)
            s[:, m, ...] *= (-1j * k0 * pixel_size[0]) * s_temp
            r = torch.fft.ifft2(torch.fft.fft2(r, norm='ortho') * as_kernel.conj(),  norm='ortho')
        grad += s.sum(axis=0).real
    print('loss', norm/u_inlet.shape[0])
    return grad/ u_inlet.shape[0], norm/u_inlet.shape[0]
