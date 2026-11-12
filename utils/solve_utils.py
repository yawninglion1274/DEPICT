import numpy as np
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io
from nibabel.viewers import OrthoSlicer3D
import argparse
from .img_utils import load_yaml, plot_volume, get_device
import os
from tqdm import tqdm
from src.optimization import FISTA
from src.prior import FGP_tv, projector_threshold, tgv_pd_3d, make_regulizer
from functools import partial
from utils.img_utils import ODT_data, get_device, load_yaml
from src.hologram_solver import holo_process_with_mask#, holo_process
from src.bpm_grad import angular_spectrum_kernel, bpm_kernel, compute_bpm_grad
from src.bpm_grad_bloch import BlochAngularSpectrumPropagator, compute_bpm_grad_bloch
from src.pps_grad import compute_pps_grad, angular_spectrum_kernel
from src.rytov_solver import RytovRec,  RytovRec_downsample
import glob
from PIL import Image
import tifffile
import warnings
import math
import torch.nn.functional as F
device = get_device()

def gaussian_3d_blur_frequency_domain(input_tensor, sigma=(1.0, 1.0, 1.0)):
    """
    在频域对3D张量进行高斯模糊
    
    参数:
        input_tensor (torch.Tensor): 输入的3D张量，形状为 [depth, height, width]
        sigma (tuple): 高斯核的标准差 (sigma_z, sigma_y, sigma_x)
        
    返回:
        torch.Tensor: 模糊后的3D张量
    """
    # 确保输入是3D张量
    assert input_tensor.dim() == 3, "输入必须是3D张量 [depth, height, width]"
    
    # 获取输入张量的形状
    depth, height, width = input_tensor.shape
    device = input_tensor.device
    
    # 创建频域网格
    z_grid, y_grid, x_grid = torch.meshgrid(
        torch.fft.fftfreq(depth, d=1.0),
        torch.fft.fftfreq(height, d=1.0),
        torch.fft.fftfreq(width, d=1.0),
        indexing='ij'
    )
    
    # 移动到设备
    z_grid = z_grid.to(device)
    y_grid = y_grid.to(device)
    x_grid = x_grid.to(device)
    
    # 计算频域高斯滤波器
    sigma_z_freq = 1.0 / (2.0 * math.pi * sigma[0])
    sigma_y_freq = 1.0 / (2.0 * math.pi * sigma[1])
    sigma_x_freq = 1.0 / (2.0 * math.pi * sigma[2])
    
    # 构建频域高斯核
    gaussian_filter = torch.exp(
        -2.0 * (math.pi ** 2) * (
            (z_grid ** 2) * (sigma_z_freq ** 2) +
            (y_grid ** 2) * (sigma_y_freq ** 2) +
            (x_grid ** 2) * (sigma_x_freq ** 2)
        )
    )
    
    # 执行FFT
    input_fft = torch.fft.fftn(input_tensor)
    
    # 应用滤波器（频域相乘等于空间域卷积）
    filtered_fft = input_fft * gaussian_filter
    
    # 执行IFFT
    result = torch.fft.ifftn(filtered_fft).real
    
    return result

def load_binary_mask_float32(image_path, threshold=128, mask_type='fiji'):
    """
    读取图片并转换为float32格式的二值化掩码

    参数:
        image_path: 图片路径
        threshold: 二值化阈值 (0-255)
        invert: 是否反转掩码 (True为反转, False为保持原样)

    返回:
        二值化掩码 (numpy数组, 类型为float32, 值为0.0或1.0)
    """
    # 以灰度模式读取图片
    img_pil = Image.open(image_path).convert('L')
    # 转换为numpy数组
    img_array = np.array(img_pil)

    # 应用阈值进行二值化 (大于阈值为1.0，否则为0.0)
    if(mask_type=='fiji'):
        binary_mask =1- (img_array > threshold).astype(np.float32)
    elif(mask_type=='bg'):
        binary_mask = (img_array > threshold).astype(np.float32)

    return binary_mask


def solve_rytov(data_config, physics_config):
    data_pack = ODT_data(data_config, physics_config)
    exp = data_pack.exp_name
    data_name = data_config['data_name']
    ## rytov recon
    warnings.warn('please check mask path')
    mask_type = data_config.get('mask_type', 'none')
    if mask_type in ['none', 'all', None]:
        print('No phase mask configured; using an all-one phase residual correction mask.')
        mask = torch.ones(tuple(data_pack.spec_size), dtype=torch.float32, device=device)
    elif(mask_type!='var'):
        mask_path = os.path.join(data_config['save_path'], data_config['expriment_name'], 'phase_mask', data_name + '.png')
        print(mask_path)
        mask = load_binary_mask_float32(mask_path, mask_type=mask_type)
        shift_idx = np.asarray(data_config['shift_idx'])
        spec_size = np.asarray(data_config['spec_size'])
        print(mask.shape)
        mask = mask[mask.shape[0]//2 + shift_idx[0]//2 - spec_size[0]//2:mask.shape[0]//2 +shift_idx[0]//2 + spec_size[0]//2, shift_idx[1]//2 - spec_size[1]//2+ mask.shape[1]//2:shift_idx[1]//2 + spec_size[1]//2 + mask.shape[1]//2]
        print(mask.shape)
    else:
        mask = data_pack.get_mask_by_var()
    data_pack, u_out, u_in, k_scan_samp, k_scan_back, check_phase_correction = holo_process_with_mask(data_pack, mask, device)
    RI_rytov, V_rytov_fft, norm_coef, f_target, f_count, rytov_phs, resolution = RytovRec(data_pack, exp)

    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'])
    os.makedirs(os.path.join(save_pth, data_name, 'scatter_potential'), exist_ok=True)
    os.makedirs(os.path.join(save_pth, 'rytov'), exist_ok=True)
    os.makedirs(os.path.join(save_pth, data_name, 'wavefront_remove_ref'), exist_ok=True)
    np.save(os.path.join(save_pth,data_name, 'scatter_potential', 'potential_fft.npy'), V_rytov_fft.detach().cpu().numpy())
    np.save(os.path.join(save_pth,data_name, 'scatter_potential', 'potential_mask.npy'), f_count.detach().cpu().numpy())
    np.save(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'u_in.npy'), u_in.detach().cpu().numpy())
    np.save(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'u_out.npy'), u_out.detach().cpu().numpy())
    
    np.save(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'k_samp.npy'), k_scan_samp.detach().cpu().numpy())
    np.save(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'k_back.npy'), k_scan_back.detach().cpu().numpy())
    np.save(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'rytov_phs.npy'), rytov_phs.detach().cpu().numpy())
    os.makedirs(os.path.join(save_pth,'rytov'), exist_ok=True)
    plot_volume(RI_rytov, os.path.join(save_pth, 'rytov', data_name+'_rytov.tif'), resolution)

def solve_rytov_downsample(data_config, physics_config):
    data_pack = ODT_data(data_config, physics_config)
    exp = data_pack.exp_name
    data_name = data_config['data_name']
    ## rytov recon
    warnings.warn('please check mask path')
    mask_path = os.path.join(data_config['save_path'], data_config['expriment_name'], 'phase_mask', data_name + '.png')
    print(mask_path)
    mask = load_binary_mask_float32(mask_path, mask_type=data_config['mask_type'])
    shift_idx = np.asarray(data_config['shift_idx'])
    spec_size = np.asarray(data_config['spec_size'])
    print(mask.shape)
    mask = mask[mask.shape[0]//2 + shift_idx[0]//2 - spec_size[0]//2:mask.shape[0]//2 +shift_idx[0]//2 + spec_size[0]//2, shift_idx[1]//2 - spec_size[1]//2+ mask.shape[1]//2:shift_idx[1]//2 + spec_size[1]//2 + mask.shape[1]//2]
    print(mask.shape)
    data_pack, u_out, u_in, k_scan_samp, k_scan_back, check_phase_correction = holo_process_with_mask(data_pack, mask, device)
    for angle_downsample in [2, 4, 8, 12, 30, 60]:
        RI_rytov, V_rytov_fft, norm_coef, f_target, f_count, rytov_phs, resolution = RytovRec_downsample(data_pack, exp, angle_downsample)

        save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'])
        os.makedirs(os.path.join(save_pth,'rytov'), exist_ok=True)
        plot_volume(RI_rytov, os.path.join(save_pth, 'rytov', data_name+f'_rytov_angledownsample{angle_downsample}.tif'), resolution)

def solve_bpm_padding(data_config, physics_config, reconstruction_config):
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    xy_padding_len = reconstruction_config['xy_padding_len']
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']
    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    padded_spec_pixel_size = spec_pixel_size * domain_size[1] / (domain_size[1] + 2 * xy_padding_len)
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    print('resolution', resolution)
    wavefront_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name,'wavefront_remove_ref')
    u_in = np.load(os.path.join(wavefront_pth, 'u_in.npy'))
    pad_array = (xy_padding_len, xy_padding_len, xy_padding_len, xy_padding_len, 0, 0)
    u_in = torch.from_numpy(u_in).to(device)
    u_in = F.pad(u_in.real, pad_array, mode='constant', value=0) + 1j * F.pad(u_in.imag, pad_array, mode='constant', value=0)
    u_out = np.load(os.path.join(wavefront_pth, 'u_out.npy'))  # .real
    u_out = torch.from_numpy(u_out).to(device)
    u_out = F.pad(u_out.real, pad_array, mode='constant', value=0) + 1j * F.pad(u_out.imag, pad_array, mode='constant', value=0)
    k_scan_samp = np.load(os.path.join(wavefront_pth, 'k_samp.npy'))  # .real
    temp = k_scan_samp * spec_pixel_size / km
    bpm_cosFactor = np.cos(np.arcsin(np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2)))
    bpm_cosFactor = torch.from_numpy(bpm_cosFactor.reshape(-1, 1)).to(device)

    ## 不能用角谱传播，只能用bpm自己的传播，不然相位对不上
    crop_z = reconstruction_config['crop_z']
    region_z = domain_size[0]
    bg_z = region_z // 2
    if crop_z != None:
        region_z = data_config['region_z']
        bg_z = data_config['bg_z']

    domain_size[1] = domain_size[1] + 2 * xy_padding_len
    domain_size[2] = domain_size[2] + 2 * xy_padding_len
    p_kernel = bpm_kernel(domain_size, padded_spec_pixel_size, resolution, km).to(device)
    print(p_kernel.shape)
    print(u_in.shape)
    u_inlet = torch.fft.ifft2(torch.fft.fft2(u_in) * (p_kernel ** (region_z - bg_z)).conj())
    u_outlet = torch.fft.ifft2(torch.fft.fft2(u_out) * (p_kernel ** (bg_z)))

    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    ROI = data_config['ROI']
    batch_size = reconstruction_config['batch_size']
    param_string = 'tv_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['tv_param'])
    param_string += 'value_range_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['value_range_param'])
    param_string += 'sparse_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['sparse_param'])
    initial_type = reconstruction_config['initial_type']
    initial_shift_idx = reconstruction_config['initial_shift_idx']
    initial_name = reconstruction_config['init_name'][:-4]
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'bpm_regu', param_string, f'step_size{step_size}_n_iter{n_iter}', initial_type +initial_name +f'_shift_idx{initial_shift_idx}', f'crop_z_{crop_z}_region_z{region_z}_bg_z{bg_z}_xy_padding{xy_padding_len}')
    out_path = save_pth
    print('save in', out_path)
    os.makedirs(out_path, exist_ok=True)
    # # resolution_pixel = physics_config['camera_pixel_size'] / physics_config['magnification']
    # # physical_param = OpticalSimulationSettings(NA=physics_config['NA'],
    # #                                             RI_bg=physics_config['n_medium'],
    # #                                             volume_sz=data_config['domain_size'],
    # #                                             wavelength=physics_config['wavelength'],
    # #                                             pixel_sz=physics_config['camera_pixel_size'],
    # #                                             mag=physics_config['magnification'], resolution=np.array([resolution_pixel, resolution_pixel, resolution_pixel]))

    regu_func = make_regulizer(reconstruction_config['tv_param'], reconstruction_config['value_range_param'], reconstruction_config['sparse_param'][0])


    init_delta_ri = torch.zeros((region_z, domain_size[1], domain_size[2]), dtype=torch.float32, device=device)
    if initial_type == 'rytov':
        rytov_ri = tifffile.imread(os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], 'rytov', data_name + '_rytov.tif'))
        init_delta_ri = torch.from_numpy(rytov_ri).to(device)
        init_delta_ri = F.pad(init_delta_ri, pad_array, mode='constant', value=0)
    if initial_type == 'constant':
        init_delta_ri = tifffile.imread(os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name , 'bpm_regu','bpm_init',reconstruction_config['init_name']))
        init_delta_ri = torch.from_numpy(init_delta_ri).to(device)
        init_delta_ri = F.pad(init_delta_ri, pad_array, mode='constant', value=0)
    print('RI shape', init_delta_ri.shape)
    s = init_delta_ri.clone()
    q_1 = 1
    x_1 = init_delta_ri.clone()
    pbar = tqdm(range(n_iter))
    os.makedirs(out_path, exist_ok=True)
    for i in pbar:
        grad, loss = compute_bpm_grad(s, u_inlet, u_outlet, batch_size, bpm_cosFactor, resolution[0], domain_size, k0,
                                      p_kernel, device)
        grad = grad.real
        pbar.set_postfix({'loss': loss.item()}, refresh=False)
        with torch.no_grad():
            z = s - grad * step_size
            del grad
            x_2 = regu_func(z)
            q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            s = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            print(i, ((x_2 - x_1)**2).sum().sqrt())
            x_1 = x_2
            q_1 = q_2
        if ((i) % 50 == 0):
            plot_volume(s[:, xy_padding_len:-xy_padding_len, xy_padding_len:-xy_padding_len], os.path.join(out_path, f'bpm_regu_step{i}.tif') , resolution)
    delta_ri = s

    vmin = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].min()
    vmax = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].max()
    print(vmin)
    print(vmax)

    os.makedirs(os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'bpm_regu'), exist_ok=True)
    plot_volume(delta_ri[:, xy_padding_len:-xy_padding_len, xy_padding_len:-xy_padding_len], os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'bpm_regu', data_name + '.tif'), resolution)
    print('check in', os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'bpm_regu', data_name + '.tif'))



def solve_bpm_bloch(data_config, physics_config, reconstruction_config):
    if(reconstruction_config['xy_padding_len']!=None):
        solve_bpm_padding(data_config, physics_config, reconstruction_config)
        return
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']

    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    print('resolution', resolution)
    wavefront_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name,'wavefront_remove_ref')
    u_in = np.load(os.path.join(wavefront_pth, 'u_in.npy'))
    u_in = torch.from_numpy(u_in).to(device)
    u_out = np.load(os.path.join(wavefront_pth, 'u_out.npy'))  # .real
    u_out = torch.from_numpy(u_out).to(device)
    k_scan_samp = torch.from_numpy(np.load(os.path.join(wavefront_pth, 'k_samp.npy'))).to(device)  # .real
    temp = k_scan_samp.cpu().numpy() * spec_pixel_size / km
    bpm_cosFactor = np.cos(np.arcsin(np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2)))
    bpm_cosFactor = torch.from_numpy(bpm_cosFactor.reshape(-1, 1)).to(device)
    prop = BlochAngularSpectrumPropagator(wavelength, resolution[0], n_medium, domain_size, device)
    ## 不能用角谱传播，只能用bpm自己的传播，不然相位对不上
    crop_z = data_config['crop_z']
    region_z = domain_size[0]
    bg_z = region_z // 2
    if crop_z != None:
        region_z = data_config['region_z']
        bg_z = data_config['bg_z']
    print(k_scan_samp.shape)
    p_kernel = prop.create_bloch_green_function_batch(k_scan_samp, (region_z - bg_z) * resolution[0])
    p_kernel = p_kernel.to(device).conj()
    u_inlet = prop.propagate_bloch_batch(u_in, (region_z - bg_z) * resolution[0], k_scan_samp, p_kernel)
    p_kernel = prop.create_bloch_green_function_batch(k_scan_samp, (bg_z) * resolution[0])
    p_kernel = p_kernel.to(device)
    u_outlet = prop.propagate_bloch_batch(u_out, (bg_z) * resolution[0], k_scan_samp, p_kernel)

    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    ROI = data_config['ROI']
    batch_size = reconstruction_config['batch_size']
    param_string = 'tv_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['tv_param'])
    param_string += 'value_range_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['value_range_param'])
    param_string += 'sparse_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['sparse_param'])
    initial_type = reconstruction_config['initial_type']
    initial_shift_idx = reconstruction_config['initial_shift_idx']
    initial_name = reconstruction_config['init_name'][:-4]
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'bpm_regu_bloch', param_string, f'step_size{step_size}_n_iter{n_iter}', initial_type +initial_name +f'_shift_idx{initial_shift_idx}', f'crop_z_{crop_z}_region_z{region_z}_bg_z{bg_z}')
    out_path = save_pth
    print('save in', out_path)
    os.makedirs(out_path, exist_ok=True)
    # # resolution_pixel = physics_config['camera_pixel_size'] / physics_config['magnification']
    # # physical_param = OpticalSimulationSettings(NA=physics_config['NA'],
    # #                                             RI_bg=physics_config['n_medium'],
    # #                                             volume_sz=data_config['domain_size'],
    # #                                             wavelength=physics_config['wavelength'],
    # #                                             pixel_sz=physics_config['camera_pixel_size'],
    # #                                             mag=physics_config['magnification'], resolution=np.array([resolution_pixel, resolution_pixel, resolution_pixel]))

    regu_func = make_regulizer(reconstruction_config['tv_param'], reconstruction_config['value_range_param'], reconstruction_config['sparse_param'][0])


    init_delta_ri = torch.zeros((region_z, domain_size[1], domain_size[2]), dtype=torch.float32, device=device)
    if initial_type == 'rytov':
        rytov_ri = tifffile.imread(os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], 'rytov', data_name + '_rytov.tif'))
        if(crop_z!=None):
            rytov_ri = rytov_ri[domain_size[0]//2 - (region_z - bg_z):domain_size[0]//2 + bg_z]
        init_delta_ri = torch.from_numpy(rytov_ri).to(device)
    if initial_type == 'constant':
        init_delta_ri = tifffile.imread(os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name , 'bpm_regu','bpm_init',reconstruction_config['init_name']))
        init_delta_ri = torch.from_numpy(init_delta_ri).to(device)
    print('RI shape', init_delta_ri.shape)
    s = init_delta_ri.clone()
    q_1 = 1
    x_1 = init_delta_ri.clone()
    pbar = tqdm(range(n_iter))
    os.makedirs(out_path, exist_ok=True)
    p_kernel = prop.create_bloch_green_function_batch(k_scan_samp, resolution[0])
    print('p kernel shape', p_kernel.shape)
    for i in pbar:
        grad, loss = compute_bpm_grad_bloch(s, u_inlet, u_outlet, batch_size, bpm_cosFactor, resolution[0], domain_size, k0,
                                      p_kernel, prop, k_scan_samp, device)
        grad = grad.real
        pbar.set_postfix({'loss': loss.item()}, refresh=False)
        with torch.no_grad():
            z = s - grad * step_size
            x_2 = regu_func(z)
            q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            s = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            print(i, ((x_2 - x_1)**2).sum().sqrt())
            x_1 = x_2
            q_1 = q_2
        if ((i) % 10 == 0):
            plot_volume(s, os.path.join(out_path, f'bpm_regu_step{i}.tif') , resolution)
    delta_ri = s

    vmin = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].min()
    vmax = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].max()
    print(vmin)
    print(vmax)

    os.makedirs(os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'bpm_regu_bloch'), exist_ok=True)
    plot_volume(delta_ri, os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'bpm_regu_bloch', data_name + '.tif'), resolution)
    print('check in', os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'bpm_regu_bloch', data_name + '.tif'))




def solve_bpm(data_config, physics_config, reconstruction_config):
    if(reconstruction_config['xy_padding_len']!=None):
        solve_bpm_padding(data_config, physics_config, reconstruction_config)
        return
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']

    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    print('resolution', resolution)
    wavefront_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name,'wavefront_remove_ref')
    u_in = np.load(os.path.join(wavefront_pth, 'u_in.npy'))
    u_in = torch.from_numpy(u_in).to(device)
    u_out = np.load(os.path.join(wavefront_pth, 'u_out.npy'))  # .real
    u_out = torch.from_numpy(u_out).to(device)
    k_scan_samp = np.load(os.path.join(wavefront_pth, 'k_samp.npy'))  # .real
    temp = k_scan_samp * spec_pixel_size / km
    bpm_cosFactor = np.cos(np.arcsin(np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2)))
    bpm_cosFactor = torch.from_numpy(bpm_cosFactor.reshape(-1, 1)).to(device)
    ## 不能用角谱传播，只能用bpm自己的传播，不然相位对不上
    crop_z = data_config['crop_z']
    region_z = domain_size[0]
    bg_z = region_z // 2
    if crop_z != None:
        region_z = data_config['region_z']
        bg_z = data_config['bg_z']
    p_kernel, _ = angular_spectrum_kernel(domain_size, spec_pixel_size, resolution, km)#.to(device)
    p_kernel = p_kernel.to(device)
    u_inlet = torch.fft.ifft2(torch.fft.fft2(u_in) * (p_kernel ** (region_z - bg_z)).conj())
    u_outlet = torch.fft.ifft2(torch.fft.fft2(u_out) * (p_kernel ** (bg_z)))
    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    ROI = data_config['ROI']
    batch_size = reconstruction_config['batch_size']
    param_string = 'tv_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['tv_param'])
    param_string += 'value_range_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['value_range_param'])
    param_string += 'sparse_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['sparse_param'])
    initial_type = reconstruction_config['initial_type']
    initial_shift_idx = reconstruction_config['initial_shift_idx']
    initial_name = reconstruction_config['init_name'][:-4]
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'bpm_regu', param_string, f'step_size{step_size}_n_iter{n_iter}', initial_type +initial_name +f'_shift_idx{initial_shift_idx}', f'crop_z_{crop_z}_region_z{region_z}_bg_z{bg_z}')
    out_path = save_pth
    print('save in', out_path)
    os.makedirs(out_path, exist_ok=True)
    # # resolution_pixel = physics_config['camera_pixel_size'] / physics_config['magnification']
    # # physical_param = OpticalSimulationSettings(NA=physics_config['NA'],
    # #                                             RI_bg=physics_config['n_medium'],
    # #                                             volume_sz=data_config['domain_size'],
    # #                                             wavelength=physics_config['wavelength'],
    # #                                             pixel_sz=physics_config['camera_pixel_size'],
    # #                                             mag=physics_config['magnification'], resolution=np.array([resolution_pixel, resolution_pixel, resolution_pixel]))

    regu_func = make_regulizer(reconstruction_config['tv_param'], reconstruction_config['value_range_param'], reconstruction_config['sparse_param'][0])


    init_delta_ri = torch.zeros((region_z, domain_size[1], domain_size[2]), dtype=torch.float32, device=device)
    if initial_type == 'rytov':
        rytov_ri = tifffile.imread(os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], 'rytov', data_name + '_rytov.tif'))
        if(crop_z!=None):
            rytov_ri = rytov_ri[domain_size[0]//2 - (region_z - bg_z):domain_size[0]//2 + bg_z]
        init_delta_ri = torch.from_numpy(rytov_ri).to(device)
    if initial_type == 'constant':
        init_delta_ri = tifffile.imread(os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name , 'bpm_regu','bpm_init',reconstruction_config['init_name']))
        init_delta_ri = torch.from_numpy(init_delta_ri).to(device)
    print('RI shape', init_delta_ri.shape)
    s = init_delta_ri.clone()
    q_1 = 1
    x_1 = init_delta_ri.clone()
    pbar = tqdm(range(n_iter))
    os.makedirs(out_path, exist_ok=True)
    for i in pbar:
        grad, loss = compute_bpm_grad(s, u_inlet, u_outlet, batch_size, bpm_cosFactor, resolution[0], domain_size, k0,
                                      p_kernel, device)
        grad = grad.real
        pbar.set_postfix({'loss': loss.item()}, refresh=False)
        with torch.no_grad():
            z = s - grad * step_size
            # with torch.cuda.device('cuda:1'):
            x_2 = regu_func(z)
            q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            s = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            print(i, ((x_2 - x_1)**2).sum().sqrt())
            x_1 = x_2
            q_1 = q_2
        if ((i) % 5 == 0):
            plot_volume(s, os.path.join(out_path, f'bpm_regu_step{i}.tif') , resolution)
    delta_ri = s

    vmin = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].min()
    vmax = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].max()
    print(vmin)
    print(vmax)

    os.makedirs(os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'bpm_regu'), exist_ok=True)
    plot_volume(delta_ri, os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'bpm_regu', data_name + '.tif'), resolution)
    print('check in', os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'bpm_regu', data_name + '.tif'))

def solve_pps_padding(data_config, physics_config, reconstruction_config):

    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']

    xy_padding_len = reconstruction_config['xy_padding_len']
    # fix me: all even size later
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    domain_size[1] = domain_size[1]-1
    domain_size[2] = domain_size[2]-1
    spec_pixel_size  = 2 * np.pi / (resolution[1] * domain_size[1])
    padded_spec_pixel_size = spec_pixel_size * domain_size[1] / (domain_size[1] + 2 * xy_padding_len)
    print('resolution', resolution)
    wavefront_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name,'wavefront_remove_ref')
    u_in = np.load(os.path.join(wavefront_pth, 'u_in.npy'))
    u_in = torch.from_numpy(u_in).to(device)[:, :-1, :-1]
    u_out = np.load(os.path.join(wavefront_pth, 'u_out.npy'))  # .real
    u_out = torch.from_numpy(u_out).to(device)[:, :-1, :-1]
    pad_array = (xy_padding_len, xy_padding_len, xy_padding_len, xy_padding_len, 0, 0)
    u_in = F.pad(u_in.real, pad_array, mode='constant', value=0) + 1j * F.pad(u_in.imag, pad_array, mode='constant',
                                                                              value=0)
    u_out = F.pad(u_out.real, pad_array, mode='constant', value=0) + 1j * F.pad(u_out.imag, pad_array, mode='constant',
                                                                              value=0)
    ## 不能用角谱传播，只能用bpm自己的传播，不然相位对不上
    crop_z = data_config['crop_z']
    region_z = domain_size[0]
    bg_z = region_z // 2
    if crop_z != None:
        region_z = data_config['region_z']
        bg_z = data_config['bg_z']
    domain_size[1] = domain_size[1] + 2 * xy_padding_len
    domain_size[2] = domain_size[2] + 2 * xy_padding_len
    as_kernel, ol_correction = angular_spectrum_kernel(domain_size, padded_spec_pixel_size, resolution, km)#.to(device)
    as_kernel = as_kernel.to(device)
    ol_correction = ol_correction.to(device).unsqueeze(0)
    u_inlet = torch.fft.ifft2(torch.fft.fft2(u_in) * (as_kernel ** (region_z - bg_z)).conj())#[0].unsqueeze(0)#[::2]#[-1].unsqueeze(0)
    u_outlet = torch.fft.ifft2(torch.fft.fft2(u_out) * (as_kernel ** (bg_z)))#[0].unsqueeze(0)#[::2]#[-1].unsqueeze(0)

    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    ROI = data_config['ROI']
    batch_size = reconstruction_config['batch_size']
    param_string = 'tv_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['tv_param'])
    param_string += 'value_range_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['value_range_param'])
    param_string += 'sparse_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['sparse_param'])
    initial_type = reconstruction_config['initial_type']
    initial_shift_idx = reconstruction_config['initial_shift_idx']
    initial_name = reconstruction_config['init_name'][:-4]
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'pps_regu', param_string, f'step_size{step_size}_n_iter{n_iter}', initial_type +initial_name +f'_shift_idx{initial_shift_idx}', f'crop_z_{crop_z}_region_z{region_z}_bg_z{bg_z}_xy_padding{xy_padding_len}')
    out_path = save_pth
    print('save in', out_path)
    os.makedirs(out_path, exist_ok=True)
    # # resolution_pixel = physics_config['camera_pixel_size'] / physics_config['magnification']
    # # physical_param = OpticalSimulationSettings(NA=physics_config['NA'],
    # #                                             RI_bg=physics_config['n_medium'],
    # #                                             volume_sz=data_config['domain_size'],
    # #                                             wavelength=physics_config['wavelength'],
    # #                                             pixel_sz=physics_config['camera_pixel_size'],
    # #                                             mag=physics_config['magnification'], resolution=np.array([resolution_pixel, resolution_pixel, resolution_pixel]))

    regu_func = make_regulizer(reconstruction_config['tv_param'], reconstruction_config['value_range_param'], reconstruction_config['sparse_param'][0])


    init_delta_ri = torch.zeros((region_z, domain_size[1], domain_size[2]), dtype=torch.float32, device=device)
    if initial_type == 'rytov':
        rytov_ri = tifffile.imread(os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], 'rytov', data_name + '_rytov.tif'))
        init_delta_ri = torch.from_numpy(rytov_ri).to(device)[:, :-1, :-1]
    if initial_type == 'constant':
        init_delta_ri = tifffile.imread(os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name , 'pps_regu','pps_init',reconstruction_config['init_name']))
        init_delta_ri = torch.from_numpy(init_delta_ri).to(device)
    init_delta_ri = F.pad(init_delta_ri, pad_array, mode='constant', value=0)
    print('RI shape', init_delta_ri.shape)
    s = init_delta_ri.clone()
    q_1 = 1
    x_1 = init_delta_ri.clone()
    pbar = tqdm(range(n_iter))
    os.makedirs(out_path, exist_ok=True)
    for i in pbar:
        grad, loss = compute_pps_grad(u_inlet, u_outlet, s, ol_correction, as_kernel, k0,resolution, batch_size, device)
        pbar.set_postfix({'loss': loss}, refresh=False)
        with torch.no_grad():
            z = s - grad * step_size
            x_2 = regu_func(z)
            q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            s = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            print(i, ((x_2 - x_1)**2).sum().sqrt())
            x_1 = x_2
            q_1 = q_2
        if ((i) % 10 == 0):
            plot_volume(s[:, xy_padding_len:-xy_padding_len, xy_padding_len:-xy_padding_len], os.path.join(out_path, f'pps_regu_step{i}.tif') , resolution)
    delta_ri = s

    vmin = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].min()
    vmax = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].max()
    print(vmin)
    print(vmax)

    os.makedirs(os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'pps_regu'), exist_ok=True)
    plot_volume(delta_ri, os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'pps_regu', data_name + '.tif'), resolution)
    print('check in', os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'pps_regu', data_name + '.tif'))

def solve_pps(data_config, physics_config, reconstruction_config):
    if(reconstruction_config['xy_padding_len']!=None):
        solve_pps_padding(data_config, physics_config, reconstruction_config)
        return
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']
    # fix me: all even size later
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    spec_pixel_size  = 2 * np.pi / (resolution[1] * domain_size[1])
    print('resolution', resolution)
    wavefront_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name,'wavefront_remove_ref')
    u_in = np.load(os.path.join(wavefront_pth, 'u_in.npy'))
    u_in = torch.from_numpy(u_in).to(device)#[:, :-1, :-1]
    u_out = np.load(os.path.join(wavefront_pth, 'u_out.npy'))  # .real
    u_out = torch.from_numpy(u_out).to(device)#[:, :-1, :-1]


    ## 不能用角谱传播，只能用bpm自己的传播，不然相位对不上
    crop_z = data_config['crop_z']
    region_z = domain_size[0]
    bg_z = region_z // 2
    if crop_z != None:
        region_z = data_config['region_z']
        bg_z = data_config['bg_z']
    print(region_z)
    print(bg_z)
    as_kernel, ol_correction = angular_spectrum_kernel(domain_size, spec_pixel_size, resolution, km)#.to(device)
    as_kernel = as_kernel.to(device)
    ol_correction = ol_correction.to(device).unsqueeze(0)
    u_inlet = torch.fft.ifft2(torch.fft.fft2(u_in) * (as_kernel ** (region_z - bg_z)).conj())#[0].unsqueeze(0)#[::2]#[-1].unsqueeze(0)
    u_outlet = torch.fft.ifft2(torch.fft.fft2(u_out) * (as_kernel ** (bg_z)))#[0].unsqueeze(0)#[::2]#[-1].unsqueeze(0)

    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    ROI = data_config['ROI']
    batch_size = reconstruction_config['batch_size']
    param_string = 'tv_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['tv_param'])
    param_string += 'value_range_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['value_range_param'])
    param_string += 'sparse_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['sparse_param'])
    initial_type = reconstruction_config['initial_type']
    initial_shift_idx = reconstruction_config['initial_shift_idx']
    initial_name = reconstruction_config['init_name'][:-4]
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'pps_regu', param_string, f'step_size{step_size}_n_iter{n_iter}', initial_type +initial_name +f'_shift_idx{initial_shift_idx}', f'crop_z_{crop_z}_region_z{region_z}_bg_z{bg_z}')
    out_path = save_pth
    print('save in', out_path)
    os.makedirs(out_path, exist_ok=True)
    # # resolution_pixel = physics_config['camera_pixel_size'] / physics_config['magnification']
    # # physical_param = OpticalSimulationSettings(NA=physics_config['NA'],
    # #                                             RI_bg=physics_config['n_medium'],
    # #                                             volume_sz=data_config['domain_size'],
    # #                                             wavelength=physics_config['wavelength'],
    # #                                             pixel_sz=physics_config['camera_pixel_size'],
    # #                                             mag=physics_config['magnification'], resolution=np.array([resolution_pixel, resolution_pixel, resolution_pixel]))

    regu_func = make_regulizer(reconstruction_config['tv_param'], reconstruction_config['value_range_param'], reconstruction_config['sparse_param'][0])


    init_delta_ri = torch.zeros((region_z, domain_size[1], domain_size[2]), dtype=torch.float32, device=device)
    if initial_type == 'rytov':
        rytov_ri = tifffile.imread(os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], 'rytov', data_name + '_rytov.tif'))
        if(crop_z!=None):
            rytov_ri = rytov_ri[domain_size[0]//2 - (region_z - bg_z):domain_size[0]//2 + bg_z]
        init_delta_ri = torch.from_numpy(rytov_ri).to(device)#[:, :-1, :-1]
    if initial_type == 'constant':
        init_delta_ri = tifffile.imread(os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name , 'pps_regu','pps_init',reconstruction_config['init_name']))
        init_delta_ri = torch.from_numpy(init_delta_ri).to(device)
    print('RI shape', init_delta_ri.shape)
    s = init_delta_ri.clone()
    q_1 = 1
    x_1 = init_delta_ri.clone()
    pbar = tqdm(range(n_iter))
    os.makedirs(out_path, exist_ok=True)
    for i in pbar:
        grad, loss = compute_pps_grad(u_inlet, u_outlet, s, ol_correction, as_kernel, k0,resolution, batch_size, device)
        pbar.set_postfix({'loss': loss}, refresh=False)
        with torch.no_grad():
            z = s - grad * step_size
            x_2 = regu_func(z)
            q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            s = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            print(i, ((x_2 - x_1)**2).sum().sqrt())
            x_1 = x_2
            q_1 = q_2
        if ((i) % 10 == 0):
            plot_volume(s, os.path.join(out_path, f'pps_regu_step{i}.tif') , resolution)
    delta_ri = s

    vmin = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].min()
    vmax = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].max()
    print(vmin)
    print(vmax)

    os.makedirs(os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'pps_regu'), exist_ok=True)
    plot_volume(delta_ri, os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'pps_regu', data_name + '.tif'), resolution)
    print('check in', os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'pps_regu', data_name + '.tif'))


def _RI2potential(RI, n_medium, km):
    return (((RI + n_medium) / n_medium) ** 2 - 1) * km ** 2

def solve_rytov_regu_fista_mask(data_config, physics_config, reconstruction_config):
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']
    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    print('resolution', resolution)
    scatter_potential_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'scatter_potential')
    f_count = np.load(os.path.join(scatter_potential_pth, 'potential_mask.npy'))
    f_mask = torch.from_numpy(f_count).to(device)# / 2
    f_spec_target= np.load(os.path.join(scatter_potential_pth, 'potential_fft.npy'))#.real
    f_spec_target = torch.from_numpy(f_spec_target).to(device)


    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    ROI = data_config['ROI']


    param_string = 'tv_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['tv_param'])
    param_string += 'value_range_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['value_range_param'])
    param_string += 'sparse_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['sparse_param'])
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'rytov_regu_fista_mask', param_string, f'step_size{step_size}_n_iter{n_iter}')
    out_path = save_pth
    print('save in', out_path)
    os.makedirs(out_path, exist_ok=True)
    # # resolution_pixel = physics_config['camera_pixel_size'] / physics_config['magnification']
    # # physical_param = OpticalSimulationSettings(NA=physics_config['NA'],
    # #                                             RI_bg=physics_config['n_medium'],
    # #                                             volume_sz=data_config['domain_size'],
    # #                                             wavelength=physics_config['wavelength'],
    # #                                             pixel_sz=physics_config['camera_pixel_size'],
    # #                                             mag=physics_config['magnification'], resolution=np.array([resolution_pixel, resolution_pixel, resolution_pixel]))
    RI2potential = partial(_RI2potential, n_medium=n_medium, km=km)
    regu_func = make_regulizer(reconstruction_config['tv_param'], reconstruction_config['value_range_param'],
                               reconstruction_config['sparse_param'][0], ROI=ROI)

    f = torch.zeros_like(f_spec_target)
    s = f.clone()
    q_1 = 1
    x_1 = f.clone()
    pbar = tqdm(range(n_iter))
    os.makedirs(out_path, exist_ok=True)
    debug_mode = reconstruction_config['debug_mode']
    for i in pbar:
        f_current = f.requires_grad_(True)
        print('f dtype', f.dtype)
        f_spec = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(f_current)))
        loss = (((f_spec * f_mask)[f_mask > 0] - f_spec_target[f_mask > 0]).abs() ** 2).sum() * 1e-4
        print(i, loss)
        grad = torch.autograd.grad(outputs=loss, inputs=f_current)[0]
        with torch.no_grad():
            f = f_current - grad * step_size
            s = f.clone()
            print('s dtype', s.dtype)
            x_2 = regu_func(s.real) + regu_func(s.imag) * 1j# + s.imag * 1j
            q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            f = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            print(i, ((x_2 - x_1)**2).sum().sqrt())
            x_1 = x_2
            q_1 = q_2
            if ((i+1) % 10 == 0 and debug_mode):
                s_ri = (n_medium * torch.sqrt(1 +  s  * (wavelength / (n_medium * 2 * np.pi))**2)) - n_medium
                plot_volume(s_ri.real, os.path.join(out_path, f'rytov_regu_step{i}_ROI{ROI}.tif'), resolution)
    delta_ri = (n_medium * torch.sqrt(1 +  f  * (wavelength / (n_medium * 2 * np.pi))**2)) - n_medium

    os.makedirs(os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'rytov_regu_fista_mask'), exist_ok=True)
    np.save(os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'rytov_regu_fista_mask', data_name + '.npy'), delta_ri.detach().cpu().numpy())

def ind2sub(ind_arr, shape):
    Y = torch.floor(ind_arr / shape[0])
    X = torch.fmod(ind_arr, shape[1])
    return X.long(), Y.long()



def solve_rytov_regu_fista_prop(data_config, physics_config, reconstruction_config):
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']
    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    print('resolution', resolution)
    scatter_potential_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'scatter_potential')
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'])
    u_Rytov = torch.from_numpy(np.load(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'rytov_phs.npy'))).to(device)
    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    ROI = data_config['ROI']
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    size_input = u_Rytov.shape
    size_Joint = data_config['domain_size']
    print('u_Rytov shape', u_Rytov.shape)

    Ax = torch.linspace(-size_input[1] // 2, size_input[1] // 2 - 1, size_input[1]).to(device)
    Ay = torch.linspace(-size_input[1] // 2, size_input[1] // 2- 1, size_input[1]).to(device)
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']

    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    kSizeZ = spec_pixel_size
    kSize = spec_pixel_size
    kScan = torch.from_numpy(np.load(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'k_samp.npy'))).to(device)
    norm_coef = size_Joint[0] * size_Joint[1] * size_Joint[2] * kSizeZ / size_input[1] / size_input[2] / (4 * np.pi * np.pi)
    f_Joint = -torch.from_numpy(np.load(os.path.join(save_pth,data_name, 'scatter_potential', 'potential_fft.npy'))).to(device) / 4/ np.pi / ( 1j * norm_coef)
    kScanZ = torch.sqrt(km**2 - (kScan[:, 0] * kSize)**2 - (kScan[:, 1] * kSize)**2)
    index_map = torch.arange(0, size_input[1] * size_input[2]).reshape(size_input[1], size_input[2]).transpose(1,0).to(device)
    NA = physics_config['NA']
    effective_NA = np.minimum(NA, n_medium)
    k_bound_pixel = np.ceil(effective_NA * 2 * np.pi / wavelength / spec_pixel_size).astype(int)
    NStack = physics_config['angle_num']
    kBound = k_bound_pixel
    print('Nstack', NStack)
    print('kScanz shape', kScanZ.shape)

    param_string = 'tv_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['tv_param'])
    param_string += 'value_range_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['value_range_param'])
    param_string += 'sparse_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['sparse_param'])
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'rytov_regu_fista_prop', param_string, f'step_size{step_size}_n_iter{n_iter}')
    out_path = save_pth
    print('save in', out_path)
    os.makedirs(out_path, exist_ok=True)
    RI2potential = partial(_RI2potential, n_medium=n_medium, km=km)
    regu_func = make_regulizer(reconstruction_config['tv_param'], reconstruction_config['value_range_param'],
                               reconstruction_config['sparse_param'][0], ROI=ROI)
    f = torch.zeros_like(f_Joint)
    I3_list = []
    I1_list = []
    I2_list = []
    kz_list = []
    for i in tqdm(range(kScanZ.shape[0])):
        SAx = Ax - kScan[i, 0]
        SAy = Ay - kScan[i, 1]
        Sx, Sy = torch.meshgrid(SAx, SAy)
        Mindex = torch.lt(torch.square(Sx / kBound) + torch.square(Sy / kBound),  1)
        kz = torch.sqrt(1 - (Sx[Mindex] / kBound)**2 - (Sy[Mindex] / kBound)**2) * km
        I3 = (torch.fmod((kz - kScanZ[i]) // kSizeZ + size_Joint[0] // 2, size_Joint[0]))
        I3 = I3.int()
        I1, I2 = ind2sub(index_map[Mindex], [size_input[1], size_input[2]])
        I3_list.append(I3.long())
        I2_list.append(I2.long())
        I1_list.append(I1.long())
        kz_list.append(kz)

    s = f.clone()
    q_1 = 1
    x_1 = f.clone()
    pbar = tqdm(range(n_iter))
    os.makedirs(out_path, exist_ok=True)
    debug_mode = reconstruction_config['debug_mode']
    for i in pbar:
        f_current = f.requires_grad_(True)
        print('f dtype', f.dtype)
        f_spec = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(f_current)))
        re_prop = []
        for _ in range(kScanZ.shape[0]):
            rytov_prop = torch.zeros_like(u_Rytov[0])
            rytov_prop[I1_list[_], I2_list[_]] += f_spec[I3_list[_], I1_list[_], I2_list[_]] / kz_list[_]
            re_prop.append(rytov_prop)
        re_prop = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(torch.stack(re_prop), dim = (-1,-2)), dim=(-1, -2)), dim = (-1, -2))
        loss = ((re_prop[:, ROI[2]:ROI[3], ROI[4]:ROI[5]] - u_Rytov[:, ROI[2]:ROI[3], ROI[4]:ROI[5]]).imag**2).sum().sqrt()
        print(i, loss)
        grad = torch.autograd.grad(outputs=loss, inputs=f_current)[0]
        with torch.no_grad():
            f = f_current - grad * step_size
            s = f.clone()
            print('s dtype', s.dtype)
            x_2 = regu_func(s.real) + regu_func(s.imag) * 1j# + s.imag * 1j
            q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            f = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            print(i, ((x_2 - x_1)**2).sum().sqrt())
            x_1 = x_2
            q_1 = q_2
            if ((i+1) % 2 == 0 and debug_mode):
                s = -s * 1j * norm_coef * 4 * np.pi
                s_ri = (n_medium * torch.sqrt(1 +  s  * (wavelength / (n_medium * 2 * np.pi))**2)) - n_medium
                plot_volume(s_ri.real, os.path.join(out_path, f'rytov_regu_step{i}_ROI{ROI}.tif'), resolution)
    delta_ri = (n_medium * torch.sqrt(1 +  f  * (wavelength / (n_medium * 2 * np.pi))**2)) - n_medium
    vmin = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].min()
    vmax = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].max()
    print(vmin)
    print(vmax)

    os.makedirs(os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'rytov_regu_fista_prop'), exist_ok=True)
    plot_volume(delta_ri, os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'rytov_regu_fista_prop', data_name + '.tif'), resolution)


def solve_rytov_regu_real_fista_mask(data_config, physics_config, reconstruction_config):
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']
    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    print('resolution', resolution)
    scatter_potential_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'scatter_potential')
    f_count = torch.from_numpy(np.load(os.path.join(scatter_potential_pth, 'potential_mask.npy'))).to(device)
    f_mask = (f_count + torch.roll(torch.flip(f_count, (0, 1, 2)), shifts=(1, 1, 1), dims=(0, 1, 2)))/2
    f_spec_target= np.load(os.path.join(scatter_potential_pth, 'potential_fft.npy'))#.real
    f_spec_target = torch.from_numpy(f_spec_target).to(device)#.real
    f_spec_target = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift((torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(f_spec_target)))).real)))


    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    ROI = data_config['ROI']


    param_string = 'tv_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['tv_param'])
    param_string += 'value_range_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['value_range_param'])
    param_string += 'sparse_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['sparse_param'])
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'rytov_regu_real_fista_mask', param_string, f'step_size{step_size}_n_iter{n_iter}')
    out_path = save_pth
    print('save in', out_path)
    os.makedirs(out_path, exist_ok=True)
    # # resolution_pixel = physics_config['camera_pixel_size'] / physics_config['magnification']
    # # physical_param = OpticalSimulationSettings(NA=physics_config['NA'],
    # #                                             RI_bg=physics_config['n_medium'],
    # #                                             volume_sz=data_config['domain_size'],
    # #                                             wavelength=physics_config['wavelength'],
    # #                                             pixel_sz=physics_config['camera_pixel_size'],
    # #                                             mag=physics_config['magnification'], resolution=np.array([resolution_pixel, resolution_pixel, resolution_pixel]))
    RI2potential = partial(_RI2potential, n_medium=n_medium, km=km)
    regu_func = make_regulizer(reconstruction_config['tv_param'], reconstruction_config['value_range_param'],
                               reconstruction_config['sparse_param'][0], ROI=ROI)

    f = torch.zeros_like(f_spec_target.real)
    s = f.clone()
    q_1 = 1
    x_1 = f.clone()
    pbar = tqdm(range(n_iter))
    os.makedirs(out_path, exist_ok=True)
    debug_mode = reconstruction_config['debug_mode']
    for i in pbar:
        f_current = f.requires_grad_(True)
        print('f dtype', f.dtype)
        f_spec = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(f_current)))
        loss = (((f_spec * f_mask)[f_mask > 0] - f_spec_target[f_mask > 0]).abs() ** 2).sum() * 1e-4
        print(i, loss)
        grad = torch.autograd.grad(outputs=loss, inputs=f_current)[0]
        with torch.no_grad():
            f = f_current - grad * step_size
            s = f.clone()
            print('s dtype', s.dtype)
            x_2 = regu_func(s) #+ regu_func(s.imag) * 1j# + s.imag * 1j
            q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            f = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            print(i, ((x_2 - x_1)**2).sum().sqrt())
            x_1 = x_2
            q_1 = q_2
            if ((i+1) % 10 == 0 and debug_mode):
                s_ri = (n_medium * torch.sqrt(1 +  s  * (wavelength / (n_medium * 2 * np.pi))**2)) - n_medium
                plot_volume(s_ri, os.path.join(out_path, f'rytov_regu_step{i}_ROI{ROI}.tif'), resolution)
    delta_ri = (n_medium * torch.sqrt(1 +  f  * (wavelength / (n_medium * 2 * np.pi))**2)) - n_medium
    vmin = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].min()
    vmax = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].max()
    print(vmin)
    print(vmax)

    os.makedirs(os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'rytov_regu_real_fista_mask'), exist_ok=True)
    plot_volume(delta_ri, os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'rytov_regu_real_fista_mask', data_name + '.tif'), resolution)




def solve_rytov_regu_fit_mask(data_config, physics_config, reconstruction_config):
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']
    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    print('resolution', resolution)
    scatter_potential_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'scatter_potential')
    f_count = np.load(os.path.join(scatter_potential_pth, 'potential_mask.npy'))
    f_mask = torch.from_numpy(f_count).to(device)# / 2
    f_spec_target= np.load(os.path.join(scatter_potential_pth, 'potential_fft.npy'))#.real
    f_spec_target = torch.from_numpy(f_spec_target).to(device)


    n_iter = reconstruction_config['n_iter']
    ROI = data_config['ROI']


    param_string = 'tv_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['tv_param'])
    param_string += 'value_range_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['value_range_param'])
    param_string += 'sparse_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['sparse_param'])
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'rytov_regu_fit_mask', param_string, f'n_iter{n_iter}')
    out_path = save_pth
    print('save in', out_path)
    os.makedirs(out_path, exist_ok=True)
    # # resolution_pixel = physics_config['camera_pixel_size'] / physics_config['magnification']
    # # physical_param = OpticalSimulationSettings(NA=physics_config['NA'],
    # #                                             RI_bg=physics_config['n_medium'],
    # #                                             volume_sz=data_config['domain_size'],
    # #                                             wavelength=physics_config['wavelength'],
    # #                                             pixel_sz=physics_config['camera_pixel_size'],
    # #                                             mag=physics_config['magnification'], resolution=np.array([resolution_pixel, resolution_pixel, resolution_pixel]))
    RI2potential = partial(_RI2potential, n_medium=n_medium, km=km)
    regu_func = make_regulizer(reconstruction_config['tv_param'], reconstruction_config['value_range_param'],
                               reconstruction_config['sparse_param'][0])

    f = torch.zeros_like(f_spec_target)
    s = f.clone()
    q_1 = 1
    x_1 = f.clone()
    pbar = tqdm(range(n_iter))
    os.makedirs(out_path, exist_ok=True)
    debug_mode = reconstruction_config['debug_mode']
    for i in pbar:
        s = f
        tmp = regu_func(s.real) + s.imag * 1j
        f_spec = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(tmp)))
        f_spec[f_mask > 0] = f_spec_target[f_mask > 0]
        f = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(f_spec)))
        if ((i+1) % 2 == 0 and debug_mode):
            s_ri = (n_medium * torch.sqrt(1 +  s  * (wavelength / (n_medium * 2 * np.pi))**2)) - n_medium
            plot_volume(s_ri.real, os.path.join(out_path, f'rytov_regu_step{i}_ROI{ROI}.tif'), resolution)
    delta_ri = (n_medium * torch.sqrt(1 +  f  * (wavelength / (n_medium * 2 * np.pi))**2)) - n_medium
    vmin = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].min()
    vmax = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].max()
    print(vmin)
    print(vmax)

    os.makedirs(os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'rytov_regu_fit_mask'), exist_ok=True)
    plot_volume(delta_ri, os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'rytov_regu_fit_mask', data_name + '.tif'), resolution)

def solve_rytov_regu_downsample(data_config, physics_config, reconstruction_config):
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']
    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    print('resolution', resolution)
    scatter_potential_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'scatter_potential')
    f_count = np.load(os.path.join(scatter_potential_pth, 'potential_mask_H.npy'))
    f_mask = torch.from_numpy(f_count).to(device) / 2
    f_spec_target= np.load(os.path.join(scatter_potential_pth, 'potential_fft.npy'))#.real
    f_spec_target = torch.from_numpy(f_spec_target).to(device)
    downsample_ROI = data_config['downsample_ROI']

    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    ROI = data_config['ROI']

    f_mask = f_mask[downsample_ROI[0]:downsample_ROI[1], downsample_ROI[2]:downsample_ROI[3], downsample_ROI[4]:downsample_ROI[5]]
    f_spec_target = f_spec_target[downsample_ROI[0]:downsample_ROI[1], downsample_ROI[2]:downsample_ROI[3], downsample_ROI[4]:downsample_ROI[5]]
    param_string = 'tv_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['tv_param'])
    param_string += 'value_range_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['value_range_param'])
    param_string += 'sparse_param'
    param_string += '_'.join(f'{i}' for i in reconstruction_config['sparse_param'])
    data_name = data_name +'_downsample'
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'], data_name, 'rytov_regu', param_string, f'step_size{step_size}_n_iter{n_iter}')
    out_path = save_pth
    print('save in', out_path)
    os.makedirs(out_path, exist_ok=True)
    # # resolution_pixel = physics_config['camera_pixel_size'] / physics_config['magnification']
    # # physical_param = OpticalSimulationSettings(NA=physics_config['NA'],
    # #                                             RI_bg=physics_config['n_medium'],
    # #                                             volume_sz=data_config['domain_size'],
    # #                                             wavelength=physics_config['wavelength'],
    # #                                             pixel_sz=physics_config['camera_pixel_size'],
    # #                                             mag=physics_config['magnification'], resolution=np.array([resolution_pixel, resolution_pixel, resolution_pixel]))
    RI2potential = partial(_RI2potential, n_medium=n_medium, km=km)
    regu_func = make_regulizer(reconstruction_config['tv_param'], reconstruction_config['value_range_param'], (reconstruction_config['sparse_param'][0]), ROI)


    print(RI2potential(reconstruction_config['sparse_param'][0]))
    f = torch.randn_like(f_spec_target.real)
    s = f.clone()
    q_1 = 1
    x_1 = f.clone()
    pbar = tqdm(range(n_iter))
    os.makedirs(out_path, exist_ok=True)
    debug_mode = reconstruction_config['debug_mode']
    for i in pbar:
        f_current = f.requires_grad_(True)

        f_spec = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(f_current)))
        loss = (((f_spec * f_mask)[f_mask > 0] - f_spec_target[f_mask > 0]).abs() ** 2).sum() * 1e-4
        grad = torch.autograd.grad(outputs=loss, inputs=f_current)[0]
        with torch.no_grad():
            f = f_current - grad * step_size
            s = f.clone()
            x_2 = f
            x_2 = regu_func(s)
            q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            f = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            x_1 = x_2
            q_1 = q_2
            if ((i+1) % 500 == 0 and debug_mode):
                s_ri = (n_medium * torch.sqrt(1 +  s  * (wavelength / (n_medium * 2 * np.pi))**2)) - n_medium
                plot_volume(s_ri, os.path.join(out_path, f'rytov_regu_step{i}_ROI{ROI}.tif'), resolution)
    delta_ri = (n_medium * torch.sqrt(1 +  f  * (wavelength / (n_medium * 2 * np.pi))**2)) - n_medium
    vmin = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].min()
    vmax = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].max()
    print(vmin)
    print(vmax)

    os.makedirs(os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'rytov_regu_test_prior'), exist_ok=True)
    plot_volume(delta_ri, os.path.join(data_config['save_path'], data_config['expriment_name'],data_config['save_name'], 'rytov_regu_test_prior', data_name + '.tif'), resolution)
