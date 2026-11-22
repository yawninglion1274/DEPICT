import argparse
import glob
import os

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from src.bpm_grad import angular_spectrum_kernel, compute_bpm_grad
from src.prior import make_regulizer
from src.rytov_solver import RytovRec_for_bpm
from utils.img_utils import load_yaml, plot_volume

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    field = amp * torch.exp(1j * phs) * torch.exp(2j * np.pi * (XS * kx / nx + YS * ky / ny))
    return field


def safe_slice_z(arr, center, region_z, bg_z):
    """Crop around the axial center, padding out-of-bounds slices with zeros."""
    start = center - (region_z - bg_z)
    end = center + bg_z

    output_shape = (region_z,) + arr.shape[1:]
    result = torch.zeros(output_shape, dtype=arr.dtype)

    src_start = max(0, start)
    src_end = min(arr.shape[0], end)

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    if src_start < src_end:
        result[dst_start:dst_end] = arr[src_start:src_end]

    return result


def load_fields(data_config, reconstruction_config):
    center = data_config['parent_size'] // 2
    spec_size = data_config['spec_size']
    shift_idx = data_config['shift_idx']
    starts = [center - size // 2 + shift for size, shift in zip(spec_size, shift_idx)]
    stops = [start + size for start, size in zip(starts, spec_size)]
    fields = {}
    for name in ('sp_amp', 'sp_phs', 'bg_amp', 'bg_phs', 'k_scan'):
        path = os.path.join(data_config['data_path'], name, data_config['data_name'] + '.npy')
        if name == 'sp_amp' and not os.path.isfile(path):
            path = path[:-4] + '.tif.npy'
        fields[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    shape = fields['sp_amp'].shape
    if len(shape) != 3 or any(fields[name].shape != shape for name in ('sp_phs', 'bg_amp', 'bg_phs')):
        raise ValueError('Sample and background amplitude/phase shapes must agree')
    if fields['k_scan'].shape != (shape[0], 2):
        raise ValueError('k_scan must have shape (observations, 2)')
    if any(start < 0 or stop > shape[axis + 1]
           for axis, (start, stop) in enumerate(zip(starts, stops))):
        raise ValueError('Crop exceeds the input field')
    frame_start = reconstruction_config.get('frame_start', 0)
    observations = reconstruction_config.get('observations', shape[0] - frame_start)
    if frame_start < 0 or observations <= 0 or frame_start + observations > shape[0]:
        raise ValueError('Requested observations exceed the input field')
    batch_size = reconstruction_config['batch_size']
    if batch_size <= 0 or observations % batch_size:
        raise ValueError('batch_size must be positive and divide the selected observation count')
    selected = slice(frame_start, frame_start + observations)
    for name in ('sp_amp', 'sp_phs', 'bg_amp', 'bg_phs'):
        fields[name] = fields[name][selected, starts[0]:stops[0], starts[1]:stops[1]]
    fields['k_scan'] = fields['k_scan'][selected] / (center * 2) * data_config['domain_size'][-1]
    return fields


def output_paths(data_config, reconstruction_config):
    sample_dir = os.path.join(data_config['save_path'], data_config['expriment_name'],
                              data_config['save_name'], data_config['data_name'])
    param_string = ''.join(name + '_'.join(str(value) for value in reconstruction_config[name])
                           for name in ('tv_param', 'value_range_param', 'sparse_param'))
    region_z = data_config['region_z'] if data_config['crop_z'] is not None else data_config['domain_size'][0]
    bg_z = data_config['bg_z'] if data_config['crop_z'] is not None else region_z // 2
    step_size, n_iter = reconstruction_config['step_size'], reconstruction_config['n_iter']
    initial = reconstruction_config['initial_type'] + reconstruction_config['init_name'][:-4]
    initial += f"_shift_idx{reconstruction_config['initial_shift_idx']}"
    geometry = f"crop_z_{data_config['crop_z']}_region_z{region_z}_bg_z{bg_z}_phase_correction0"
    out_path = os.path.join(sample_dir, 'bpm_regu', param_string,
                            f'step_size{step_size}_n_iter{n_iter}', initial, geometry)
    final_path = os.path.join(data_config['save_path'], data_config['expriment_name'],
                              data_config['save_name'], 'bpm_regu', data_config['data_name'] + '.tif')
    return out_path, os.path.join(sample_dir, 'rytov.tif'), final_path


def solve_bpm_clean_style(data_config, physics_config, reconstruction_config, dry_run=False):
    fields = load_fields(data_config, reconstruction_config)
    out_path, rytov_path, final_path = output_paths(data_config, reconstruction_config)
    print('observations', fields['sp_amp'].shape[0])
    print('save in', out_path)
    print('final volume', final_path)
    if dry_run:
        return
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    domain_size = data_config['domain_size']

    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    print('resolution', resolution)
    sp_amp, sp_phs, bg_amp, bg_phs = (
        torch.from_numpy(np.array(fields[name], copy=True))
        for name in ('sp_amp', 'sp_phs', 'bg_amp', 'bg_phs')
    )
    k_scan_samp = fields['k_scan']
    del fields
    initial_type = reconstruction_config['initial_type']
    if initial_type == 'rytov':
        u_Rytov = torch.log(sp_amp / bg_amp) + 1.0j * (sp_phs - bg_phs)
    sp_amp = sp_amp.to(device)
    sp_phs = sp_phs.to(device)
    bg_amp = bg_amp.to(device)
    bg_phs = bg_phs.to(device)
    temp = k_scan_samp * spec_pixel_size / km
    bpm_cosFactor = np.cos(np.arcsin(np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2)))
    bpm_cosFactor = torch.from_numpy(bpm_cosFactor.reshape(-1, 1)).to(device)
    crop_z = data_config['crop_z']
    region_z = domain_size[0]
    bg_z = region_z // 2
    if crop_z is not None:
        region_z = data_config['region_z']
        bg_z = data_config['bg_z']
    p_kernel, _ = angular_spectrum_kernel(domain_size, spec_pixel_size, resolution[0], km)
    p_kernel = p_kernel.to(device)
    k_scan_samp = torch.from_numpy(k_scan_samp).to(device)
    u_in = compute_wavefront(bg_amp, bg_phs, k_scan_samp)
    u_out = compute_wavefront(sp_amp, sp_phs, k_scan_samp)
    u_inlet = torch.fft.ifft2(torch.fft.fft2(u_in) * (p_kernel ** (region_z - bg_z)).conj())
    u_outlet = torch.fft.ifft2(torch.fft.fft2(u_out) * (p_kernel ** (bg_z)))
    k_bound_pixel = np.ceil(physics_config['NA'] * 2 * np.pi / wavelength / spec_pixel_size).astype(int)

    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    ROI = data_config.get('ROI') or [None] * 6
    batch_size = reconstruction_config['batch_size']
    save_every = reconstruction_config.get('save_every', 30)
    os.makedirs(out_path, exist_ok=True)

    regu_func = make_regulizer(
        reconstruction_config['tv_param'], reconstruction_config['value_range_param'],
        reconstruction_config['sparse_param'][0], step_size=step_size,
    )

    init_delta_ri = torch.zeros((region_z, domain_size[1], domain_size[2]), dtype=torch.float32, device=device)
    if initial_type == 'rytov':
        rytov_ri = RytovRec_for_bpm(
            u_Rytov, domain_size, spec_pixel_size, k_scan_samp.cpu().numpy(),
            k_bound_pixel, n_medium, wavelength, 'cpu',
        )
        print('crop rytov ri', rytov_ri.shape, domain_size[0] // 2 - (region_z - bg_z), domain_size[0] // 2 + bg_z)
        if crop_z is not None:
            rytov_ri = safe_slice_z(rytov_ri, domain_size[0] // 2, region_z, bg_z)
        init_delta_ri = rytov_ri.to(device)
        del u_Rytov
    del u_in, u_out, bg_phs, sp_phs, bg_amp, sp_amp
    tifffile.imwrite(rytov_path, init_delta_ri.cpu().numpy())
    print('RI shape', init_delta_ri.shape)
    s = init_delta_ri.clone()
    q_1 = 1
    x_1 = init_delta_ri.clone()
    pbar = tqdm(range(n_iter))
    for i in pbar:
        grad, loss = compute_bpm_grad(
            s, u_inlet, u_outlet, batch_size, bpm_cosFactor, resolution[0],
            domain_size, k0, p_kernel, device,
        )
        grad = grad.real
        pbar.set_postfix({'loss': loss.item()}, refresh=False)
        with torch.no_grad():
            z = s - grad * step_size
            x_2 = regu_func(z)
            q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            s = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            x_1 = x_2
            q_1 = q_2
        if i % save_every == 0:
            plot_volume(s, os.path.join(out_path, f'bpm_regu_step{i}.tif'), resolution)
    delta_ri = s
    vmin = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].min()
    vmax = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].max()
    print(vmin)
    print(vmax)

    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    plot_volume(delta_ri, final_path, resolution)
    print('check in', final_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, required=True)
    parser.add_argument('--physics_config', type=str, required=True)
    parser.add_argument('--save_name', default=None, type=str)
    parser.add_argument('--reconstruction_config', type=str, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Check inputs and print output paths without reconstructing')
    args = parser.parse_args()

    physics_config = load_yaml(args.physics_config)
    reconstruction_config = load_yaml(args.reconstruction_config)
    if args.save_name is not None:
        paths = list(glob.iglob(args.data_config))
        paths.sort()
        print(paths)
        for pth in paths:
            sub_data_config = load_yaml(pth)
            if sub_data_config['save_name'] == args.save_name:
                solve_bpm_clean_style(sub_data_config, physics_config, reconstruction_config, dry_run=args.dry_run)
    else:
        data_config = load_yaml(args.data_config)
        solve_bpm_clean_style(data_config, physics_config, reconstruction_config, dry_run=args.dry_run)
