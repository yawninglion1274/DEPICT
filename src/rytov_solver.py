import numpy as np
import torch
import os
import tifffile
from utils.img_utils import ODT_data, plot_volume
from tqdm import tqdm

def RytovRec_downsample(ODT_data, exp, angle_downsample):
    u_Samp_amp = ODT_data.sp_amp
    u_Samp_phs = ODT_data.sp_phs
    u_Back_amp = ODT_data.bg_amp
    u_Back_phs = ODT_data.bg_phs
    km = ODT_data.km

    u_Rytov = torch.log(u_Samp_amp / u_Back_amp) + 1.0j * (u_Samp_phs - u_Back_phs)
    kScan = ODT_data.k_scan_sp.cpu()
    if (angle_downsample == 30):
        u_Rytov = u_Rytov[-96::12]
        kScan = kScan[-96::12]
    elif (angle_downsample == 60):
        u_Rytov = u_Rytov[-96::24]
        kScan = kScan[-96::24]
    else:
        u_Rytov = u_Rytov[::angle_downsample]
        kScan = kScan[::angle_downsample]
    # del ODT_data.sp_amp
    # del ODT_data.sp_phs
    # del ODT_data.bg_amp
    # del ODT_data.bg_phs
    # for i in range(u_Rytov.shape[0]):
    # return
    size_input = u_Rytov.shape
    size_Joint = ODT_data.domain_size
    u_Rytov = torch.fft.fftshift(torch.fft.fftshift(u_Rytov, 1), 2)
    u_Rytov = torch.fft.fft2(u_Rytov)
    u_Rytov = torch.fft.fftshift(torch.fft.fftshift(u_Rytov, 1), 2)
    print('u_Rytov shape', u_Rytov.shape)
    f_Joint = torch.zeros(size_Joint, dtype = torch.cfloat, device=u_Rytov.device)
    f_count = torch.zeros(size_Joint, dtype=torch.int)
    Ax = torch.linspace(-size_input[1] // 2, size_input[1] // 2 - 1, size_input[1])
    Ay = torch.linspace(-size_input[1] // 2, size_input[1] // 2- 1, size_input[1])
    kSizeZ = ODT_data.spec_pixel_size
    kSize = ODT_data.spec_pixel_size

    norm_coef = size_Joint[0] * size_Joint[1] * size_Joint[2] * kSizeZ / size_input[1] / size_input[2] / (4 * np.pi * np.pi)
    kScanZ = torch.sqrt(km**2 - (kScan[:, 0] * kSize)**2 - (kScan[:, 1] * kSize)**2)
    index_map = torch.arange(0, size_input[1] * size_input[2]).reshape(size_input[1], size_input[2]).transpose(1,0)
    NStack = ODT_data.angle_num
    kBound = ODT_data.k_bound_pixel
    print('Nstack', NStack)
    print('kScanz shape', kScanZ.shape)
    for i in tqdm(range(kScanZ.shape[0])):
        SAx = Ax - kScan[i, 0]
        SAy = Ay - kScan[i, 1]
        Sx, Sy = torch.meshgrid(SAx, SAy)
        Mindex = torch.lt(torch.square(Sx / kBound) + torch.square(Sy / kBound),  1)
        kz = torch.sqrt(1 - (Sx[Mindex] / kBound)**2 - (Sy[Mindex] / kBound)**2) * km
        I3 = (torch.fmod((kz - kScanZ[i]) // kSizeZ + size_Joint[0] // 2, size_Joint[0]))

        I3 = I3.int()
        I1, I2 = ind2sub(index_map[Mindex], [size_input[1], size_input[2]])
        f_Joint[I3.long(), I1.long(), I2.long()] += u_Rytov[i, I1.long(), I2.long()] * kz.to(u_Rytov.device)
        f_count[I3.long(), I1.long(), I2.long()] += 1
    f_Joint = f_Joint.cpu()
    f_Joint[f_count > 0] = f_Joint[f_count > 0] / f_count[f_count > 0]
    f_count[f_count > 0] = 1
    # del I3, I1, I2, u_Rytov
    f = -torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(f_Joint))) * 1.0j * norm_coef
    print('f real min', f.real.min(), 'f real max', f.real.max())
    print('f imag min', f.imag.min(), 'f imag max', f.imag.max())
    f = f * 4 * np.pi
    deltaRI = (ODT_data.n_medium * torch.sqrt(1 +  f  * (ODT_data.wavelength / (ODT_data.n_medium * 2 * np.pi))**2)) - ODT_data.n_medium
    deltaRI = deltaRI.real
    f_count[f_count > 0.5] = 1
    f_count[f_count < 0.5] = 0
    f_count_hermite = f_count + torch.flip(f_count, dims=[0, 1, 2])
    f_Joint = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(f)))
    resolution = 1 / np.asarray(size_Joint) * np.max(np.asarray(size_Joint))
    return deltaRI, f_Joint, norm_coef, f, f_count, (torch.log(u_Samp_amp / u_Back_amp) + 1.0j * (u_Samp_phs - u_Back_phs)), resolution

def RytovRec_for_bpm(u_Rytov, domain_size, spec_pixel_size, k_scan_sp, k_bound_pixel, n_medium, wave_length, device):
    km = 2 * np.pi / wave_length * n_medium
    # # OrthoSlicer3D(u_Rytov.imag.numpy()).show()
    # del ODT_data.sp_amp
    # del ODT_data.sp_phs
    # del ODT_data.bg_amp
    # del ODT_data.bg_phs
    size_input = u_Rytov.shape
    size_Joint = domain_size
    u_Rytov = torch.fft.fftshift(torch.fft.fftshift(u_Rytov, 1), 2)
    u_Rytov = torch.fft.fft2(u_Rytov)
    u_Rytov = torch.fft.fftshift(torch.fft.fftshift(u_Rytov, 1), 2)
    print('u_Rytov shape', u_Rytov.shape)
    f_Joint = torch.zeros(size_Joint, dtype = torch.cfloat, device=u_Rytov.device)
    f_count = torch.zeros(size_Joint, dtype=torch.int)
    Ax = torch.linspace(-size_input[1] // 2, size_input[1] // 2 - 1, size_input[1])
    Ay = torch.linspace(-size_input[1] // 2, size_input[1] // 2- 1, size_input[1])
    kSizeZ = spec_pixel_size
    kSize = spec_pixel_size
    temp = k_scan_sp * kSize / km
    bpm_cosFactor = np.cos(np.arcsin(np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2)))
    bpm_cosFactor = torch.from_numpy(bpm_cosFactor.reshape(-1, 1)).to(device)
    norm_coef = size_Joint[0] * size_Joint[1] * size_Joint[2] * kSizeZ / size_input[1] / size_input[2] / (4 * np.pi * np.pi)
    k_scan_sp = torch.from_numpy(k_scan_sp).to(device)
    kScanZ = torch.sqrt(km**2 - (k_scan_sp[:, 0] * kSize)**2 - (k_scan_sp[:, 1] * kSize)**2)
    index_map = torch.arange(0, size_input[1] * size_input[2]).reshape(size_input[1], size_input[2]).transpose(1,0)
    filtered_u_Rytov = torch.zeros_like(u_Rytov)
    NStack = size_input[0]
    kBound = k_bound_pixel
    print('Nstack', NStack)
    print('kScanz shape', kScanZ.shape)

    for i in tqdm(range(kScanZ.shape[0])):
        if(torch.isnan(bpm_cosFactor[i])):
            continue
        SAx = Ax - k_scan_sp[i, 0]
        SAy = Ay - k_scan_sp[i, 1]
        Sx, Sy = torch.meshgrid(SAx, SAy)
        Mindex = torch.lt(torch.square(Sx / kBound) + torch.square(Sy / kBound),  1)
        kz = torch.sqrt(1 - (Sx[Mindex] / kBound)**2 - (Sy[Mindex] / kBound)**2) * km
        I3 = (torch.fmod((kz - kScanZ[i]) // kSizeZ + size_Joint[0] // 2, size_Joint[0]))

        I3 = I3.int()
        I1, I2 = ind2sub(index_map[Mindex], [size_input[1], size_input[2]])
        f_Joint[I3.long(), I1.long(), I2.long()] += u_Rytov[i, I1.long(), I2.long()] * kz.to(u_Rytov.device)
        filtered_u_Rytov[i, I1, I2] = u_Rytov[i, I1, I2]
        f_count[I3.long(), I1.long(), I2.long()] += 1
    f_Joint = f_Joint.cpu()
    f_Joint[f_count > 0] = f_Joint[f_count > 0] / f_count[f_count > 0]
    f_count[f_count > 0] = 1
    # del I3, I1, I2, u_Rytov
    f = -torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(f_Joint))) * 1.0j * norm_coef
    print('f real min', f.real.min(), 'f real max', f.real.max())
    print('f imag min', f.imag.min(), 'f imag max', f.imag.max())
    f = f * 4 * np.pi 
    deltaRI = (n_medium * torch.sqrt(1 +  f  * (wave_length / (n_medium * 2 * np.pi))**2)) - n_medium
    deltaRI = deltaRI.real

    return deltaRI


def ind2sub(ind_arr, shape):
    Y = torch.floor(ind_arr / shape[0])
    X = torch.fmod(ind_arr, shape[1])
    return X.long(), Y.long()


def RytovRec(ODT_data, exp):
    u_Samp_amp = ODT_data.sp_amp
    u_Samp_phs = ODT_data.sp_phs
    u_Back_amp = ODT_data.bg_amp
    u_Back_phs = ODT_data.bg_phs
    km = ODT_data.km

    u_Rytov = torch.log(u_Samp_amp / u_Back_amp) + 1.0j * (u_Samp_phs - u_Back_phs)
    del ODT_data.sp_amp
    del ODT_data.sp_phs
    del ODT_data.bg_amp
    del ODT_data.bg_phs
    # for i in range(u_Rytov.shape[0]):
    # return
    size_input = u_Rytov.shape
    size_Joint = ODT_data.domain_size
    u_Rytov = torch.fft.fftshift(torch.fft.fftshift(u_Rytov, 1), 2)
    u_Rytov = torch.fft.fft2(u_Rytov)
    u_Rytov = torch.fft.fftshift(torch.fft.fftshift(u_Rytov, 1), 2)
    print('u_Rytov shape', u_Rytov.shape)
    f_Joint = torch.zeros(size_Joint, dtype = torch.cfloat, device=u_Rytov.device)
    f_count = torch.zeros(size_Joint, dtype=torch.int)
    Ax = torch.linspace(-size_input[1] // 2, size_input[1] // 2 - 1, size_input[1])
    Ay = torch.linspace(-size_input[1] // 2, size_input[1] // 2- 1, size_input[1])
    kSizeZ = ODT_data.spec_pixel_size
    kSize = ODT_data.spec_pixel_size
    kScan = ODT_data.k_scan_sp.cpu()
    norm_coef = size_Joint[0] * size_Joint[1] * size_Joint[2] * kSizeZ / size_input[1] / size_input[2] / (4 * np.pi * np.pi)
    kScanZ = torch.sqrt(km**2 - (kScan[:, 0] * kSize)**2 - (kScan[:, 1] * kSize)**2)
    index_map = torch.arange(0, size_input[1] * size_input[2]).reshape(size_input[1], size_input[2]).transpose(1,0)
    NStack = ODT_data.angle_num
    kBound = ODT_data.k_bound_pixel
    print('Nstack', NStack)
    print('kScanz shape', kScanZ.shape)
    for i in tqdm(range(kScanZ.shape[0])):
        SAx = Ax - kScan[i, 0]
        SAy = Ay - kScan[i, 1]
        Sx, Sy = torch.meshgrid(SAx, SAy)
        Mindex = torch.lt(torch.square(Sx / kBound) + torch.square(Sy / kBound),  1)
        kz = torch.sqrt(1 - (Sx[Mindex] / kBound)**2 - (Sy[Mindex] / kBound)**2) * km
        I3 = (torch.fmod((kz - kScanZ[i]) // kSizeZ + size_Joint[0] // 2, size_Joint[0]))

        I3 = I3.int()
        I1, I2 = ind2sub(index_map[Mindex], [size_input[1], size_input[2]])
        f_Joint[I3.long(), I1.long(), I2.long()] += u_Rytov[i, I1.long(), I2.long()] * kz.to(u_Rytov.device)
        f_count[I3.long(), I1.long(), I2.long()] += 1
    f_Joint = f_Joint.cpu()
    f_Joint[f_count > 0] = f_Joint[f_count > 0] / f_count[f_count > 0]
    f_count[f_count > 0] = 1
    # del I3, I1, I2, u_Rytov
    f = -torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(f_Joint))) * 1.0j * norm_coef
    print('f real min', f.real.min(), 'f real max', f.real.max())
    print('f imag min', f.imag.min(), 'f imag max', f.imag.max())
    f = f * 4 * np.pi
    deltaRI = (ODT_data.n_medium * torch.sqrt(1 +  f  * (ODT_data.wavelength / (ODT_data.n_medium * 2 * np.pi))**2)) - ODT_data.n_medium
    deltaRI = deltaRI.real
    f_count[f_count > 0.5] = 1
    f_count[f_count < 0.5] = 0
    f_count_hermite = f_count + torch.flip(f_count, dims=[0, 1, 2])
    f_Joint = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(f)))
    resolution = 1 / np.asarray(size_Joint) * np.max(np.asarray(size_Joint))
    return deltaRI, f_Joint, norm_coef, f, f_count, (torch.log(u_Samp_amp / u_Back_amp) + 1.0j * (u_Samp_phs - u_Back_phs)), resolution


def checkRytovProp(data_config, physics_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_name = data_config['data_name']
    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'])
    u_Rytov = torch.from_numpy(np.load(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'rytov_phs.npy'))).to(device)
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
    re_prop = []
    for i in tqdm(range(kScanZ.shape[0])):
        SAx = Ax - kScan[i, 0]
        SAy = Ay - kScan[i, 1]
        Sx, Sy = torch.meshgrid(SAx, SAy)
        Mindex = torch.lt(torch.square(Sx / kBound) + torch.square(Sy / kBound),  1)
        kz = torch.sqrt(1 - (Sx[Mindex] / kBound)**2 - (Sy[Mindex] / kBound)**2) * km
        I3 = (torch.fmod((kz - kScanZ[i]) // kSizeZ + size_Joint[0] // 2, size_Joint[0]))


        I3 = I3.int()
        I1, I2 = ind2sub(index_map[Mindex], [size_input[1], size_input[2]])
        rytov_prop = torch.zeros_like(u_Rytov[0])
        rytov_prop[I1.long(), I2.long()] = f_Joint[I3.long(), I1.long(), I2.long()] / kz
        print(rytov_prop.shape)
        rytov_prop = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(rytov_prop)))
        re_prop.append(rytov_prop)
    re_prop = torch.stack(re_prop)
    error = re_prop - u_Rytov
    re_prop_path = os.path.join(save_pth,  data_name, 'rytov_prop')
    os.makedirs(re_prop_path, exist_ok=True)
    np.save(os.path.join(re_prop_path, 'reprop_wf.npy'), re_prop.cpu().numpy())
    np.save(os.path.join(re_prop_path, 'reprop_error.npy'), error.cpu().numpy())
    tifffile.imsave(os.path.join(re_prop_path, 'reprop_wf.tif'), re_prop.abs().cpu().numpy())
    tifffile.imsave(os.path.join(re_prop_path, 'reprop_error.tif'), error.abs().cpu().numpy())
    tifffile.imsave(os.path.join(re_prop_path, 'rytov_phase_wf.tif'), u_Rytov.abs().cpu().numpy())





if __name__ == '__main__':
    exp = 'PMC'
    data_config = f'configs/rytov_{exp}/data_config.yaml'
    physics_config = f'configs/rytov_{exp}/physics_config.yaml'
    device = torch.device('cuda')
    cos7_1 = ODT_data(data_config, physics_config)
    cos7_1 = holo_process(cos7_1, device)
    delta_RI, mask_RI = RytovRec(cos7_1, exp)#.real
    delta_RI = delta_RI.real
    print(delta_RI[20:-20, 20:-20, 20:-20].max(), delta_RI[20:-20, 20:-20, 20:-20].min())
    plot_volume(delta_RI, f'result/with_edge_rytov_{exp}_{cos7_1.domain_size[0]}')
    plot_volume(mask_RI[20:-20, 20:-20, 20:-20], f'result/masked_rytov_{exp}_{cos7_1.domain_size[0]}')




