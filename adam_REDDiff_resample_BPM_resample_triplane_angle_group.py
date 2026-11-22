import os
import torch
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tifffile
from functools import partial
import argparse
import yaml
from utils.img_utils import plot_volume
from src.bpm_grad import compute_bpm_grad, angular_spectrum_kernel
from src.rytov_solver import RytovRec_for_bpm
from torch import optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_FLAG = True

def multi_sample_x0(model, xt, scheduler, batch_size, t, views, generator=None):
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}')

    x0_sample = torch.zeros_like(xt)
    count = torch.zeros_like(xt)
    sub_imgs = []
    for h_start, h_end, w_start, w_end in views:
        sub_imgs.append(xt[:, :, h_start:h_end, w_start:w_end])
    sub_imgs = torch.concat(sub_imgs, 0)
    sub_x_pred = []
    # Batch the complete set of 2-D view patches.
    for start in range(0, sub_imgs.shape[0], batch_size):
        batch_sub_imgs = sub_imgs[start:start + batch_size]
        timesteps = torch.tensor([t] * batch_sub_imgs.shape[0], device=xt.device)
        batch_sub_noise_pred = model(batch_sub_imgs, timesteps).sample
        batch_sub_x_pred = scheduler.step(
            batch_sub_noise_pred, t, batch_sub_imgs, generator=generator
        ).pred_original_sample
        sub_x_pred.append(batch_sub_x_pred)
    sub_x_pred = torch.tensor_split(torch.concat(sub_x_pred, 0), len(views))
    for i, (h_start, h_end, w_start, w_end) in enumerate(views):
        x0_sample[:, :, h_start:h_end, w_start:w_end] += sub_x_pred[i]
        count[:, :, h_start:h_end, w_start:w_end] += 1
    return x0_sample / count

def get_grad_scale(mu, vmin, vmax, rescaler_from, rescaler_to):
    """
    Compute the Jacobian scaling factor for mapping gradients from latent space
    back to the physical-space variable mu.
    """
    device = mu.device
    
    # --- 1. Derivative of inverse_scaler ---
    d_step2 = 2.0 / (vmax - vmin)
    
    # --- 2. Derivative of second_inverse_scaler (scaler2img) ---
    # This is a piecewise-linear function, so its derivative is piecewise constant.
    from0, from1 = rescaler_from
    to0, to1 = rescaler_to
    
    inv_scale_ratio = (from1 - from0) / (to1 - to0)
    
    # Compute the local derivative for each value range of mu.
    # y < to0: derivative is 1
    # to0 <= y <= to1: derivative is inv_scale_ratio
    # y > to1: derivative is 1 (the slope of ... + (y - to1) is 1).
    d_step1 = torch.where(
        mu < to0,
        torch.tensor(1.0, device=device),
        torch.where(
            mu <= to1,
            torch.tensor(inv_scale_ratio, device=device),
            torch.tensor(1.0, device=device)
        )
    )
    
    # --- 3. Apply the chain rule ---
    total_scale = d_step2 * d_step1
    
    return total_scale

def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return (x + 1) / 2


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def inverse_scaler_function(volume, v_min, v_max):
    return ((volume - v_min) / (v_max - v_min)) * 2 - 1


def scaler_function(volume, v_min, v_max):
    return (volume + 1) / 2 * (v_max - v_min) + v_min


def posterior_sampling(
    init_ri_x0,            # Initial physical-space estimate (initial value of mu)
    model,                 # Diffusion model
    scheduler,             # DDPMScheduler
    grad_func,             # Physical-space gradient function
    scaler, inverse_scaler, # First-stage scalers
    second_scaler, second_inverse_scaler, # Second-stage scalers
    lambda_scheduling_type,
    grad_scale,
    views,                 # View configuration for the xz/yz directions
    xy_views,              # View configuration for the xy direction
    batch_size,            # Batch size
    save_path,
    resample_times,        # Number of RED-Diff resampling iterations
    resample_start,        # Timestep at which RED-Diff resampling starts
    lr=0.05,               # Adam learning rate
    lambda_scale=1.0,      # RED regularization weight (balances physical and prior gradients)
    data_weight=0.1, 
    device='cuda',
    resample_end=1,
    axial_count=None,
    ave_start=0,           # Enable three-direction averaging below this timestep
    yz_views=None,         # Optional separate views for y/z when the ROI is not square
    roi=None,              # Compute the RED prior only in this ROI; None means the full volume
    start_time=999,
    save_timesteps=None,
    seed=0,
):
    if not 0 <= start_time < len(scheduler.alphas_cumprod):
        raise ValueError(f'start_time={start_time} is outside the diffusion schedule')
    if save_timesteps is not None:
        save_timesteps = frozenset(save_timesteps)
        if any(t < 0 or t >= len(scheduler.alphas_cumprod) for t in save_timesteps):
            raise ValueError('save_timesteps must be within the diffusion schedule')
    # --- 1. Initialize optimization variables (RED-Diff logic) ---
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    # Compute sigma
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sigma_ks = sqrt_1m_alphas_cumprod / sqrt_alphas_cumprod
    
    if axial_count is None:
        calc_dir = lambda val, count: val % 3
    else:
        calc_dir = lambda val, count: val % 2 if count < axial_count else 2
        
    # Define the scheduling function
    if lambda_scheduling_type == 'linear':
        lambda_fn = lambda sigma: sigma * lambda_scale
    elif lambda_scheduling_type == 'sqrt':
        lambda_fn = lambda sigma: np.sqrt(sigma) * lambda_scale
    elif lambda_scheduling_type == 'constant':
        lambda_fn = lambda sigma: lambda_scale
    else:
        raise NotImplementedError(f"Unknown schedule: {lambda_scheduling_type}")
        
    mu = init_ri_x0.clone().detach().to(device)
    # Advance one local RNG throughout the run, including scheduler variance draws.
    generator = torch.Generator(device=mu.device).manual_seed(seed)
    mu.requires_grad_(True)
    roi_slices, roi_shape = make_roi_slices(roi, mu.shape)
    use_roi_prior = roi_slices is not None
    if use_roi_prior:
        print(f"RED prior ROI enabled: roi={roi}, cropped shape={roi_shape}")
    else:
        print("RED prior ROI disabled: using full volume for diffusion prior.")
    xz_views = views
    yz_views = yz_views if yz_views is not None else views
    
    # Use the Adam optimizer
    optimizer = optim.Adam([mu], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    
    print(f"Start Hybrid Optimization: Adam Solver with 3-Direction (xz/yz/xy) Transforms.")

    # --- 2. Solve using the RED-Diff annealing loop ---
    # A restart can begin inside the interval that already uses resampling.
    current_resample_times = resample_times if resample_end < start_time <= resample_start else 1
    count = 0
    
    for t in tqdm(range(start_time, -1, -1), total=start_time + 1):
        should_save = False
        if save_timesteps is not None:
            should_save = t in save_timesteps
        elif t < 10:
            # t < 10: save every timestep
            should_save = True
        elif t > 300:
            # t > 300: save only t=900
            should_save = (t == 900)
        else:
            # 10 <= t <= 300: save sparsely, every 100 steps (100, 200, 300)
            should_save = (t % 100 == 0)
        should_save = should_save and SAVE_FLAG
        if should_save:
            plot_volume((mu.detach()), os.path.join(save_path, f'iso_{t}.tif'), [1, 1, 1])
        
        # Adjust the number of resampling iterations dynamically.
        if t == resample_start:
            current_resample_times = resample_times
        if t == resample_end:
            current_resample_times = 1
            
        for jj in range(current_resample_times):
            optimizer.zero_grad()
            
            # Physical space -> latent space
            if use_roi_prior:
                mu_prior = mu.detach()[roi_slices]
            else:
                mu_prior = mu.detach()
            x_0_fusion = inverse_scaler(second_inverse_scaler(mu_prior)).unsqueeze(1)
            
            # Construct the noisy input required by RED-Diff.
            timesteps = torch.tensor([t] * x_0_fusion.shape[0], device=device)
            noise = torch.empty_like(x_0_fusion).normal_(generator=generator)
            
            xt = scheduler.add_noise(x_0_fusion, noise, timesteps)
            
            # Choose single-direction prediction or three-direction averaging based on t < ave_start.
            with torch.no_grad():
                # Branch A: average the scores from all three directions when t < ave_start.
                if t < ave_start:
                    accumulated_x0_pred = 0.0
                    
                    # Always iterate over all three directions: 0=xz, 1=yz, 2=xy.
                    for direction_idx in range(3):
                        # 1. Permute
                        if direction_idx == 0: # xz
                            noisy_volume = xt.permute(2, 1, 3, 0)
                            current_views = xz_views
                        elif direction_idx == 1: # yz
                            noisy_volume = xt.permute(3, 1, 2, 0)
                            current_views = yz_views
                        else: # xy
                            noisy_volume = xt
                            current_views = xy_views
                        
                        # 2. Network Prediction
                        pred_temp = multi_sample_x0(
                            model, noisy_volume, scheduler, batch_size, t, current_views, generator
                        )
                        
                        # 3. Inverse permute
                        if direction_idx == 0:
                            pred_temp = pred_temp.permute(3, 1, 0, 2)
                        elif direction_idx == 1:
                            pred_temp = pred_temp.permute(3, 1, 2, 0)
                        # No inverse permutation is needed for xy.
                        
                        # 4. Accumulate
                        accumulated_x0_pred += pred_temp
                    
                    # Average to obtain the final x0 prediction.
                    x_0_pred_latent = accumulated_x0_pred / 3.0
                    
                # Branch B: retain the original single-direction rotation when t >= ave_start.
                else:
                    direction = calc_dir(jj + t, count)
                    
                    # 1. Permute
                    if direction == 0:
                        noisy_volume = xt.permute(2, 1, 3, 0)
                        current_views = xz_views
                        count += 1
                    elif direction == 1:
                        noisy_volume = xt.permute(3, 1, 2, 0)
                        current_views = yz_views
                        count += 1
                    else: # direction == 2
                        noisy_volume = xt
                        current_views = xy_views
                        count = 0
                        
                    # 2. Network Prediction
                    x_0_pred_latent = multi_sample_x0(
                        model, noisy_volume, scheduler, batch_size, t, current_views, generator
                    )

                    # 3. Inverse Permute
                    if direction == 0:
                        x_0_pred_latent = x_0_pred_latent.permute(3, 1, 0, 2)
                    elif direction == 1:
                        x_0_pred_latent = x_0_pred_latent.permute(3, 1, 2, 0)


            # Latent space -> physical space and gradient calculation
            
            # A. Compute the data consistency gradient.
            data_grad, data_loss = grad_func(init_delta_ri=mu.detach())
            data_grad = data_grad.real * data_weight 
            
            # 1. Add the physical-space gradient.
            mu.grad = data_grad
            
            # 2. Add the RED regularization gradient (prior term).
            alpha_prod_t = alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            
            alpha_sqrt = alpha_prod_t ** 0.5
            beta_sqrt = beta_prod_t ** 0.5

            with torch.no_grad():
                # Use the x_0_pred_latent computed above (averaged or from one direction).
                # Compute the corresponding predicted noise.
                pred_noise = (xt - alpha_sqrt * x_0_pred_latent) / beta_sqrt
                
                # Compute the RED gradient direction: (pred_noise - noise).
                # When t < ave_start, pred_noise combines information from all three directions.
                noise_grad_latent = pred_noise - noise
            
            grad_scale_factor = grad_scale(mu_prior)
            red_grad_roi = noise_grad_latent.squeeze(1) * grad_scale_factor
            current_lambda = lambda_fn(sigma_ks[t].item())
            if use_roi_prior:
                red_grad = torch.zeros_like(mu)
                red_grad[roi_slices] = red_grad_roi
            else:
                red_grad = red_grad_roi
            mu.grad += current_lambda * red_grad
            
            # Update the optimizer.
            optimizer.step()
            mu.data = torch.clamp(mu.data, min=0., max=0.5)
            
        # Print progress.
        if t % 50 == 0:
            direction_names = ['xz', 'yz', 'xy']
            # Show the current mode.
            if t < ave_start:
                curr_dir_str = "Averaged (xz+yz+xy)"
            else:
                curr_dir_str = direction_names[(jj + t) % 3]
                
            print(f't={t} | Data Loss: {data_loss:.6f} | Direction: {curr_dir_str}')

    return mu.detach()


def scaler2RI(x, rescaler_from=[0.09, 0.1], rescaler_to=[0.09, 0.14]):
    """
    Piecewise linear: unchanged for x < from[0], mapped linearly to the to
    interval for from[0] <= x <= from[1], and linearly extrapolated for x > from[1].
    """
    from0, from1 = rescaler_from
    to0, to1 = rescaler_to
    scale = (to1 - to0) / (from1 - from0)
    y = torch.where(
        x < from0,
        x,
        torch.where(
            x <= from1,
            to0 + (x - from0) * scale,
            to0 + (from1 - from0) * scale + (x - from1)  # Linear extrapolation
        )
    )
    return y

def scaler2img(y, rescaler_from=[0.09, 0.1], rescaler_to=[0.09, 0.14]):
    """
    Piecewise-linear inverse: unchanged for y < to[0], mapped linearly back to
    the from interval for to[0] <= y <= to[1], and linearly extrapolated for y > to[1].
    """
    from0, from1 = rescaler_from
    to0, to1 = rescaler_to
    inv_scale = (from1 - from0) / (to1 - to0)
    x = torch.where(
        y < to0,
        y,
        torch.where(
            y <= to1,
            from0 + (y - to0) * inv_scale,
            from0 + (to1 - to0) * inv_scale + (y - to1)  # Linear extrapolation
        )
    )
    return x

def safe_slice_z(arr, center, region_z, bg_z):
    """
    Extract a region from arr centered at center, padding out-of-bounds areas with zeros.
    
    arr: Input array
    center: Center index (usually domain_size[0] // 2)
    region_z: Output size along the z dimension
    bg_z: Distance from the center to the boundary
    
    Returns an array with shape (region_z, *arr.shape[1:]).
    """
    start = center - (region_z - bg_z)
    end = center + bg_z
    
    # Create the output array, padded with zeros.
    output_shape = (region_z,) + arr.shape[1:]
    result = torch.zeros(output_shape, dtype=arr.dtype)
    
    # Compute the valid source index range.
    src_start = max(0, start)
    src_end = min(arr.shape[0], end)
    
    # Compute the corresponding destination index range.
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    
    # Copy the valid portion.
    if src_start < src_end:
        result[dst_start:dst_end] = arr[src_start:src_end]
    
    return result


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
    return field#, amp * torch.exp(2j * np.pi * (XS * kx / nx + YS * ky / ny))


def generate_views(target_h, target_w, sub_resolution, add_on=1):
    """
    Helper function to generate a list of views.
    
    Args:
        target_h: Target resolution along the height dimension
        target_w: Target resolution along the width dimension
        sub_resolution: Sub-image resolution (model input size)
        add_on: Additional stride increment
    
    Returns:
        views: [(h_start, h_end, w_start, w_end), ...]
    """
    # Height direction
    if target_h <= sub_resolution:
        h_start_arr = [0]
    else:
        stride_h = target_h // sub_resolution + add_on
        if stride_h > 1:
            spacing_h = (target_h - sub_resolution) // (stride_h - 1)
        else:
            spacing_h = 0
        h_start_arr = [spacing_h * j for j in range(stride_h - 1)]
        h_start_arr.append(target_h - sub_resolution)
    
    # Width direction
    if target_w <= sub_resolution:
        w_start_arr = [0]
    else:
        stride_w = target_w // sub_resolution + add_on
        if stride_w > 1:
            spacing_w = (target_w - sub_resolution) // (stride_w - 1)
        else:
            spacing_w = 0
        w_start_arr = [spacing_w * j for j in range(stride_w - 1)]
        w_start_arr.append(target_w - sub_resolution)
    
    # Generate views.
    views = []
    for h in h_start_arr:
        for w in w_start_arr:
            h_end = h + sub_resolution if h + sub_resolution < target_h else None
            w_end = w + sub_resolution if w + sub_resolution < target_w else None
            views.append((h, h_end, w, w_end))
    
    # Special case: both dimensions equal sub_resolution.
    if target_h == sub_resolution and target_w == sub_resolution:
        views = [(None, None, None, None)]
    
    return views


def make_roi_slices(roi, volume_shape):
    if roi is None:
        return None, tuple(volume_shape)
    if len(roi) != 6:
        raise ValueError(f"ROI must have 6 entries [z0,z1,x0,x1,y0,y1], got {roi}")
    if all(v is None for v in roi):
        return None, tuple(volume_shape)

    slices = (slice(roi[0], roi[1]), slice(roi[2], roi[3]), slice(roi[4], roi[5]))
    roi_shape = []
    for axis, (slc, dim) in enumerate(zip(slices, volume_shape)):
        start, stop, step = slc.indices(dim)
        if step != 1:
            raise ValueError(f"ROI step must be 1 on axis {axis}, got {step}")
        length = max(0, stop - start)
        if length <= 0:
            raise ValueError(
                f"ROI axis {axis} is empty after applying {slc} to dimension {dim}"
            )
        roi_shape.append(length)
    return slices, tuple(roi_shape)


def normalize_roi_config(roi):
    if roi is None:
        return [None, None, None, None, None, None]
    if len(roi) != 6:
        raise ValueError(f"ROI must have 6 entries [z0,z1,x0,x1,y0,y1], got {roi}")
    return roi


def build_roi_triplane_views(roi_shape, sub_resolution, add_on=1):
    z_len, x_len, y_len = roi_shape
    xz_views = generate_views(y_len, z_len, sub_resolution, add_on=add_on)
    yz_views = generate_views(x_len, z_len, sub_resolution, add_on=add_on)
    xy_views = generate_views(x_len, y_len, sub_resolution, add_on=add_on)
    return xz_views, yz_views, xy_views


def normalize_angle_group_mode(angle_group_mode):
    mode = angle_group_mode.strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'sparse': 'sparse_view',
        'sparse_view': 'sparse_view',
        'fast': 'fast_acq',
        'fast_acq': 'fast_acq',
    }
    if mode not in aliases:
        raise ValueError(
            f'angle_group_mode must be one of sparse_view or fast_acq, got {angle_group_mode}'
        )
    return aliases[mode]


def select_angle_group_observations(
    field,
    angle_group,
    frame_idx,
    total_observations=240,
    angle_group_mode='sparse_view',
):
    """
    Select dynamic observations from measurements.

    In sparse_view mode, the input observation count is assumed to be
    total_observations. For angle_group < total_observations, use:

        field[frame_idx::total_observations // angle_group]

    In fast_acq mode, use the actual input observation count and select
    contiguous acquisition groups in acquisition order:

        field[frame_idx * angle_group:(frame_idx + 1) * angle_group]

    fast_acq does not require the observation count to be 240 or divisible by
    angle_group; any trailing incomplete group is ignored.
    """
    angle_group_mode = normalize_angle_group_mode(angle_group_mode)
    actual_observations = field.shape[0]

    if angle_group_mode == 'fast_acq':
        if angle_group <= 0 or angle_group > actual_observations:
            raise ValueError(
                f'angle_group must be in [1, {actual_observations}] for fast_acq, got {angle_group}'
            )

        if angle_group == actual_observations:
            return field

        num_groups = actual_observations // angle_group
        if frame_idx < 0 or frame_idx >= num_groups:
            raise ValueError(
                f'frame_idx must be in [0, {num_groups - 1}] for fast_acq '
                f'with angle_group={angle_group} and {actual_observations} observations, got {frame_idx}'
            )

        start = frame_idx * angle_group
        selected = field[start:start + angle_group]
        if selected.shape[0] != angle_group:
            raise RuntimeError(
                f'Expected {angle_group} selected observations, got {selected.shape[0]}'
            )
        return selected

    if actual_observations != total_observations:
        raise ValueError(
            f'Expected {total_observations} observations, got {actual_observations}'
        )

    if angle_group <= 0 or angle_group > total_observations:
        raise ValueError(
            f'angle_group must be in [1, {total_observations}], got {angle_group}'
        )

    if angle_group == total_observations:
        return field

    if total_observations % angle_group != 0:
        raise ValueError(
            f'total_observations={total_observations} must be divisible by angle_group={angle_group}'
        )

    num_groups = total_observations // angle_group
    if frame_idx < 0 or frame_idx >= num_groups:
        raise ValueError(
            f'frame_idx must be in [0, {num_groups - 1}] for angle_group={angle_group}, got {frame_idx}'
        )

    selected = field[frame_idx::num_groups]

    if selected.shape[0] != angle_group:
        raise RuntimeError(
            f'Expected {angle_group} selected observations, got {selected.shape[0]}'
        )
    return selected


def save_k_scan_scatter_plot(
    k_scan_samp,
    save_dir,
    angle_group,
    frame_idx,
    total_observations=240,
    angle_group_mode='sparse_view',
):
    angle_group_mode = normalize_angle_group_mode(angle_group_mode)

    if isinstance(k_scan_samp, torch.Tensor):
        k_scan_np = k_scan_samp.detach().cpu().numpy()
    else:
        k_scan_np = np.asarray(k_scan_samp)

    if k_scan_np.ndim != 2 or k_scan_np.shape[1] < 2:
        raise ValueError(
            f'Expected k_scan_samp with shape (N, >=2), got {k_scan_np.shape}'
        )

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    scatter = ax.scatter(
        k_scan_np[:, 0],
        k_scan_np[:, 1],
        c=np.arange(k_scan_np.shape[0]),
        cmap='viridis',
        s=36,
        linewidths=0
    )
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    if angle_group == total_observations:
        title = f'Full-angle k-scan scatter ({total_observations} angles)'
    else:
        title = (
            f'Selected k-scan scatter '
            f'({angle_group} angles, mode={angle_group_mode}, frame_idx={frame_idx})'
        )
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label='Selected angle index')
    fig.tight_layout()

    save_path = os.path.join(save_dir, 'selected_k_scan_scatter.png')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved selected k-scan scatter plot to {save_path}')


def load_initial_volume(paths, expected_shape, device):
    """Load physical delta-RI volumes and average aligned inputs without rescaling."""
    initial_volume = np.zeros(tuple(expected_shape), dtype=np.float32)
    for path in paths:
        volume = tifffile.imread(path)
        if volume.shape != tuple(expected_shape):
            raise ValueError(f'Initialization shape {volume.shape} != {expected_shape}: {path}')
        if np.iscomplexobj(volume) or not np.isfinite(volume).all():
            raise ValueError(f'Initialization must contain finite real delta-RI values: {path}')
        initial_volume += volume.astype(np.float32, copy=False) / len(paths)
    if not paths or not np.isfinite(initial_volume).all():
        raise ValueError('Initialization must contain at least one finite volume')
    print(f'Loaded mean of {len(paths)} physical initialization volume(s): {paths}')
    return torch.from_numpy(initial_volume).to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--physics_config', type=str)
    parser.add_argument('--reconstruction_config', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--lambda_scale', type=float)
    parser.add_argument('--step_size', type=float)
    parser.add_argument('--resample_times', type=int)
    parser.add_argument('--resample_start', type=int)
    parser.add_argument('--prior_batch_size', type=int, default=22,
                        help='Maximum number of 2-D prior patches passed through the diffusion UNet at once. '
                             'Lower this if CUDA OOM occurs.')
    parser.add_argument('--bpm_batch_size', type=int, default=6,
                        help='Number of observations per BPM gradient batch.')
    parser.add_argument('--axial_count', type=int, default=None)
    parser.add_argument('--init_tif', nargs='+', default=None,
                        help='Physical delta-RI TIFF(s), averaged without rescaling for initialization.')
    parser.add_argument('--start_time', type=int, default=None,
                        help='First diffusion timestep, inclusive. Required with --init_tif. '
                             'Defaults to 999.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Explicit result directory.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for the reconstruction noise sequence (default: 0).')
    parser.add_argument('--seed_in_output_path', action='store_true',
                        help='Append a seed subdirectory to the result path.')
    parser.add_argument('--require_cuda', action='store_true',
                        help='Fail if the selected GPU is unavailable instead of using the CPU.')
    parser.add_argument('--save_timesteps', nargs='+', type=int, default=None,
                        help='Save intermediate iso_T.tif volumes only at these timesteps. '
                             'The final iso.tif is always saved.')
    parser.add_argument('--skip_init_save', action='store_true',
                        help='Do not write the initialization volume init_x0.tif.')
    parser.add_argument('--angle_group', type=int, required=True,
                        help='Number of angles used for one dynamic reconstruction. '
                             'In sparse_view mode it must divide 240. In fast_acq mode it can be '
                             'any positive value up to the actual number of observations.')
    parser.add_argument('--angle_group_mode', type=normalize_angle_group_mode, default='sparse_view',
                        choices=['sparse_view', 'fast_acq'],
                        help='Angle-group selection mode. sparse_view keeps the original interleaved '
                             'selection field[frame_idx::240//angle_group]; fast_acq uses contiguous '
                             'groups in acquisition order such as [0:12], [12:24], ... for angle_group=12.')
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='Angle group index. In sparse_view mode it is the interleaved frame offset; '
                             'in fast_acq mode it selects contiguous group [idx*angle_group:(idx+1)*angle_group]. '
                             'Ignored when angle_group equals the active observation count.')
    parser.add_argument('--stop_before_reconstruction', action='store_true',
                        help='Run preprocessing and argument/data checks, then stop immediately before '
                             'posterior_sampling starts.')

    args = parser.parse_args()
    if not 0 <= args.seed < 2**64:
        parser.error('--seed must be in [0, 2**64)')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'Reconstruction seed: {args.seed}')
    if args.require_cuda and device.type != 'cuda':
        parser.error('--require_cuda was set but the selected CUDA device is unavailable')
    if args.init_tif and args.start_time is None:
        parser.error('--init_tif requires an explicit --start_time')
    if args.start_time is not None and args.start_time < 0:
        parser.error('--start_time must be non-negative')
    if args.bpm_batch_size <= 0:
        parser.error('--bpm_batch_size must be positive')
    axial_count = args.axial_count
    angle_group = args.angle_group
    angle_group_mode = args.angle_group_mode
    frame_idx = args.frame_idx
    data_config = load_yaml(args.data_config)
    physics_config = load_yaml(args.physics_config)
    reconstruction_config = load_yaml(args.reconstruction_config)

    model_config = load_yaml(args.model_config)
    model_name = model_config['model_name']
    model_dir = model_config['model_dir']
    model = torch.load(model_dir, map_location=device, weights_only=False).to(device)
    model.eval()

    resample_start = args.resample_start
    resample_end = reconstruction_config['resample_end']
    start_time = reconstruction_config['start_time']
    data_weight = reconstruction_config['data_weight']
    resample_times = args.resample_times
    ave_start = reconstruction_config['ave_start']
    data_name = data_config['data_name']
    ROI = normalize_roi_config(data_config.get('ROI', [None, None, None, None, None, None]))
    num_inference_steps = model_config['num_inference']
    sampling_start_time = args.start_time if args.start_time is not None else 999
    if sampling_start_time >= num_inference_steps:
        raise ValueError(f'start_time must be less than num_inference={num_inference_steps}')
    if args.start_time is not None:
        start_time = args.start_time

    vmin = 0
    vmax = 0.1
    inverse_scaler = partial(inverse_scaler_function, v_min=vmin, v_max=vmax)
    scaler = partial(scaler_function, v_min=vmin, v_max=vmax)
    print('RI vmin', vmin, 'potential vmax', vmax)
    rescaler_from = [0.08, 0.1]
    rescaler_to = [0.08, 0.15]
    grad_scale = partial(get_grad_scale, vmin=vmin, vmax=vmax, rescaler_from=rescaler_from, rescaler_to=rescaler_to)
    second_scaler = partial(scaler2RI, rescaler_from=rescaler_from, rescaler_to=rescaler_to)
    second_inverse_scaler = partial(scaler2img, rescaler_from=rescaler_from, rescaler_to=rescaler_to)
    ## load bpm wavefront
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']
    raw_size = data_config['raw_size']
    parent_size = data_config['parent_size']
    resolution = pixelsize * raw_size / parent_size * np.ones_like(np.array(domain_size),dtype=np.float32)
    spec_pixel_size = 2 * np.pi / (resolution[-1] * domain_size[-1])
    print('resolution', resolution)
    save_pth = data_config['data_path']
    sp_name = data_config['data_name']
    center = data_config['parent_size']//2
    shift_idx = data_config['shift_idx']
    if(center == spec_size[0]):
        print('no crop')
        sp_amp = np.load(os.path.join(save_pth, 'sp_amp',sp_name + '.npy'))#[:, center-spec_size[0]//2 + shift_idx[0]:center+spec_size[0]//2 + shift_idx[0], center-spec_size[1]//2+shift_idx[1]:center+spec_size[1]//2 + shift_idx[1]]
        sp_phs = np.load(os.path.join(save_pth, 'sp_phs', sp_name + '.npy'))#[:, center-spec_size[0]//2 + shift_idx[0]:center+spec_size[0]//2 + shift_idx[0], center-spec_size[1]//2+shift_idx[1]:center+spec_size[1]//2 + shift_idx[1]]
        bg_amp = np.load(os.path.join(save_pth, 'bg_amp', sp_name + '.npy'))#[:, center-spec_size[0]//2 + shift_idx[0]:center+spec_size[0]//2 + shift_idx[0], center-spec_size[1]//2+shift_idx[1]:center+spec_size[1]//2 + shift_idx[1]]
        bg_phs = np.load(os.path.join(save_pth, 'bg_phs', sp_name + '.npy'))#[:, center-spec_size[0]//2 + shift_idx[0]:center+spec_size[0]//2 + shift_idx[0], center-spec_size[1]//2+shift_idx[1]:center+spec_size[1]//2 + shift_idx[1]]
        k_scan_samp = np.load(os.path.join(save_pth, 'k_scan', sp_name + '.npy')) #/ (center * 2) * domain_size[-1]

    else:
        print('crop')
        sp_amp = np.load(os.path.join(save_pth, 'sp_amp',sp_name + '.npy'))[:, center-spec_size[0]//2 + shift_idx[0]:center+spec_size[0]//2 + shift_idx[0], center-spec_size[1]//2+shift_idx[1]:center+spec_size[1]//2 + shift_idx[1]]
        sp_phs = np.load(os.path.join(save_pth, 'sp_phs', sp_name + '.npy'))[:, center-spec_size[0]//2 + shift_idx[0]:center+spec_size[0]//2 + shift_idx[0], center-spec_size[1]//2+shift_idx[1]:center+spec_size[1]//2 + shift_idx[1]]
        bg_amp = np.load(os.path.join(save_pth, 'bg_amp', sp_name + '.npy'))[:, center-spec_size[0]//2 + shift_idx[0]:center+spec_size[0]//2 + shift_idx[0], center-spec_size[1]//2+shift_idx[1]:center+spec_size[1]//2 + shift_idx[1]]
        bg_phs = np.load(os.path.join(save_pth, 'bg_phs', sp_name + '.npy'))[:, center-spec_size[0]//2 + shift_idx[0]:center+spec_size[0]//2 + shift_idx[0], center-spec_size[1]//2+shift_idx[1]:center+spec_size[1]//2 + shift_idx[1]]
        k_scan_samp = np.load(os.path.join(save_pth, 'k_scan', sp_name + '.npy')) / (center * 2) * domain_size[-1]

    configured_total_observations = 240
    actual_observations = int(sp_amp.shape[0])
    total_observations = (
        actual_observations if angle_group_mode == 'fast_acq'
        else configured_total_observations
    )
    if angle_group == total_observations:
        print(
            f'angle_group={angle_group}, using full-angle reconstruction '
            f'with all {total_observations} observations.'
        )
        if frame_idx != 0:
            print(f'frame_idx={frame_idx} is ignored because angle_group={angle_group}.')
    else:
        if angle_group_mode == 'sparse_view':
            num_groups = total_observations // angle_group
            print(
                f'Applying sparse_view dynamic observation selection: '
                f'field[{frame_idx}::{num_groups}] with angle_group={angle_group}'
            )
        else:
            if angle_group <= 0 or angle_group > total_observations:
                raise ValueError(
                    f'angle_group must be in [1, {total_observations}] for fast_acq, got {angle_group}'
                )
            num_groups = total_observations // angle_group
            selection_start = frame_idx * angle_group
            selection_end = selection_start + angle_group
            print(
                f'Applying fast_acq dynamic observation selection: '
                f'field[{selection_start}:{selection_end}] with angle_group={angle_group}. '
                f'Using actual observation count {total_observations}; '
                f'valid full-group frame_idx range is 0..{num_groups - 1}.'
            )
        sp_amp = select_angle_group_observations(
            sp_amp, angle_group, frame_idx, total_observations, angle_group_mode
        )
        sp_phs = select_angle_group_observations(
            sp_phs, angle_group, frame_idx, total_observations, angle_group_mode
        )
        bg_amp = select_angle_group_observations(
            bg_amp, angle_group, frame_idx, total_observations, angle_group_mode
        )
        bg_phs = select_angle_group_observations(
            bg_phs, angle_group, frame_idx, total_observations, angle_group_mode
        )
        k_scan_samp = select_angle_group_observations(
            k_scan_samp, angle_group, frame_idx, total_observations, angle_group_mode
        )
        print(f'Selected dynamic observations, resulting shape: {sp_amp.shape}')

    sp_amp = torch.from_numpy(sp_amp).to(device)
    sp_phs = torch.from_numpy(sp_phs).to(device)
    bg_amp = torch.from_numpy(bg_amp).to(device)
    bg_phs = torch.from_numpy(bg_phs).to(device)
    temp = k_scan_samp * spec_pixel_size / km
    bpm_cosFactor = np.cos(np.arcsin(np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2)))
    bpm_cosFactor = torch.from_numpy(bpm_cosFactor.reshape(-1, 1)).to(device)


    ## Use BPM's own propagation instead of angular-spectrum propagation to keep the phase consistent.
    crop_z = data_config['crop_z']
    region_z = domain_size[0]
    bg_z = region_z // 2
    if crop_z != None:
        region_z = data_config['region_z']
        bg_z = data_config['bg_z']
    p_kernel, _ = angular_spectrum_kernel(domain_size, spec_pixel_size, resolution[0], km)#.to(device)
    p_kernel = p_kernel.to(device)
    k_scan_samp = torch.from_numpy(k_scan_samp).to(device)
    u_in = compute_wavefront(bg_amp, bg_phs, k_scan_samp)
    u_out = compute_wavefront(sp_amp, sp_phs, k_scan_samp)
    print('u_in', u_in.shape, 'u_out', u_out.shape, p_kernel.shape)
    u_inlet = torch.fft.ifft2(torch.fft.fft2(u_in) * (p_kernel ** (region_z - bg_z)).conj())
    u_outlet = torch.fft.ifft2(torch.fft.fft2(u_out) * (p_kernel ** (bg_z)))
    k_bound_pixel = np.ceil(physics_config['NA'] * 2 * np.pi / wavelength / spec_pixel_size).astype(int)

    if sp_amp.shape[0] % args.bpm_batch_size:
        raise ValueError('Selected observation count must be divisible by bpm_batch_size')
    grad_func = partial(compute_bpm_grad, u_in=u_inlet, u_out=u_outlet, batch_size=args.bpm_batch_size, cos_factor=bpm_cosFactor,
                        dz=resolution[0], domain_size=domain_size, k0=k0, p_kernel=p_kernel, device=device)

    ## PMC - Generate views

    print('vmin', vmin, 'vmax', vmax)
    inverse_scaler = partial(inverse_scaler_function, v_min=vmin, v_max=vmax)
    scaler = partial(scaler_function, v_min=vmin, v_max=vmax)
    sub_resolution = model_config['image_size']
    domain_size = data_config['domain_size']
    print(ROI)
    
    add_on = 1
    
    # Views for the xz/yz directions
    # xz: (X, 1, Y, Z) -> h=Y, w=Z
    # yz: (Y, 1, X, Z) -> h=X, w=Z
    # Use region_z as the Z dimension when generating views along w. When
    # region_z > image_size, sliding windows cover the full volume along Z
    # instead of processing only the first 256 slices.
    xz_target_h = domain_size[2]
    yz_target_h = domain_size[1]
    z_target_w = region_z
    print('xz target resolution (h=Y, w=Z):', xz_target_h, z_target_w)
    print('yz target resolution (h=X, w=Z):', yz_target_h, z_target_w)
    
    xz_views = generate_views(xz_target_h, z_target_w, sub_resolution, add_on=add_on)
    yz_views = generate_views(yz_target_h, z_target_w, sub_resolution, add_on=add_on)
    print('xz views:', xz_views)
    print('yz views:', yz_views)
    
    # Views for the xy direction
    # xy: (Z, 1, X, Y) -> h=X, w=Y
    # Use domain_size[1] and domain_size[2].
    xy_target_h = domain_size[1]  # X dimension
    xy_target_w = domain_size[2]  # Y dimension
    print('xy target resolution (h=X):', xy_target_h, '(w=Y):', xy_target_w)
    
    # Generate xy_views with the helper function.
    xy_views = generate_views(xy_target_h, xy_target_w, sub_resolution, add_on=add_on)
    print('xy views:', xy_views)

    roi_slices, roi_shape = make_roi_slices(ROI, (region_z, domain_size[1], domain_size[2]))
    if roi_slices is None:
        red_xz_views = xz_views
        red_yz_views = yz_views
        red_xy_views = xy_views
        posterior_roi = None
        print('RED prior views use full volume.')
    else:
        red_xz_views, red_yz_views, red_xy_views = build_roi_triplane_views(
            roi_shape, sub_resolution, add_on=add_on
        )
        posterior_roi = ROI
        print('RED prior ROI slices:', roi_slices, 'shape:', roi_shape)
        print('RED prior xz views:', red_xz_views)
        print('RED prior yz views:', red_yz_views)
        print('RED prior xy views:', red_xy_views)
    
    # Scheduler and initialization
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_inference_steps, beta_schedule="linear",
                                    variance_type='fixed_large')
    print('domain_size:', domain_size)
    init_type = 'tif' if args.init_tif else reconstruction_config['init_type']
    init_x0 = torch.zeros((region_z, domain_size[1], domain_size[2]), dtype=torch.float32, device=device)
    if init_type == 'tif':
        init_x0 = load_initial_volume(args.init_tif, init_x0.shape, device)
    elif init_type == 'rytov':
        num_init_angles = sp_amp.shape[0]
        print(f'Initializing rytov with all selected observations: {num_init_angles} angles')
        u_Rytov = torch.log(sp_amp / bg_amp) + 1.0j * (sp_phs - bg_phs)
        rytov_ri = RytovRec_for_bpm(
            u_Rytov.cpu(),
            domain_size,
            spec_pixel_size,
            k_scan_samp.cpu().numpy(),
            k_bound_pixel,
            n_medium,
            wavelength,
            'cpu'
        )
        print('crop rytov ri', rytov_ri.shape, domain_size[0] // 2 - (region_z - bg_z), domain_size[0] // 2 + bg_z)
        if (crop_z != None):
            rytov_ri = safe_slice_z(rytov_ri, domain_size[0] // 2, region_z, bg_z)
        init_x0 = rytov_ri.to(device)
        del u_Rytov
    else:
        print(f'Using default zero initialization for init_type={init_type}')
    del u_in, u_out, bg_phs, sp_phs, bg_amp, sp_amp
    batch_size = args.prior_batch_size
    # for lambda_scale in [1,2.5,5,7.5]:
    coef_type = 'sqrt'
    for lambda_scale in [args.lambda_scale]: #0.03, 0.006, 0.003,step size 0.005, 
    # for lambda_scale in [50]:
        # for step_size in [0.0001, 0.00005]:
        for step_size in [args.step_size]:#0.0002
            if angle_group == total_observations:
                dynamic_obs_str = f'_angle_group{angle_group}_full'
            else:
                dynamic_obs_str = f'_angle_group{angle_group}_{angle_group_mode}_frame_idx{frame_idx}'
 
            save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'],
                                    data_name, f'adam_REDdiff_3dir_resample_{coef_type}_inittype{init_type}_axial_count{axial_count}_ave_start{ave_start}{dynamic_obs_str}',
                                    model_name,
                                    f'resample_start{resample_start}_resample_times{resample_times}_start_time{start_time}_end_time{resample_end}_data_weight{data_weight}',
                                    f'lambda{lambda_scale}_step_size{step_size}_recover_volume_with_ini_add_on{add_on}')
            if args.output_dir is not None:
                save_pth = args.output_dir
            if args.seed_in_output_path:
                save_pth = os.path.join(save_pth, f'seed{args.seed}')
            if args.stop_before_reconstruction:
                print('[PRE-RECON CHECK] Reached reconstruction loop entry.')
                print(f'[PRE-RECON CHECK] Would save results to: {save_pth}')
                print('[PRE-RECON CHECK] Exiting before posterior_sampling.')
                continue
            os.makedirs(save_pth, exist_ok=True)
            print(save_pth)
            save_k_scan_scatter_plot(
                k_scan_samp, save_pth, angle_group, frame_idx, total_observations, angle_group_mode
            )
            if not args.skip_init_save:
                plot_volume(init_x0, os.path.join(save_pth, 'init_x0.tif'), [1, 1, 1])
            recover_volume = posterior_sampling(
                init_ri_x0=init_x0, # Corresponds to mu
                model=model,
                scheduler=noise_scheduler,
                grad_func=grad_func,
                scaler=scaler,
                inverse_scaler=inverse_scaler,
                second_scaler=second_scaler,
                second_inverse_scaler=second_inverse_scaler,
                grad_scale=grad_scale,
                views=red_xz_views,
                xy_views=red_xy_views,  # Pass xy_views
                batch_size=batch_size,
                save_path=save_pth,
                resample_times=resample_times,
                resample_start=resample_start,
                lambda_scheduling_type=coef_type,
                lr=step_size,
                lambda_scale=lambda_scale,
                device=device,
                data_weight=data_weight,
                resample_end=resample_end,
                axial_count=axial_count,
                ave_start=ave_start,
                yz_views=red_yz_views,
                roi=posterior_roi,
                start_time=sampling_start_time,
                save_timesteps=args.save_timesteps,
                seed=args.seed,
            )
            print(save_pth)
            plot_volume((recover_volume), os.path.join(save_pth, f'iso.tif'), [1, 1, 1])
