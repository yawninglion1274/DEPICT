"""Unweighted affine OLS over every supplied mask pixel; positive divergence."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .masks import validate_masks
from .volume_io import write_csv, write_json


def prepare_support(mask):
    yy, xx = np.nonzero(mask)
    if len(yy) < 3:
        raise ValueError("The mask must support a rank-three in-plane affine fit")
    positions = np.column_stack([xx, yy]).astype(np.float64)
    center = positions.mean(axis=0)
    design = np.column_stack([np.ones(len(yy)), positions-center])
    if len(yy) < 3 or np.linalg.matrix_rank(design) != 3:
        raise ValueError("The mask must support a rank-three in-plane affine fit")
    return yy, xx, center, design


def fit_areal_strain_rate(flow, mask, *, fps=32.5, components=(1, 2)):
    """Fit [u_column,u_row]=b+B*[column,row]; rate=+fps*trace(B).

    Displacements are native voxels/frame. Default (row,column) components
    are (dy,dx); pass (dz,dx)=(0,2) for a displayed XZ section.
    """
    flow = np.asarray(flow)
    if np.asarray(mask).ndim != 2:
        raise ValueError("fit_areal_strain_rate requires a single 2D mask")
    mask = validate_masks(mask, flow.shape[-2:])[0]
    if flow.shape != (3, *mask.shape) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Expected a (dz,dy,dx) section and a positive frame rate")
    yy, xx, center, design = prepare_support(mask)
    row, column = components
    values = np.column_stack([flow[column, yy, xx], flow[row, yy, xx]]).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite in-mask displacement; no samples may be dropped")
    coef, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank != 3:
        raise ValueError("Rank-deficient OLS fit")
    gradient = coef[1:].T
    residual = values-design@coef
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return dict(coefficients=coef, B=gradient, center_xy=center,
                rate_s=float(fps*np.trace(gradient)), rmse_voxel_frame=rmse)


def analyze_and_save(flows, masks, *, native_left_frames, output, fps=32.5,
                     time_origin=0, region_names=None, components=(1, 2)):
    """Stream through fields, save numerical NPZ/CSV/JSON; no plotting/smoothing."""
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None]
    masks = validate_masks(masks, masks.shape[-2:])
    frames = np.asarray(native_left_frames, dtype=int)
    names = region_names if region_names is not None else [f"region_{i}" for i in range(len(masks))]
    if len(names) != len(masks) or len(set(names)) != len(names) or not len(frames):
        raise ValueError("Supply frame indices and unique names for all masks")
    fits = []
    for flow in flows:
        fits.append([fit_areal_strain_rate(flow, mask, fps=fps, components=components) for mask in masks])
    if len(fits) != len(frames):
        raise ValueError("Frame indices and flow count differ")
    rate = np.array([[fit["rate_s"] for fit in row] for row in fits]).T
    gradients = np.array([[fit["B"] for fit in row] for row in fits]).transpose(1, 0, 2, 3)
    coefficients = np.array([[fit["coefficients"] for fit in row] for row in fits]).transpose(1, 0, 2, 3)
    rmses = np.array([[fit["rmse_voxel_frame"] for fit in row] for row in fits]).T
    times = (frames-time_origin+.5)/fps
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "areal_strain_rate.npz", masks=masks, region_names=np.array(names),
                        native_pairs=np.column_stack([frames, frames+1]), time_s=times,
                        rate_s=rate, B=gradients, coefficients=coefficients, rmse_voxel_frame=rmses,
                        center_xy=np.array([fit["center_xy"] for fit in fits[0]]), fps=fps)
    write_csv(output / "areal_strain_rate.csv", [dict(region=name, left_native_frame=int(frame),
        right_native_frame=int(frame+1), time_mid_s=float(times[j]), areal_strain_rate_s=float(rate[i, j]),
        rmse_voxel_frame=float(rmses[i, j])) for i, name in enumerate(names) for j, frame in enumerate(frames)])
    write_json(output / "areal_strain_rate.json", dict(method="unweighted affine ordinary least squares",
        definition="positive divergence: +fps*trace(B)", fps=fps,
        row_column_components=list(components), all_mask_pixels=True, smoothing=False, plotting=False))
    return rate
