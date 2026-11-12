"""Figure 3 crop, orientation and diagonal-comparison functions."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

VOXEL_SIZE_UM = 0.09
TARGET_VOLUME_SHAPE = (256, 450, 450)
VMIN, VMAX = 0.0, 0.06
CELL_CONFIGS = {}

def center_crop_volume(volume, target_shape=TARGET_VOLUME_SHAPE):
    slices = []
    starts = []
    for axis_len, target_len in zip(volume.shape, target_shape):
        if axis_len < target_len:
            raise ValueError(f"Cannot crop shape {volume.shape} to {target_shape}")
        start = (axis_len - target_len) // 2
        starts.append(start)
        slices.append(slice(start, start + target_len))
    return volume[tuple(slices)], tuple(starts)

def load_volume(path):
    try:
        volume = tifffile.memmap(path)
    except ValueError:
        volume = tifffile.imread(path)
    if len(volume.shape) != 3:
        raise ValueError(f"Expected a 3D volume for {path}, got {volume.shape}")
    return center_crop_volume(volume)

def axial_source_axis(swap_xy=False, rotate_xy=True):
    # Fix an XY display column so both panels share the same vertical direction.
    axis = 1 if rotate_xy else 2
    return 3 - axis if swap_xy else axis

def resolve_slice_indices(volume, crop_starts, z_idx, y_idx, swap_xy=False, rotate_xy=True):
    # None selects the central plane after cropping; integers remain source indices.
    z_idx = volume.shape[0] // 2 if z_idx is None else z_idx - crop_starts[0]
    lateral_axis = axial_source_axis(swap_xy, rotate_xy)
    y_idx = volume.shape[lateral_axis] // 2 if y_idx is None else y_idx - crop_starts[lateral_axis]
    return z_idx, y_idx

def load_slices(path, z_idx, y_idx, include_xy=False, offset=0.0, swap_xy=False, rotate_xy=True):
    volume, crop_starts = load_volume(path)
    z_idx, y_idx = resolve_slice_indices(volume, crop_starts, z_idx, y_idx, swap_xy, rotate_xy)
    if swap_xy:
        volume = np.swapaxes(volume, 1, 2)
    if offset:
        volume = np.asarray(volume, dtype=np.float32) + offset
    if not (0 <= z_idx < volume.shape[0]):
        raise ValueError(f"z_idx {z_idx} is outside cropped shape {volume.shape} for {path}")
    if not (0 <= y_idx < volume.shape[1 if rotate_xy else 2]):
        raise ValueError(f"y_idx {y_idx} is outside cropped shape {volume.shape} for {path}")

    if rotate_xy:
        volume = np.rot90(volume, k=1, axes=(1, 2))
    axial_slice = volume[:, :, y_idx]
    slice_xy = np.asarray(volume[z_idx, :, :], dtype=np.float32) if include_xy else None
    slice_xz = np.asarray(axial_slice, dtype=np.float32).transpose()
    return slice_xy, slice_xz

def combine_rytov_bpm(rytov_xz, bpm_xz):
    height, width = rytov_xz.shape
    yy, xx = np.indices((height, width))
    split_y = 0.25 * (height - 1) + 0.5 * (height - 1) * xx / (width - 1)
    return np.where(yy <= split_y, rytov_xz, bpm_xz).astype(np.float32)
