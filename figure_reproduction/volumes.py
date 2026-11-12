"""Three-view volume cropping and display transforms."""
from pathlib import Path

import numpy as np
import tifffile

TARGET_SHAPE = (256, 450, 450)
PLANES = ("xy", "xz", "yz")
OFFSETS = {"xy": 0, "xz": 450, "yz": 706}
VOXEL_SIZE_UM = 0.09


def read_volume(path):
    path = Path(path)
    try:
        volume = tifffile.memmap(path)
    except ValueError:
        volume = tifffile.imread(path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a complete 3D TIFF: {path}, {volume.shape}")
    return volume


def three_slices(path, config):
    raw = read_volume(path)
    starts = tuple((a - b) // 2 for a, b in zip(raw.shape, TARGET_SHAPE))
    if min(starts) < 0:
        raise ValueError(f"Cannot crop {raw.shape} to {TARGET_SHAPE}")
    volume = raw[tuple(slice(s, s + n) for s, n in zip(starts, TARGET_SHAPE))]
    swap = config.get("swap_xy", False)
    rotate = config.get("rotate_xy", True)
    if swap:
        volume = np.swapaxes(volume, 1, 2)
    z = config.get("z_idx")
    z = volume.shape[0] // 2 if z is None else z - starts[0]
    primary_axis = (2 if rotate else 1) if swap else (1 if rotate else 2)
    complementary_axis = (1 if rotate else 2) if swap else (2 if rotate else 1)
    y = config.get("y_idx")
    y = volume.shape[2 if rotate else 1] // 2 if y is None else y - starts[primary_axis]
    x = config.get("x_idx", config.get("y_idx"))
    x = volume.shape[2 if rotate else 1] // 2 if x is None else x - starts[complementary_axis]
    display = np.rot90(volume, k=1, axes=(1, 2)) if rotate else volume
    # The second axial plane follows the XY display orientation; the third is complementary.
    axial = display[:, :, y]
    complementary = volume[:, :, x] if rotate else volume[:, x, :]
    images = {
        "xy": np.asarray(display[z], dtype=np.float32),
        "xz": np.asarray(axial, dtype=np.float32).T,
        "yz": np.asarray(complementary, dtype=np.float32).T,
    }
    for plane in PLANES:
        if config.get("offset_tide", 0):
            images[plane] = images[plane] + config["offset_tide"]
        if config.get("flip_ud", False):
            images[plane] = np.flipud(images[plane])
        expected = (450, 450 if plane == "xy" else 256)
        if images[plane].shape != expected:
            raise ValueError(f"Unexpected {plane} shape: {images[plane].shape}")
    return images


def analysis_volume(path):
    volume = read_volume(path)
    if volume.shape[0] != 256:
        raise ValueError(f"Expected 256 Z planes: {path}")
    return np.rot90(np.asarray(volume[64:-64], dtype=np.float32), k=1, axes=(1, 2))
