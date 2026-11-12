"""Sample fixed native XY coordinates from a cached Kalman volume series."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def sample_kymograph(volumes, *, volume_frames, native_frames, z, coordinates_yx):
    """Return T x distance RI, averaging the supplied transverse sample points."""
    volume_frames = np.asarray(volume_frames, dtype=int)
    frames = np.asarray(native_frames, dtype=int)
    coordinates = np.asarray(coordinates_yx, dtype=np.float64)
    if (volumes.ndim != 4 or len(volumes) != len(volume_frames)
            or len(np.unique(volume_frames)) != len(volume_frames)
            or frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) != 1)
            or not np.isin(frames, volume_frames).all() or not 0 <= z < volumes.shape[1]
            or coordinates.ndim != 3 or coordinates.shape[0] != 2
            or not np.isfinite(coordinates).all()):
        raise ValueError("Invalid TZYX cache, native frames or 2 x distance x width coordinates")
    for axis, size in enumerate(volumes.shape[-2:]):
        if np.any(coordinates[axis] < 0) or np.any(coordinates[axis] > size-1):
            raise ValueError("Sampling coordinates fall outside the native XY section")
    lookup = {int(frame): i for i, frame in enumerate(volume_frames)}
    rows = []
    for frame in frames:
        image = np.array(volumes[lookup[int(frame)], z])
        rows.append(map_coordinates(image, coordinates, order=1, mode="nearest").mean(-1))
    values = np.stack(rows)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite sampled RI")
    return values
