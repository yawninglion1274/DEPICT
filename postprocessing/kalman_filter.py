"""Native-plane temporal Kalman RTS; the established Figure 5 equations."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .volume_io import digest, read_array, signature, write_json


def kalman_rts_smooth(
    stack: np.ndarray, map_smooth: float = 3.0, qr_scale: float = 2.0
) -> np.ndarray:
    """Per-pixel RTS (Rauch-Tung-Striebel) smoother for a (T, Y, X) stack.

    Models each pixel's time series as a constant-position random walk
    observed in noise: x_t = x_{t-1} + w_t (Var w = Q), z_t = x_t + v_t
    (Var v = R). Q and R are estimated per pixel from the data via the
    method of moments on the first differences d_t = z_t - z_{t-1}:

        R = -Cov(d_t, d_{t-1})   (anti-correlated lag-1 == white flicker)
        Q =  Var(d_t) - 2R       (the smoothly-varying real signal)

    Because R varies per pixel, the Kalman gain is per-pixel: noisy regions
    (low Q/R) are smoothed hard, while high-SNR moving structure (high Q/R)
    follows the observations and keeps its real motion -- unlike a fixed
    Gaussian kernel, which blurs motion everywhere. The forward filter plus
    backward RTS pass is non-causal (zero phase lag), appropriate for offline
    data. Vectorized over all pixels.

    map_smooth: spatial Gaussian sigma (px) applied to the Q/R maps to
        stabilize the short-sequence moment estimates (0 disables).
    qr_scale: multiplier on Q/R; <1 smooths more, >1 preserves more motion.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3 or len(stack) < 2 or not np.isfinite(stack).all():
        raise ValueError("Expected a finite (T,Y,X) stack with at least two frames")
    if map_smooth < 0 or not np.isfinite(map_smooth) or qr_scale <= 0 or not np.isfinite(qr_scale):
        raise ValueError("Invalid Kalman map_smooth or qr_scale")
    n_frames = stack.shape[0]
    stack = stack.astype(np.float64, copy=False)

    diff = np.diff(stack, axis=0)
    var_diff = diff.var(axis=0)
    if n_frames > 2:
        lag1 = (diff[1:] * diff[:-1]).mean(0) - diff[1:].mean(0) * diff[:-1].mean(0)
    else:
        lag1 = np.zeros_like(var_diff)
    noise = np.clip(-lag1, 0.0, None)            # R: white flicker power
    process = np.clip(var_diff - 2.0 * noise, 0.0, None)  # Q: real change power

    if map_smooth and map_smooth > 0:
        noise = gaussian_filter(noise, map_smooth)
        process = gaussian_filter(process, map_smooth)

    # Floor R away from zero so the gain stays finite where the estimate is noisy.
    eps = max(float(var_diff.mean()) * 1e-6, 1e-12)
    R = noise + eps
    Q = np.clip(process * float(qr_scale), 0.0, None) + eps



    # Forward pass (predict/update); P0 = var_diff is a diffuse-ish prior.
    x_filt = np.empty_like(stack)
    P_filt = np.empty_like(stack)
    P_pred = np.empty_like(stack)
    x = stack[0].copy()
    P = var_diff + eps
    for t in range(n_frames):
        if t > 0:
            x = x_filt[t - 1]
            P = P_filt[t - 1] + Q
        P_pred[t] = P
        gain = P / (P + R)
        x_filt[t] = x + gain * (stack[t] - x)
        P_filt[t] = (1.0 - gain) * P

    # Backward RTS smoothing pass.
    x_smooth = x_filt.copy()
    for t in range(n_frames - 2, -1, -1):
        c = P_filt[t] / P_pred[t + 1]
        x_smooth[t] = x_filt[t] + c * (x_smooth[t + 1] - x_filt[t])

    return x_smooth.astype(np.float32, copy=False)


def _filter_plane(task):
    series, path, stored_frames, z, parameters, plane_dir, keep_plane = task
    filtered = kalman_rts_smooth(series.plane(z), **parameters)
    indices = np.searchsorted(series.frames, stored_frames)
    volume = np.load(path, mmap_mode="r+")
    volume[:, z] = filtered[indices]
    volume.flush()
    if keep_plane:
        np.save(Path(plane_dir) / f"xy_z{z}_context.npy", filtered)
    done = np.load(Path(plane_dir) / "kalman_done.npy", mmap_mode="r+")
    done[z] = True
    done.flush()
    return z


class FilteredSeries:
    def __init__(self, path, frames, context_planes):
        self.path = Path(path)
        self.frames = np.asarray(frames, dtype=int)
        self.context_planes = context_planes

    @property
    def array(self):
        return np.load(self.path, mmap_mode="r")

    def volume(self, frame):
        indices = np.flatnonzero(self.frames == frame)
        if len(indices) != 1:
            raise ValueError(f"Frame {frame} is absent from filtered storage")
        return self.array[int(indices[0])]


def prepare(series, output, *, stored_frames, xy_planes, parameters=None,
            workers=1, external_cache=None):
    """Filter every native XY plane using the complete temporal context.

    An explicit read-only cache may store fewer frames than the context. Each
    requested XY plane is independently filtered from raw inputs and compared
    bit for bit with its stored cache plane before reuse is recorded.
    """
    parameters = dict(map_smooth=3., qr_scale=2.) if parameters is None else parameters
    frames = np.asarray(stored_frames, dtype=int)
    if not len(frames) or not np.isin(frames, series.frames).all() or len(set(frames)) != len(frames):
        raise ValueError("Stored frames must be distinct input native frame indices")
    output = Path(output)
    contract = dict(inputs=series.signatures, context_frames=series.frames.tolist(),
                    parameters=parameters, stored_frames=frames.tolist(),
                    implementation=signature(__file__))
    cache_dir = output / digest(contract)[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    context_planes = {}
    if external_cache:
        if external_cache["parameters"] != parameters:
            raise ValueError("External Kalman cache parameters differ; use --no-external-cache")
        path = Path(external_cache["path"])
        volume = np.load(path, mmap_mode="r")
        first = int(external_cache["first_frame"])
        cache_frames = np.arange(first, first+len(volume))
        if (volume.shape[1:] != series.shape or volume.dtype != np.float32
                or not np.isin(frames, cache_frames).all() or not np.isin(cache_frames, series.frames).all()):
            raise ValueError("External Kalman cache shape, dtype or native frame coverage differs")
        if external_cache.get("done") and not np.load(external_cache["done"]).all():
            raise ValueError("External Kalman volume is incomplete")
        contract["external"] = signature(path)
    else:
        path = cache_dir / "volume_kalman_float32.npy"
        cache_frames = frames
        shape = (len(frames), *series.shape)
        if not path.exists():
            volume = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
            volume.flush()
            del volume
            np.save(cache_dir / "kalman_done.npy", np.zeros(series.shape[0], dtype=bool))
        if np.load(path, mmap_mode="r").shape != shape:
            raise ValueError("Local Kalman cache shape differs")
        done = np.load(cache_dir / "kalman_done.npy")
        tasks = [(series, path, frames, int(z), parameters, cache_dir, int(z) in xy_planes)
                 for z in np.flatnonzero(~done)]
        if workers < 1:
            raise ValueError("workers must be positive")
        if workers == 1:
            for task in tasks:
                _filter_plane(task)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for _ in pool.map(_filter_plane, tasks):
                    pass
        if not np.load(cache_dir / "kalman_done.npy").all():
            raise RuntimeError("Incomplete Kalman output")
    validation_path = cache_dir / "cache_validation.json"
    previous = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    verified = previous.get("verified_z", []) if previous.get("contract") == contract else []
    cache_indices = np.searchsorted(series.frames, cache_frames)
    for z in sorted(set(xy_planes)):
        plane_path = cache_dir / f"xy_z{z}_context.npy"
        if z not in verified or not plane_path.exists():
            filtered = kalman_rts_smooth(series.plane(z), **parameters)
            np.testing.assert_array_equal(filtered[cache_indices], np.load(path, mmap_mode="r")[:, z],
                                          err_msg="Raw-input Kalman plane differs from the explicit cache")
            np.save(plane_path, filtered)
            verified.append(z)
        context_planes[z] = plane_path
    write_json(validation_path, dict(contract=contract, verified_z=sorted(set(verified)),
                                    raw_context_plane_bit_identical=True))
    return FilteredSeries(path, cache_frames, context_planes)
