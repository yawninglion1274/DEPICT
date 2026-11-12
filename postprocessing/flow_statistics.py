"""Spatial statistics of full 3D speed on a supplied section mask."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .masks import validate_masks
from .optical_flow import speed_um_s


def section_speed_signal(flows, *, fps, voxel_um, smoothing_sigma_t_y_x):
    """Calibrated float32 3D magnitudes on complete sections, smoothed once."""
    sigma = np.asarray(smoothing_sigma_t_y_x, dtype=float)
    if sigma.shape != (3,) or not np.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError("Supply nonnegative finite Gaussian sigmas for (time,y,x)")
    if not np.isfinite([fps, voxel_um]).all() or fps <= 0 or voxel_um <= 0:
        raise ValueError("Frame rate and voxel calibration must be positive")
    values = []
    for field in flows:
        field = np.asarray(field, dtype=np.float32)
        if field.ndim != 3 or field.shape[0] != 3 or not np.isfinite(field).all():
            raise ValueError("Supply finite (dz,dy,dx) sections")
        values.append(np.linalg.norm(field, axis=0)*np.float32(fps*voxel_um))
    if not values:
        raise ValueError("Supply at least one displacement section")
    return gaussian_filter(np.stack(values), sigma=sigma, mode="nearest", truncate=4)


def summarize_speed(flows, mask, *, native_left_frames, fps, voxel_um, time_origin):
    """Reduce |(dz,dy,dx)| within the mask, without smoothing or drift removal."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Speed statistics require a single 2D mask")
    mask = validate_masks(mask, mask.shape)[0]
    frames = np.asarray(native_left_frames, dtype=int)
    if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) != 1):
        raise ValueError("Supply consecutive native left-frame indices")
    rows = []
    for field in flows:
        field = np.asarray(field)
        if field.shape != (3, *mask.shape):
            raise ValueError("Expected a (dz,dy,dx) displacement section matching the mask")
        values = speed_um_s(field[:, mask].astype(np.float64), fps=fps, voxel_um=voxel_um)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite in-mask speed; samples may not be dropped")
        rows.append(np.percentile(values, [25, 50, 75]))
    if len(rows) != len(frames):
        raise ValueError("Native frame and displacement counts differ")
    quantiles = np.array(rows)
    return dict(native_pairs=np.column_stack([frames, frames+1]),
                time_s=(frames-time_origin+.5)/fps, p25_um_s=quantiles[:, 0],
                median_um_s=quantiles[:, 1], p75_um_s=quantiles[:, 2],
                mask_pixels=np.array(mask.sum()), fps=np.array(fps), voxel_um=np.array(voxel_um))


def speed_histograms(speeds, masks, *, bin_edges_um_s, threshold_um_s):
    """Count all fixed-mask pixels per time/bin; percentages sum to 100 per row.

    Speeds are already calibrated (T,Y,X) magnitudes. This function applies no
    smoothing or threshold-based selection. The strict threshold fraction is
    counted from individual speeds, independently of histogram bin boundaries.
    """
    speeds = np.asarray(speeds)
    if speeds.ndim != 3 or not len(speeds):
        raise ValueError("Supply a nonempty (T,Y,X) speed array")
    masks = validate_masks(masks, speeds.shape[1:])
    edges = np.asarray(bin_edges_um_s, dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("Speed bin edges must be finite, nonnegative and strictly increasing")
    if not np.isfinite(threshold_um_s) or threshold_um_s < 0:
        raise ValueError("The speed threshold must be finite and nonnegative")
    counts = np.empty((len(masks), len(speeds), len(edges)-1), dtype=np.int64)
    above = np.empty((len(masks), len(speeds)), dtype=np.int64)
    pixels = masks.sum(axis=(1, 2))
    for region, mask in enumerate(masks):
        for time, field in enumerate(speeds):
            # Use float64 comparisons so thresholds and edges keep their precision.
            values = field[mask].astype(np.float64)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("In-mask speeds must be finite and nonnegative")
            if values.min() < edges[0] or values.max() > edges[-1]:
                raise ValueError("Speed bins must cover every in-mask pixel; no samples may be dropped")
            counts[region, time] = np.histogram(values, bins=edges)[0]
            above[region, time] = np.count_nonzero(values > threshold_um_s)
    return dict(counts=counts, pixel_percentage=counts/pixels[:, None, None]*100,
                above_threshold_counts=above, above_threshold_percentage=above/pixels[:, None]*100,
                mask_pixels=pixels, speed_bin_edges_um_s=edges,
                threshold_um_s=np.array(threshold_um_s))


def regional_speed_onsets(percentages, time_ms, *, minimum_percentage,
                          consecutive_frames=2, after_ms=0):
    """First post-reference run with sufficient above-threshold mask coverage.

    Times are native sample centers. This region-level criterion does not
    interpolate onset times or require the same pixels to remain above speed
    threshold across samples.
    """
    percentages = np.asarray(percentages, dtype=np.float64)
    times = np.asarray(time_ms, dtype=np.float64)
    if (percentages.ndim != 2 or times.ndim != 1 or not len(times)
            or percentages.shape[1] != len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not np.isfinite(percentages).all()
            or np.any((percentages < 0) | (percentages > 100))):
        raise ValueError("Supply finite region/time percentages and increasing native sample times")
    if (not np.isfinite(minimum_percentage) or not 0 < minimum_percentage <= 100
            or not np.isfinite(after_ms) or not np.isfinite(consecutive_frames)
            or consecutive_frames < 1 or int(consecutive_frames) != consecutive_frames):
        raise ValueError("Invalid coverage, reference time or consecutive-frame requirement")
    consecutive_frames = int(consecutive_frames)
    indices = np.full(len(percentages), -1, dtype=int)
    for region, row in enumerate(percentages):
        qualifies = (row >= minimum_percentage) & (times > after_ms)
        for first in range(len(times)-consecutive_frames+1):
            if qualifies[first:first+consecutive_frames].all():
                indices[region] = first
                break
    onsets = np.full(len(percentages), np.nan)
    detected = indices >= 0
    onsets[detected] = times[indices[detected]]
    return dict(regional_onset_index=indices, regional_onset_time_ms=onsets,
                onset_minimum_percentage=np.array(minimum_percentage),
                onset_consecutive_frames=np.array(consecutive_frames),
                onset_after_ms=np.array(after_ms))
