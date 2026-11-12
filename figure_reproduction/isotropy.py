"""Measure axial and lateral FWHM at fixed lipid-droplet coordinates."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
import pandas as pd


def reference_ratios(reference):
    """Compute three-method ratios from the shared FWHM table."""
    tables = []
    for method, label in (("TIDE", "DEPICT"), ("BPM+TV", "BPM+TV"), ("Rytov", "Rytov")):
        data = reference.loc[reference["valid_"+label]].copy()
        widths = [data[f"fwhm_{axis}_um_{label}"] for axis in ("x", "y", "z")]
        ratios = widths[2] / ((widths[0]+widths[1])/2)
        np.testing.assert_allclose(ratios, data["ratio_"+label], rtol=1e-12, atol=1e-12)
        result = data[["cell", "cell_display", "group", "lipid_id", "z", "y", "x"]].copy()
        result["method"] = method
        for axis, width in zip(("x", "y", "z"), widths):
            result["fwhm_"+axis+"_um"] = width
        result["axial_lateral_ratio"] = ratios
        tables.append(result)
    return pd.concat(tables, ignore_index=True)

@dataclass(frozen=True)
class RunConfig:
    z_crop_start: int
    z_crop_stop: int
    voxel_size_um: float
    threshold: float
    connectivity: int
    min_voxels: int
    max_voxels: int
    split_erosion_iters: int
    split_expand_distance: float
    centroid_mode: str
    profile_half_width_px: float
    profile_step_px: float
    profile_smooth_sigma_px: float
    peak_search_radius_px: float
    baseline_fraction: float
    min_profile_contrast: float
    max_fwhm_px: float
    max_component_fwhm_um: float
    max_component_max_min_ratio: float


def interpolate_crossing(x0: float, y0: float, x1: float, y1: float, threshold: float) -> float:
    if abs(y1 - y0) < 1e-12:
        return float((x0 + x1) * 0.5)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def fwhm_from_profile(t: np.ndarray, profile: np.ndarray, config: RunConfig, *, apply_qc: bool = True) -> dict:
    smooth = ndi.gaussian_filter1d(profile.astype(np.float32), sigma=config.profile_smooth_sigma_px)
    n_edge = max(2, int(round(config.baseline_fraction * smooth.size)))
    baseline = float(np.median(np.r_[smooth[:n_edge], smooth[-n_edge:]]))
    peak_mask = np.abs(t) <= float(config.peak_search_radius_px)
    if not np.any(peak_mask):
        peak_mask = np.ones_like(t, dtype=bool)
    peak_candidates = np.where(peak_mask)[0]
    peak_idx = int(peak_candidates[np.argmax(smooth[peak_candidates])])
    peak = float(smooth[peak_idx])
    contrast = peak - baseline
    if apply_qc and contrast < config.min_profile_contrast:
        return {"valid": False, "invalid_reason": "low_contrast", "fwhm_px": np.nan, "fwhm_um": np.nan, "baseline": baseline, "peak": peak, "contrast": contrast}
    if contrast <= 0:
        return {"valid": False, "invalid_reason": "nonpositive_peak_contrast", "fwhm_px": np.nan, "fwhm_um": np.nan, "baseline": baseline, "peak": peak, "contrast": contrast}
    threshold = baseline + 0.5 * contrast

    left_cross = np.nan
    for i in range(peak_idx - 1, -1, -1):
        if smooth[i] <= threshold <= smooth[i + 1] or smooth[i] >= threshold >= smooth[i + 1]:
            left_cross = interpolate_crossing(float(t[i]), float(smooth[i]), float(t[i + 1]), float(smooth[i + 1]), threshold)
            break
    right_cross = np.nan
    for i in range(peak_idx, smooth.size - 1):
        if smooth[i] >= threshold >= smooth[i + 1] or smooth[i] <= threshold <= smooth[i + 1]:
            right_cross = interpolate_crossing(float(t[i]), float(smooth[i]), float(t[i + 1]), float(smooth[i + 1]), threshold)
            break
    if not np.isfinite(left_cross) or not np.isfinite(right_cross):
        return {"valid": False, "invalid_reason": "missing_halfmax_crossing", "fwhm_px": np.nan, "fwhm_um": np.nan, "baseline": baseline, "peak": peak, "contrast": contrast}
    fwhm_px = right_cross - left_cross
    if fwhm_px <= 0 or (apply_qc and fwhm_px > config.max_fwhm_px):
        return {"valid": False, "invalid_reason": "fwhm_out_of_range", "fwhm_px": fwhm_px, "fwhm_um": fwhm_px * config.voxel_size_um, "baseline": baseline, "peak": peak, "contrast": contrast}
    return {"valid": True, "invalid_reason": "", "fwhm_px": fwhm_px, "fwhm_um": fwhm_px * config.voxel_size_um, "baseline": baseline, "peak": peak, "contrast": contrast}


def sample_axis_profile(volume: np.ndarray, z: float, y: float, x: float, axis: str, config: RunConfig, *, apply_qc: bool = True) -> dict:
    t = np.arange(-config.profile_half_width_px, config.profile_half_width_px + 0.5 * config.profile_step_px, config.profile_step_px, dtype=np.float32)
    if axis == "x":
        coords = np.vstack([np.full_like(t, z), np.full_like(t, y), x + t])
    elif axis == "y":
        coords = np.vstack([np.full_like(t, z), y + t, np.full_like(t, x)])
    elif axis == "z":
        coords = np.vstack([z + t, np.full_like(t, y), np.full_like(t, x)])
    else:
        raise ValueError(axis)
    if np.any(coords[0] < 0) or np.any(coords[0] > volume.shape[0] - 1) or np.any(coords[1] < 0) or np.any(coords[1] > volume.shape[1] - 1) or np.any(coords[2] < 0) or np.any(coords[2] > volume.shape[2] - 1):
        return {"valid": False, "invalid_reason": "profile_outside_volume", "fwhm_px": np.nan, "fwhm_um": np.nan, "baseline": np.nan, "peak": np.nan, "contrast": np.nan}
    profile = ndi.map_coordinates(volume, coords, order=1, mode="nearest")
    return fwhm_from_profile(t, profile, config, apply_qc=apply_qc)
