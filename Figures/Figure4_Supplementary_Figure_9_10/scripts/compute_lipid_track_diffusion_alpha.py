"""Three-dimensional TA-MSD and log-OLS functions."""
from __future__ import annotations
import argparse
from typing import Any
import numpy as np
import pandas as pd

def z_pixel_size_um(args: argparse.Namespace) -> float:
    return float(args.pixel_size_um if args.z_pixel_size_um is None else args.z_pixel_size_um)

def fit_log_msd(msd: pd.DataFrame, min_fit_lags: int) -> dict[str, float | int | str]:
    valid = (
        np.isfinite(msd["tau_s"].to_numpy(dtype=np.float64))
        & np.isfinite(msd["msd_3d_um2"].to_numpy(dtype=np.float64))
        & (msd["tau_s"].to_numpy(dtype=np.float64) > 0)
        & (msd["msd_3d_um2"].to_numpy(dtype=np.float64) > 0)
    )
    fit_df = msd.loc[valid].copy()
    if len(fit_df) < int(min_fit_lags):
        return {
            "alpha": np.nan,
            "D_um2_s_alpha": np.nan,
            "log_intercept_ln_6D": np.nan,
            "fit_r2_log": np.nan,
            "fit_lags": int(len(fit_df)),
            "status": "insufficient_fit_lags",
        }

    x = np.log(fit_df["tau_s"].to_numpy(dtype=np.float64))
    y = np.log(fit_df["msd_3d_um2"].to_numpy(dtype=np.float64))
    alpha, intercept = np.polyfit(x, y, 1)
    pred = alpha * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "alpha": float(alpha),
        "D_um2_s_alpha": float(np.exp(intercept) / 6.0),
        "log_intercept_ln_6D": float(intercept),
        "fit_r2_log": float(r2),
        "fit_lags": int(len(fit_df)),
        "status": "fit_ok",
    }

def track_tamsd(group: pd.DataFrame, fps: float, max_lag: int, min_pairs_per_lag: int) -> pd.DataFrame:
    group = group.sort_values("frame")
    base = group[["frame", "x_um", "y_um", "z_um"]].copy()
    rows: list[dict[str, Any]] = []
    for lag in range(1, int(max_lag) + 1):
        shifted = base.copy()
        shifted["frame"] = shifted["frame"] - lag
        merged = base.merge(shifted, on="frame", suffixes=("_0", "_1"))
        n_pairs = int(len(merged))
        if n_pairs < int(min_pairs_per_lag):
            rows.append({"lag_frames": lag, "tau_s": lag / fps, "n_pairs": n_pairs, "msd_3d_um2": np.nan})
            continue
        dx = merged["x_um_1"].to_numpy(dtype=np.float64) - merged["x_um_0"].to_numpy(dtype=np.float64)
        dy = merged["y_um_1"].to_numpy(dtype=np.float64) - merged["y_um_0"].to_numpy(dtype=np.float64)
        dz = merged["z_um_1"].to_numpy(dtype=np.float64) - merged["z_um_0"].to_numpy(dtype=np.float64)
        dr2 = dx * dx + dy * dy + dz * dz
        rows.append(
            {
                "lag_frames": lag,
                "tau_s": lag / fps,
                "n_pairs": n_pairs,
                "msd_3d_um2": float(np.mean(dr2)),
                "msd_x_um2": float(np.mean(dx * dx)),
                "msd_y_um2": float(np.mean(dy * dy)),
                "msd_z_um2": float(np.mean(dz * dz)),
            }
        )
    return pd.DataFrame(rows)

def track_size_summary(group: pd.DataFrame, args: argparse.Namespace) -> dict[str, float]:
    voxel_volume_um3 = float(args.pixel_size_um) * float(args.pixel_size_um) * z_pixel_size_um(args)
    voxel_count = group["voxel_count"].to_numpy(dtype=np.float64)
    volume_um3 = voxel_count * voxel_volume_um3
    equiv_diameter_um = 2.0 * np.cbrt((3.0 * volume_um3) / (4.0 * np.pi))

    row = {
        "mean_voxel_count": float(np.nanmean(voxel_count)),
        "median_voxel_count": float(np.nanmedian(voxel_count)),
        "mean_volume_um3": float(np.nanmean(volume_um3)),
        "median_volume_um3": float(np.nanmedian(volume_um3)),
        "mean_equiv_diameter_um": float(np.nanmean(equiv_diameter_um)),
        "median_equiv_diameter_um": float(np.nanmedian(equiv_diameter_um)),
        "median_equiv_radius_um": float(np.nanmedian(equiv_diameter_um) / 2.0),
    }
    if {"fwhm_x_um", "fwhm_y_um", "fwhm_z_um"}.issubset(group.columns):
        fwhm_geo = np.cbrt(
            group["fwhm_x_um"].to_numpy(dtype=np.float64)
            * group["fwhm_y_um"].to_numpy(dtype=np.float64)
            * group["fwhm_z_um"].to_numpy(dtype=np.float64)
        )
        row["median_fwhm_geomean_um"] = float(np.nanmedian(fwhm_geo))
    return row

def compute_per_track(tracks: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    msd_rows: list[pd.DataFrame] = []
    for track_id, group in tracks.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        n_det = int(len(group))
        if n_det < int(args.min_track_length):
            continue
        msd = track_tamsd(group, float(args.fps), int(args.max_lag), int(args.min_pairs_per_lag))
        fit = fit_log_msd(msd.dropna(subset=["msd_3d_um2"]), int(args.min_fit_lags))
        size = track_size_summary(group, args)
        first = group.iloc[0]
        last = group.iloc[-1]
        duration_s = float((last["frame"] - first["frame"]) / float(args.fps))
        msd.insert(0, "track_id", int(track_id))
        msd_rows.append(msd)
        summary_rows.append(
            {
                "track_id": int(track_id),
                "n_detections": n_det,
                "first_frame": int(first["frame"]),
                "last_frame": int(last["frame"]),
                "duration_s": duration_s,
                "fit_max_lag_frames": int(args.max_lag),
                "fit_max_tau_s": float(args.max_lag / float(args.fps)),
                "min_pairs_per_lag": int(args.min_pairs_per_lag),
                **size,
                **fit,
            }
        )
    return pd.DataFrame(summary_rows), pd.concat(msd_rows, ignore_index=True) if msd_rows else pd.DataFrame()
