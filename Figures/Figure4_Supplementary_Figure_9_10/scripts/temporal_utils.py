"""Localization linking, identity-association helpers and motion statistics."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
from laptrack_compat import LapTrack
import compute_lipid_track_diffusion_alpha as diffusion
FPS=16.25
VOXEL=0.09
XYZ=["z","y","x"]
XYZ_UM=["z_um","y_um","x_um"]

def link(detections, angle, destination):
    fps = FPS * 12 / angle
    cutoff = 5.0 / fps / VOXEL
    tracker = LapTrack(metric="sqeuclidean", cutoff=cutoff ** 2,
        gap_closing_metric="sqeuclidean", gap_closing_cutoff=(2 * cutoff) ** 2,
        gap_closing_max_frame_count=2, splitting_cutoff=False, merging_cutoff=False)
    tracked, _, _ = tracker.predict_dataframe(detections, coordinate_cols=XYZ,
                                             frame_col="frame", only_coordinate_cols=False)
    tracked = tracked.sort_values(["track_id", "frame"]).reset_index(drop=True)
    tracked.to_csv(destination / "laptrack_tracks.csv", index=False)
    return tracked

def reference_predictions(reference, frames, angle, frame_selection="within_common"):
    q = angle // 12
    rows = []
    for track_id, group in reference.groupby("track_id"):
        group = group.set_index("frame")
        for row in frames.loc[frames[frame_selection]].itertuples():
            first = row.global_frame * q - 20
            position = group.loc[np.arange(first, first + q), XYZ_UM].mean().to_numpy(float)
            rows.append(dict(reference_track_id=track_id, global_frame=row.global_frame,
                **dict(zip(XYZ_UM, position))))
    return pd.DataFrame(rows)

def assign_with_rejection(costs, valid):
    """Maximize valid pair count before minimizing the total geometric cost."""
    n = len(costs)
    rejection = (n + 1) * (float(np.max(costs[valid])) + 1.0) if valid.any() else 1.0
    r, c = linear_sum_assignment(np.c_[np.where(valid, costs, np.inf), np.full((n, n), rejection)])
    return [(int(i), int(j)) for i, j in zip(r, c) if j < costs.shape[1] and valid[i, j]]

def statistics(tracks, label="reconstructed"):
    speed_rows, alpha_rows, msd_rows = [], [], []
    for angle, data in tracks.groupby("angle_group"):
        q, fps = int(angle) // 12, FPS * 12 / angle
        args = SimpleNamespace(fps=fps, pixel_size_um=VOXEL, z_pixel_size_um=VOXEL,
            min_track_length=100 if angle == 12 else 5, max_lag=10 if angle == 12 else 4,
            min_pairs_per_lag=5 if angle == 12 else 3, min_fit_lags=4 if angle == 12 else 3)
        fwhm_columns = ["fwhm_x_um", "fwhm_y_um", "fwhm_z_um"]
        fit_input = data
        if set(fwhm_columns).issubset(data.columns) and data[fwhm_columns].isna().all().all():
            fit_input = data.drop(columns=fwhm_columns)
        fits, msd = diffusion.compute_per_track(fit_input, args)
        for table in (fits, msd):
            table["angle_group"] = angle
            table["downsample_factor"] = q
            table["volume_rate_vps"] = fps
            table["analysis_source"] = label
        alpha_rows.append(fits)
        msd_rows.append(msd)
        for track_id, g in data.groupby("track_id"):
            g = g.sort_values("time_s")
            dt = np.diff(g.time_s)
            speeds = np.linalg.norm(np.diff(g[XYZ_UM].to_numpy(float), axis=0), axis=1) / dt * 60
            speed_rows.append(dict(track_id=track_id, angle_group=angle, downsample_factor=q,
                volume_rate_vps=fps, n_steps=len(speeds), mean_speed_um_min=speeds.mean(),
                analysis_source=label))
    return pd.DataFrame(speed_rows), pd.concat(alpha_rows, ignore_index=True), pd.concat(msd_rows, ignore_index=True)
