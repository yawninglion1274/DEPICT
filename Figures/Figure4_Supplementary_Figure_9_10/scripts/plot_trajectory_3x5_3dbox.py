#!/usr/bin/env python3
"""Representative baseline trajectories in three rows of five.

The first row contains the five largest D values among |alpha - 1| <= 0.1;
the second uses the five largest D values among alpha > 1.4. The third shows
the five slowest confined trajectories. Alpha and D use a lag-10 MSD fit.
All panels share a 1.425-um cube, with the same camera, elapsed-time colors
and a 1-um scale bar. Labels report computed mean speeds in um/min."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "panel_d"
FIG4 = HERE.parent
TRACKS_CSV = FIG4 / "analysis" / "tracks_by_resolution.csv"
LEGACY = PROJECT / "scripts"
ANALYSIS = PROJECT / "scripts"
sys.path.insert(0, str(ANALYSIS))

from compute_lipid_track_diffusion_alpha import compute_per_track  # noqa: E402

from common import load_reference_tracks

FPS = 16.25
N_PER_ROW = 5
FINAL_GROUPS_CSV = PROJECT / "analysis" / "motion_classification.csv"
SELECTION_CSV =HERE / "trajectory_3x5_3dbox_selected_tracks.csv"
SOURCE_STEM = "trajectory_2x5_3dbox_speed_only_scalebar_nooutline"
TARGET_STEM = "trajectory_3x5_3dbox_speed_only_scalebar_nooutline"


def mean_speed_um_min(group: pd.DataFrame) -> float:
    """Mean of successive 3-D step speeds, as in the plotting script."""
    group = group.sort_values("frame")
    xyz = group[["x_um", "y_um", "z_um"]].to_numpy(dtype=np.float64)
    dt = np.diff(group["frame"].to_numpy(dtype=np.float64)) / FPS
    return float(np.mean(np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt) * 60.0)


def confined_track_ids() -> list[int]:
    """First five panels of the confined sheet: speed, then track ID."""
    groups = pd.read_csv(FINAL_GROUPS_CSV)
    confined = groups[groups["final_group"].eq("confined")]
    confined = confined.sort_values(["speed_1x_um_min", "track_id"])
    return confined["track_id"].astype(int).head(N_PER_ROW).tolist()


def select_groups(tracks: pd.DataFrame) -> pd.DataFrame:
    fit_args = SimpleNamespace(
        fps=FPS, pixel_size_um=0.09, z_pixel_size_um=0.09,
        min_track_length=100, max_lag=10, min_pairs_per_lag=5, min_fit_lags=4,
    )
    per_track, _ = compute_per_track(tracks, fit_args)
    per_track = per_track[per_track["status"].eq("fit_ok")].copy()
    speeds = {int(k): mean_speed_um_min(g) for k, g in tracks.groupby("track_id")}
    per_track["mean_speed_um_min"] = per_track["track_id"].map(speeds)

    near1 = per_track[(per_track["alpha"] - 1.0).abs() <= 0.1]
    rows = (
        ("alpha_near1_w0p1", "alpha near 1 (|alpha-1| <= 0.1), top 5 by D",
         near1.sort_values("D_um2_s_alpha", ascending=False)),
        ("alpha_gt1p4", "alpha > 1.4, top 5 by D",
         per_track[per_track["alpha"] > 1.4].sort_values("D_um2_s_alpha", ascending=False)),
        ("confined_first5", "first 5 confined tracks by speed",
         per_track.set_index("track_id").loc[confined_track_ids()].reset_index()),
    )
    tables = []
    for key, label, ranked in rows:
        top = ranked.head(N_PER_ROW).copy()
        top.insert(0, "group_rank", np.arange(1, len(top) + 1))
        top.insert(0, "group_label", label)
        top.insert(0, "group_key", key)
        tables.append(top)
    columns = ["group_key", "group_label", "group_rank", "track_id", "alpha",
               "D_um2_s_alpha", "fit_r2_log", "mean_speed_um_min", "n_detections",
               "first_frame", "last_frame", "duration_s"]
    return pd.concat(tables, ignore_index=True)[columns]


def main() -> None:
    tracks = load_reference_tracks()
    tracks["track_id"] = tracks["track_id"].astype(int)
    groups = select_groups(tracks)
    groups.to_csv(SELECTION_CSV, index=False)
    print(groups[["group_key", "group_rank", "track_id", "alpha",
                  "D_um2_s_alpha", "mean_speed_um_min"]].to_string(index=False))

    with tempfile.TemporaryDirectory(prefix="fig4d_3x5_") as temporary:
        build_dir = Path(temporary)
        (build_dir / "matplotlib").mkdir()
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(build_dir / "matplotlib")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        tracks.to_csv(build_dir / "reference_tracks.csv", index=False)
        command = [
            sys.executable, str(LEGACY / "plot_D_alpha_top5_trajectory_3dbox_2x5.py"),
            "--tracks-csv", str(build_dir / "reference_tracks.csv"),
            "--groups-csv", str(SELECTION_CSV),
            "--groups", "alpha_near1_w0p1", "alpha_gt1p4", "confined_first5",
            "--output-dir", str(build_dir),
            "--label-mode", "speed",
            "--speed-unit", "um_min",
            "--font-scale", "1.5",
            "--common-cube-from", "2,2",
            "--clip-to-cube",
            "--no-trajectory-outline",
            "--scale-bar-scope", "figure",
            "--scale-bar-um", "1",
            "--output-tag", "_scalebar",
            "--formats", "pdf",
            "--dpi", "600",
        ]
        subprocess.run(command, check=True, env=env)
        # The plotting script also writes a colour bar; publish only the two figures.
        for variant in ("with_text",):
            target = HERE / "panel_d_with_text.pdf"
            shutil.copyfile(build_dir / f"{SOURCE_STEM}_{variant}.pdf", target)
            print(f"Saved: {target}")
    print(f"Saved: {SELECTION_CSV}")


if __name__ == "__main__":
    main()
