#!/usr/bin/env python3
"""Motion classification and trajectory sheets for 83 complete baseline tracks.

Each sheet displays one motion class, sorted from slowest to fastest, with
3-D boxes, elapsed-time colors, speed and alpha labels, and a shared 1-um bar.

Classification uses lag-10 alpha and mean successive-step speed at 16.25 vps:
alpha < 1.2 defines diffusing tracks and alpha > 1.2 defines directional tracks.
Diffusing tracks with speed rounded to two decimals <= 21.15 um/min become
confined. Manual assignments set track 39 to directional and tracks 23 and 93
to confined. The per-track table records the classification rule."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-panel-d")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "panel_d"
FIG4 = HERE.parent
TRACKS_CSV = FIG4 / "analysis" / "tracks_by_resolution.csv"
LEGACY = PROJECT / "scripts"
ANALYSIS = PROJECT / "scripts"
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(ANALYSIS))

import plot_D_alpha_top5_trajectory_3dbox_2x5 as traj  # noqa: E402
from compute_lipid_track_diffusion_alpha import compute_per_track  # noqa: E402

from common import load_reference_tracks

FPS = 16.25
ALPHA_CUTOFF = 1.2
CONFINED_SPEED_UM_MIN = 21.15  # inclusive, on the displayed two-decimal speed
TO_DIRECTIONAL = (39,)
TO_CONFINED = (23, 93)
BASE_CUBE_UM = 1.425
COLUMNS = 12
SPEED_DECIMALS = 2  # keeps the 21.15 um/min confined cutoff readable
SPEED_UNIT = r"($\mathrm{\mu m\,min^{-1}}$)"
GROUPS_CSV = PROJECT / "analysis" / "motion_classification.csv"
ARIAL_FONT = PROJECT / "inputs/fonts/Arial.TTF"
COMPACT_WIDTH_CM = 17.845
FINAL_GROUPS = (
    ("diffusing", "Final diffusing", "#0072B2"),
    ("directional", "Final directional", "#D55E00"),
    ("confined", "Confined", "#66754D"),
)


def drawing_options(cube_um: float) -> SimpleNamespace:
    """Camera, box and color settings for trajectory sheets."""
    return SimpleNamespace(
        fps=FPS, elev=24.0, azim=-62.0, focal_length=1.0,
        padding_frac=0.10, common_cube_um=cube_um, clip_to_cube=False,
        face_mode="none", floor_grid=True, grid_grey=0.88,
        grid_line_width=0.25, projection_mode="floor+left-wall",
        projection_grey=0.78, projection_scale=0.62, line_width=0.85,
        box_line_width=0.65, box_grey=0.18, time_cmap="turbo",
        time_cmap_trim=0.05, font_size=8.0,
    )


def mean_speed_um_min(points: pd.DataFrame) -> float:
    points = points.sort_values("frame")
    xyz = points[["x_um", "y_um", "z_um"]].to_numpy(dtype=np.float64)
    dt = np.diff(points["frame"].to_numpy(dtype=np.float64)) / FPS
    return float(np.mean(np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt) * 60.0)


def classify_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    fit_args = SimpleNamespace(
        fps=FPS, pixel_size_um=0.09, z_pixel_size_um=0.09,
        min_track_length=100, max_lag=10, min_pairs_per_lag=5, min_fit_lags=4,
    )
    per_track, _ = compute_per_track(tracks, fit_args)
    groups = per_track.loc[
        per_track["status"].eq("fit_ok") & np.isfinite(per_track["alpha"]),
        ["track_id", "alpha", "D_um2_s_alpha", "fit_r2_log"],
    ].copy()
    if groups["alpha"].eq(ALPHA_CUTOFF).any():
        raise ValueError("A track has alpha exactly equal to 1.2")
    for track_id, points in tracks.groupby("track_id"):
        if np.any(np.diff(np.sort(points["frame"].to_numpy(dtype=int))) > 1):
            raise ValueError(f"Track {track_id} has internal missing frames")
    speeds = {int(k): mean_speed_um_min(g) for k, g in tracks.groupby("track_id")}
    groups["speed_1x_um_min"] = groups["track_id"].map(speeds)
    groups["initial_group"] = np.where(groups["alpha"] < ALPHA_CUTOFF, "diffusing", "directional")
    groups["final_group"] = groups["initial_group"]
    groups["rule"] = np.where(groups["alpha"] < ALPHA_CUTOFF, "alpha < 1.2", "alpha > 1.2")
    displayed = groups["speed_1x_um_min"].round(SPEED_DECIMALS)
    slow = groups["initial_group"].eq("diffusing") & displayed.le(CONFINED_SPEED_UM_MIN)
    groups.loc[slow, "final_group"] = "confined"
    groups.loc[slow, "rule"] = f"diffusing, speed <= {CONFINED_SPEED_UM_MIN:g} um/min"
    for ids, initial, final in ((TO_DIRECTIONAL, "diffusing", "directional"),
                                (TO_CONFINED, "directional", "confined")):
        for track_id in ids:
            target = groups["track_id"].eq(track_id)
            if target.sum() != 1 or groups.loc[target, "initial_group"].iloc[0] != initial:
                raise ValueError(f"Track {track_id} is not in the initial {initial} group")
            groups.loc[target, "final_group"] = final
            groups.loc[target, "rule"] = f"manual: {initial} -> {final}"
    return groups.sort_values("track_id").reset_index(drop=True)


def build_panels(groups: pd.DataFrame, tracks: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    points_by_id = dict(tuple(tracks.groupby("track_id", sort=False)))
    panels: dict[str, list[dict[str, object]]] = {name: [] for name, _, _ in FINAL_GROUPS}
    for row in groups.itertuples(index=False):
        points = points_by_id[row.track_id]
        xyz = points[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)
        # Only an unusually long trajectory needs a larger box; never clip points.
        extent_um = float(np.ptp(xyz, axis=0).max())
        cube_um = (math.ceil(1.2 * extent_um * 100) / 100
                   if extent_um > BASE_CUBE_UM else BASE_CUBE_UM)
        opts = drawing_options(cube_um)
        panel = traj.build_panel(pd.Series(row._asdict()), points, opts)
        if not math.isclose(panel["mean_speed"] * 60, row.speed_1x_um_min, abs_tol=1e-9):
            raise ValueError(f"Speed mismatch for track {row.track_id}")
        panel["cube_um"] = cube_um
        panel["options"] = opts
        panels[row.final_group].append(panel)
    for sheet in panels.values():
        sheet.sort(key=lambda p: (p["mean_speed"], p["track_id"]))
    return panels


def add_speed_label(fig: plt.Figure, x: float, y: float, speed: str, *,
                    fontsize: float, with_unit: bool) -> None:
    """Place an Arial speed label with a raised, smaller -1 in the unit."""
    if not with_unit:
        fig.text(x, y, speed, ha="center", va="center", fontsize=fontsize, color="black")
        return
    parts = ((f"{speed} (μm min", fontsize, 0.0),
             ("-1", fontsize * 0.65, fontsize * 0.35),
             (")", fontsize, 0.0))
    renderer = fig.canvas.get_renderer()
    widths = [renderer.get_text_width_height_descent(
        label, FontProperties(family="Arial", size=size), ismath=False)[0]
        for label, size, _ in parts]
    x_px = x * fig.bbox.width - sum(widths) / 2
    for (label, size, rise_pt), width in zip(parts, widths):
        fig.text(x_px / fig.bbox.width,
                 y + rise_pt * fig.dpi / (72 * fig.bbox.height),
                 label, ha="left", va="center", fontsize=size, color="black")
        x_px += width


def draw_sheet(name: str, title: str, color: str, panels: list[dict[str, object]],
               *, duration_s: float, with_text: bool, compact: bool = False) -> None:
    """Render one trajectory sheet with shared box geometry."""
    if not panels:
        raise ValueError(f"No {name} tracks to draw")
    opts = drawing_options(BASE_CUBE_UM)
    panel_w = 1.55
    panel_h = panel_w / traj.canonical_aspect(opts.elev, opts.azim, opts.focal_length, traj.DATA_PAD)
    nrows = math.ceil(len(panels) / COLUMNS)
    left, right, gap_x = 0.20, 0.20, 0.10
    title_h, caption_h, foot_h, gap_y, footer_h = 0.67, 0.24, 0.25, 0.11, 0.70
    fig_w = left + COLUMNS * panel_w + (COLUMNS - 1) * gap_x + right
    scale = (COMPACT_WIDTH_CM / 2.54 / fig_w) if compact else 1.0
    left, right, gap_x, title_h, caption_h, foot_h, gap_y, footer_h, panel_w, panel_h = (
        dimension * scale for dimension in
        (left, right, gap_x, title_h, caption_h, foot_h, gap_y, footer_h, panel_w, panel_h)
    )
    if compact:
        caption_h = foot_h = max(caption_h, foot_h, 0.13)
    fig_w *= scale
    fig_h = title_h + nrows * (caption_h + panel_h + foot_h) + (nrows - 1) * gap_y + footer_h
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    if with_text and not compact:
        fig.text(left / fig_w, 1 - 0.25 / fig_h, f"{title}  |  n = {len(panels)}",
                 ha="left", va="center", fontsize=11, weight="semibold", color=color)
        fig.text(1 - right / fig_w, 1 - 0.25 / fig_h, r"Slow to fast  $\rightarrow$",
                 ha="right", va="center", fontsize=9, color="#333333")
        fig.text(left / fig_w, 1 - 0.51 / fig_h,
                 "Mean step speed above each track; 16.25 volumes/s",
                 ha="left", va="center", fontsize=8, color="#555555")

    cmap = traj.time_colormap(opts)
    time_norm = Normalize(0, duration_s)
    reference = None
    for idx, panel in enumerate(panels):
        row, col = divmod(idx, COLUMNS)
        x0 = left + col * (panel_w + gap_x)
        y_top = fig_h - title_h - row * (caption_h + panel_h + foot_h + gap_y)
        panel_top = y_top - caption_h
        ax = fig.add_axes((x0 / fig_w, (panel_top - panel_h) / fig_h,
                           panel_w / fig_w, panel_h / fig_h))
        panel_opts = panel["options"]
        if compact:
            panel_opts = SimpleNamespace(**vars(panel_opts))
            panel_opts.line_width *= scale
            panel_opts.box_line_width *= scale
            panel_opts.grid_line_width *= scale
        traj.draw_box(ax, panel, args=panel_opts)
        traj.draw_trajectory(ax, panel, time_cmap=cmap, norm=time_norm,
                             line_width=panel_opts.line_width, outline_scale=0)
        if compact:
            for marker in ax.lines:
                marker.set_markersize(marker.get_markersize() * scale)
                marker.set_markeredgewidth(marker.get_markeredgewidth() * scale)
        xlim, ylim = traj.panel_data_limits(panel, traj.DATA_PAD)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        if with_text or compact:
            speed = f"{panel['mean_speed'] * 60:.{SPEED_DECIMALS}f}"
            x_mid = (x0 + panel_w / 2) / fig_w
            speed_y = (panel_top + caption_h / 2) / fig_h
            if compact:
                add_speed_label(fig, x_mid, speed_y, speed, fontsize=6,
                                with_unit=idx == len(panels) - 1)
            else:
                if col == COLUMNS - 1 or idx == len(panels) - 1:
                    speed += " " + SPEED_UNIT
                fig.text(x_mid, speed_y, speed, ha="center", va="center",
                         fontsize=9.2, color="#111111")
            label = (f"α = {panel['alpha']:.3f}" if compact
                     else rf"$\alpha$ = {panel['alpha']:.3f}")
            if not compact and panel["cube_um"] > BASE_CUBE_UM:
                label += rf"  |  {panel['cube_um']:g} $\mu$m box"
            fig.text((x0 + panel_w / 2) / fig_w, (panel_top - panel_h - foot_h / 2) / fig_h,
                     label, ha="center", va="center", fontsize=6 if compact else 7.3,
                     color="black" if compact else "#555555")
        if reference is None and panel["cube_um"] == BASE_CUBE_UM:
            reference = (ax, panel)

    if reference is None:
        raise ValueError("No standard-size cube for the shared scale bar")
    fig.canvas.draw()
    ax, panel = reference
    x0_data, _ = ax.get_xlim()
    y0_data, y1_data = ax.get_ylim()
    y_mid = (y0_data + y1_data) / 2
    anchor = ax.transData.transform((x0_data, y_mid))
    endpoint = ax.transData.transform((x0_data + traj.scale_bar_length_ndc(panel, 1.0), y_mid))
    bar_frac = abs(endpoint[0] - anchor[0]) / (fig_w * fig.dpi)
    bar_x, bar_y = (left + 0.10 * scale) / fig_w, 0.37 * scale / fig_h
    fig.add_artist(Line2D([bar_x, bar_x + bar_frac], [bar_y, bar_y],
                          transform=fig.transFigure, linewidth=3.2 * scale,
                          color="black", solid_capstyle="butt"))
    if with_text and not compact:
        fig.text(bar_x + bar_frac + 0.03 / fig_w, bar_y, r"1 $\mu$m",
                 ha="left", va="center", fontsize=8, color="black")
        fig.text(bar_x, 0.14 / fig_h, r"1.425 $\mu$m box unless labeled otherwise",
                 ha="left", va="center", fontsize=7.5, color="#555555")

    # Time colour bar: no ticks; the text version labels only the two ends.
    bar_left, bar_width, bar_h = 0.56, 0.23, 0.11 * scale / fig_h
    bar_bottom = 0.30 * scale / fig_h
    color_ax = fig.add_axes((bar_left, bar_bottom, bar_width, bar_h))
    colorbar = fig.colorbar(ScalarMappable(norm=time_norm, cmap=cmap), cax=color_ax,
                            orientation="horizontal")
    colorbar.set_ticks([])
    colorbar.ax.tick_params(length=0)
    colorbar.outline.set_linewidth(0.7 * scale)
    if with_text and not compact:
        gap = 0.05 / fig_w
        fig.text(bar_left - gap, bar_bottom + bar_h / 2, "0 s",
                 ha="right", va="center", fontsize=7.5, color="black")
        fig.text(bar_left + bar_width + gap, bar_bottom + bar_h / 2, f"{duration_s:.2f} s",
                 ha="left", va="center", fontsize=7.5, color="black")

    variant = "with_text" if with_text else "no_text"
    target = (HERE / f"final_{name}_trajectories_speed_sorted_17p845cm_arial6_speed_alpha.pdf"
              if compact else HERE / f"final_{name}_trajectories_speed_sorted_{variant}.pdf")
    if compact or with_text:
        fig.savefig(target, facecolor="white")
    else:
        # Without the title and footer text the page is cropped to the drawn content.
        fig.savefig(target, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {target} ({len(panels)} tracks)")
