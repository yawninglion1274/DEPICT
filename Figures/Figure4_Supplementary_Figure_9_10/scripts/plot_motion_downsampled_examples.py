#!/usr/bin/env python3
"""Camera, geometry and annotations for paired measured trajectories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig4-supp-motion")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "Supplementary_Figure_10"
FIG4 = HERE.parent
PANEL_D = FIG4 / "panel_d"
LEGACY = PROJECT / "scripts"
sys.path.insert(0, str(PANEL_D))
sys.path.insert(0, str(LEGACY))

import plot_D_alpha_top5_trajectory_3dbox_2x5 as traj  # noqa: E402
from plot_final_motion_trajectories import drawing_options  # noqa: E402

FPS = 16.25
FACTORS = (1, 20)
CUBE_UM = 1.425
GROUPS = (
    ("diffusing", "Diffusing", "alpha_near1_w0p1", (56, 143), "#0072B2"),
    ("directional", "Directional", "alpha_gt1p4", (35, 54), "#D55E00"),
    ("confined", "Confined", "confined_first5", (14, 8), "#66754D"),
)
OUTPUT_STEM = "final_motion_downsampled_examples_2x6"
ARIAL_FONT = PROJECT / "inputs/fonts/Arial.TTF"
SPEED_ALPHA_WIDTH_CM = 12.0


def sampled_panel(full: dict, points: pd.DataFrame, alpha: float) -> dict:
    """Keep the full-track camera and centre while replacing its time samples."""
    xyz = points[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)
    frames = points["frame"].to_numpy(dtype=float)
    centred = xyz - full["centre"]
    times = (frames - full["first_frame"]) / FPS
    runs = traj.clip_polylines_to_cube(centred, times, full["half_size"])
    projected_runs = [traj.rl.project_points(run, full["matrix"]) for run, _ in runs]
    projected = traj.rl.project_points(centred, full["matrix"])
    inside = np.all(np.abs(centred) <= full["half_size"] + 1e-12, axis=1)
    panel = full.copy()
    panel.update({
        "alpha": alpha,
        "mean_speed": traj.mean_speed_um_s(xyz, frames, FPS),
        "centred": centred,
        "projected": projected,
        "runs": runs,
        "projected_runs": projected_runs,
        "segments": [run[i:i + 2] for run in projected_runs for i in range(len(run) - 1)],
        "segment_times": [0.5 * (t[i] + t[i + 1]) for _, t in runs for i in range(len(t) - 1)],
        "start_xy": projected[0] if inside[0] else None,
        "end_xy": projected[-1] if inside[-1] else None,
        "sample_times": times,
        "sample_inside": inside,
        "n_detections": len(points),
    })
    return panel




def draw_figure(examples: list[dict], metadata: pd.DataFrame, *, with_text: bool,
                speed_alpha_12cm: bool = False) -> plt.Figure:
    opts = drawing_options(CUBE_UM)
    opts.clip_to_cube = True
    opts.font_size = 6.2 * 1.5
    traj.style_rcparams(opts)
    panel_w = 1.34
    panel_h = panel_w / traj.canonical_aspect(opts.elev, opts.azim, opts.focal_length, traj.DATA_PAD)
    gap_x = 0.09
    if with_text:
        left, right = 0.55, 0.17
        header_h, caption_h, foot_h, gap_y, footer_h = 0.55, 0.26, 0.25, 0.17, 0.61
    else:
        left, right = 0.07, 0.07
        header_h, caption_h, foot_h, gap_y, footer_h = 0.04, 0.01, 0.01, 0.08, 0.22
    fig_w = left + 6 * panel_w + 5 * gap_x + right
    scale = SPEED_ALPHA_WIDTH_CM / 2.54 / fig_w if speed_alpha_12cm else 1.0
    if speed_alpha_12cm:
        panel_w, panel_h, gap_x, left, right, header_h, gap_y, footer_h = (
            dimension * scale for dimension in
            (panel_w, panel_h, gap_x, left, right, header_h, gap_y, footer_h)
        )
        caption_h = foot_h = 0.15  # Room for the 7 pt labels around each scaled box.
        fig_w *= scale
        opts.line_width *= scale
        opts.box_line_width *= scale
        opts.grid_line_width *= scale
        font_manager.fontManager.addfont(str(ARIAL_FONT))
        matplotlib.rcParams["font.family"] = "Arial"
        matplotlib.rcParams["font.size"] = 7
    fig_h = header_h + 2 * (caption_h + panel_h + foot_h) + gap_y + footer_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor="white")
    cmap = traj.time_colormap(opts)
    norm = Normalize(0, max(e["panels"][1]["duration_s"] for e in examples))

    for boundary in (2, 4):
        x = left + boundary * panel_w + (boundary - 0.5) * gap_x
        line_bottom = footer_h if with_text else footer_h + foot_h
        line_top = fig_h - (0.10 if with_text else header_h + caption_h)
        fig.add_artist(Line2D([x / fig_w] * 2,
                              [line_bottom / fig_h, line_top / fig_h],
                              transform=fig.transFigure, color="#999999",
                              linewidth=0.8 * scale,
                              linestyle=(0, (4 * scale, 4 * scale))))
    if with_text:
        for index, (_, title, _, _, color) in enumerate(GROUPS):
            x = left + (2 * index + 1) * panel_w + (2 * index + 0.5) * gap_x
            fig.text(x / fig_w, 1 - 0.17 / fig_h, title, ha="center", va="center",
                     fontsize=9.3, color=color)
        for col, example in enumerate(examples):
            x = left + (col + 0.5) * panel_w + col * gap_x
            fig.text(x / fig_w, 1 - 0.39 / fig_h, f"Track {example['track_id']}",
                     ha="center", va="center", fontsize=7.3, color="#333333")

    reference = None
    for row, factor in enumerate(FACTORS):
        y_top = fig_h - header_h - row * (caption_h + panel_h + foot_h + gap_y)
        panel_top = y_top - caption_h
        if with_text:
            fig.text(0.25 / fig_w, (panel_top - panel_h / 2) / fig_h,
                     f"{FPS / factor:g}\nvps", ha="center", va="center", fontsize=8.2,
                     color="#333333")
        for col, example in enumerate(examples):
            x0 = left + col * (panel_w + gap_x)
            ax = fig.add_axes((x0 / fig_w, (panel_top - panel_h) / fig_h,
                               panel_w / fig_w, panel_h / fig_h))
            panel = example["panels"][factor]
            traj.draw_box(ax, panel, args=opts)
            traj.draw_trajectory(ax, panel, time_cmap=cmap, norm=norm,
                                 line_width=opts.line_width, outline_scale=0)
            if speed_alpha_12cm:
                for marker in ax.lines:
                    marker.set_markersize(marker.get_markersize() * scale)
                    marker.set_markeredgewidth(marker.get_markeredgewidth() * scale)
            if factor > 1:
                inside = panel["sample_inside"]
                ax.scatter(panel["projected"][inside, 0], panel["projected"][inside, 1],
                           c=panel["sample_times"][inside], cmap=cmap, norm=norm,
                           s=8 * scale ** 2, edgecolors="white",
                           linewidths=0.3 * scale, zorder=5)
            xlim, ylim = traj.panel_data_limits(panel, traj.DATA_PAD)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            if speed_alpha_12cm:
                x_mid = (x0 + panel_w / 2) / fig_w
                speed = f"{panel['mean_speed'] * 60:.1f}"
                fig.text(x_mid, (panel_top + caption_h / 2) / fig_h,
                         speed,
                         ha="center", va="center", fontsize=7, color="black")
                fig.text(x_mid, (panel_top - panel_h - foot_h / 2) / fig_h,
                         f"α = {panel['alpha']:.3f}",
                         ha="center", va="center", fontsize=7, color="black")
            elif with_text:
                speed = f"{panel['mean_speed'] * 60:.1f}"
                if col == len(examples) - 1:
                    speed += r" ($\mathrm{\mu m\,min^{-1}}$)"
                x_mid = (x0 + panel_w / 2) / fig_w
                fig.text(x_mid, (panel_top + caption_h / 2) / fig_h, speed,
                         ha="center", va="center", fontsize=9.3, color="black")
                r2 = metadata.loc[
                    metadata["track_id"].eq(example["track_id"])
                    & metadata["downsample_factor"].eq(factor), "alpha_fit_r2"
                ].iloc[0]
                flag = "*" if r2 < 0.5 else ""
                fig.text(x_mid, (panel_top - panel_h - foot_h / 2) / fig_h,
                         rf"$\alpha$ = {panel['alpha']:.3f}{flag}",
                         ha="center", va="center", fontsize=7.3, color="#555555")
            if reference is None:
                reference = (ax, panel)

    fig.canvas.draw()
    ax, panel = reference
    x_lo, _ = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    mid_y = (y_lo + y_hi) / 2
    p0 = ax.transData.transform((x_lo, mid_y))
    p1 = ax.transData.transform((x_lo + traj.scale_bar_length_ndc(panel, 1.0), mid_y))
    bar_length = abs(p1[0] - p0[0]) / (fig_w * fig.dpi)
    bar_x = (0.65 if with_text else left + 0.10 * scale) / fig_w
    bar_y = (0.32 if with_text else 0.10 * scale) / fig_h
    fig.add_artist(Line2D([bar_x, bar_x + bar_length], [bar_y, bar_y],
                          transform=fig.transFigure, linewidth=3.2 * scale,
                          color="black", solid_capstyle="butt"))
    if with_text:
        fig.text(bar_x + bar_length + 0.04 / fig_w, bar_y, r"1 $\mu$m",
                 ha="left", va="center", fontsize=8)

    if with_text:
        color_ax = fig.add_axes((0.70, 0.27 / fig_h, 0.19, 0.10 / fig_h))
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax,
                                orientation="horizontal")
        colorbar.set_ticks([])
        colorbar.ax.tick_params(length=0)
        colorbar.outline.set_linewidth(0.7)
        fig.text(0.70 - 0.04 / fig_w, 0.32 / fig_h, "0 s", ha="right", va="center", fontsize=7.5)
        fig.text(0.89 + 0.04 / fig_w, 0.32 / fig_h, f"{norm.vmax:.2f} s",
                 ha="left", va="center", fontsize=7.5)
        flagged = metadata.loc[metadata["alpha_fit_r2"].lt(0.5)]
        if not flagged.empty:
            ids = ", ".join(f"{r.track_id} ({r.alpha_fit_r2:.2f})" for r in flagged.itertuples())
            fig.text(0.35 / fig_w, 0.09 / fig_h, f"* Low alpha fit R²: track {ids}",
                     ha="left", va="center", fontsize=7, color="#555555")
    return fig
