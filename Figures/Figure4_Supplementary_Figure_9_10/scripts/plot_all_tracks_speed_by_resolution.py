#!/usr/bin/env python3
"""All-track box summaries and annotated speed-distribution rendering."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig4-supp-speeds")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "Supplementary_Figure_10"
TRACKS_CSV = HERE.parent / "lipid_tracking" / "complete_tracks.csv"
FPS = 16.25
FACTORS = (1, 2, 4, 20)
RATE_LABELS = ("16.25", "8.13", "4.06", "0.81")
POSITIONS = (4, 3, 2, 1)
BOX_WIDTH = 0.50
BOX_COLORS = ("#6F8793", "#0077B6", "#D1A33A", "#B45B48")
INK = "#1A1A1A"
GRID = "#E8E8E8"
DPI = 320

PLOT_WIDTH = 5.08 * 0.979
PLOT_HEIGHT = (2.424 * 1.5) * (1 - 0.035 / 1.5 - 0.009 / 1.5)
TEXTLESS_PLOT_WIDTH = 6.0 * (0.975 - 0.145)
TEXTLESS_PLOT_HEIGHT = 3.3 * (0.955 - 0.20)






def summarize(speeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in FACTORS:
        values = speeds.loc[speeds["downsample_factor"].eq(factor),
                            "mean_speed_um_min"].to_numpy()
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        iqr = q3 - q1
        inliers = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
        rows.append({
            "downsample_factor": factor,
            "volume_rate_vps": FPS / factor,
            "n_tracks": len(values),
            "mean_um_min": values.mean(),
            "q1_um_min": q1,
            "median_um_min": median,
            "q3_um_min": q3,
            "whisker_low_um_min": inliers.min(),
            "whisker_high_um_min": inliers.max(),
            "n_outliers": len(values) - len(inliers),
        })
    return pd.DataFrame(rows)


def plot(speeds: pd.DataFrame, *, with_text: bool) -> plt.Figure:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    if with_text:
        left, right, bottom, top = 1.30, 0.14, 0.64, 0.24
        plot_width, plot_height = PLOT_WIDTH, PLOT_HEIGHT
    else:
        left = bottom = 0.075
        right = top = 0.02
        plot_width, plot_height = TEXTLESS_PLOT_WIDTH, TEXTLESS_PLOT_HEIGHT
    width = left + plot_width + right
    height = bottom + plot_height + top
    fig = plt.figure(figsize=(width, height), dpi=DPI)
    ax = fig.add_axes((left / width, bottom / height,
                       plot_width / width, plot_height / height))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.75)

    values = [speeds.loc[speeds["downsample_factor"].eq(factor),
                         "mean_speed_um_min"].to_numpy() for factor in FACTORS]
    artists = ax.boxplot(
        values, positions=POSITIONS, vert=False, widths=BOX_WIDTH, whis=1.5,
        patch_artist=True, manage_ticks=False,
        boxprops={"edgecolor": INK, "linewidth": 1.3},
        medianprops={"linewidth": 0},
        whiskerprops={"color": INK, "linewidth": 1.3, "solid_capstyle": "round"},
        capprops={"color": INK, "linewidth": 1.3, "solid_capstyle": "round"},
        flierprops={"marker": "o", "markersize": 2.4, "markerfacecolor": INK,
                    "markeredgecolor": "none", "alpha": 0.45},
    )
    for box, color in zip(artists["boxes"], BOX_COLORS):
        box.set_facecolor(color)
    median_half = BOX_WIDTH * (0.085 / 0.23)
    for position, value in zip(POSITIONS, values):
        median = float(np.median(value))
        for color, line_width, zorder in ((INK, 3.5, 4), ("white", 2.1, 5)):
            ax.plot([median, median], [position - median_half, position + median_half],
                    color=color, linewidth=line_width, solid_capstyle="round", zorder=zorder)

    ax.set_xlim(0, max(66, speeds["mean_speed_um_min"].max() + 2.4))
    ax.set_ylim(0.45, 4.55)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(12))
    ax.set_yticks(POSITIONS)
    if with_text:
        ax.set_yticklabels([rf"{rate} ({factor}$\times$)" for rate, factor in
                            zip(RATE_LABELS, FACTORS)])
        ax.set_xlabel(r"Mean track speed ($\mu$m min$^{-1}$)", fontsize=9, labelpad=6)
        ax.set_ylabel(r"Volume rate (volumes s$^{-1}$)", fontsize=9, labelpad=9)
        ax.text(0.98, 0.96, f"All tracks  (n={len(values[0])})", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=INK)
    else:
        ax.xaxis.set_major_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_formatter(mticker.NullFormatter())
    ax.tick_params(axis="both", direction="out",
                   labelsize=8, width=1.1, length=4.5, color=INK, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.3)
        spine.set_color(INK)
    return fig
