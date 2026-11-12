#!/usr/bin/env python3
"""Shared-bin density and annotated alpha-distribution rendering functions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig4-supp-alpha")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "Supplementary_Figure_10"
FIG4 = HERE.parent
ANALYSIS = PROJECT / "scripts"
sys.path.insert(0, str(ANALYSIS))

from compute_lipid_track_diffusion_alpha import compute_per_track  # noqa: E402

TRACKS_CSV = FIG4 / "lipid_tracking" / "complete_tracks.csv"
FPS = 16.25
FACTORS = (1, 2, 4, 20)  # Same temporal resolutions as panel E.
RATE_LABELS = ("16.25", "8.13", "4.06", "0.81")
COLORS = ("#6F8793", "#0077B6", "#D1A33A", "#B45B48")
BIN_WIDTH = 0.05
SMOOTHING_KERNEL = np.array([1, 2, 3, 2, 1], dtype=float) / 9
KDE_GRID_POINTS = 1001
KDE_SCOTT_MULTIPLIER = 0.75
KDE_SUFFIX = "_kde_scott_x0p75"
INK = "#1A1A1A"






def density_profiles(fits: pd.DataFrame) -> pd.DataFrame:
    lower = min(-0.5, BIN_WIDTH * np.floor(fits["alpha"].min() / BIN_WIDTH))
    upper = max(2.0, BIN_WIDTH * np.ceil(fits["alpha"].max() / BIN_WIDTH))
    edges = np.arange(lower, upper + BIN_WIDTH / 2, BIN_WIDTH)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for factor in FACTORS:
        values = fits.loc[fits["downsample_factor"].eq(factor), "alpha"].to_numpy()
        density, _ = np.histogram(values, bins=edges, density=True)
        density = np.convolve(density, SMOOTHING_KERNEL, mode="same")
        rows.append(pd.DataFrame({
            "downsample_factor": factor,
            "volume_rate_vps": FPS / factor,
            "bin_left": edges[:-1],
            "bin_center": centers,
            "bin_right": edges[1:],
            "smoothed_density": density,
        }))
    return pd.concat(rows, ignore_index=True)




def plot(profiles: pd.DataFrame, fits: pd.DataFrame, *, with_text: bool,
         kde: bool = False, fill: bool = False, fill_alpha: float = 0.20) -> plt.Figure:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    if with_text:
        fig, ax = plt.subplots(figsize=(6.0, 3.3), dpi=320)
        fig.subplots_adjust(left=0.145, right=0.975, bottom=0.20, top=0.955)
    else:
        plot_width = 6.0 * (0.975 - 0.145)
        plot_height = 3.3 * (0.955 - 0.20)
        left = bottom = 0.075
        right = top = 0.02
        width, height = plot_width + left + right, plot_height + bottom + top
        fig = plt.figure(figsize=(width, height), dpi=320)
        ax = fig.add_axes((left / width, bottom / height,
                           plot_width / width, plot_height / height))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.75)
    for factor, rate_label, color in zip(FACTORS, RATE_LABELS, COLORS):
        profile = profiles.loc[profiles["downsample_factor"].eq(factor)]
        count = fits["downsample_factor"].eq(factor).sum()
        x_column, y_column = ("alpha", "density") if kde else ("bin_center", "smoothed_density")
        if fill:
            ax.fill_between(profile[x_column].to_numpy(dtype=float), 0,
                            profile[y_column].to_numpy(dtype=float),
                            color=color, alpha=fill_alpha, linewidth=0, zorder=1)
        ax.plot(profile[x_column], profile[y_column],
                color=color, lw=2.0, solid_capstyle="round", solid_joinstyle="round",
                label=f"{rate_label} volumes/s  (n={count})", zorder=2)

    if kde:
        ax.set_xlim(profiles["alpha"].min(), profiles["alpha"].max())
    else:
        ax.set_xlim(profiles["bin_left"].min(), profiles["bin_right"].max())
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    if with_text:
        ax.set_xlabel(r"Anomalous exponent, $\alpha$", fontsize=9, labelpad=6)
        ax.set_ylabel("Probability density", fontsize=9, labelpad=8)
    else:
        ax.xaxis.set_major_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_formatter(mticker.NullFormatter())
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.3)
    ax.tick_params(axis="both", direction="out",
                   width=1.1, length=4.5,
                   color=INK, labelcolor=INK, labelsize=8)
    if with_text:
        ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=2.4,
                  labelspacing=0.45, borderaxespad=0.8)
    return fig
