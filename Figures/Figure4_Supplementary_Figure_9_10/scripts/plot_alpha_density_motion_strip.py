#!/usr/bin/env python3
"""Baseline alpha KDE, raw-sample Gaussian mixture and fixed motion-class strip."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig4-panel-f")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import gaussian_kde, norm
from sklearn.mixture import GaussianMixture
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "panel_f"
FIG4 = HERE.parent

GROUPS_CSV = PROJECT / "analysis/motion_classification.csv"
DPI = 320
DENSITY_COLOR = "#6F8793"
# Muted fit colors stay distinct from the three motion classes and the density curve.
COMPONENT_COLORS = ("#9A668E", "#BD6969")
# Motion colors of panel E; rows listed top to bottom.
MOTIONS = (
    ("diffusing", "Diffusing", "#90BE6D"),
    ("directional", "Directional", "#FAA307"),
    ("confined", "Confined", "#0077B6"),
)
X_MIN, X_MAX, X_STEP = 0.2, 1.8, 0.5
BINS = 28
KDE_GRID_POINTS = 1001
KDE_SCOTT_MULTIPLIER = 0.75
KDE_SUFFIX = "kde_scott_x0p75"
INK = "#1A1A1A"
GRID = "#E8E8E8"
# Frame, tick and grid strokes of panel E (points), so the two panels match when placed.
LW_SPINE, TICK_WIDTH, TICK_LENGTH, LW_GRID = 1.3, 1.1, 4.5, 0.75
LW_DENSITY, LW_COMPONENT, LW_MEAN = 2.0, 1.5, 1.0
PANEL_E_PAGE = (5.08, 2.424 * 1.5)
PAGE_W = PANEL_E_PAGE[0]
PAGE_H = 1.903 * 1.5 * PAGE_W / 4.597
MARGIN_LEFT = 0.016 * PAGE_W
MARGIN_RIGHT = (1 - 0.016 - 0.979) * PAGE_W
MARGIN_TOP = 0.009 / 1.5 * PANEL_E_PAGE[1]
MARGIN_BOTTOM = 0.035 / 1.5 * PANEL_E_PAGE[1]
PLOT_W = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT
STRIP_GAP = 0.08
PLOT_H = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM - STRIP_GAP
DENSITY_H = 0.62 * PLOT_H
STRIP_H = PLOT_H - DENSITY_H
LANE_STEP = 0.34
ROW_STEP = 3.0
MARKER_SIZE = 17
MIN_SEPARATION = 0.019


def beeswarm_lanes(alpha: np.ndarray) -> np.ndarray:
    """Smallest |lane| for each dot (in alpha order) that avoids a close neighbour."""
    lanes = np.zeros(len(alpha), dtype=int)
    placed: list[int] = []
    candidates = [0] + [lane for level in range(1, len(alpha) + 1) for lane in (level, -level)]
    for index in np.argsort(alpha, kind="stable"):
        for lane in candidates:
            if all(lane != lanes[j] or abs(alpha[index] - alpha[j]) >= MIN_SEPARATION
                   for j in placed):
                lanes[index] = lane
                placed.append(int(index))
                break
    return lanes


def clip_below(x: np.ndarray, y: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    keep = y >= floor
    first, last = int(np.argmax(keep)), int(len(keep) - np.argmax(keep[::-1]))
    return x[first:last], y[first:last]

def strip_points(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row_no, (name, _, _) in enumerate(MOTIONS):
        center = (len(MOTIONS) - 1 - row_no) * ROW_STEP
        subset = groups.loc[groups["final_group"].eq(name)].sort_values(["alpha", "track_id"])
        lanes = beeswarm_lanes(subset["alpha"].to_numpy())
        for track, lane in zip(subset.itertuples(index=False), lanes):
            rows.append({"track_id": track.track_id, "final_group": name,
                         "alpha": track.alpha, "display_lane": int(lane),
                         "display_y": center + LANE_STEP * lane})
    return pd.DataFrame(rows)


def style_frame(ax: plt.Axes) -> None:
    """Closed frame, outward ticks and vertical grid as in panel E."""
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=LW_GRID)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LW_SPINE)
        spine.set_color(INK)
    ax.tick_params(axis="both", direction="out", width=TICK_WIDTH, length=TICK_LENGTH,
                   color=INK, labelcolor=INK, labelsize=8)


def draw(x: np.ndarray, density: np.ndarray, fit: dict, points: pd.DataFrame,
         *, with_text: bool) -> plt.Figure:
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    # Text version: same axes size in inches, with room for the labels.
    if with_text:
        left, right, top, bottom = 1.30, 0.10, 0.08, 0.50
    else:
        left, right, top, bottom = MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM
    width = left + PLOT_W + right
    height = bottom + PLOT_H + STRIP_GAP + top
    fig = plt.figure(figsize=(width, height), dpi=DPI)
    strip_ax = fig.add_axes((left / width, bottom / height, PLOT_W / width, STRIP_H / height))
    dens_ax = fig.add_axes((left / width, (bottom + STRIP_H + STRIP_GAP) / height,
                            PLOT_W / width, DENSITY_H / height), sharex=strip_ax)
    for ax in (dens_ax, strip_ax):
        style_frame(ax)

    # Density stays foremost; each fit and its mean share a muted color.
    y_top = 1.15 * max(density.max(), fit["component1"].max(), fit["component2"].max())
    for color, mu, comp in zip(COMPONENT_COLORS,
                               (fit["mu1"], fit["mu2"]),
                               (fit["component1"], fit["component2"])):
        dens_ax.axvline(mu, color=color, lw=LW_MEAN, ls=(0, (2.5, 2.0)),
                        alpha=0.65, zorder=2)
        g, y = clip_below(fit["grid"], comp, 0.01 * y_top)
        dens_ax.plot(g, y, color=color, lw=LW_COMPONENT, ls=(0, (4.0, 2.0)),
                     dash_capstyle="round", zorder=3)
    dens_ax.plot(x, density, color=DENSITY_COLOR, lw=LW_DENSITY, zorder=4,
                 solid_capstyle="round", solid_joinstyle="round")
    dens_ax.set_ylim(0, y_top)
    dens_ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    # Density values only (text version); no tick marks on the left edge.
    dens_ax.tick_params(axis="y", length=0, pad=4)
    dens_ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # Motion strip: one row per class, dots at their exact alpha.
    for name, _, color in MOTIONS:
        sub = points.loc[points["final_group"].eq(name)]
        strip_ax.scatter(sub["alpha"], sub["display_y"], s=MARKER_SIZE, c=color,
                         edgecolors="white", linewidths=0.35, zorder=3)
    row_centers = [(len(MOTIONS) - 1 - i) * ROW_STEP for i in range(len(MOTIONS))]
    lane_room = LANE_STEP * points["display_lane"].abs().max() + 0.5 * ROW_STEP
    strip_ax.set_ylim(-lane_room, row_centers[0] + lane_room)
    strip_ax.set_xlim(X_MIN, X_MAX)
    strip_ax.xaxis.set_major_locator(mticker.MultipleLocator(X_STEP))
    strip_ax.set_yticks(row_centers)
    # Row labels only (text version); no tick marks on the strip's left edge.
    strip_ax.tick_params(axis="y", length=0, pad=4)

    if with_text:
        counts = points["final_group"].value_counts()
        strip_ax.set_yticklabels([f"{label} ($n={counts[name]}$)" for name, label, _ in MOTIONS])
        strip_ax.set_xlabel(r"Anomalous exponent, $\alpha$", fontsize=9, labelpad=6)
        dens_ax.set_ylabel("Probability density", fontsize=9, labelpad=9)
        for mu in (fit["mu1"], fit["mu2"]):
            dens_ax.annotate(f"{mu:.2f}", xy=(mu, 1), xycoords=dens_ax.get_xaxis_transform(),
                             xytext=(3, -3), textcoords="offset points", ha="left", va="top",
                             fontsize=7.5, color="#666666")
    else:
        for ax in (dens_ax, strip_ax):
            ax.xaxis.set_major_formatter(mticker.NullFormatter())
            ax.yaxis.set_major_formatter(mticker.NullFormatter())
    return fig


def kde_density_profile(values: np.ndarray) -> pd.DataFrame:
    kde = gaussian_kde(
        values, bw_method=lambda estimator: KDE_SCOTT_MULTIPLIER * estimator.scotts_factor()
    )
    grid = np.linspace(X_MIN, X_MAX, KDE_GRID_POINTS)
    return pd.DataFrame({
        "alpha": grid,
        "density": kde(grid),
        "scott_multiplier": KDE_SCOTT_MULTIPLIER,
        "bandwidth_used": float(np.sqrt(kde.covariance[0, 0])),
    })

def fit_two_gaussians_to_samples(values: np.ndarray, grid: np.ndarray) -> dict:
    """Fit a normalized two-component Gaussian mixture to the raw alpha values."""
    samples = values.reshape(-1, 1)
    model = GaussianMixture(n_components=2, n_init=20, random_state=0,
                            reg_covar=1e-6).fit(samples)
    single = GaussianMixture(n_components=1, n_init=20, random_state=0,
                             reg_covar=1e-6).fit(samples)
    order = np.argsort(model.means_.ravel())
    weights = model.weights_[order]
    means = model.means_.ravel()[order]
    sigmas = np.sqrt(model.covariances_.reshape(-1)[order])
    components = [weight * norm.pdf(grid, mu, sigma)
                  for weight, mu, sigma in zip(weights, means, sigmas)]
    return {
        "method": "raw_sample_gaussian_mixture_mle",
        "n": len(values),
        "w1": float(weights[0]), "mu1": float(means[0]), "sigma1": float(sigmas[0]),
        "w2": float(weights[1]), "mu2": float(means[1]), "sigma2": float(sigmas[1]),
        "aic_1_component": float(single.aic(samples)),
        "aic_2_components": float(model.aic(samples)),
        "bic_1_component": float(single.bic(samples)),
        "bic_2_components": float(model.bic(samples)),
        "grid": grid,
        "component1": components[0], "component2": components[1],
        "total": components[0] + components[1],
    }
