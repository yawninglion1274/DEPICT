#!/usr/bin/env python3
"""Three-class boxplot style parameters and summary function."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig4-panel-e")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "panel_e"
FIG4 = HERE.parent
GROUPS_CSV = PROJECT / "analysis/motion_classification.csv"
TRACKS_CSV = FIG4 / "lipid_tracking" / "complete_tracks.csv"
FPS = 16.25
FACTORS = (1, 2, 4, 20)
RATES = ("16.25", "8.13", "4.06", "0.81")
INK = "#1A1A1A"
GRID = "#E8E8E8"
DPI = 320
# Top to bottom within each row; motion colors match panel F.
GROUPS = (
    ("diffusing", "#90BE6D", 0.27),
    ("directional", "#FAA307", 0.0),
    ("confined", "#0077B6", -0.27),
)
HEIGHT_SCALE = 1.5
OLD_W, OLD_H = 2155 / DPI, 1279 / DPI
OLD_AXES = (0.235, 0.18, 0.73, 0.68)
TEXTLESS_SIZE = (5.08, 2.424 * HEIGHT_SCALE)
TEXTLESS_AXES = (0.016, 0.035 / HEIGHT_SCALE, 0.979, 1 - 0.035 / HEIGHT_SCALE - 0.009 / HEIGHT_SCALE)
ROW_STEP = 1.0
ROW_CENTERS = (4, 3, 2, 1)
BOX_WIDTH = 0.23
MEDIAN_HALF = 0.085




def summarize(speeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, _, _ in GROUPS:
        for factor in FACTORS:
            v = speeds.loc[speeds["motion_group"].eq(group)
                           & speeds["downsample_factor"].eq(factor), "mean_speed_um_min"].to_numpy()
            q1, median, q3 = np.percentile(v, [25, 50, 75])
            iqr = q3 - q1
            inliers = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
            rows.append({
                "motion_group": group, "downsample_factor": factor,
                "volume_rate_vps": FPS / factor, "n_tracks": len(v),
                "mean_um_min": v.mean(), "q1_um_min": q1, "median_um_min": median,
                "q3_um_min": q3, "whisker_low_um_min": inliers.min(),
                "whisker_high_um_min": inliers.max(), "n_outliers": len(v) - len(inliers),
            })
    return pd.DataFrame(rows)
