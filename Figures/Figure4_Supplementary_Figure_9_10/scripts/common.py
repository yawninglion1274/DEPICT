"""Shared package paths and the trajectory table used by all panels."""
from __future__ import annotations

import os
from pathlib import Path
import site
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parents[1] / "manifest/Figure4_Supplementary_Figure_9_10"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/depict-figure4-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Prefer isolated tracking dependencies when available.
for directory in site.getsitepackages():
    private = Path(directory) / "depict_tracking_deps"
    if private.is_dir():
        sys.path.insert(0, str(private))
        break


def load_tracks():
    import pandas as pd
    return pd.read_csv(ROOT / "analysis/tracks_by_resolution.csv", low_memory=False)


def load_reference_tracks():
    table = load_tracks()
    table = table.loc[table.angle_group.eq(12)].copy()
    # Baseline tables do not measure FWHM; concatenation adds empty columns.
    return table.drop(columns=[c for c in table.columns if table[c].isna().all()])


def downsample_tracks(tracks, factor):
    table = tracks.loc[tracks.frame.mod(factor).eq(0)].copy()
    table["frame"] = table.frame // factor
    return table
