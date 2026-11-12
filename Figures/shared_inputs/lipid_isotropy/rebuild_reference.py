#!/usr/bin/env python3
"""Remeasure shared lipid widths and ratios from the latest complete 3D TIFFs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from DEPICT.figure_reproduction.lipid_reference import rebuild

if __name__ == "__main__":
    rebuild(Path(__file__).resolve().parent)
