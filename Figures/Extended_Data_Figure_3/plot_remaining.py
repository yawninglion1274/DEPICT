#!/usr/bin/env python3
"""Render panels c/d from this figure's local analysis caches and config."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from DEPICT.postprocessing.plotting import main

if __name__ == "__main__":
    main(Path(__file__).with_name("config.json"))
