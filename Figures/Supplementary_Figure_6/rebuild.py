#!/usr/bin/env python3
"""Rebuild the six annotated SVG panels from complete local 3D TIFF inputs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from DEPICT.figure_reproduction.pipeline import main

if __name__ == "__main__":
    main(Path(__file__).with_name("config.json"))
