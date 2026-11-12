#!/usr/bin/env python3
"""Load this figure's config and delegate processing to the shared library."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from DEPICT.postprocessing.pipeline import main

if __name__ == "__main__":
    main(Path(__file__).with_name("config.json"))
