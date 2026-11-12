#!/usr/bin/env python3
"""Replot this panel from the shared trajectory analysis."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "rebuild.py").is_file())
sys.path.insert(0, str(ROOT / "scripts"))
from reproduce import plot, record_run
if __name__ == "__main__":
    plot(['9c'])
    record_run("plot:" + '9c')
