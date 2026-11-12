#!/usr/bin/env python3
"""Reproduce Figure 3 a-h from complete local volumes and supplied coordinates."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/depict-figure3-matplotlib")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))

from scripts.analyze_isotropy import build_tables
from scripts.plot_panels import assemble, save_boxplot, save_image_svg, setup_style
from scripts.render_panels_with_zoom import build_images


def main():
    config = json.loads((ROOT / "config.json").read_text())
    labels = [p["label"] for p in config["panels"]]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", choices=labels, help="Default: all eight panels")
    args = parser.parse_args()
    selected = args.panels or labels
    if len(selected) != len(set(selected)):
        parser.error("Duplicate panel letters")
    for methods in config["volumes"].values():
        for path in methods.values():
            if not (ROOT / path).is_file():
                raise FileNotFoundError(ROOT / path)
    setup_style()
    geometries = build_images(ROOT, config, selected)
    for panel in config["panels"]:
        if panel["label"] in geometries:
            save_image_svg(ROOT, panel, geometries[panel["label"]]["geometry"])
    tables = {}
    for panel in config["panels"]:
        if panel["kind"] == "isotropy" and panel["label"] in selected:
            tables[panel["label"]] = build_tables(ROOT, config, panel)
            save_boxplot(ROOT, config, panel, tables[panel["label"]])
        if panel["label"] in selected:
            print("Figure3: panel " + panel["label"] + " complete", flush=True)
    if set(selected) == set(labels):
        assemble(ROOT, config, geometries, tables)
        print("Figure3_reproduced.pdf complete", flush=True)


if __name__ == "__main__":
    main()
