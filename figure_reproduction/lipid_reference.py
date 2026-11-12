"""Rebuild the shared fixed-coordinate lipid reference from current 3D volumes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .isotropy import RunConfig, reference_ratios, sample_axis_profile
from .volumes import analysis_volume

METHODS = {"depict": "DEPICT", "bpm": "BPM+TV", "rytov": "Rytov"}


def measure(root, config, cells):
    centers = pd.read_csv(root / config["reference_droplets"])
    settings = RunConfig(**config["fwhm_settings"])
    rows = []
    for cell in cells:
        selected = centers.loc[centers.cell == cell].sort_values("lipid_id")
        if selected.empty or selected.lipid_id.duplicated().any():
            raise ValueError(f"Missing or duplicated droplet coordinates: {cell}")
        if not np.isfinite(selected[["z", "y", "x"]].to_numpy()).all():
            raise ValueError(f"Nonfinite droplet coordinates: {cell}")
        for method, label in METHODS.items():
            volume = analysis_volume(root / config["volumes"][cell][method])
            for center in selected.itertuples(index=False):
                row = dict(cell=cell, cell_display=config["cell_names"][cell],
                           method=label, lipid_id=center.lipid_id,
                           z=center.z, y=center.y, x=center.x)
                widths = {}
                for axis in ("x", "y", "z"):
                    result = sample_axis_profile(volume, center.z, center.y, center.x,
                                                 axis, settings, apply_qc=True)
                    row.update({axis + "_" + key: value for key, value in result.items()})
                    width = result["fwhm_um"] if result["valid"] else np.nan
                    row["fwhm_" + axis + "_um"] = widths[axis] = width
                row["valid_xyz"] = bool(np.isfinite(list(widths.values())).all())
                row["axial_lateral_ratio"] = (
                    widths["z"] / ((widths["x"] + widths["y"]) / 2)
                    if row["valid_xyz"] else np.nan
                )
                rows.append(row)
            del volume
    return pd.DataFrame(rows)


def rebuild(root):
    root = Path(root).resolve()
    config = json.loads((root / "analysis_config.json").read_text())
    coordinates = pd.read_csv(root / config["reference_droplets"])
    table = measure(root, config, list(config["volumes"]))
    table.to_csv(root / "fwhm_measurements_240.csv", index=False)
    valid_ids = table.loc[(table.method == "DEPICT") & table.valid_xyz,
                          ["cell", "lipid_id"]]
    common = coordinates.merge(valid_ids, on=["cell", "lipid_id"], validate="one_to_one")
    rows = []
    indexed = table.set_index(["cell", "lipid_id", "method"])
    for center in common.to_dict("records"):
        row = dict(center)
        for label in METHODS.values():
            measured = indexed.loc[(center["cell"], center["lipid_id"], label)]
            valid = bool(measured.valid_xyz)
            row["valid_" + label] = valid
            row["ratio_" + label] = measured.axial_lateral_ratio if valid else np.nan
            for axis in ("x", "y", "z"):
                row[f"fwhm_{axis}_um_{label}"] = measured[f"fwhm_{axis}_um"] if valid else np.nan
        row["valid_all_three"] = all(row["valid_" + m] for m in METHODS.values())
        rows.append(row)
    reference = pd.DataFrame(rows)
    reference.to_csv(root / "reference_droplets_240.csv", index=False)
    ratios = reference_ratios(reference)
    ratios.to_csv(root / "reference_ratios_240.csv", index=False)
    print(f"Rebuilt {len(reference)} shared coordinates and {len(ratios)} valid method ratios from complete TIFFs.")
    return reference
