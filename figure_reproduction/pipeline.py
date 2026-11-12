"""Rebuild supplementary panels directly from complete local TIFF volumes."""
import argparse
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/depict-figure-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import ndimage

from .isotropy import RunConfig, sample_axis_profile
from .volumes import OFFSETS, PLANES, VOXEL_SIZE_UM, analysis_volume, three_slices

DISPLAY_PLANES = {"xy": "xy", "xz": "yz", "yz": "xz"}
COLORS = {"bpm": "#D55E00", "rytov": "#0072B2", 240: "#D55E00", 24: "#ecb78c"}


def setup_style():
    font = Path(__file__).resolve().parents[1] / "postprocessing/fonts/Arial.TTF"
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "Arial", "font.size": 6,
                         "axes.linewidth": 0.5, "svg.fonttype": "none",
                         "svg.hashsalt": "DEPICT", "savefig.transparent": True})


def resolve(root, name):
    path = (root / name).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def save_svg(fig, output):
    fig.savefig(output, format="svg", dpi=300, metadata={"Date": None})
    plt.close(fig)
    # Explicit font properties also work in SVG readers without CSS shorthand support.
    def expand_font(match):
        tokens = match.group(1).strip().split()
        size_index = next(i for i, token in enumerate(tokens) if token.endswith("px"))
        styles = [f"font-size: {tokens[size_index]}", "font-family: " + " ".join(tokens[size_index+1:])]
        for token in tokens[:size_index]:
            styles.append(("font-style: " if token in ("italic", "oblique") else "font-weight: ") + token)
        return "; ".join(styles)
    text = re.sub(r'font:\s*([^;\"]+)', expand_font, output.read_text())
    output.write_text(text)


def image_panel(root, panel, output, line_coordinates=None):
    images = three_slices(resolve(root, panel["input"]), panel["display"])
    canvas = np.concatenate([images[p] for p in PLANES], axis=1)
    fig = plt.figure(figsize=(3.55, 2.06))
    ax = fig.add_axes([0.045, 0.18, 0.91, 0.73])
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=0.06, origin="upper", interpolation="nearest")
    ax.axis("off")
    for x in (449.5, 705.5):
        ax.plot([x, x], [-0.5, 449.5], color="white", linewidth=0.45)
    for plane in PLANES:
        x, y = OFFSETS[plane] + 19, 425
        ax.annotate("", (x+65, y), (x, y), arrowprops={"arrowstyle": "->", "color": "white", "lw": 0.45, "mutation_scale": 4})
        ax.annotate("", (x, y-65), (x, y), arrowprops={"arrowstyle": "->", "color": "white", "lw": 0.45, "mutation_scale": 4})
        horizontal, vertical = ("x", "y") if plane == "xy" else ("z", "y" if plane == "xz" else "x")
        ax.text(x+70, y+2, horizontal, color="white", fontsize=5, va="center")
        ax.text(x, y-75, vertical, color="white", fontsize=5, ha="center")
    if panel.get("scale_bar_um"):
        length = panel["scale_bar_um"] / VOXEL_SIZE_UM
        ax.plot([432-length, 432], [28, 28], color="white", lw=1.2, solid_capstyle="butt")
        ax.text(432-length/2, 51, f"{panel['scale_bar_um']:g} \u00b5m", color="white", ha="center", fontsize=5)
    if line_coordinates is not None:
        subset = line_coordinates.loc[line_coordinates.cell == panel["cell"]]
        for row in subset.itertuples(index=False):
            x = np.array([row.start_x_px, row.end_x_px]) + OFFSETS[row.plane]
            y = [row.start_y_px, row.end_y_px]
            ax.plot(x, y, color=COLORS[panel["method"]], lw=0.9,
                    linestyle="--" if panel["method"] == "rytov" else "-")
    fig.text(0.02, 0.955, panel["label"], weight="bold", va="top")
    fig.text(0.51, 0.955, panel["title"], ha="center", va="top")
    cax = fig.add_axes([0.28, 0.115, 0.44, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(1.33, 1.39), cmap="gray"), cax=cax, orientation="horizontal", ticks=[1.33, 1.39])
    cb.ax.tick_params(length=0, pad=2, labelsize=5)
    cb.set_label("Refractive index", labelpad=1, fontsize=5)
    save_svg(fig, output / f"panel_{panel['label']}_with_text.svg")


def line_profiles(root, config, cell, coordinates):
    images = {}
    for method, path in config["profile_volumes"][cell].items():
        images[method] = three_slices(resolve(root, path), config["profile_display"][cell])
    rows = []
    for line in coordinates.loc[coordinates.cell == cell].itertuples(index=False):
        start = np.array([line.start_x_px, line.start_y_px])
        end = np.array([line.end_x_px, line.end_y_px])
        positions = start + np.linspace(0, 1, 179)[:, None] * (end-start)
        distance = np.linspace(0, np.linalg.norm(end-start)*VOXEL_SIZE_UM, 179)
        values = {m: ndimage.map_coordinates(images[m][line.plane],
                  [positions[:, 1], positions[:, 0]], order=1, prefilter=False) for m in images}
        for i in range(len(distance)):
            rows.append({"cell": cell, "plane": line.plane, "sample_index": i,
                         "distance_um": distance[i], "slice_x_px": positions[i, 0],
                         "slice_y_px": positions[i, 1],
                         "bpm_delta_n": float(values["bpm"][i]),
                         "rytov_delta_n": float(values["rytov"][i]),
                         "bpm_n": float(values["bpm"][i])+1.33,
                         "rytov_n": float(values["rytov"][i])+1.33})
    return pd.DataFrame(rows)


def profile_panel(root, config, panel, output, coordinates):
    table = line_profiles(root, config, panel["cell"], coordinates)
    table.to_csv(output / "line_profile_samples.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(3.55, 1.85), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.26, top=0.83, wspace=0.08)
    for ax, plane in zip(axes, PLANES):
        sample = table.loc[table.plane == plane]
        for method in ("bpm", "rytov"):
            ax.plot(sample.distance_um, sample[method+"_n"], color=COLORS[method], lw=0.65,
                    ls="--" if method == "rytov" else "-", label=f"DEPICT ({method.upper() if method == 'bpm' else 'Rytov'})")
        ax.set(xlim=(0, 4), ylim=(1.33, 1.41), xticks=[0, 1, 2, 3, 4],
               yticks=[1.33, 1.35, 1.37, 1.39, 1.41])
        ax.set_xlabel(DISPLAY_PLANES[plane] + " distance (\u00b5m)", labelpad=1)
        ax.tick_params(length=2, width=0.4, pad=1, labelsize=5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Refractive index", labelpad=1)
    axes[0].legend(frameon=False, fontsize=4.5, loc="upper left", handlelength=2.2, borderaxespad=0.2)
    fig.text(0.02, 0.96, panel["label"], weight="bold", va="top")
    fig.text(0.53, 0.96, panel["title"], ha="center", va="top")
    save_svg(fig, output / f"panel_{panel['label']}_with_text.svg")


def isotropy_measurements(root, config, cells):
    reference = pd.read_csv(resolve(root, config["reference_droplets"]))
    coordinates_24 = None
    if "analysis_coordinates_24" in config:
        coordinates_24 = pd.read_csv(resolve(root, config["analysis_coordinates_24"]))
        keys = ["cell", "lipid_id"]
        axes = ["z", "y", "x"]
        if coordinates_24.empty or coordinates_24.duplicated(keys).any():
            raise ValueError("Missing or duplicated final 24-angle coordinates")
        if not np.isfinite(coordinates_24[axes].to_numpy()).all():
            raise ValueError("Nonfinite final 24-angle coordinates")
        matched = coordinates_24.merge(reference[keys + axes], on=keys, how="left",
                                       suffixes=("", "_reference"), validate="one_to_one")
        np.testing.assert_allclose(matched[axes], matched[[axis + "_reference" for axis in axes]],
                                   rtol=0, atol=1e-12,
                                   err_msg="Final 24-angle coordinates differ from shared reference")
    settings = RunConfig(**config["fwhm_settings"])
    rows = []
    for cell in cells:
        for angle in (240, 24):
            coordinate_source = coordinates_24 if angle == 24 and coordinates_24 is not None else reference
            centers = coordinate_source.loc[coordinate_source.cell == cell].sort_values("lipid_id")
            if centers.empty or centers.lipid_id.duplicated().any():
                raise ValueError(f"Invalid droplet coordinates for {cell}/{angle}")
            volume = analysis_volume(resolve(root, config["isotropy_volumes"][cell][str(angle)]))
            qc = angle == 240 or cell in ("293t", "hela", "cos7")
            for center in centers.itertuples(index=False):
                row = {"cell": cell, "cell_display": config["cell_names"][cell],
                       "angle_count": angle, "lipid_id": center.lipid_id,
                       "z": center.z, "y": center.y, "x": center.x, "quality_control_applied": qc}
                widths = {}
                for axis in ("x", "y", "z"):
                    measured = sample_axis_profile(volume, center.z, center.y, center.x, axis, settings, apply_qc=qc)
                    row.update({axis+"_"+k: v for k, v in measured.items()})
                    widths[axis] = measured["fwhm_um"] if measured["valid"] else np.nan
                    row["fwhm_"+axis+"_um"] = widths[axis]
                row["valid_xyz"] = bool(np.all(np.isfinite(list(widths.values()))))
                row["axial_lateral_ratio"] = widths["z"] / ((widths["x"]+widths["y"])/2) if row["valid_xyz"] else np.nan
                row["measurement_source"] = "complete_local_TIFF"
                if angle == 240:
                    cached_widths = [getattr(center, f"fwhm_{axis}_um_DEPICT") for axis in ("x", "y", "z")]
                    cached_ratio = cached_widths[2] / ((cached_widths[0]+cached_widths[1])/2)
                    np.testing.assert_allclose(cached_ratio, center.ratio_DEPICT, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(row["axial_lateral_ratio"], center.ratio_DEPICT, rtol=1e-7, atol=1e-9,
                                               err_msg=f"Shared 240-angle reference differs: {cell}/{center.lipid_id}")
                rows.append(row)
            del volume
    table = pd.DataFrame(rows)
    if coordinates_24 is not None:
        if not table.valid_xyz.all():
            invalid = table.loc[~table.valid_xyz, ["cell", "angle_count", "lipid_id"]]
            raise ValueError(f"Final analysis coordinates contain invalid measurements:\n{invalid}")
        table = table.drop(columns=[axis + "_invalid_reason" for axis in ("x", "y", "z")])
    return table


def isotropy_panel(root, config, panel, output):
    table = isotropy_measurements(root, config, panel["cells"])
    table.to_csv(output / "fwhm_measurements.csv", index=False)
    valid = table.loc[table.valid_xyz].copy()
    valid.to_csv(output / "boxplot_values.csv", index=False)
    summary = []
    for (cell, angle), data in table.groupby(["cell", "angle_count"], sort=False):
        values = data.loc[data.valid_xyz, "axial_lateral_ratio"]
        summary.append({"cell": cell, "angle_count": angle, "n_measured": len(data),
                        "n_valid": len(values), "n_uncomputable": len(data)-len(values),
                        "median": values.median(), "q25": values.quantile(.25), "q75": values.quantile(.75)})
    pd.DataFrame(summary).to_csv(output / "sample_sizes_and_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.55, 1.95))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.22, top=0.83)
    for angle, offset in ((240, -0.12), (24, 0.12)):
        data = [valid.loc[(valid.cell == c) & (valid.angle_count == angle), "axial_lateral_ratio"].to_numpy() for c in panel["cells"]]
        bp = ax.boxplot(data, positions=np.arange(3)+offset, widths=0.20,
                        whis=1.5, patch_artist=True, showfliers=False, manage_ticks=False,
                        medianprops={"color": "black", "linewidth": .5},
                        whiskerprops={"color": "0.35", "linewidth": .45},
                        capprops={"color": "0.35", "linewidth": .45})
        for box in bp["boxes"]:
            box.set(facecolor=COLORS[angle], edgecolor="0.35", linewidth=.45)
    ax.axhline(1, color="0.5", ls="--", lw=.5, zorder=0)
    ax.set(xlim=(-.55, 2.55), ylim=(0, 3.2), yticks=[1, 3], xticks=np.arange(3))
    ax.set_xticklabels([config["cell_names"][c] for c in panel["cells"]], fontsize=5)
    ax.set_ylabel("Axial-to-lateral FWHM ratio", fontsize=5.5, labelpad=2)
    ax.tick_params(length=2, width=.4, pad=2, labelsize=5)
    ax.legend(handles=[Patch(facecolor=COLORS[a], edgecolor="0.35", linewidth=.4,
              label=f"DEPICT ({a} angles)") for a in (240, 24)], frameon=False,
              loc="upper left", fontsize=5, handlelength=2)
    fig.text(.02, .96, panel["label"], weight="bold", va="top")
    fig.text(.56, .96, panel["title"], ha="center", va="top")
    save_svg(fig, output / f"panel_{panel['label']}_with_text.svg")


def main(config_path):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", help="Panel letters to rebuild; default: all")
    args = parser.parse_args()
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = json.loads(config_path.read_text())
    labels = [p["label"] for p in config["panels"]]
    selected = args.panels or labels
    if len(selected) != len(set(selected)) or set(selected)-set(labels):
        parser.error("Unknown or duplicate panel letters")
    setup_style()
    coordinates = pd.read_csv(resolve(root, config["line_coordinates"])) if "line_coordinates" in config else None
    for panel in config["panels"]:
        if panel["label"] not in selected:
            continue
        output = root / ("panel_"+panel["label"])
        output.mkdir(exist_ok=True)
        if panel["kind"] == "image":
            image_panel(root, panel, output, coordinates)
        elif panel["kind"] == "line_profiles":
            profile_panel(root, config, panel, output, coordinates)
        elif panel["kind"] == "isotropy":
            isotropy_panel(root, config, panel, output)
        else:
            raise ValueError(panel["kind"])
        print(f"{config['figure']}: panel {panel['label']} -> {output.name}/panel_{panel['label']}_with_text.svg", flush=True)
