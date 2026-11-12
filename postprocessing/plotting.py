"""Config-driven annotated SVG panels from existing postprocessing caches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/depict_postprocessing_mpl")

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import flow_statistics, kymograph
from .volume_io import cut_flow, resolve, signature, write_csv, write_json


def configure_style(settings, base):
    for path in settings.get("font_files", []):
        font_manager.fontManager.addfont(str(resolve(path, base)))
    family = settings.get("font_family", "Arial")
    font_manager.findfont(family, fallback_to_default=False)
    size = settings.get("font_size_pt", 6)
    plt.rcParams.update({
        "font.family": family, "font.size": size, "axes.labelsize": size,
        "axes.titlesize": size, "xtick.labelsize": size, "ytick.labelsize": size,
        "legend.fontsize": size, "axes.linewidth": .65,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.major.width": .65, "ytick.major.width": .65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "svg.fonttype": "none", "svg.image_inline": True,
        "svg.hashsalt": "DEPICT-postprocessing", "figure.facecolor": "white",
        "savefig.facecolor": "white", "mathtext.fontset": "custom",
        "mathtext.rm": family, "mathtext.it": family, "mathtext.bf": family+":bold",
        "mathtext.cal": family, "mathtext.sf": family, "mathtext.tt": family,
        "mathtext.default": "regular",
    })


def axes_inches(fig, box):
    x, y, w, h = box
    return fig.add_axes([x/fig.get_figwidth(), y/fig.get_figheight(),
                         w/fig.get_figwidth(), h/fig.get_figheight()])


def panel_letter(fig, letter, *, settings=None):
    settings = settings or {}
    position = settings.get("position", [.055/fig.get_figwidth(), 1-.20/fig.get_figheight()])
    fig.text(*position, letter, fontweight="bold", fontsize=settings.get("font_size_pt", plt.rcParams["font.size"]),
             va=settings.get("vertical_alignment", "baseline"))


def rectangular_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(".2")
        spine.set_linewidth(.7)
    ax.tick_params(direction="out", length=3, width=.65, pad=1.5)


def save_svg(fig, path):
    """Keep every label live and every curve/marker as a vector object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(match=Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if not (fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)):
                raise ValueError(f"Text extends outside SVG canvas: {artist.get_text()!r}")
    fig.savefig(path, format="svg", dpi=300, metadata={"Date": None})
    ns = {"svg": "http://www.w3.org/2000/svg"}
    root = ET.parse(path).getroot()
    ids = [e.attrib["id"] for e in root.iter() if "id" in e.attrib]
    if len(ids) != len(set(ids)) or not root.findall(".//svg:text", ns):
        raise ValueError("SVG must contain live text and unique element IDs")


def _check_signature(record):
    actual = signature(record["path"])
    if any(actual[key] != record[key] for key in actual):
        raise ValueError(f"Cached input changed: {record['path']}; rebuild the image panels first")


def _dataset_context(config, base, panel, *, cache_only=False):
    matches = [d for d in config["datasets"] if d["panel"] == panel]
    if len(matches) != 1:
        raise ValueError(f"Need one image dataset for panel {panel}")
    path = base / f"panel_{panel}/all_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["dataset"] != matches[0]:
        raise ValueError(f"Dataset config changed since {path}; rebuild the image panels first")
    if not cache_only:
        for record in manifest["source_signatures"]:
            _check_signature(record)
        _check_signature(manifest["filtered_volume"])
    return manifest


def load_rate(base, spec, manifest):
    section = spec.get("section", "xy")
    directory = base / f"panel_{spec['source_panel']}"
    source = directory / f"{section}_strain_rate/areal_strain_rate.npz"
    mask_path = directory / f"{section}_masks.npz"
    with np.load(source) as saved:
        names = saved["region_names"].tolist()
        region = names.index(spec["region"]) if "region" in spec else 0
        raw = saved["rate_s"][region].copy()
        times, pairs = saved["time_s"].copy(), saved["native_pairs"].copy()
        with np.load(mask_path) as support:
            np.testing.assert_array_equal(saved["masks"], support["masks"])
        if float(saved["fps"]) != manifest["fps"]:
            raise ValueError("Strain cache frame rate differs from the image pipeline")
    first, last = manifest["dataset"]["analysis_frame_range"]
    np.testing.assert_array_equal(pairs, np.column_stack([np.arange(first, last), np.arange(first+1, last+1)]))
    origin = manifest["dataset"].get("time_origin", first)
    np.testing.assert_allclose(times, (pairs[:, 0]-origin+.5)/manifest["fps"], rtol=0, atol=1e-14)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite strain cache")
    sigma = float(spec.get("display_sigma_frames", 0))
    if sigma < 0:
        raise ValueError("Display smoothing sigma must be nonnegative")
    rate = gaussian_filter1d(raw, sigma, mode="nearest", truncate=4) if sigma else raw.copy()
    return dict(raw_rate_s=raw, rate_s=rate, time_s=times, native_pairs=pairs,
                duration_s=(last-origin)/manifest["fps"],
                sources=[signature(source), signature(mask_path)], display_sigma_frames=sigma)


def draw_rate(fig, data, spec, *, x_origin=0):
    box = spec.get("axes_box_inches", [.70, .55, 3.18, 2.35])
    ax = axes_inches(fig, [box[0]+x_origin, *box[1:]])
    for start, end in spec.get("event_spans_s", []):
        ax.axvspan(start, end, facecolor=".97", edgecolor=".8", lw=.6, zorder=0)
    for start, end in spec.get("highlight_spans_s", []):
        ax.axvspan(start, end, facecolor="#c9fff8", edgecolor=".8", lw=.6, zorder=.1)
    t, rate = data["time_s"], data["rate_s"]
    ax.plot(t, rate, color=".2", lw=1.2, zorder=2, gid=f"strain_curve_{spec['panel']}")
    ax.axhline(0, color=".65", lw=.7, zorder=1)
    limit = float(spec.get("ylim_abs", 1.45*np.max(abs(rate))))
    if limit <= 0:
        limit = 1.
    ax.set(xlim=(0, data["duration_s"]), ylim=(-limit, limit), xlabel="Time (s)",
           ylabel=r"Areal strain rate (s$^{-1}$)")
    ax.set_xticks(np.arange(0, data["duration_s"], .5))
    ticks = plt.MaxNLocator(5).tick_values(-limit, limit)
    ax.set_yticks(ticks[(ticks >= -limit) & (ticks <= limit)])
    native = data["native_pairs"][:, 0]
    selected = []
    for frame in spec["marker_frames"]:
        matches = np.flatnonzero(native == frame)
        if len(matches) != 1:
            raise ValueError(f"Selected strain frame {frame} missing from cache")
        i = int(matches[0])
        left = spec.get("positive_labels_left", False) and rate[i] > 0
        offset = (-6 if left else 6, 7)
        ax.scatter(t[i], rate[i], s=16, facecolors="white", edgecolors="black", linewidths=.7, zorder=4)
        ax.annotate(f"{t[i]:.3f} s", (t[i], rate[i]), xytext=offset, textcoords="offset points",
                    ha="right" if left else "left", va="bottom", zorder=5,
                    bbox=dict(boxstyle="square,pad=.1", facecolor="white", edgecolor="none"))
        selected.append(dict(native_left_frame=int(frame), time_s=float(t[i]), rate_s=float(rate[i])))
    extrema = []
    for first, last in spec.get("interval_search_windows", []):
        indices = np.flatnonzero((native >= first) & (native <= last))
        if not len(indices):
            raise ValueError("An interval search window does not overlap the strain cache")
        extrema.append(int(indices[np.argmin(rate[indices])]))
    intervals = []
    for a, b in zip(extrema[:-1], extrema[1:]):
        ax.annotate("", xy=(t[a], .09), xytext=(t[b], .09),
                    xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                    arrowprops=dict(arrowstyle="|-|", color=".45", lw=.7, mutation_scale=4))
        ax.text((t[a]+t[b])/2, .045, f"{t[b]-t[a]:.3f} s", transform=ax.get_xaxis_transform(),
                ha="center", va="center", color=".2")
        intervals.append(dict(start_time_s=float(t[a]), end_time_s=float(t[b]), interval_s=float(t[b]-t[a])))
    rectangular_axes(ax)
    return dict(panel=spec["panel"], kind="areal_strain_rate", samples=len(t),
                marker_times=selected, intervals=intervals, display_sigma_frames=data["display_sigma_frames"],
                rate_definition="+fps*trace(B)", refitted=False)


def draw_rate_regions(fig, rows, spec):
    """Overlay named cached regional rates with the matching speed-frame times."""
    if len(rows) != len(spec["regions"]) or len(rows) < 2:
        raise ValueError("Supply at least two named strain-rate regions")
    for row in rows[1:]:
        np.testing.assert_array_equal(row["time_s"], rows[0]["time_s"])
        np.testing.assert_array_equal(row["native_pairs"], rows[0]["native_pairs"])
    ax = axes_inches(fig, spec["axes_box_inches"])
    ax.axhline(0, color="#a0a0a0", lw=.55)
    native_times, pairs = rows[0]["time_s"], rows[0]["native_pairs"]
    unit = spec.get("time_unit", "s")
    if unit not in ("s", "ms"):
        raise ValueError("Plot time units must be s or ms")
    zero = float(spec.get("zero_time_s", 0))
    if not np.isfinite(zero):
        raise ValueError("Plot time zero must be finite")
    times = (native_times-zero)*(1000 if unit == "ms" else 1)
    xlim = spec.get(f"time_window_{unit}", [float(times[0]), float(times[-1])])
    markers = []
    for index, frame in enumerate(spec["marker_frames"]):
        found = np.flatnonzero(pairs[:, 0] == frame)
        if len(found) != 1:
            raise ValueError(f"Missing strain marker frame {frame}")
        time = float(times[found[0]])
        ax.axvline(time, color="#777777", lw=.6, ls=(0, (3, 2)), zorder=1,
            gid=f"speed-time-{frame}", label="3D speed times" if index == 0 else None)
        markers.append(dict(native_pair=[int(frame), int(frame+1)], time_s=float(native_times[found[0]]),
            display_time=time, display_unit=unit, visible=xlim[0] <= time <= xlim[1]))
    for row, region in zip(rows, spec["regions"]):
        ax.plot(times, row["rate_s"], color=region["color"], lw=1.1, label=region["label"],
                gid=f"strain_curve_{region['region']}")
    ax.set(xlabel=f"Time ({unit})", ylabel=r"Areal strain rate (s$^{-1}$)", xlim=xlim)
    if f"xticks_{unit}" in spec:
        ax.set_xticks(spec[f"xticks_{unit}"])
        ax.set_xlim(xlim)
    for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        ticks = axis.get_majorticklocs()
        axis.set_ticks(ticks[(ticks >= limits[0]) & (ticks <= limits[1])])
    handles, labels = ax.get_legend_handles_labels()
    order = [*range(1, len(rows)+1), 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False,
              loc="lower right", fontsize=5.5, labelspacing=.35)
    rectangular_axes(ax)
    ax.tick_params(length=2, pad=2)
    return dict(panel=spec["panel"], kind="areal_strain_rate_regions", samples=len(times),
        regions=[r["region"] for r in spec["regions"]], marker_times=markers,
        time_unit=unit, zero_time_s=zero, display_xlim=list(ax.get_xlim()), display_xticks=ax.get_xticks().tolist(),
        display_sigma_frames=rows[0]["display_sigma_frames"],
        rate_definition="positive divergence: +fps*trace(B)", refitted=False)


def load_speed(base, spec, manifest, output):
    source_panel = spec["source_panel"]
    section = next(s["section"] for s in manifest["sections"]
                   if s["section"]["name"] == spec.get("section", "xy"))
    mask_path = base / f"panel_{source_panel}/{section['name']}_masks.npz"
    with np.load(mask_path) as saved:
        if "region" in spec:
            names = saved["region_names"].tolist()
            if spec["region"] not in names:
                raise ValueError(f"Unknown speed-statistics region: {spec['region']}")
            mask = saved["masks"][names.index(spec["region"])].copy()
        else:
            mask = saved["masks"].any(0)
    first, last = manifest["dataset"]["analysis_frame_range"]
    frames = np.arange(first, last)
    records = [manifest["flow_fields"][str(frame)] for frame in frames]
    for record in records:
        _check_signature(record)
    sources = [signature(mask_path), *[signature(r["path"]) for r in records]]
    origin = manifest["dataset"].get("time_origin", first)
    contract = dict(sources=sources, section=section, native_left_frames=frames.tolist(),
                    fps=manifest["fps"], voxel_um=manifest["voxel_um"], time_origin=origin,
                    implementation=signature(flow_statistics.__file__))
    if "region" in spec:
        contract["region"] = spec["region"]
    npz_path, meta_path = output / "speed_statistics.npz", output / "speed_statistics.json"
    if npz_path.exists() and meta_path.exists() and json.loads(meta_path.read_text()).get("contract") == contract:
        with np.load(npz_path) as saved:
            data = {k: saved[k].copy() for k in saved.files}
    else:
        def fields():
            for record in records:
                volume = np.load(record["path"], mmap_mode="r")
                if volume.ndim != 4 or volume.shape[0] != 3:
                    raise ValueError("Speed statistics require cached full-volume 3D flow")
                yield cut_flow(volume, section)
        data = flow_statistics.summarize_speed(fields(), mask, native_left_frames=frames,
            fps=manifest["fps"], voxel_um=manifest["voxel_um"], time_origin=origin)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **data)
        write_csv(output / "speed_statistics.csv", [dict(left_native_frame=int(frame),
            right_native_frame=int(frame+1), time_mid_s=float(data["time_s"][i]),
            p25_um_s=float(data["p25_um_s"][i]), median_um_s=float(data["median_um_s"][i]),
            p75_um_s=float(data["p75_um_s"][i])) for i, frame in enumerate(frames)])
        write_json(meta_path, dict(contract=contract,
            definition="norm(dz,dy,dx)*voxel_um*fps, then spatial p25/median/p75 within the fixed RI mask",
            smoothing=False, optical_flow_recomputed=False))
    data.update(sources=sources, duration_s=(last-origin)/manifest["fps"])
    return data


def draw_speed(fig, data, spec, *, x_origin=0):
    box = spec.get("axes_box_inches", [.70, .55, 3.18, 2.10])
    ax = axes_inches(fig, [box[0]+x_origin, *box[1:]])
    t, median = data["time_s"], data["median_um_s"]
    color = spec["color"]
    ax.fill_between(t, data["p25_um_s"], data["p75_um_s"], color=color, alpha=.22, linewidth=0, step="mid")
    ax.plot(t, median, color=color, linewidth=.95, solid_capstyle="round", label=spec["label"],
            gid=f"speed_curve_{spec['panel']}")
    markers = []
    for j, frame in enumerate(spec.get("marker_frames", [])):
        matches = np.flatnonzero(data["native_pairs"][:, 0] == frame)
        if len(matches) != 1:
            raise ValueError(f"Selected speed frame {frame} missing from cache")
        i = int(matches[0])
        ax.scatter(t[i], median[i], s=12, facecolor="white", edgecolor="black", linewidth=.6, zorder=4)
        ax.annotate(f"{t[i]:.3f} s", (t[i], median[i]), xytext=(0 if j == 0 else 5, 9),
                    textcoords="offset points", ha="center" if j == 0 else "left", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.5))
        markers.append(dict(native_left_frame=int(frame), time_s=float(t[i]), median_um_s=float(median[i])))
    ax.set(xlim=spec.get("xlim", [0, data["duration_s"]]), ylim=spec.get("ylim", [0, 3.2]), xlabel="Time (s)",
           ylabel=r"Speed ($\mu$m$\cdot$s$^{-1}$)")
    ax.set_xticks(spec.get("xticks", np.arange(0, data["duration_s"], .4)))
    ax.set_yticks(spec.get("yticks", np.arange(.5, 3.01, .5)))
    ax.legend(loc="upper right", frameon=False, handlelength=1.7, handletextpad=.5)
    rectangular_axes(ax)
    return dict(panel=spec["panel"], kind="speed_statistics", samples=len(t), marker_times=markers,
                full_3d_magnitude=True, smoothing=False)


def draw_speed_regions(fig, rows, spec):
    """Aligned regional distributions with common scales and separate IQR bands."""
    if len(rows) != len(spec["regions"]) or len(rows) < 2:
        raise ValueError("Supply at least two named speed distributions")
    for row in rows[1:]:
        np.testing.assert_array_equal(row["time_s"], rows[0]["time_s"])
        np.testing.assert_array_equal(row["native_pairs"], rows[0]["native_pairs"])
    upper = 1.12*max(float(row["p75_um_s"].max()) for row in rows)
    ticks = plt.MaxNLocator(4).tick_values(0, upper)
    ylim = spec.get("ylim", [0, float(ticks[-1])])
    ticks = ticks[(ticks >= ylim[0]) & (ticks <= ylim[1])]
    times, fps = rows[0]["time_s"], float(rows[0]["fps"])
    xlim = spec.get("xlim", [float(times[0]-.5/fps), float(times[-1]+.5/fps)])
    step = float(spec.get("time_tick_step_s", .2))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Time tick spacing must be positive")
    xticks = spec.get("xticks", np.arange(np.ceil(xlim[0]/step)*step, xlim[1]+1e-10, step))
    boxes = spec["axes_boxes_inches"]
    if len(boxes) != len(rows):
        raise ValueError("Provide one axes box for each region")
    records = []
    axes = []
    for index, (data, region, box) in enumerate(zip(rows, spec["regions"], boxes)):
        settings = {**spec, **region, "panel": f"{spec['panel']}_{region['region']}",
                    "axes_box_inches": box, "ylim": ylim, "yticks": ticks,
                    "xlim": xlim, "xticks": xticks}
        records.append(draw_speed(fig, data, settings))
        ax = fig.axes[-1]
        if axes:
            ax.sharex(axes[0])
            ax.sharey(axes[0])
        axes.append(ax)
        if index < len(rows)-1:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
    return dict(panel=spec["panel"], kind="speed_regions", regions=records,
                common_ylim=list(ylim), common_xlim=list(axes[0].get_xlim()),
                layout="aligned rows with shared time and speed axes", overlapping_bands=False)


def prepare_speed_signal_cache(base, spec, manifest, *, config_base):
    """Materialize a local section-speed signal from manifest-registered 3D flow.

    Rebuilding reads the full-section vectors from existing flow fields, never
    solves new optical flow. A matching local cache can be used without opening
    any original volumes or external flow files.
    """
    settings = spec["speed_cache_parameters"]
    cache = {key: resolve(path, config_base) for key, path in spec["speed_cache"].items()}
    section = next(s["section"] for s in manifest["sections"]
                   if s["section"]["name"] == spec.get("section", "xy"))
    if section["plane"] != "xy" or section.get("rotation_cw", 0):
        raise ValueError("Section-speed caches require unrotated native XY sections")
    mask_path = base / f"panel_{spec['source_panel']}/{section['name']}_masks.npz"
    first, last = manifest["dataset"]["analysis_frame_range"]
    frames = np.arange(first, last)
    records = [manifest["flow_fields"][str(frame)] for frame in frames]
    origin = manifest["dataset"].get("time_origin", first)
    times = (frames-origin+.5)/manifest["fps"]
    contract = dict(flow_fields=records, mask=signature(mask_path), section=section,
        native_pairs=np.column_stack([frames, frames+1]).tolist(), fps=manifest["fps"],
        voxel_um=manifest["voxel_um"], time_origin=origin, parameters=settings,
        implementation=signature(flow_statistics.__file__))
    if all(path.exists() for path in cache.values()):
        saved = json.loads(cache["metadata"].read_text())
        if saved.get("contract") == contract:
            for record in saved["outputs"]:
                _check_signature(record)
            return
    print(f"Preparing local {section['name']} speed signal from {len(records)} cached 3D fields", flush=True)
    def fields():
        for record in records:
            _check_signature(record)
            field = np.load(record["path"], mmap_mode="r")
            if field.ndim != 4 or field.shape[0] != 3:
                raise ValueError("Speed signal requires cached full-volume 3D flow")
            yield cut_flow(field, section)
    speeds = flow_statistics.section_speed_signal(fields(), fps=manifest["fps"],
        voxel_um=manifest["voxel_um"], smoothing_sigma_t_y_x=settings["smoothing_sigma_t_y_x"])
    with np.load(mask_path) as saved:
        masks, names = saved["masks"].copy(), saved["region_names"].copy()
    if speeds.shape != (len(frames), *masks.shape[1:]):
        raise ValueError("Speed fields, native time range and masks do not align")
    quiet = (times >= settings["quiet_window_s"][0]) & (times <= settings["quiet_window_s"][1])
    if not quiet.any():
        raise ValueError("The quiet window must contain native samples")
    baseline_values = speeds[quiet][:, masks.any(0)]
    baseline, sd = float(baseline_values.mean()), float(baseline_values.std())
    for path in cache.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache["values"].with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, speeds)
    temporary.replace(cache["values"])
    np.savez_compressed(cache["alignment"], masks=masks, region_names=names, time_s=times,
                        native_pairs=np.column_stack([frames, frames+1]))
    write_json(cache["metadata"], dict(contract=contract,
        outputs=[signature(cache["values"]), signature(cache["alignment"])],
        fps=manifest["fps"], voxel_um=manifest["voxel_um"], native_z=section["index"],
        signal="full 3D dense speed at native XY section", grid_shape=list(masks.shape[1:]),
        speed_smoothing_sigma_t_y_x=settings["smoothing_sigma_t_y_x"],
        smoothing_mode="nearest", smoothing_truncate=4, smoothing_applications=1,
        quiet_window_s=settings["quiet_window_s"], baseline_um_s=baseline, sd_um_s=sd,
        threshold_um_s=baseline+3*sd,
        threshold_definition="pooled mean + 3 population SD, both fixed native masks",
        detection="regional coverage onset is defined in the plot config", optical_flow_recomputed=False))


def load_speed_distribution(base, spec, manifest, output, *, config_base):
    """Histogram an aligned cached onset signal using the image pipeline's masks."""
    section = next(s["section"] for s in manifest["sections"]
                   if s["section"]["name"] == spec.get("section", "xy"))
    if section["plane"] != "xy" or section.get("rotation_cw", 0):
        raise ValueError("Cached onset speeds require a native, unrotated XY section")
    mask_path = base / f"panel_{spec['source_panel']}/{section['name']}_masks.npz"
    cache = {key: resolve(path, config_base) for key, path in spec["speed_cache"].items()}
    sources = [signature(path) for path in [mask_path, *cache.values()]]
    metadata = json.loads(cache["metadata"].read_text())
    with np.load(mask_path) as saved:
        masks, names = saved["masks"].copy(), saved["region_names"].tolist()
    with np.load(cache["alignment"]) as saved:
        np.testing.assert_array_equal(saved["masks"], masks)
        times = saved["time_s"].copy()
    first, last = manifest["dataset"]["analysis_frame_range"]
    frames = np.arange(first, last)
    origin = manifest["dataset"].get("time_origin", first)
    fps = manifest["fps"]
    np.testing.assert_allclose(times, (frames-origin+.5)/fps, rtol=0, atol=1e-14)
    if (metadata["fps"] != fps or metadata["voxel_um"] != manifest["voxel_um"]
            or metadata["native_z"] != section["index"]):
        raise ValueError("Onset cache calibration or native section differs from the image pipeline")
    speeds = np.load(cache["values"], mmap_mode="r")
    if speeds.shape != (len(frames), *masks.shape[1:]):
        raise ValueError("Speed cache, masks and frame range must align exactly")
    selected = [names.index(region["region"]) for region in spec["regions"]]
    if len(set(selected)) != len(selected):
        raise ValueError("Select each region only once")
    lo, hi = spec["speed_bins_um_s"]["range"]
    width = float(spec["speed_bins_um_s"]["width"])
    if not np.isfinite([lo, hi, width]).all() or width <= 0 or hi <= lo:
        raise ValueError("Invalid speed bin range or width")
    bins = int(round((hi-lo)/width))
    if bins < 1 or not np.isclose(bins*width, hi-lo, rtol=0, atol=1e-10):
        raise ValueError("The speed bin width must divide the requested range")
    threshold = float(spec.get("threshold_um_s", metadata["threshold_um_s"]))
    zero = float(spec.get("zero_time_s", metadata["quiet_window_s"][1]))
    data = flow_statistics.speed_histograms(speeds, masks[selected],
        bin_edges_um_s=np.linspace(lo, hi, bins+1), threshold_um_s=threshold)
    data.update(time_s=times, time_ms=1000*(times-zero),
        time_bin_edges_ms=1000*((np.r_[frames, last]-origin)/fps-zero),
        native_pairs=np.column_stack([frames, frames+1]),
        region_names=np.array([names[index] for index in selected]), masks=masks[selected],
        fps=np.array(fps), voxel_um=np.array(manifest["voxel_um"]), zero_time_s=np.array(zero))
    onset_records = []
    if "regional_onset" in spec:
        settings = spec["regional_onset"]
        data.update(flow_statistics.regional_speed_onsets(data["above_threshold_percentage"], data["time_ms"],
            minimum_percentage=settings["minimum_percentage"],
            consecutive_frames=settings.get("consecutive_frames", 2), after_ms=settings.get("after_ms", 0)))
        for region, name in enumerate(data["region_names"]):
            index = int(data["regional_onset_index"][region])
            onset_records.append(dict(region=str(name), detected=index >= 0,
                onset_time_ms=float(data["time_ms"][index]) if index >= 0 else None,
                left_native_frame=int(frames[index]) if index >= 0 else None,
                above_threshold_percentage=float(data["above_threshold_percentage"][region, index]) if index >= 0 else None))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "speed_distribution.npz", **data)
    write_csv(output / "above_threshold.csv", [dict(region=str(name),
        left_native_frame=int(frames[t]), right_native_frame=int(frames[t]+1),
        time_s=float(times[t]), time_ms=float(data["time_ms"][t]), mask_pixels=int(data["mask_pixels"][r]),
        threshold_um_s=threshold, above_threshold_pixels=int(data["above_threshold_counts"][r, t]),
        above_threshold_percentage=float(data["above_threshold_percentage"][r, t]))
        for r, name in enumerate(data["region_names"]) for t in range(len(frames))])
    write_json(output / "speed_distribution.json", dict(
        sources=sources, implementation=signature(flow_statistics.__file__),
        histogram_definition="100 * count in speed bin / all fixed-region mask pixels at each time",
        bin_interval="left-inclusive, right-exclusive; last bin includes the right edge",
        threshold_definition=("fixed threshold from plot config" if "threshold_um_s" in spec
                              else metadata["threshold_definition"]), threshold_um_s=threshold,
        source_onset_threshold_um_s=metadata["threshold_um_s"],
        above_threshold_definition="instantaneous speed > threshold; no consecutive-frame condition",
        regional_onset=(dict(minimum_percentage=float(data["onset_minimum_percentage"]),
            consecutive_frames=int(data["onset_consecutive_frames"]), after_ms=float(data["onset_after_ms"]),
            definition="first native sample with sufficient above-threshold region coverage throughout the required run",
            same_pixels_required_across_frames=False, results=onset_records) if onset_records else None),
        onset_map_detection=metadata["detection"], quiet_window_s=metadata["quiet_window_s"],
        existing_speed_smoothing_sigma_t_y_x=metadata["speed_smoothing_sigma_t_y_x"],
        additional_smoothing=False, optical_flow_recomputed=False, zero_time_s=zero,
        section=section, region_names=data["region_names"].tolist(),
        mask_pixels=data["mask_pixels"].tolist(),
        speed_bin_edges_um_s=data["speed_bin_edges_um_s"].tolist(),
        time_range_ms=data["time_bin_edges_ms"][[0, -1]].tolist()))
    data["sources"] = sources
    return data


def draw_speed_distribution(fig, data, spec):
    """Two regional time/speed histograms with identical bins and color mapping."""
    boxes = spec["axes_boxes_inches"]
    if len(boxes) != len(spec["regions"]) or len(boxes) != len(data["region_names"]):
        raise ValueError("Supply one axes box per speed-distribution region")
    percentage = data["pixel_percentage"]
    vmax = float(spec.get("percentage_vmax", np.ceil(percentage.max())))
    gamma = float(spec.get("percentage_gamma", 1))
    if not np.isfinite([vmax, gamma]).all() or vmax < percentage.max() or gamma <= 0:
        raise ValueError("The common color scale must cover all percentages and have a positive gamma")
    norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
    xlim = spec.get("time_window_ms", data["time_bin_edges_ms"][[0, -1]].tolist())
    ylim = spec.get("speed_window_um_s", data["speed_bin_edges_um_s"][[0, -1]].tolist())
    axes = []
    hidden = []
    interpolation = spec.get("heatmap_interpolation", "cells")
    if interpolation not in ("cells", "nearest", "bilinear"):
        raise ValueError("Use cells, nearest or bilinear heatmap display")
    for index, (region, box) in enumerate(zip(spec["regions"], boxes)):
        ax = axes_inches(fig, box)
        if interpolation == "cells":
            image = ax.pcolormesh(data["time_bin_edges_ms"], data["speed_bin_edges_um_s"],
                percentage[index].T, cmap=spec.get("colormap", "magma"), norm=norm,
                shading="flat", edgecolors="none", antialiased=False,
                gid=f"speed_distribution_{region['region']}")
        else:
            for edges in (data["time_bin_edges_ms"], data["speed_bin_edges_um_s"]):
                if not np.allclose(np.diff(edges), np.diff(edges)[0], rtol=1e-10, atol=1e-10):
                    raise ValueError("Image interpolation requires uniform time and speed bins")
            image = ax.imshow(percentage[index].T, origin="lower", aspect="auto",
                extent=(*data["time_bin_edges_ms"][[0, -1]], *data["speed_bin_edges_um_s"][[0, -1]]),
                cmap=spec.get("colormap", "magma"), norm=norm,
                interpolation=interpolation, interpolation_stage="data", resample=True,
                gid=f"speed_distribution_{region['region']}")
        if spec.get("show_zero_line", True):
            ax.axvline(0, color="white", ls=(0, (3, 3)), lw=.65, alpha=.85)
        ax.axhline(float(data["threshold_um_s"]), color="white", ls=(0, (4, 2)), lw=.85)
        label_seconds = spec.get("absolute_time_labels_s", False)
        ax.set(xlim=xlim, ylim=ylim, xlabel="Time (s)" if label_seconds else "Time (ms)",
               ylabel=r"Speed magnitude ($\mu$m/s)")
        ax.set_xticks(spec.get("xticks_ms", plt.MaxNLocator(6).tick_values(*xlim)))
        if label_seconds:
            # Relabel the existing coordinates without moving data or onset markers.
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda value, _: f"{float(data['zero_time_s'])+value/1000:.1f}"))
        ax.set_yticks(spec.get("yticks_um_s", plt.MaxNLocator(5).tick_values(*ylim)))
        # set_ticks can expand limits; preserve the explicitly configured view.
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_title(region["label"], loc="left", pad=4)
        ax.set_title(rf"Threshold: {float(data['threshold_um_s']):.2f} $\mu$m/s", loc="right", pad=4)
        if axes:
            ax.sharex(axes[0])
            ax.sharey(axes[0])
        axes.append(ax)
        if index < len(boxes)-1:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        rectangular_axes(ax)
        keep = (data["speed_bin_edges_um_s"][:-1] >= ylim[0]) & (data["speed_bin_edges_um_s"][1:] <= ylim[1])
        excluded = 100-percentage[index][:, keep].sum(axis=1)
        hidden.append(dict(region=region["region"], max_percentage_outside_speed_view=float(np.maximum(0, excluded).max())))
    onset_annotations = []
    if "regional_onset" in spec:
        settings = spec["regional_onset"]
        reference = data["region_names"].tolist().index(settings["reference_region"])
        reference_time = float(data["regional_onset_time_ms"][reference])
        for index, (ax, region) in enumerate(zip(axes, spec["regions"])):
            onset = float(data["regional_onset_time_ms"][index])
            if np.isfinite(reference_time) and index != reference:
                ax.axvline(reference_time, color="white", lw=1, ls=(0, (4, 3)), alpha=.9,
                    gid=f"onset_reference_{region['region']}")
            if not np.isfinite(onset):
                continue
            ax.axvline(onset, color="white", lw=1.1, gid=f"regional_onset_{region['region']}")
            i = int(data["regional_onset_index"][index])
            coverage = float(data["above_threshold_percentage"][index, i])
            left = np.isfinite(reference_time) and onset < reference_time
            ax.text(onset+(-8 if left else 8), .89,
                f"{region.get('onset_label', region['label']+' onset')}: {onset:.1f} ms\n{coverage:.1f}% above threshold",
                transform=ax.get_xaxis_transform(), color="white", va="top",
                ha="right" if left else "left", linespacing=1.4,
                gid=f"onset_label_{region['region']}")
            if np.isfinite(reference_time) and onset < reference_time:
                ax.annotate("", (onset, .53), (reference_time, .53),
                    xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                    arrowprops=dict(arrowstyle="|-|", lw=.9, color="white", mutation_scale=4))
                ax.text((onset+reference_time)/2, .57, f"{reference_time-onset:.1f} ms earlier",
                    transform=ax.get_xaxis_transform(), color="white", ha="center", va="bottom")
            onset_annotations.append(dict(region=region["region"], time_ms=onset,
                above_threshold_percentage=coverage, left_native_frame=int(data["native_pairs"][i, 0])))
        fig.text(*spec["onset_definition_position"],
            rf"Onset: $\geq${float(data['onset_minimum_percentage']):g}% of region pixels with speed $>$ {float(data['threshold_um_s']):.2f} $\mu$m/s"
            "\n"+rf"for $\geq${int(data['onset_consecutive_frames'])} consecutive samples after 0 ms (sampling: {1000/float(data['fps']):.1f} ms).",
            ha="left", va="top", linespacing=1.5)
    bar = fig.colorbar(image, cax=axes_inches(fig, spec["colorbar_box_inches"]))
    bar.solids.set_rasterized(False)
    bar.set_ticks(spec.get("percentage_ticks", [0, 1, 5, 10, 20, vmax]))
    bar.set_label("Pixel percentage per speed bin (%)", labelpad=4)
    bar.ax.tick_params(length=3, width=.65, pad=2)
    bar.outline.set_linewidth(.7)
    fig.text(*spec["reference_text_position"], f"0 ms = {float(data['zero_time_s']):.3f} s",
             ha="left", va="center")
    return dict(panel=spec["panel"], kind="speed_distribution", samples=len(data["time_ms"]),
        regions=data["region_names"].tolist(), mask_pixels=data["mask_pixels"].tolist(),
        common_xlim_ms=list(xlim), common_ylim_um_s=list(ylim),
        common_color_scale=dict(normalization="power", gamma=gamma, vmin=0, vmax=vmax),
        colormap=spec.get("colormap", "magma"), heatmap_interpolation=interpolation,
        interpolation_used_for_analysis=False, regional_onset_annotations=onset_annotations,
        threshold_um_s=float(data["threshold_um_s"]), zero_time_s=float(data["zero_time_s"]),
        all_pixels_in_denominator=True, hidden_speed_tails=hidden,
        speed_bin_count=percentage.shape[-1], additional_smoothing=False)


def load_kymograph(base, spec, config):
    sampling_path = resolve(spec["sampling_cache"], base)
    manifest = _dataset_context(config, base, spec["source_panel"])
    section = next(s["section"] for s in manifest["sections"]
                   if s["section"]["name"] == spec.get("section", "xy"))
    if section["plane"] != "xy":
        raise ValueError("Kymograph coordinates must be native XY coordinates")
    with np.load(sampling_path) as saved:
        coordinates = saved["sample_coordinates_yx"].copy()
        distance, keep = saved["distance_um"].copy(), saved["focus_keep"].copy()
    native = np.array(manifest["native_context_frames"])
    source = manifest["filtered_volume"]
    volumes = np.load(source["path"], mmap_mode="r")
    if len(volumes) != len(native):
        raise ValueError("Kymograph cache must contain the complete native temporal context")
    values = kymograph.sample_kymograph(volumes, volume_frames=native, native_frames=native,
        z=section["index"], coordinates_yx=coordinates)
    if len(distance) != values.shape[1] or keep.dtype != bool or keep.shape != distance.shape or not keep.any():
        raise ValueError("Invalid focused kymograph coordinates")
    origin = manifest["dataset"].get("time_origin", int(native[0]))
    return dict(ri_contrast=values, time_s=(native-origin)/manifest["fps"], distance_um=distance,
                focus_keep=keep, native_frames=native, sample_coordinates_yx=coordinates,
                fps=np.array(manifest["fps"]), sources=[signature(sampling_path), source])


def draw_kymographs(fig, rows, spec):
    left, bottom, width = .70, .55, 3.18
    row_height, gap = .95, .40
    image = None
    for row, y in zip(rows, [bottom+row_height+gap, bottom]):
        ax = axes_inches(fig, [left, y, width, row_height])
        keep = row["focus_keep"]
        d, t = row["distance_um"][keep], row["time_s"]
        distance_step, time_step = np.diff(row["distance_um"])[0], 1/float(row["fps"])
        lo, hi = spec.get("ri_contrast_range", [0, .1])
        offset = spec.get("ri_baseline", 1.33)
        # Only the colorbar uses absolute RI; saved numerical RI contrast stays unchanged.
        image = ax.imshow(row["ri_contrast"][:, keep].T+offset, cmap="gray", vmin=lo+offset, vmax=hi+offset,
            extent=(t[0]-time_step/2, t[-1]+time_step/2, d[-1]+distance_step/2, d[0]-distance_step/2),
            aspect="auto", interpolation="nearest")
        ax.set(xlim=(t[0]-time_step/2, t[-1]+time_step/2), ylabel=r"Distance ($\mu$m)", xlabel="Time (s)")
        ax.set_xticks(np.arange(0, t[-1]+1e-8, .5))
        ax.set_yticks([4, 5, 6])
        ax.set_title(row["label"], loc="left", pad=2)
        rectangular_axes(ax)
    bar_ax = axes_inches(fig, [4.15, bottom+.40, .14, 1.55])
    bar = fig.colorbar(image, cax=bar_ax)
    bar.set_ticks([offset+lo, offset+hi])
    bar.set_ticklabels([f"{offset+lo:.2f}", f"{offset+hi:.2f}"])
    bar.set_label("Refractive index", labelpad=3)
    bar.ax.tick_params(length=0, pad=3)
    return dict(panel=spec["panel"], kind="kymograph", rows=[dict(label=r["label"],
        frames=len(r["time_s"]), focused_samples=int(r["focus_keep"].sum())) for r in rows],
        colorbar_absolute_ri=True, ri_baseline=offset, ri_contrast_range=[lo, hi])


def run(config_path, *, panels=None):
    config_path = Path(config_path).resolve()
    base, config = config_path.parent, json.loads(config_path.read_text())
    source_path = resolve(config.get("source_config", config_path), base)
    source_base = source_path.parent
    source_config = config if source_path == config_path else json.loads(source_path.read_text())
    specs = config.get("plots", [])
    if not specs or len({s["panel"] for s in specs}) != len(specs):
        raise ValueError("Each SVG plot must have a unique panel")
    if panels and not set(panels).issubset({s["panel"] for s in specs}):
        raise ValueError("Requested SVG panel is absent from config")
    configure_style(config.get("plot_style", {}), base)
    records = []
    for spec in specs:
        if spec.get("shared_time_axis", False):
            spec = {**config["plot_time_axis"], **spec}
        if panels and spec["panel"] not in panels:
            continue
        panel = spec["panel"]
        output = resolve(spec.get("output_directory", f"panel_{panel}"), base)
        output.mkdir(parents=True, exist_ok=True)
        kind = spec["kind"]
        fig = plt.figure(figsize=spec.get("page_inches", [3.96, 3.25]))
        if kind == "speed_regions":
            manifest = _dataset_context(source_config, source_base, spec["source_panel"])
            rows = [load_speed(source_base, {**spec, **region}, manifest, output / region["region"])
                    for region in spec["regions"]]
            record = draw_speed_regions(fig, rows, spec)
            sources = [source for row in rows for source in row["sources"]]
        elif kind == "speed_distribution":
            manifest = _dataset_context(source_config, source_base, spec["source_panel"],
                                        cache_only=spec.get("cache_only", False))
            if "speed_cache_parameters" in spec:
                prepare_speed_signal_cache(source_base, spec, manifest, config_base=base)
            data = load_speed_distribution(source_base, spec, manifest, output, config_base=base)
            record = draw_speed_distribution(fig, data, spec)
            sources = data["sources"]
        elif kind == "areal_strain_rate_regions":
            manifest = _dataset_context(source_config, source_base, spec["source_panel"],
                                        cache_only=spec.get("cache_only", False))
            if spec["marker_frames"] != manifest["dataset"]["flow_frames"]:
                raise ValueError("Strain markers must identify the existing speed-image pairs")
            rows = [load_rate(source_base, {**spec, **region}, manifest) for region in spec["regions"]]
            record = draw_rate_regions(fig, rows, spec)
            sources = [source for row in rows for source in row["sources"]]
            np.savez_compressed(output / "strain_rate_plot_data.npz",
                time_s=rows[0]["time_s"], native_pairs=rows[0]["native_pairs"],
                region_names=np.array([r["region"] for r in spec["regions"]]),
                raw_rate_s=np.array([row["raw_rate_s"] for row in rows]),
                displayed_rate_s=np.array([row["rate_s"] for row in rows]),
                display_time=(rows[0]["time_s"]-record["zero_time_s"])*(1000 if record["time_unit"] == "ms" else 1),
                display_time_unit=np.array(record["time_unit"]), zero_time_s=record["zero_time_s"],
                display_sigma_frames=rows[0]["display_sigma_frames"])
        elif kind == "kymograph":
            rows = [dict(**load_kymograph(source_base, row, source_config), label=row["label"])
                    for row in spec["rows"]]
            record = draw_kymographs(fig, rows, spec)
            sources = [source for row in rows for source in row["sources"]]
            for i, row in enumerate(rows, 1):
                np.savez_compressed(output / f"kymograph_e{i}_data.npz",
                                    **{k: v for k, v in row.items() if k not in ("sources", "label")})
        else:
            manifest = _dataset_context(source_config, source_base, spec["source_panel"])
            if kind == "areal_strain_rate":
                if spec["marker_frames"] != manifest["dataset"]["flow_frames"]:
                    raise ValueError("Strain markers must identify the existing speed-image pairs")
                data = load_rate(source_base, spec, manifest)
                record = draw_rate(fig, data, spec)
            elif kind == "speed_statistics":
                if spec["marker_frames"] != manifest["dataset"]["flow_frames"]:
                    raise ValueError("Speed markers must identify the existing speed-image pairs")
                data = load_speed(source_base, spec, manifest, output)
                record = draw_speed(fig, data, spec)
            else:
                raise ValueError(f"Unknown plot kind: {kind}")
            sources = data["sources"]
        panel_letter(fig, panel, settings=spec.get("panel_letter"))
        path = output / spec.get("filename", f"panel_{panel}_with_text.svg")
        save_svg(fig, path)
        plt.close(fig)
        for source in sources:
            _check_signature(source)
        record.update(output=signature(path), sources=sources, config=spec,
                      code=signature(__file__), editable_svg_text=True)
        write_json(output / "plot_manifest.json", record)
        records.append(record)
        print(f"Panel {panel}: {path}", flush=True)
    return records


def main(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--panels", nargs="+")
    args = parser.parse_args()
    run(args.config, panels=args.panels)


if __name__ == "__main__":
    main()
