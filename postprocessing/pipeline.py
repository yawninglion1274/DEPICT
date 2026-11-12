"""Config-driven volume -> Kalman -> masks/flow -> clean TIFF/strain pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from DEPICT.postprocessing import areal_strain_rate, kalman_filter, masks, optical_flow, visualize
from DEPICT.postprocessing.volume_io import (VolumeSeries, crop, cut_flow, cut_scalar, digest, resolve,
                                            signature, write_json)


def _external_cache(config, base, kind):
    cached = config.get("cache", {}).get(kind)
    if not cached:
        return None
    cached = dict(cached)
    for name in ("path", "pattern", "done"):
        if cached.get(name):
            cached[name] = str(resolve(cached[name], base))
    return cached


def _section_mean(series, filtered, section, parameters, output):
    contract = dict(inputs=series.signatures, context_frames=series.frames.tolist(),
        filtered_volume=signature(filtered.path), kalman=parameters,
        coordinates={k: section.get(k, 0) for k in ("plane", "index", "rotation_cw")})
    path = output / "cache/mean_sections" / (digest(contract)[:20]+".npy")
    if path.exists():
        return np.load(path)
    # Completed exports also contain this mean; reuse only matching provenance.
    previous = output / "all_manifest.json"
    if previous.exists():
        metadata = json.loads(previous.read_text())
        if (metadata["source_signatures"] == series.signatures and metadata["kalman"] == parameters
                and metadata["filtered_volume"] == contract["filtered_volume"]):
            for old in metadata["sections"]:
                coordinates = {k: old["section"].get(k, 0) for k in ("plane", "index", "rotation_cw")}
                saved = output / f"{old['section']['name']}_masks.npz"
                if coordinates == contract["coordinates"] and saved.exists():
                    with np.load(saved) as arrays:
                        mean = arrays["mean_ri"].copy()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, mean)
                    return mean
    if section["plane"] == "xy":
        plane = np.load(filtered.context_planes[int(section["index"])], mmap_mode="r")
        if section.get("rotation_cw", 0):
            plane = np.rot90(plane, -1, axes=(-2, -1))
        mean = plane.mean(0)
    elif np.array_equal(filtered.frames, series.frames):
        mean = cut_scalar(filtered.array, section).mean(0)
    else:
        # XZ masks still use full context when storage contains only an event window.
        rows = []
        for z in range(series.shape[0]):
            plane_mean = kalman_filter.kalman_rts_smooth(series.plane(z), **parameters).mean(0)
            rows.append(plane_mean[:, int(section["index"])][::-1] if section.get("rotation_cw", 0)
                        else plane_mean[int(section["index"]), :])
        mean = np.array(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mean)
    return mean


def _section_masks(mean, section, settings, base):
    local = {**settings, **section.get("mask", {})}
    if "path" in local:
        path = resolve(local["path"], base)
        if path.suffix == ".npz":
            with np.load(path) as saved:
                array = saved[local.get("key", "masks")].copy()
        else:
            array = np.load(path)
        array = masks.validate_masks(array, mean.shape)
        names = local.get("region_names", [f"region_{i}" for i in range(len(array))])
        if len(names) != len(array) or len(set(names)) != len(names):
            raise ValueError("Provide unique names for each supplied mask")
        return array, names, dict(source=signature(path), method="explicit supplied masks")
    array, names = masks.generate_masks(mean, threshold=local.get("threshold", .007),
        sigma=local.get("sigma", 1.), regions=local.get("regions"))
    return array, names, dict(method="Gaussian(temporal-mean Kalman RI), threshold, largest component, hole fill",
        threshold=local.get("threshold", .007), sigma=local.get("sigma", 1.),
        regions=local.get("regions"), full_temporal_context=True, flow_independent=True)


def _section_display_mask(mean, section, settings, base, region_masks):
    if "display_mask" not in section:
        return region_masks.any(0), dict(method="union of analysis masks")
    inherited = {key: section.get("mask", {})[key] for key in ("threshold", "sigma")
                 if key in section.get("mask", {})}
    display_section = dict(mask={**inherited, **section["display_mask"]})
    display_masks, _, method = _section_masks(mean, display_section, settings, base)
    return display_masks.any(0), method


def _refresh_completed_render(output, current):
    """Keep completed-run provenance current after a display-only mask change."""
    path = output / "all_manifest.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text())

    def analysis_dataset(dataset):
        return {**dataset, "sections": [{k: v for k, v in s.items() if k != "display_mask"}
                                         for s in dataset["sections"]]}

    if analysis_dataset(previous["dataset"]) != analysis_dataset(current["dataset"]):
        return
    for key in ("source_signatures", "native_context_frames", "filtered_volume", "fps", "voxel_um",
                "kalman", "optical_flow", "masks"):
        if previous[key] != current[key]:
            return
    for old, new in zip(previous["sections"], current["sections"]):
        if any(old[key] != new[key] for key in ("mask_generation", "region_names", "mask_pixels")):
            return
    fields = {**previous["flow_fields"], **current["flow_fields"]}
    for key in ("dataset", "sections", "visualize", "arrow_visibility", "library_code"):
        previous[key] = current[key]
    previous["flow_fields"] = fields
    previous["latest_render_manifest"] = signature(output / "render_manifest.json")
    write_json(path, previous)


def _rate_frames(dataset, series):
    if "analysis_frame_range" in dataset:
        first, stop = map(int, dataset["analysis_frame_range"])
        if stop <= first:
            raise ValueError("analysis_frame_range must contain at least two volumes")
        frames = np.arange(first, stop)
    else:
        frames = series.frames[:-1]
    if not np.isin(frames, series.frames).all() or not np.isin(frames+1, series.frames).all():
        raise ValueError("Rate frame pairs are not present in the native input series")
    return frames


def run(config_path, *, panels=None, stage="all", workers=1, device="auto", use_external_cache=True):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    base = config_path.parent
    if config.get("schema_version") != 1:
        raise ValueError("Expected config schema_version=1")
    datasets = config["datasets"]
    if not datasets or len({d["panel"] for d in datasets}) != len(datasets):
        raise ValueError("Each dataset must map to a distinct output panel")
    if panels and not set(panels).issubset({d["panel"] for d in datasets}):
        raise ValueError("Requested panel is absent from config")
    if stage not in ("all", "kalman", "masks", "flow", "render", "rates"):
        raise ValueError(f"Unknown pipeline stage: {stage}")
    settings = config.get("settings", {})
    kalman = {"map_smooth": 3., "qr_scale": 2., **settings.get("kalman", {})}
    flow_parameters = {**optical_flow.DEFAULTS, **settings.get("optical_flow", {})}
    render_settings = {**visualize.DEFAULTS, **settings.get("visualize", {})}
    mask_settings = {"threshold": .007, "sigma": 1., **settings.get("mask", {})}
    results = []
    for dataset in datasets:
        panel = dataset["panel"]
        if panels and panel not in panels:
            continue
        if panel not in ("a", "b"):
            raise ValueError("Figure reconstruction currently supports panel a and b")
        output = base / f"panel_{panel}"
        output.mkdir(parents=True, exist_ok=True)
        series = VolumeSeries(dataset["source"], base)
        fps = float(dataset.get("fps", settings.get("fps", 32.5)))
        voxel_um = float(dataset.get("voxel_um", settings.get("voxel_um", .0900811958)))
        if not np.isfinite(fps) or fps <= 0 or not np.isfinite(voxel_um) or voxel_um <= 0:
            raise ValueError("fps and voxel_um must be positive")
        selected = np.array(dataset["flow_frames"], dtype=int)
        if not len(selected) or len(set(selected)) != len(selected):
            raise ValueError("Select distinct native left frames for the speed tiles")
        rate_frames = _rate_frames(dataset, series)
        do_rates = stage in ("rates", "all") and dataset.get("strain_rate", True)
        used_pairs = np.unique(np.r_[selected, rate_frames if dataset.get("strain_rate", True) else []]).astype(int)
        stored_frames = np.unique(np.r_[dataset["ri_frame"], used_pairs, used_pairs+1]).astype(int)
        sections = dataset["sections"]
        if not sections or len({s["name"] for s in sections}) != len(sections):
            raise ValueError("Sections need unique names")
        xy_planes = [int(s["index"]) for s in sections if s["plane"] == "xy"]
        if not xy_planes:
            xy_planes = [series.shape[0]//2]
        print(f"Panel {panel}: {dataset['id']}; {len(series.frames)} context volumes", flush=True)
        filtered = kalman_filter.prepare(series, output / "cache/kalman", stored_frames=stored_frames,
            xy_planes=xy_planes, parameters=kalman, workers=workers,
            external_cache=_external_cache(dataset, base, "kalman") if use_external_cache else None)
        if stage == "kalman":
            continue
        store = optical_flow.FlowStore(filtered, output / "cache/flow", parameters=flow_parameters, device=device,
            external_cache=_external_cache(dataset, base, "flow") if use_external_cache else None)
        if stage == "flow":
            for frame in used_pairs:
                store.get(frame)
            write_json(output / "flow_manifest.json", dict(parameters=flow_parameters, fields=store.used))
            continue
        panel_rows, section_meta = [], []
        origin = int(dataset.get("time_origin", int(series.frames[0])))
        for section in sections:
            name = section["name"]
            print(f"Panel {panel}/{name}: preparing fixed RI mask", flush=True)
            mean = _section_mean(series, filtered, section, kalman, output)
            region_masks, region_names, mask_meta = _section_masks(mean, section, mask_settings, base)
            np.savez_compressed(output / f"{name}_masks.npz", masks=region_masks,
                region_names=np.array(region_names), mean_ri=mean, context_native_frames=series.frames)
            full_display_mask, display_mask_meta = _section_display_mask(
                mean, section, mask_settings, base, region_masks)
            if "display_mask" in section:
                np.savez_compressed(output / f"{name}_display_mask.npz", mask=full_display_mask)
            components = (1, 2) if section["plane"] == "xy" else (0, 2)
            record = dict(section=section, mask_generation=mask_meta,
                          region_names=region_names, mask_pixels=region_masks.sum(axis=(1, 2)).tolist(),
                          display_mask_generation=display_mask_meta,
                          display_mask_pixels=int(full_display_mask.sum()))
            print(f"Panel {panel}/{name}: masks ready", flush=True)
            if stage in ("all", "render"):
                bounds = section.get("crop")
                ri = crop(cut_scalar(filtered.volume(dataset["ri_frame"]), section), bounds)
                display_mask = crop(full_display_mask, bounds)
                if not display_mask.any():
                    raise ValueError(f"No tissue remains inside the display crop for {panel}/{name}")
                coordinate = ("z" if section["plane"] == "xy" else "y") + str(section["index"])
                images = [visualize.render_clean(ri, limits=render_settings["ri_range"],
                    tile_width_px=render_settings["tile_width_px"], dpi=render_settings["dpi"])]
                visualize.save_clean_tiff(output / f"ri_frame{dataset['ri_frame']:03d}_{name}_{coordinate}_clean.tif",
                                          images[0], dpi=render_settings["dpi"])
                flows, speeds, statistics = [], [], []
                for frame in selected:
                    flow = crop(cut_flow(store.get(frame), section), bounds)
                    rgb, geometry, speed = visualize.render_speed(flow, display_mask, voxel_um=voxel_um,
                        fps=fps, components=components, settings=render_settings)
                    images.append(rgb)
                    flows.append(flow)
                    speeds.append(speed)
                    statistics.append(geometry["statistics"])
                    visualize.save_clean_tiff(output / f"speed_{frame:03d}_{frame+1:03d}_{name}_{coordinate}_clean.tif",
                                              rgb, dpi=render_settings["dpi"])
                row = visualize.join_tiles(images, gap_px=render_settings["gap_px"])
                visualize.save_clean_tiff(output / f"panel_{panel}_{name}_clean.tif", row, dpi=render_settings["dpi"])
                panel_rows.append(row)
                np.savez_compressed(output / f"{name}_section_data.npz", ri=ri, mask=display_mask,
                    dense_displacement_zyx=np.array(flows), speed_um_s=np.array(speeds),
                    native_pairs=np.column_stack([selected, selected+1]),
                    time_s=(selected-origin+.5)/fps, voxel_um=voxel_um, fps=fps)
                record.update(geometry_statistics=statistics, cropped_shape=list(ri.shape),
                    speed_saturation_fraction=[float(np.mean(s[display_mask] > render_settings["speed_range"][1]))
                                               for s in speeds])
            if do_rates and section["plane"] == "xy":
                fields = (cut_flow(store.get(frame), section) for frame in rate_frames)
                areal_strain_rate.analyze_and_save(fields, region_masks, native_left_frames=rate_frames,
                    output=output / f"{name}_strain_rate", fps=fps, time_origin=origin,
                    region_names=region_names, components=components)
            section_meta.append(record)
        if panel_rows:
            rgb = visualize.join_tiles(panel_rows, gap_px=render_settings["gap_px"], axis=0)
            visualize.save_clean_tiff(output / f"panel_{panel}_clean.tif", rgb, dpi=render_settings["dpi"])
        manifest = dict(dataset=dataset, source_signatures=series.signatures, stage=stage,
            native_context_frames=series.frames.tolist(), filtered_volume=signature(filtered.path),
            fps=fps, voxel_um=voxel_um, kalman=kalman, optical_flow=flow_parameters,
            masks=mask_settings, visualize=render_settings, arrow_visibility=visualize.visibility_policy(),
            sections=section_meta, flow_fields=store.used,
            numerical_units="native voxel/frame; speed um/s; areal strain rate 1/s",
            library_code=[signature(p) for p in sorted(Path(__file__).parent.glob("*.py"))])
        write_json(output / f"{stage}_manifest.json", manifest)
        if stage == "render":
            _refresh_completed_render(output, manifest)
        results.append(manifest)
        print(f"Panel {panel}: {stage} complete -> {output}", flush=True)
    return results


def main(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--panels", nargs="+", choices=["a", "b"])
    parser.add_argument("--stage", choices=["all", "kalman", "masks", "flow", "render", "rates"], default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-external-cache", action="store_true",
                        help="Recompute from native inputs into local caches")
    args = parser.parse_args()
    run(args.config, panels=args.panels, stage=args.stage, workers=args.workers, device=args.device,
        use_external_cache=not args.no_external_cache)


if __name__ == "__main__":
    main()
