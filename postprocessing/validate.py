"""Validate reconstructed clean figures against native fields and saved masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import tifffile

from . import masks, visualize
from .optical_flow import speed_um_s
from .volume_io import crop, cut_flow, cut_scalar, signature, write_json


def _check_geometry(flow, mask, *, voxel_um, fps, components):
    geometry = visualize.prepare_flow_geometry(flow, mask, voxel_um=voxel_um, fps=fps, components=components)
    height, width = mask.shape
    factor = 430/(width-1)
    columns = np.linspace(0, width-1, 431)
    rows = columns if height == width else np.arange(int(np.floor((height-1)*factor))+1)/factor
    rr, cc = np.meshgrid(rows, columns, indexing="ij")
    sample = lambda array: ndi.map_coordinates(array, [rr, cc], order=1, prefilter=False)
    domain = ((sample(mask.astype(float)) > .999)
              & (sample(np.hypot(flow[components[0]], flow[components[1]])*voxel_um*fps) >= .25))

    def check(points):
        points = points*factor
        inside = ndi.map_coordinates(domain.astype(float), [points[:, 1], points[:, 0]],
            order=1, mode="constant", cval=0., prefilter=False)
        assert inside.min() >= .998, "Arrow leaves the tissue/speed domain"

    for path in geometry["paths"]:
        assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum()*factor >= 26.-1e-10
        check(path)
        check((path[:-1]+path[1:])/2)
    for vertices in geometry["triangles"]:
        step = max(2, int(np.ceil(np.linalg.norm(vertices-np.roll(vertices, 1, axis=0), axis=1).max()*factor/.3)))
        i, j = np.indices((step+1, step+1))
        take = i+j <= step
        check(vertices[0]+i[take, None]/step*(vertices[1]-vertices[0])
              + j[take, None]/step*(vertices[2]-vertices[0]))
    stats = geometry["statistics"]
    if stats["filled_triangles"]:
        assert stats["arrow_field_min_cosine"] > .999999
    return stats


def validate(config_path):
    config_path = Path(config_path).resolve()
    base = config_path.parent
    config = json.loads(config_path.read_text())
    report = dict(status="passed", panels={})
    for dataset in config["datasets"]:
        panel = dataset["panel"]
        out = base / f"panel_{panel}"
        manifest = json.loads((out / "all_manifest.json").read_text())
        assert manifest["dataset"] == dataset
        assert manifest["arrow_visibility"] == visualize.visibility_policy()
        for record in manifest["source_signatures"] + [manifest["filtered_volume"]]:
            assert signature(record["path"]) == record
        for record in manifest["flow_fields"].values():
            assert signature(record["path"]) == {k: record[k] for k in ("path", "bytes", "mtime_ns")}
        checks = []
        volume = np.load(manifest["filtered_volume"]["path"], mmap_mode="r")
        cache = dataset.get("cache", {}).get("kalman")
        first = int(cache["first_frame"]) if cache else None
        for section, meta in zip(dataset["sections"], manifest["sections"]):
            name = section["name"]
            with np.load(out / f"{name}_masks.npz") as saved:
                region_masks, names, mean = saved["masks"], saved["region_names"], saved["mean_ri"]
            masks.validate_masks(region_masks, mean.shape)
            if "source" not in meta["mask_generation"]:
                method = meta["mask_generation"]
                expected, expected_names = masks.generate_masks(mean, threshold=method["threshold"],
                    sigma=method["sigma"], regions=method["regions"])
                np.testing.assert_array_equal(region_masks, expected)
                np.testing.assert_array_equal(names, expected_names)
                for mask in region_masks:
                    assert ndi.label(mask)[1] == 1
                    np.testing.assert_array_equal(mask, ndi.binary_fill_holes(mask))
            components = (1, 2) if section["plane"] == "xy" else (0, 2)
            display_mask = region_masks.any(0)
            if "display_mask" in section:
                method = meta["display_mask_generation"]
                with np.load(out / f"{name}_display_mask.npz") as saved:
                    display_mask = saved["mask"]
                if "source" not in method:
                    expected, _ = masks.generate_masks(mean, threshold=method["threshold"],
                        sigma=method["sigma"], regions=method["regions"])
                    np.testing.assert_array_equal(display_mask, expected.any(0))
                else:
                    assert signature(method["source"]["path"]) == method["source"]
                assert int(display_mask.sum()) == meta["display_mask_pixels"]
            with np.load(out / f"{name}_section_data.npz") as saved:
                expected_pairs = np.array([[f, f+1] for f in dataset["flow_frames"]])
                np.testing.assert_array_equal(saved["native_pairs"], expected_pairs)
                np.testing.assert_array_equal(saved["mask"], crop(display_mask, section.get("crop")))
                for index, (frame, _) in enumerate(expected_pairs):
                    source = np.load(manifest["flow_fields"][str(frame)]["path"], mmap_mode="r")
                    flow = crop(cut_flow(source, section), section.get("crop"))
                    np.testing.assert_array_equal(saved["dense_displacement_zyx"][index], flow)
                    np.testing.assert_array_equal(saved["speed_um_s"][index], speed_um_s(flow,
                        voxel_um=manifest["voxel_um"], fps=manifest["fps"]))
                    stats = _check_geometry(flow, saved["mask"], voxel_um=manifest["voxel_um"],
                        fps=manifest["fps"], components=components)
                    assert stats == meta["geometry_statistics"][index]
                if first is not None:
                    expected_ri = crop(cut_scalar(volume[dataset["ri_frame"]-first], section), section.get("crop"))
                    np.testing.assert_array_equal(saved["ri"], expected_ri)
            if section["plane"] == "xy" and dataset.get("strain_rate", True):
                with np.load(out / f"{name}_strain_rate/areal_strain_rate.npz") as rate:
                    np.testing.assert_array_equal(rate["masks"], region_masks)
                    np.testing.assert_allclose(rate["rate_s"], manifest["fps"]*np.trace(rate["B"], axis1=2, axis2=3), atol=0, rtol=0)
                    for j in (0, len(rate["native_pairs"])-1):
                        frame = int(rate["native_pairs"][j, 0])
                        flow = cut_flow(np.load(manifest["flow_fields"][str(frame)]["path"], mmap_mode="r"), section)
                        for i, mask in enumerate(region_masks):
                            y, x = np.nonzero(mask)
                            positions = np.column_stack([x, y]).astype(float)
                            positions -= positions.mean(0)
                            values = np.column_stack([flow[2, y, x], flow[1, y, x]]).astype(float)
                            values -= values.mean(0)
                            independent = np.linalg.solve(positions.T@positions, positions.T@values).T
                            np.testing.assert_allclose(rate["B"][i, j], independent, atol=1e-10, rtol=0)
            checks.append(dict(section=name, masks_verified=True, native_ri_flow_speed_verified=True,
                geometry_length_boundary_direction_verified=True,
                heads=[g["filled_triangles"] for g in meta["geometry_statistics"]]))
        files = sorted(out.glob("*.tif"))
        assert files and (out / f"panel_{panel}_clean.tif").exists()
        for path in files:
            assert path.name.endswith("_clean.tif")
            with tifffile.TiffFile(path) as handle:
                page = handle.pages[0]
                assert page.dtype == np.uint8 and page.shape[-1] == 3
                assert page.tags["XResolution"].value == (manifest["visualize"]["dpi"], 1)
        assert not list(out.glob("*.png")) + list(out.glob("*.svg")) + list(out.glob("*.pdf"))
        report["panels"][panel] = dict(clean_rgb_tiffs=len(files), sections=checks,
            source_and_cache_identities_unchanged=True, independent_affine_covariance_fits_verified=True)
    write_json(base / "validation_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.config), indent=2))
