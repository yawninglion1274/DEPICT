"""Numerical and raw-input integration checks for the shared pipeline."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import ndimage as ndi
import tifffile

from DEPICT.postprocessing import areal_strain_rate as strain
from DEPICT.postprocessing import kalman_filter, masks, optical_flow, pipeline, visualize, _streamlines
from DEPICT.postprocessing.volume_io import cut_flow, cut_scalar


class NumericalTests(unittest.TestCase):
    def test_kalman_matches_established_implementation(self):
        path = Path(__file__).resolve().parents[3] / "helper/generate_video/make_timeseries_slice_avi.py"
        if not path.exists():
            self.skipTest("Original reference source is not installed")
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kalman_rts_smooth")
        namespace = dict(np=np, gaussian_filter=ndi.gaussian_filter)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        rng = np.random.default_rng(41)
        stack = rng.normal(.01, .003, (9, 17, 19)).astype(np.float32)
        expected = namespace["kalman_rts_smooth"](stack)
        actual = kalman_filter.kalman_rts_smooth(stack)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        with self.assertRaises(ValueError):
            kalman_filter.kalman_rts_smooth(stack[:1])

    def test_tvl1_matches_skimage_and_preserves_axis_order(self):
        import torch
        from skimage.registration import optical_flow_tvl1
        torch.set_num_threads(2)
        rng = np.random.default_rng(418)
        ref = ndi.gaussian_filter(rng.random((40, 41, 42), dtype=np.float32), 1.)
        mov = ndi.shift(ref, (.6, -.8, 1.1), order=1, mode="nearest")
        actual = optical_flow.dense_flow(ref*.06, mov*.06, device="cpu")
        expected = optical_flow_tvl1(ref, mov, attachment=15., tightness=.3,
                                    num_warp=5, num_iter=10, tol=1e-4, prefilter=False)
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=0)
        np.testing.assert_allclose(np.median(actual[:, 6:-6, 6:-6, 6:-6], axis=(1, 2, 3)),
                                   [.6, -.8, 1.1], atol=.25, rtol=0)

    def test_mask_uses_ri_and_respects_disjoint_regions(self):
        mean = np.zeros((40, 50), np.float32)
        mean[3:17, 4:20] = .02
        mean[8:11, 10:13] = 0
        mean[25:37, 25:46] = .02
        mean[1, 1] = .1
        regions = [dict(name="upper", roi=[0, 20, 0, 50]), dict(name="lower", roi=[20, 40, 0, 50])]
        result, names = masks.generate_masks(mean, threshold=.007, sigma=0, regions=regions)
        self.assertEqual(names, ["upper", "lower"])
        self.assertTrue(result[0, 9, 11])
        self.assertFalse(result[:, 1, 1].any())
        self.assertFalse((result[0] & result[1]).any())
        for mask in result:
            self.assertEqual(ndi.label(mask)[1], 1)
            np.testing.assert_array_equal(mask, ndi.binary_fill_holes(mask))

    def test_affine_rate_sign_all_mask_pixels_and_time(self):
        yy, xx = np.indices((19, 23))
        mask = ((xx-8)**2/110+(yy-7)**2/50 < 1) & (xx > yy/3)
        B = np.array([[.02, .07], [-.03, -.005]])
        flow = np.stack([np.zeros_like(xx), 3+B[1, 0]*xx+B[1, 1]*yy,
                         -2+B[0, 0]*xx+B[0, 1]*yy])
        fit = strain.fit_areal_strain_rate(flow, mask)
        np.testing.assert_allclose(fit["B"], B, atol=1e-14, rtol=0)
        self.assertAlmostEqual(fit["rate_s"], 32.5*np.trace(B))
        with self.assertRaises(ValueError):
            strain.fit_areal_strain_rate(flow, np.stack([mask, mask]))
        with tempfile.TemporaryDirectory() as tmp:
            strain.analyze_and_save([flow, flow], mask, native_left_frames=[40, 41],
                                    output=tmp, time_origin=40, region_names=["cell"])
            with np.load(Path(tmp) / "areal_strain_rate.npz") as saved:
                np.testing.assert_array_equal(saved["time_s"], np.array([.5, 1.5])/32.5)
                np.testing.assert_array_equal(saved["masks"][0], mask)
                self.assertEqual(saved["rate_s"].shape, (1, 2))

    def test_rotation_transforms_vector_endpoints_and_xz(self):
        scalar = np.arange(2*5*7).reshape(2, 5, 7)
        flow = np.stack([scalar+1, scalar+2, scalar+3])
        section = dict(plane="xy", index=1, rotation_cw=90)
        rotated = cut_flow(flow, section)
        transform = lambda p: np.array([p[0], p[2], 4-p[1]])
        for y, x in ((0, 0), (2, 3), (4, 6)):
            p = np.array([1, y, x])
            np.testing.assert_array_equal(rotated[:, x, 4-y], transform(p+flow[:, 1, y, x])-transform(p))
        section = dict(plane="xz", index=3, rotation_cw=90)
        np.testing.assert_array_equal(cut_scalar(scalar, section), scalar[:, :, 3][:, ::-1])
        expected = flow[:, :, :, 3][..., ::-1]
        np.testing.assert_array_equal(cut_flow(flow, section), np.stack([expected[0], expected[2], -expected[1]]))

    def test_three_rules_and_rectangular_streamlines(self):
        field = np.zeros((431, 431, 2))
        field[..., 1] = 1.
        small = np.zeros((431, 431), bool)
        small[199:206, 99:113] = True
        self.assertLess(small.sum(), 100)
        g = _streamlines.build_geometry(field, small, triangle_length=3., minimum_path_length=8.)
        self.assertTrue(g["triangles"])
        short = np.zeros((431, 431), bool)
        short[100:180, 100:120] = True
        self.assertFalse(_streamlines.build_geometry(field, short, triangle_length=3.)["paths"])
        flow = np.zeros((3, 60, 150), np.float32)
        flow[2] = 1.
        mask = np.ones((60, 150), bool)
        g = visualize.prepare_flow_geometry(flow, mask, voxel_um=.09, fps=32.5, components=(0, 2))
        self.assertTrue(g["triangles"])
        self.assertGreater(g["statistics"]["arrow_field_min_cosine"], .999999)
        self.assertGreaterEqual(g["statistics"]["minimum_retained_path_length"], 26.)
        self.assertFalse(g["statistics"]["component_area_filter"])
        self.assertFalse(g["statistics"]["curvature_filter"])
        self.assertFalse(g["statistics"]["head_spacing_filter"])
        for points in g["paths"] + g["triangles"]:
            self.assertTrue((points[:, 0] >= 0).all() and (points[:, 0] < 150).all())
            self.assertTrue((points[:, 1] >= 0).all() and (points[:, 1] < 60).all())


class IntegrationTests(unittest.TestCase):
    def test_raw_volume_pipeline_without_external_caches(self):
        import torch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            zz, yy, xx = np.indices((6, 16, 20))
            raw = np.array([.002+.025*np.exp(-((xx-9-.2*t)**2/25+(yy-8)**2/20+(zz-3)**2/10))
                            for t in range(5)], dtype=np.float32)
            source = directory / "input.npy"
            np.save(source, raw)
            before = source.read_bytes()
            config = dict(schema_version=1, settings=dict(optical_flow=dict(num_warp=2, num_iter=2),
                visualize=dict(tile_width_px=200)), datasets=[dict(id="synthetic", panel="a",
                source=dict(path="input.npy", first_frame=101), ri_frame=102, flow_frames=[101, 103],
                analysis_frame_range=[101, 105], sections=[dict(name="xy", plane="xy", index=3,
                    mask=dict(regions=[dict(name="upper", roi=[0, 7, 0, 20]),
                                       dict(name="lower", roi=[9, 16, 0, 20])])),
                dict(name="xz", plane="xz", index=8, mask=dict(threshold=.006))])])
            path = directory / "config.json"
            path.write_text(json.dumps(config))
            result = pipeline.run(path, device="cpu", use_external_cache=False)
            mask_stage = pipeline.run(path, stage="masks", device="cpu", use_external_cache=False)
            self.assertEqual(result[0]["filtered_volume"], mask_stage[0]["filtered_volume"])
            self.assertFalse(mask_stage[0]["flow_fields"])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(result), 1)
            out = directory / "panel_a"
            self.assertTrue((out / "panel_a_clean.tif").exists())
            self.assertFalse(list(out.glob("*.png")) + list(out.glob("*.pdf")))
            with tifffile.TiffFile(out / "panel_a_clean.tif") as handle:
                self.assertEqual(handle.pages[0].dtype, np.uint8)
                self.assertEqual(handle.pages[0].shape[-1], 3)
                self.assertEqual(handle.pages[0].tags["XResolution"].value, (300, 1))
            with np.load(out / "xy_strain_rate/areal_strain_rate.npz") as saved:
                self.assertEqual(saved["rate_s"].shape, (2, 4))
            self.assertTrue(all(not f["explicit_external_cache"] for f in result[0]["flow_fields"].values()))
            rate_path = out / "xy_strain_rate/areal_strain_rate.npz"
            before_rate = rate_path.read_bytes()
            config["datasets"][0]["sections"][0]["display_mask"] = dict(regions=None)
            path.write_text(json.dumps(config))
            rendered = pipeline.run(path, stage="render", device="cpu", use_external_cache=False)
            self.assertEqual(before_rate, rate_path.read_bytes())
            with np.load(out / "xy_masks.npz") as analysis, np.load(out / "xy_section_data.npz") as display:
                self.assertFalse(analysis["masks"][:, 8].any())
                self.assertTrue(display["mask"][8, 10])
                with np.load(rate_path) as rates:
                    np.testing.assert_array_equal(rates["masks"], analysis["masks"])
            completed = json.loads((out / "all_manifest.json").read_text())
            self.assertEqual(completed["dataset"], config["datasets"][0])
            self.assertEqual(completed["sections"], rendered[0]["sections"])
            self.assertEqual(set(completed["flow_fields"]), {"101", "102", "103", "104"})


if __name__ == "__main__":
    unittest.main()
