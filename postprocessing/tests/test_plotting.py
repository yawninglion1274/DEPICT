"""Independent numerical checks for speed reduction and native RI sampling."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from DEPICT.postprocessing.flow_statistics import regional_speed_onsets, speed_histograms, summarize_speed
from DEPICT.postprocessing.kymograph import sample_kymograph
from DEPICT.postprocessing.plotting import (draw_rate_regions, draw_speed_distribution,
                                           load_speed, load_speed_distribution, prepare_speed_signal_cache)
from DEPICT.postprocessing.volume_io import signature


class PlotDataTests(unittest.TestCase):
    def test_local_speed_cache_rebuilds_from_three_components_then_works_without_external_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            panel = base / "panel_b"
            panel.mkdir()
            masks = np.zeros((2, 2, 3), bool)
            masks[0, 0] = True
            masks[1, 1] = True
            mask_path = panel / "xy_masks.npz"
            np.savez_compressed(mask_path, masks=masks, region_names=np.array(["upper", "lower"]))
            records = {}
            for index, frame in enumerate([40, 41], 1):
                path = base / f"flow_{frame}.npy"
                field = np.ones((3, 1, 2, 3), np.float32)*np.array([3., 4., 12.], np.float32)[:, None, None, None]
                np.save(path, index*field)
                records[str(frame)] = signature(path)
            manifest = dict(fps=5., voxel_um=.2, flow_fields=records,
                sections=[dict(section=dict(name="xy", plane="xy", index=0))],
                dataset=dict(analysis_frame_range=[40, 42], time_origin=40))
            spec = dict(source_panel="b", speed_cache=dict(values="cache/speed.npy",
                alignment="cache/alignment.npz", metadata="cache/speed.json"),
                speed_cache_parameters=dict(smoothing_sigma_t_y_x=[0, 0, 0], quiet_window_s=[0, 1]),
                speed_bins_um_s=dict(range=[0, 30], width=1), threshold_um_s=.54,
                regions=[dict(region="upper"), dict(region="lower")])
            prepare_speed_signal_cache(base, spec, manifest, config_base=base)
            expected = np.broadcast_to(np.array([13., 26.])[:, None, None], (2, 2, 3))
            np.testing.assert_array_equal(np.load(base / "cache/speed.npy"), expected)
            for record in records.values():
                Path(record["path"]).unlink()
            # A plot rebuild must only read the local signal/alignment/mask caches.
            prepare_speed_signal_cache(base, spec, manifest, config_base=base)
            data = load_speed_distribution(base, spec, manifest, base / "panel_c", config_base=base)
            np.testing.assert_array_equal(data["mask_pixels"], [3, 3])
            np.testing.assert_array_equal(data["native_pairs"], [[40, 41], [41, 42]])
            np.testing.assert_array_equal(data["counts"].sum(-1), [[3, 3], [3, 3]])
            np.testing.assert_array_equal(data["above_threshold_percentage"], np.full((2, 2), 100.))
            masks[0, 0, 0] = False
            np.savez_compressed(mask_path, masks=masks, region_names=np.array(["upper", "lower"]))
            with self.assertRaises(FileNotFoundError):
                prepare_speed_signal_cache(base, spec, manifest, config_base=base)

    def test_regional_rate_plot_preserves_curves_and_uses_speed_pair_midpoints(self):
        from matplotlib import pyplot as plt
        rows = [dict(time_s=np.array([.1, .3, .5]), native_pairs=np.array([[40, 41], [41, 42], [42, 43]]),
                     rate_s=values, display_sigma_frames=1)
                for values in [np.array([-.1, .2, .1]), np.array([-.2, .1, .3])]]
        spec = dict(panel="d", axes_box_inches=[.616, .456, 2.10, 1.8552], marker_frames=[40, 42],
                    regions=[dict(region="upper", label="Upper region", color="#0072B2"),
                             dict(region="lower", label="Lower region", color="#D55E00")])
        fig = plt.figure(figsize=(2.8, 2.4))
        try:
            record = draw_rate_regions(fig, rows, spec)
            self.assertEqual([m["time_s"] for m in record["marker_times"]], [.1, .5])
            for row, region in zip(rows, spec["regions"]):
                curve = next(line for line in fig.axes[0].lines if line.get_gid() == f"strain_curve_{region['region']}")
                np.testing.assert_array_equal(curve.get_ydata(), row["rate_s"])
            self.assertFalse(record["refitted"])
        finally:
            plt.close(fig)

    def test_regional_onset_excludes_rest_and_isolated_spikes(self):
        percentages = np.array([[30, 40, 50, 11, 1, 10, 10, 12],
                                [30, 40, 50, 10, 10, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0, 20]])
        data = regional_speed_onsets(percentages, np.arange(-2, 6),
                                     minimum_percentage=10, consecutive_frames=2)
        np.testing.assert_array_equal(data["regional_onset_index"], [5, 3, -1])
        np.testing.assert_allclose(data["regional_onset_time_ms"], [3, 1, np.nan])
        for minimum, frames in [(0, 2), (101, 2), (10, 0), (10, 1.5)]:
            with self.subTest(minimum=minimum, frames=frames), self.assertRaises(ValueError):
                regional_speed_onsets(percentages, np.arange(-2, 6),
                    minimum_percentage=minimum, consecutive_frames=frames)

    def test_rate_curve_and_speed_markers_share_the_configured_millisecond_axis(self):
        from matplotlib import pyplot as plt
        times = np.array([.1, .3, .5])
        pairs = np.array([[40, 41], [41, 42], [42, 43]])
        rows = [dict(time_s=times.copy(), native_pairs=pairs.copy(), rate_s=np.array(values),
                     display_sigma_frames=1) for values in [[-.1, .2, .1], [-.2, .1, .3]]]
        spec = dict(panel="d", axes_box_inches=[.7, .5, 3.65, 1.5], marker_frames=[40, 42],
            time_unit="ms", zero_time_s=.2, time_window_ms=[-150, 400],
            xticks_ms=[-100, 0, 100, 200, 300, 400],
            regions=[dict(region="upper", label="Upper region", color="#0072B2"),
                     dict(region="lower", label="Lower region", color="#D55E00")])
        fig = plt.figure(figsize=(5.2, 2.4))
        try:
            record = draw_rate_regions(fig, rows, spec)
            ax = fig.axes[0]
            self.assertEqual(ax.get_xlabel(), "Time (ms)")
            np.testing.assert_array_equal(ax.get_xlim(), [-150, 400])
            np.testing.assert_array_equal(ax.get_xticks(), spec["xticks_ms"])
            for row, region in zip(rows, spec["regions"]):
                curve = next(line for line in ax.lines if line.get_gid() == f"strain_curve_{region['region']}")
                np.testing.assert_allclose(curve.get_xdata(), [-100, 100, 300], atol=1e-12)
                np.testing.assert_array_equal(curve.get_ydata(), row["rate_s"])
                np.testing.assert_array_equal(row["time_s"], times)
            for frame, expected in [(40, -100), (42, 300)]:
                marker = next(line for line in ax.lines if line.get_gid() == f"speed-time-{frame}")
                np.testing.assert_allclose(marker.get_xdata(), [expected, expected], atol=1e-12)
            self.assertEqual(record["zero_time_s"], .2)
        finally:
            plt.close(fig)

    def test_display_interpolation_preserves_cells_and_native_onset_positions(self):
        from matplotlib import pyplot as plt
        data = dict(pixel_percentage=np.tile([[25., 75.]], (2, 5, 1)),
            time_ms=np.arange(-1., 4.), time_bin_edges_ms=np.arange(-1.5, 4.),
            speed_bin_edges_um_s=np.array([0., 1., 2.]), threshold_um_s=np.array(.54),
            zero_time_s=np.array(1.8), region_names=np.array(["upper", "lower"]),
            mask_pixels=np.array([100, 200]), fps=np.array(1000.),
            native_pairs=np.column_stack([np.arange(40, 45), np.arange(41, 46)]),
            above_threshold_percentage=np.array([[0, 0, 15, 20, 30], [0, 0, 5, 15, 30]]))
        settings = dict(minimum_percentage=10, consecutive_frames=2, reference_region="lower")
        data.update(regional_speed_onsets(data["above_threshold_percentage"], data["time_ms"],
            minimum_percentage=10, consecutive_frames=2))
        original = data["pixel_percentage"].copy()
        spec = dict(panel="c", regions=[dict(region="upper", label="Upper region"),
                                       dict(region="lower", label="Lower region")],
            axes_boxes_inches=[[.7, 2.4, 3., 1.4], [.7, .5, 3., 1.4]],
            colorbar_box_inches=[4., 1., .1, 2.], reference_text_position=[.15, .95],
            onset_definition_position=[.15, .05], regional_onset=settings,
            percentage_gamma=.5, percentage_vmax=100, percentage_ticks=[0, 25, 100],
            heatmap_interpolation="bilinear", colormap="Blues_r")
        fig = plt.figure(figsize=(5, 4.4))
        try:
            record = draw_speed_distribution(fig, data, spec)
            upper, lower = fig.axes[:2]
            self.assertIs(upper.images[0].norm, lower.images[0].norm)
            self.assertEqual(upper.images[0].get_interpolation(), "bilinear")
            np.testing.assert_array_equal(upper.images[0].get_array(), original[0].T)
            np.testing.assert_array_equal(data["pixel_percentage"], original)
            for ax, name, expected in [(upper, "upper", 1.), (lower, "lower", 2.)]:
                marker = next(line for line in ax.lines if line.get_gid() == f"regional_onset_{name}")
                np.testing.assert_array_equal(marker.get_xdata(), [expected, expected])
            self.assertFalse(record["interpolation_used_for_analysis"])
        finally:
            plt.close(fig)

    def test_histograms_count_all_mask_pixels_and_threshold_independently(self):
        masks = np.array([[[True, True, True, False, False]],
                          [[False, False, False, True, False]]])
        speeds = np.array([[[0., .5, 2., 1.25, np.nan]],
                           [[1., .75, 1.5, 0., np.nan]]])
        data = speed_histograms(speeds, masks, bin_edges_um_s=[0, 1, 2], threshold_um_s=.5)
        np.testing.assert_array_equal(data["counts"], [[[2, 1], [1, 2]], [[0, 1], [1, 0]]])
        np.testing.assert_array_equal(data["mask_pixels"], [3, 1])
        np.testing.assert_allclose(data["pixel_percentage"].sum(-1), 100, rtol=0, atol=1e-13)
        # .5 is not above the threshold, despite sharing a bin with 0 and .75.
        np.testing.assert_array_equal(data["above_threshold_counts"], [[1, 3], [1, 0]])
        np.testing.assert_allclose(data["above_threshold_percentage"], [[100/3, 100], [100, 0]])

    def test_histograms_reject_incomplete_bins_and_invalid_in_mask_values(self):
        mask = np.ones((1, 2), bool)
        for values, edges in [([0., 2.1], [0, 1, 2]), ([0., 1.], [.1, 2]),
                              ([0., np.nan], [0, 1]), ([0., -1.], [0, 1]),
                              ([0., 1.], [0, 1, 1])]:
            with self.subTest(values=values, edges=edges), self.assertRaises(ValueError):
                speed_histograms(np.array(values).reshape(1, 1, 2), mask,
                                 bin_edges_um_s=edges, threshold_um_s=.5)

    def test_distribution_heatmaps_share_axes_and_the_same_color_normalization(self):
        from matplotlib import pyplot as plt
        data = dict(pixel_percentage=np.array([[[10., 90.], [40., 60.]],
                                               [[75., 25.], [30., 70.]]]),
            time_ms=np.array([0., 1.]), time_bin_edges_ms=np.array([-.5, .5, 1.5]),
            speed_bin_edges_um_s=np.array([0., 1., 2.]), threshold_um_s=np.array(.5),
            zero_time_s=np.array(1.8), region_names=np.array(["upper", "lower"]),
            mask_pixels=np.array([100, 200]))
        spec = dict(panel="c", regions=[dict(region="upper", label="Upper region"),
                                       dict(region="lower", label="Lower region")],
            axes_boxes_inches=[[.7, 2.4, 3., 1.4], [.7, .5, 3., 1.4]],
            colorbar_box_inches=[4., 1., .1, 2.], reference_text_position=[.15, .95],
            percentage_gamma=.5, percentage_vmax=100, percentage_ticks=[0, 25, 100])
        fig = plt.figure(figsize=(5, 4.4))
        try:
            record = draw_speed_distribution(fig, data, spec)
            upper, lower = fig.axes[:2]
            self.assertIs(upper.collections[0].norm, lower.collections[0].norm)
            self.assertEqual(upper.get_xlim(), lower.get_xlim())
            self.assertEqual(upper.get_ylim(), lower.get_ylim())
            self.assertTrue(upper.get_shared_x_axes().joined(upper, lower))
            self.assertTrue(upper.get_shared_y_axes().joined(upper, lower))
            np.testing.assert_array_equal(upper.collections[0].get_array().ravel(),
                                          data["pixel_percentage"][0].T.ravel())
            self.assertTrue(record["all_pixels_in_denominator"])
        finally:
            plt.close(fig)

    def test_region_selection_is_part_of_the_speed_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            panel = base / "panel_b"
            panel.mkdir()
            regions = np.zeros((2, 2, 3), bool)
            regions[0, 0] = True
            regions[1, 1] = True
            np.savez_compressed(panel / "xy_masks.npz", masks=regions,
                                region_names=np.array(["upper", "lower"]))
            field = np.zeros((3, 1, 2, 3))
            field[:, 0, 0] = np.array([3., 4., 12.])[:, None]
            field[2, 0, 1] = 2.
            records = {}
            for i, frame in enumerate([40, 41], 1):
                path = base / f"flow_{frame}.npy"
                np.save(path, i*field)
                records[str(frame)] = signature(path)
            manifest = dict(fps=5., voxel_um=.2, flow_fields=records,
                sections=[dict(section=dict(name="xy", plane="xy", index=0))],
                dataset=dict(analysis_frame_range=[40, 42], time_origin=40))
            cache = base / "statistics"
            for region, expected in [("upper", [13., 26.]), ("lower", [2., 4.])]:
                data = load_speed(base, dict(source_panel="b", region=region), manifest, cache)
                np.testing.assert_allclose(data["median_um_s"], expected, atol=1e-13, rtol=0)
                contract = json.loads((cache / "speed_statistics.json").read_text())["contract"]
                self.assertEqual(contract["region"], region)

    def test_speed_uses_three_components_and_only_supplied_mask(self):
        mask = np.array([[True, False, True], [False, True, True]])
        field = np.full((3, 2, 3), np.nan)
        field[:, mask] = np.array([3., 4., 12.])[:, None]*np.arange(1, 5)
        data = summarize_speed([field, 2*field], mask, native_left_frames=[40, 41],
                               time_origin=40, fps=5., voxel_um=.2)
        np.testing.assert_array_equal(data["native_pairs"], [[40, 41], [41, 42]])
        np.testing.assert_array_equal(data["time_s"], [.1, .3])
        np.testing.assert_allclose(data["p25_um_s"], [22.75, 45.5], rtol=0, atol=1e-13)
        np.testing.assert_allclose(data["median_um_s"], [32.5, 65.], rtol=0, atol=1e-13)
        np.testing.assert_allclose(data["p75_um_s"], [42.25, 84.5], rtol=0, atol=1e-13)
        self.assertEqual(int(data["mask_pixels"]), 4)
        with self.assertRaises(ValueError):
            summarize_speed([field], mask, native_left_frames=[40, 41], time_origin=40, fps=5., voxel_um=.2)

    def test_kymograph_interpolates_native_coordinates_and_keeps_frame_order(self):
        t, z, y, x = np.indices((3, 2, 5, 6))
        volumes = (2*t+3*z+5*y+7*x).astype(np.float32)
        coordinates = np.array([[[.5, 1.5], [2.25, 2.75], [3., 3.5]],
                                [[1., 1.5], [2., 3.], [4., 4.5]]])
        actual = sample_kymograph(volumes, volume_frames=[90, 91, 92], native_frames=[91, 92],
                                   z=1, coordinates_yx=coordinates)
        expected_profile = 3+5*coordinates[0].mean(-1)+7*coordinates[1].mean(-1)
        np.testing.assert_allclose(actual, expected_profile[None]+np.array([2, 4])[:, None],
                                   rtol=0, atol=1e-6)
        with self.assertRaises(ValueError):
            sample_kymograph(volumes, volume_frames=[90, 91, 92], native_frames=[91, 92],
                              z=1, coordinates_yx=coordinates+10)


if __name__ == "__main__":
    unittest.main()
