"""Rebuild tracking, statistics and annotated panels from saved localizations."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from common import ROOT, MANIFEST, load_reference_tracks, load_tracks

import numpy as np
import pandas as pd


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def track():
    import association
    import temporal_utils
    import tracking_base as base

    detections = pd.read_csv(ROOT / "inputs/angle12/localizations.csv")
    if detections.detection_id.duplicated().any():
        raise ValueError("Duplicate baseline detection IDs")
    tracked = base.link_detections(detections)
    groups = base.track_groups(tracked)
    last_frame = int(detections.frame.max())
    candidates = base.find_overlap_joins(groups, last_frame)
    joins = [p for p in candidates
             if int(groups[p["early_track_id"]].frame.iloc[0]) == 0
             and int(groups[p["late_track_id"]].frame.iloc[-1]) == last_frame]
    joined_ids = {p[k] for p in joins for k in ("early_track_id", "late_track_id")}
    rows = []
    for identity, group in groups.items():
        if identity not in joined_ids and int(group.frame.iloc[0]) == 0 and int(group.frame.iloc[-1]) == last_frame:
            rows.extend(base.position_row(row, identity) for _, row in group.iterrows())
    for pair in joins:
        rows.extend(base.join_pair(groups[pair["early_track_id"]], groups[pair["late_track_id"]], pair["late_track_id"]))
    reference = pd.DataFrame(rows, columns=base.FINAL_COLUMNS)
    reference = reference.groupby("track_id", group_keys=False).filter(
        lambda g: len(g) == last_frame + 1 and set(g.frame) == set(range(last_frame + 1)))
    reference = reference.sort_values(["track_id", "frame"]).reset_index(drop=True)
    folder = ROOT / "analysis/angle12"
    folder.mkdir(parents=True, exist_ok=True)
    tracked.to_csv(folder / "laptrack_tracks.csv", index=False, float_format="%.12g")
    pd.DataFrame(joins).to_csv(folder / "fragment_joins.csv", index=False)
    # Serialize baseline positions with 12 significant digits before association.
    reference.to_csv(folder / "complete_tracks.csv", index=False, float_format="%.12g")
    reference = pd.read_csv(folder / "complete_tracks.csv")
    results = []
    for angle in (24, 48, 240):
        destination = ROOT / f"analysis/angle{angle}"
        destination.mkdir(parents=True, exist_ok=True)
        measured = pd.read_csv(ROOT / f"inputs/angle{angle}/localizations.csv")
        native = temporal_utils.link(measured, angle, destination).set_index("detection_id").track_id
        result, audit = association.associate(
            angle, reference, detections=measured, native=native, destination=destination,
            frame_selection="within_common" if angle == 240 else "within_reference")
        if angle == 240:
            result["position_origin"] = "raw_reconstruction_multithreshold_geometric_centroid"
        results.append(result)
        print(f"{angle} angles: {result.track_id.nunique()} complete tracks", flush=True)
    reference["angle_group"], reference["downsample_factor"], reference["volume_rate_vps"] = 12, 1, 16.25
    reference["global_frame"], reference["time_s"] = reference.frame + 20, reference.frame / 16.25
    rebuilt = pd.concat([reference] + results, ignore_index=True)
    keys = ["angle_group", "track_id", "frame"]
    actual = rebuilt.sort_values(keys).reset_index(drop=True)
    snapshot = MANIFEST / "server_snapshots/temporal/reconstructed_tracks_by_resolution.csv"
    matches_source = None
    if snapshot.exists():
        expected = pd.read_csv(snapshot, low_memory=False).sort_values(keys).reset_index(drop=True)
        np.testing.assert_array_equal(actual[keys], expected[keys])
        np.testing.assert_allclose(actual[["x_um", "y_um", "z_um", "time_s"]],
                                   expected[["x_um", "y_um", "z_um", "time_s"]], rtol=0, atol=2e-10)
        matches_source = True
    rebuilt.to_csv(ROOT / "analysis/tracks_by_resolution.csv", index=False)
    write_json(MANIFEST / "tracking_validation.json", {
        "measured_positions_match_server": matches_source,
        "coordinate_tolerance_um": 2e-10,
        "n_baseline_detections": len(detections), "n_baseline_native_tracks": len(groups),
        "n_baseline_fragment_joins": len(joins),
        "tracks_by_angle": {str(a): int(g.track_id.nunique()) for a, g in actual.groupby("angle_group")},
        "positions_per_track": {str(a): sorted(g.groupby("track_id").size().unique().tolist()) for a, g in actual.groupby("angle_group")},
        "laptrack_version": importlib.metadata.version("laptrack"),
        "source_laptrack_version": "0.17.1",
    })


def analyze():
    import temporal_utils
    import plot_final_motion_trajectories as motion

    tracks = load_tracks()
    groups = motion.classify_tracks(load_reference_tracks())
    groups.to_csv(ROOT / "analysis/motion_classification.csv", index=False)
    speeds, fits, msd = temporal_utils.statistics(tracks, "per_resolution_detected_tracks")
    labels = groups[["track_id", "final_group"]].rename(columns={"final_group": "motion_group"})
    speeds = speeds.merge(labels, on="track_id", validate="many_to_one")
    fits = fits.merge(labels, on="track_id", validate="many_to_one")
    speeds.to_csv(ROOT / "analysis/track_mean_speeds.csv", index=False)
    fits.to_csv(ROOT / "analysis/alpha_by_temporal_resolution.csv", index=False)
    msd.to_csv(ROOT / "analysis/per_track_msd.csv", index=False)
    common = set.intersection(*(set(g.track_id) for _, g in tracks.groupby("angle_group")))
    write_json(ROOT / "analysis/paired_identities.json", {"track_ids": sorted(map(int, common))})
    summary = speeds.groupby(["angle_group", "volume_rate_vps"]).agg(
        n_tracks=("track_id", "nunique"), median_speed_um_min=("mean_speed_um_min", "median")).reset_index()
    summary = summary.merge(fits.groupby("angle_group").agg(
        median_alpha=("alpha", "median"), successful_alpha_fits=("status", lambda x: x.eq("fit_ok").sum()),
        min_fit_lags=("fit_lags", "min")), on="angle_group")
    summary.to_csv(ROOT / "analysis/analysis_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    checks = {}
    for name, actual, columns in [
        ("track_mean_speeds.csv", speeds, ["mean_speed_um_min", "n_steps"]),
        ("alpha_by_temporal_resolution.csv", fits, ["alpha", "D_um2_s_alpha", "fit_r2_log", "fit_lags"]),
    ]:
        snapshot = MANIFEST / "server_snapshots/temporal" / name
        if not snapshot.exists():
            continue
        source = pd.read_csv(snapshot)
        keys = ["angle_group", "track_id"]
        actual = actual.sort_values(keys).reset_index(drop=True)
        source = source.sort_values(keys).reset_index(drop=True)
        np.testing.assert_array_equal(actual[keys], source[keys])
        np.testing.assert_allclose(actual[columns], source[columns], rtol=1e-8, atol=1e-9)
        checks[name] = float(np.max(np.abs(actual[columns].to_numpy() - source[columns].to_numpy())))
    write_json(MANIFEST / "analysis_validation.json", {
        "matches_server_statistics": True if checks else None, "maximum_absolute_differences": checks,
        "group_counts_baseline": {k: int(v) for k, v in groups.final_group.value_counts().items()},
        "paired_identities": len(common),
        "manual_motion_assignments": {"39": "directional", "23": "confined", "93": "confined"},
    })


def save(fig, directory, name):
    import matplotlib.pyplot as plt
    path = ROOT / directory / f"{name}_with_text.pdf"
    fig.savefig(path, dpi=320, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Saved", path.relative_to(ROOT), flush=True)


def plot(selected=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import plot_final_motion_trajectories as motion
    import plot_trajectory_3x5_3dbox as representative
    import plot_alpha_density_motion_strip as alpha_strip
    import plot_alpha_temporal_resolutions as alpha_plot
    import plot_all_tracks_speed_by_resolution as all_speed
    import plot_motion_downsampled_examples as examples
    import temporal_plots as temporal

    font_manager.fontManager.addfont(str(ROOT / "inputs/fonts/Arial.TTF"))
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    tracks = load_tracks()
    groups = pd.read_csv(ROOT / "analysis/motion_classification.csv")
    speeds = pd.read_csv(ROOT / "analysis/track_mean_speeds.csv")
    fits = pd.read_csv(ROOT / "analysis/alpha_by_temporal_resolution.csv")
    selected = set(selected or ("d", "e", "f", "9a", "9b", "9c", "10a", "10b", "10c"))
    if "d" in selected:
        representative.main()
    if "e" in selected:
        save(temporal.plot_group_speeds(speeds, textless=False), "panel_e", "panel_e")
        temporal.panel_e.summarize(speeds).to_csv(ROOT / "panel_e/speed_boxplot_summary.csv", index=False)

    if "f" in selected:
        values = groups.alpha.to_numpy(float)
        density = alpha_strip.kde_density_profile(values)
        gaussian = alpha_strip.fit_two_gaussians_to_samples(values, density.alpha.to_numpy())
        points = alpha_strip.strip_points(groups)
        fig = alpha_strip.draw(density.alpha.to_numpy(), density.density.to_numpy(), gaussian, points, with_text=True)
        from matplotlib.lines import Line2D
        fig.axes[1].legend(handles=[
            Line2D([], [], color=alpha_strip.DENSITY_COLOR, label=f"16.25 vps (n = {len(values)})"),
            Line2D([], [], color=alpha_strip.COMPONENT_COLORS[0], linestyle="--", label="GMM fitting 1"),
            Line2D([], [], color=alpha_strip.COMPONENT_COLORS[1], linestyle="--", label="GMM fitting 2"),
        ], loc="upper left", fontsize=7, frameon=False)
        save(fig, "panel_f", "panel_f")
        fields = ["method", "n", "w1", "mu1", "sigma1", "w2", "mu2", "sigma2", "aic_1_component",
                  "aic_2_components", "bic_1_component", "bic_2_components"]
        pd.DataFrame([{k: gaussian[k] for k in fields}]).to_csv(ROOT / "panel_f/gaussian_fit.csv", index=False)
        density.to_csv(ROOT / "panel_f/density_profile.csv", index=False)
        pd.DataFrame({"alpha": gaussian["grid"], "component1": gaussian["component1"],
            "component2": gaussian["component2"], "total": gaussian["total"]}).to_csv(ROOT / "panel_f/gaussian_curves.csv", index=False)
        points.to_csv(ROOT / "panel_f/strip_points.csv", index=False)

    panels = motion.build_panels(groups, load_reference_tracks()) if selected & {"9a", "9b", "9c"} else None
    duration = (load_reference_tracks().frame.max() - load_reference_tracks().frame.min()) / motion.FPS
    for key, group in [("a", "confined"), ("b", "diffusing"), ("c", "directional")]:
        if f"9{key}" not in selected:
            continue
        motion.HERE = ROOT / f"Supplementary_Figure_9/panel_{key}"
        motion.draw_sheet(group, group.capitalize(), "black", panels[group], duration_s=duration, with_text=True, compact=True)
        original = motion.HERE / f"final_{group}_trajectories_speed_sorted_17p845cm_arial6_speed_alpha.pdf"
        original.replace(motion.HERE / f"panel_{key}_with_text.pdf")

    if "10a" in selected:
        common = set.intersection(*(set(g.track_id) for _, g in tracks.groupby("angle_group")))
        selection = temporal.choose_examples(groups.loc[groups.track_id.isin(common)], speeds)
        selection.to_csv(ROOT / "Supplementary_Figure_10/panel_a/example_selection.csv", index=False)
        panel_data, metrics = temporal.example_panels(tracks, fits, selection)
        metrics.to_csv(ROOT / "Supplementary_Figure_10/panel_a/example_statistics.csv", index=False)
        entries = [dict(track_id=int(row.track_id), panels=panel_data[row.track_id]) for row in selection.itertuples()]
        save(examples.draw_figure(entries, metrics.loc[metrics.downsample_factor.isin((1, 20))], with_text=True),
            "Supplementary_Figure_10/panel_a", "panel_a")
    if "10b" in selected:
        success = fits.loc[fits.status.eq("fit_ok") & np.isfinite(fits.alpha)]
        histogram = alpha_plot.density_profiles(success)
        histogram.to_csv(ROOT / "Supplementary_Figure_10/panel_b/density_profiles.csv", index=False)
        save(alpha_plot.plot(histogram, success, with_text=True, kde=False), "Supplementary_Figure_10/panel_b", "panel_b")
    if "10c" in selected:
        fig = all_speed.plot(speeds, with_text=True)
        counts = speeds.groupby("downsample_factor").track_id.nunique()
        fig.axes[0].set_yticklabels([f"{16.25/q:.5g} ({a} angles)\nn={counts.loc[q]}"
                                    for q, a in zip((1, 2, 4, 20), (12, 24, 48, 240))])
        for text in fig.axes[0].texts:
            if text.get_text().startswith("All tracks"):
                text.set_text("All detected tracks")
        save(fig, "Supplementary_Figure_10/panel_c", "panel_c")
        all_speed.summarize(speeds).to_csv(ROOT / "Supplementary_Figure_10/panel_c/speed_boxplot_summary.csv", index=False)


def record_run(stage):
    import platform
    inventory = [p for p in ROOT.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts]
    write_json(MANIFEST / "run_manifest.json", {
        "stage": stage, "python": sys.executable, "python_version": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "scipy", "pandas", "matplotlib", "laptrack", "scikit-learn"]},
        "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size}
                  for p in sorted(inventory)],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "track", "analyze", "plot"), default="all")
    parser.add_argument("--panels", nargs="+", choices=("d", "e", "f", "9a", "9b", "9c", "10a", "10b", "10c"))
    args = parser.parse_args()
    if args.stage in ("all", "track"):
        track()
    if args.stage in ("all", "analyze"):
        analyze()
    if args.stage in ("all", "plot"):
        plot(args.panels)
    record_run(args.stage)


if __name__ == "__main__":
    main()
