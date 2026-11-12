#!/usr/bin/env python3
"""Render rows of three-dimensional trajectories in perspective boxes.

Trajectories are colored by elapsed time and projected onto the box floor and
side wall. Rendering options control camera orientation, common physical box
size, clipping, scale bars, annotations and output format."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import projection as rl  # noqa: E402

ANALYSIS_DIR = HERE / "lipid_analysis"
MERGED_DIR = HERE / "lipid_D_alpha_top5_trajectory_only_merged"
DEFAULT_TRACKS_CSV = ANALYSIS_DIR / "sam2_lipid_gt0p1_cpu_tracks_minlen100_merged.csv"
DEFAULT_GROUPS_CSV = ANALYSIS_DIR / "sam2_lipid_gt0p1_cpu_figure_groups_merged_trajectories.csv"
DEFAULT_OUTPUT_DIR = MERGED_DIR / "combine_2x5_3dbox"

# ``render_lipid_mip_time_trajectory.box_corners`` enumerates the eight cube
# corners as [x, y, z] with x varying slowest and z fastest:
#   0=(x_lo,y_lo,z_lo) 1=(x_lo,y_lo,z_hi) 2=(x_lo,y_hi,z_lo) 3=(x_lo,y_hi,z_hi)
#   4=(x_hi,y_lo,z_lo) 5=(x_hi,y_lo,z_hi) 6=(x_hi,y_hi,z_lo) 7=(x_hi,y_hi,z_hi)
# Each face is stored as (corner ids walked around the quad, outward normal).
CUBE_FACES: tuple[tuple[tuple[int, int, int, int], tuple[float, float, float]], ...] = (
    ((0, 2, 6, 4), (0.0, 0.0, -1.0)),  # floor   (z = z_lo)
    ((1, 3, 7, 5), (0.0, 0.0, 1.0)),   # ceiling (z = z_hi)
    ((0, 1, 3, 2), (-1.0, 0.0, 0.0)),  # wall    (x = x_lo)
    ((4, 5, 7, 6), (1.0, 0.0, 0.0)),   # wall    (x = x_hi)
    ((0, 1, 5, 4), (0.0, -1.0, 0.0)),  # wall    (y = y_lo)
    ((2, 3, 7, 6), (0.0, 1.0, 0.0)),   # wall    (y = y_hi)
)

# Fractional padding applied around the projected box when fixing the 2-D limits.
DATA_PAD = 0.015

# The 12 cube edges; every one of them is stroked with the same width + colour.
# Per-label-mode layout (inches).  Only the bands needed by a mode are kept,
# so the "speed" figure stays compact while "full" and "none" share geometry.
MODE_LAYOUT: dict[str, dict[str, float]] = {
    "full": {"caption_band": 0.30, "footnote_band": 0.26, "row_gap": 0.16,
             "left_margin": 0.44, "top_margin": 0.10, "bottom_margin": 0.06},
    "speed": {"caption_band": 0.26, "footnote_band": 0.02, "row_gap": 0.10,
              "left_margin": 0.10, "top_margin": 0.08, "bottom_margin": 0.04},
}
MODE_LAYOUT["none"] = dict(MODE_LAYOUT["full"])

# Annotation font sizes for the "full" mode (px offsets relative to --font-size).

CUBE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (2, 3), (4, 5), (6, 7),  # along z
    (0, 2), (1, 3), (4, 6), (5, 7),  # along y
    (0, 4), (1, 5), (2, 6), (3, 7),  # along x
)

# Light-grey fill for the box faces, keyed by the outward face normal.
FACE_GREY: dict[tuple[float, float, float], float] = {
    (0.0, 0.0, -1.0): 0.930,  # floor
    (0.0, 0.0, 1.0): 0.972,  # ceiling
    (0.0, -1.0, 0.0): 0.958,
    (0.0, 1.0, 0.0): 0.905,
    (-1.0, 0.0, 0.0): 0.942,
    (1.0, 0.0, 0.0): 0.918,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--tracks-csv", type=Path, default=None,
                        help="Defaults to the merged long-track detection table.")
    parser.add_argument("--groups-csv", type=Path, default=DEFAULT_GROUPS_CSV)
    parser.add_argument("--groups", nargs="+", default=("alpha_near1_w0p1", "alpha_gt1p4"),
                        help="One group key per row, top row first.")
    parser.add_argument("--row-labels", nargs="*", default=None,
                        help="Optional short label per row (default: derived from the group label).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=float, default=16.25)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-62.0)
    parser.add_argument("--focal-length", type=float, default=1.0,
                        help="mplot3d perspective focal length (smaller = stronger 3-D).")
    parser.add_argument("--padding-frac", type=float, default=0.10)
    parser.add_argument("--line-width", type=float, default=0.85)
    parser.add_argument("--output-tag", default="", help="Suffix added to every output file name.")
    parser.add_argument("--mode", choices=("panels", "xyz", "colorbar", "both"), default="panels",
                        help="'panels': 2x5 figure + colour bar; 'xyz': only the orientation triad; "
                             "'colorbar': only the colour bar; 'both': panels + triad.")
    parser.add_argument("--xyz-size", type=float, default=1.7, help="Side length of the orientation figure (in).")
    parser.add_argument("--trajectory-outline-scale", type=float, default=1.9,
                        help="Width multiplier of the black halo stroked under every trajectory "
                             "(default: 1.9); 0 disables the halo.")
    parser.add_argument("--trajectory-outline", dest="trajectory_outline", action="store_true", default=True,
                        help="Stroke the black halo under the coloured tracks (default).")
    parser.add_argument("--no-trajectory-outline", dest="trajectory_outline", action="store_false",
                        help="Stroke the coloured trajectories without their black outline and "
                             "append '_nooutline' to the output names.")
    parser.add_argument("--scale-bar-scope", choices=("panel", "figure"), default="panel",
                        help="'panel': one exact bar in the lower-left corner of every panel; "
                             "'figure': one bar only, drawn at the bottom-left panel scale "
                             "(exact for other panels only with --common-cube-um).")
    parser.add_argument("--common-cube-from", default=None,
                        help="ROW,COL (1-based) of the panel whose own zoom is imposed on every "
                             "panel, e.g. '2,2' copies the cube size of row 2 / column 2.")
    parser.add_argument("--clip-to-cube", dest="clip_to_cube", action="store_true", default=False,
                        help="Cut tracks that leave the cube at the cube surface instead of "
                             "letting them run outside the box.")
    parser.add_argument("--no-clip-to-cube", dest="clip_to_cube", action="store_false")
    parser.add_argument("--common-cube-um", type=float, default=None,
                        help="Force every panel to show the same physical cube size (um). "
                             "Only then is one figure-wide scale bar exact for all panels.")
    parser.add_argument("--time-cmap", default="turbo")
    parser.add_argument("--time-cmap-trim", type=float, default=0.05)
    parser.add_argument("--panel-width", type=float, default=1.34, help="Panel width in inches.")
    parser.add_argument("--col-gap", type=float, default=0.05)
    parser.add_argument("--row-gap", type=float, default=None, help="Default depends on --label-mode.")
    parser.add_argument("--left-margin", type=float, default=None, help="Default depends on --label-mode.")
    parser.add_argument("--right-margin", type=float, default=0.06)
    parser.add_argument("--top-margin", type=float, default=None, help="Default depends on --label-mode.")
    parser.add_argument("--bottom-margin", type=float, default=None, help="Default depends on --label-mode.")
    parser.add_argument("--caption-band", type=float, default=None,
                        help="Space for the text above a panel (in); depends on --label-mode.")
    parser.add_argument("--footnote-band", type=float, default=None,
                        help="Space for the text below a panel (in); depends on --label-mode.")
    parser.add_argument("--speed-offset", type=float, default=0.205, help="Mean-speed text height above the panel (in).")
    parser.add_argument("--alpha-offset", type=float, default=0.085, help="Alpha text height above the panel (in).")
    parser.add_argument("--track-offset", type=float, default=0.070, help="Track-id text depth below the panel (in).")
    parser.add_argument("--d-offset", type=float, default=0.165, help="D text depth below the panel (in).")
    parser.add_argument("--face-mode", choices=("hidden", "visible", "none"), default="none",
                        help="Which box faces receive a light shading fill (default: none, edges only).")
    parser.add_argument("--box-line-width", type=float, default=0.65,
                        help="Line width shared by all 12 box edges (pt).")
    parser.add_argument("--box-grey", type=float, default=0.18, help="Grey level of the box edges (0=black).")
    parser.add_argument("--projection-mode",
                        choices=("floor+left-wall", "all", "floor", "walls", "left-wall", "none"),
                        default="floor+left-wall",
                        help="Hidden faces that receive an orthographic projection of the track "
                             "(default: floor = xy plane plus the left-hand wall = yz plane).")
    parser.add_argument("--projection-grey", type=float, default=0.78,
                        help="Grey level of the projected track copies (0=black).")
    parser.add_argument("--projection-scale", type=float, default=0.62,
                        help="Projection line width relative to --line-width.")
    parser.add_argument("--scale-bar-um", type=float, default=1.0,
                        help="Length of the black scale bar in the lower-left corner (um).")
    parser.add_argument("--scale-bar-color", default="black",
                        help="Colour of the scale bar (default: black).")
    parser.add_argument("--scale-bar-line-width", type=float, default=3.2,
                        help="Scale-bar thickness in points.")
    parser.add_argument("--scale-bar-x-frac", type=float, default=0.08,
                        help="Bar left edge as a fraction of the bottom-left panel width.")
    parser.add_argument("--scale-bar-y-frac", type=float, default=0.06,
                        help="Bar height as a fraction of the bottom-left panel height, "
                             "measured up from the bottom edge of that panel (--scale-bar-scope panel).")
    parser.add_argument("--scale-bar-band", type=float, default=0.30,
                        help="Empty band reserved under the panels for the single figure-level "
                             "scale bar (--scale-bar-scope figure), in inches.")
    parser.add_argument("--no-scale-bar", dest="scale_bar", action="store_false",
                        help="Do not draw the scale bar.")
    parser.add_argument("--grid-grey", type=float, default=0.88, help="Grey level of the floor grid.")
    parser.add_argument("--grid-line-width", type=float, default=0.25, help="Floor grid line width (pt).")
    parser.add_argument("--floor-grid", dest="floor_grid", action="store_true", default=True,
                        help="Draw a faint grid on the box floor (default).")
    parser.add_argument("--no-floor-grid", dest="floor_grid", action="store_false")
    parser.add_argument("--font-size", type=float, default=6.2)
    parser.add_argument("--font-scale", type=float, default=1.0,
                        help="Multiply every annotation font size by this factor.")
    parser.add_argument("--label-mode", choices=("full", "speed", "none"), default="full",
                        help="'full': v-bar + alpha above, track/D below, row labels; "
                             "'speed': one mean-speed number per panel, unit in brackets on the "
                             "last column only; 'none': no text at all.")
    parser.add_argument("--speed-unit", choices=("um_s", "um_min"), default="um_s",
                        help="Unit used by --label-mode speed (values are converted accordingly).")
    parser.add_argument("--cbar-height", type=float, default=1.55, help="Standalone colour-bar height (in).")
    parser.add_argument("--cbar-plain", dest="cbar_plain", action="store_true", default=False,
                        help="Colour bar without its black outline and without the two time labels.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", nargs="+", default=("png", "pdf", "svg"))
    return parser.parse_args()


def time_colormap(args: argparse.Namespace) -> ListedColormap:
    base = plt.get_cmap(args.time_cmap)
    trim = float(args.time_cmap_trim)
    if not (0.0 <= trim < 0.5):
        raise ValueError(f"time_cmap_trim must be in [0, 0.5), got {trim}")
    if trim > 0.0:
        return ListedColormap(base(np.linspace(trim, 1.0 - trim, 256)))
    return base


def style_rcparams(args: argparse.Namespace) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": float(args.font_size),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def view_direction(elev: float, azim: float) -> np.ndarray:
    """Unit vector pointing from the box centre towards the camera."""
    e, a = np.radians(float(elev)), np.radians(float(azim))
    return np.asarray(
        [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)],
        dtype=np.float64,
    )


def cube_corners(limits3d) -> np.ndarray:
    return rl.box_corners(*limits3d)


def load_groups(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, tuple[str, pd.DataFrame]]]:
    groups_csv = Path(args.groups_csv)
    tracks_csv = Path(args.tracks_csv) if args.tracks_csv is not None else DEFAULT_TRACKS_CSV
    if not groups_csv.exists():
        raise FileNotFoundError(groups_csv)
    if not tracks_csv.exists():
        raise FileNotFoundError(tracks_csv)
    table = pd.read_csv(groups_csv)
    tracks = pd.read_csv(tracks_csv)
    tracks["track_id"] = tracks["track_id"].astype(int)

    by_key: dict[str, tuple[str, pd.DataFrame]] = {}
    for group_key, group in table.groupby("group_key", sort=False):
        group = group.sort_values("group_rank").reset_index(drop=True)
        by_key[str(group_key)] = (str(group["group_label"].iloc[0]), group)
    missing = [key for key in args.groups if key not in by_key]
    if missing:
        raise KeyError(f"Unknown group key(s) {missing}; available: {sorted(by_key)}")
    return tracks, {key: by_key[key] for key in args.groups}


def mean_speed_um_s(xyz_um: np.ndarray, frames: np.ndarray, fps: float) -> float:
    """Time-averaged instantaneous speed along the track (um/s)."""
    if xyz_um.shape[0] < 2:
        return float("nan")
    step_um = np.linalg.norm(np.diff(xyz_um, axis=0), axis=1)
    dt_s = np.diff(frames) / float(fps)
    valid = dt_s > 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(step_um[valid] / dt_s[valid]))


def build_panel(track_row: pd.Series, track_points: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    track_points = track_points.sort_values("frame").reset_index(drop=True)
    xyz_um = track_points[["x_um", "y_um", "z_um"]].to_numpy(dtype=np.float64)
    frames = track_points["frame"].to_numpy(dtype=np.float64)
    if xyz_um.shape[0] < 2:
        raise ValueError(f"Track {int(track_row['track_id'])} has fewer than two detections")

    centre = 0.5 * (xyz_um.min(axis=0) + xyz_um.max(axis=0))
    centred = xyz_um - centre
    half_size = float(np.max(np.abs(centred))) * (1.0 + 2.0 * float(args.padding_frac))
    half_size = max(half_size, 0.25)
    common_cube_um = getattr(args, "common_cube_um", None)
    if common_cube_um:
        half_size = 0.5 * abs(float(common_cube_um))
        reach = float(np.max(np.abs(centred)))
        if reach > half_size:
            action = (
                "the part outside the box is cut off (--clip-to-cube)"
                if bool(getattr(args, "clip_to_cube", False))
                else "the trajectory will stick out of its box (use --clip-to-cube to cut it)"
            )
            print(
                f"[warn] track {int(track_row['track_id'])} reaches {reach:.2f} um from its centre "
                f"but the common cube gives a half-size of {half_size:.2f} um; {action}."
            )
    limits3d = ((-half_size, half_size), (-half_size, half_size), (-half_size, half_size))

    matrix = rl.make_projection_matrix(
        *limits3d,
        box_aspect=(1.0, 1.0, 1.0),
        elev=float(args.elev),
        azim=float(args.azim),
        focal_length=float(args.focal_length),
    )
    projected = rl.project_points(centred, matrix)
    projected_corners = rl.project_points(cube_corners(limits3d), matrix)

    times_s = (frames - frames[0]) / float(args.fps)
    duration_s = float((frames[-1] - frames[0]) / float(args.fps))

    if bool(getattr(args, "clip_to_cube", False)):
        runs = clip_polylines_to_cube(centred, times_s, half_size)
        if not runs:
            raise ValueError(
                f"Track {int(track_row['track_id'])} lies completely outside the "
                f"{2.0 * half_size:.2f} um cube"
            )
    else:
        runs = [(centred, times_s)]

    inside = np.all(np.abs(centred) <= half_size + 1e-12, axis=1)
    clipped = bool(getattr(args, "clip_to_cube", False))
    projected_runs = [rl.project_points(points_, matrix) for points_, _ in runs]
    segments = [proj[idx : idx + 2] for proj in projected_runs for idx in range(proj.shape[0] - 1)]
    segment_times = [
        0.5 * (t[idx] + t[idx + 1]) for _, t in runs for idx in range(t.shape[0] - 1)
    ]

    return {
        "track_id": int(track_row["track_id"]),
        "alpha": float(track_row["alpha"]),
        "D": float(track_row["D_um2_s_alpha"]),
        "mean_speed": mean_speed_um_s(xyz_um, frames, float(args.fps)),
        "duration_s": max(duration_s, 1.0 / float(args.fps)),
        "n_detections": int(track_points.shape[0]),
        "segments": segments,
        "segment_times": segment_times,
        "runs": runs,
        "projected_runs": projected_runs,
        # Start/end markers keep their meaning (dot = first detection, cross =
        # last detection); when a clipped track enters/leaves the cube the
        # corresponding marker is dropped instead of moving to the cut.
        "start_xy": None if (clipped and not inside[0]) else projected_runs[0][0],
        "end_xy": None if (clipped and not inside[-1]) else projected_runs[-1][-1],
        "projected": projected,
        "projected_corners": projected_corners,
        "centred": centred,
        "limits3d": limits3d,
        "matrix": matrix,
        "centre": centre,
        "half_size": half_size,
    }


def _slab_interval(start: np.ndarray, delta: np.ndarray, half_size: float) -> tuple[float, float]:
    """Parameter interval [t0, t1] of the segment start -> start+delta inside the cube."""
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        if abs(float(delta[axis])) < 1e-15:
            if start[axis] < -half_size or start[axis] > half_size:
                return 1.0, 0.0
            continue
        ta = (-half_size - float(start[axis])) / float(delta[axis])
        tb = (half_size - float(start[axis])) / float(delta[axis])
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
        if t0 > t1:
            return 1.0, 0.0
    return t0, t1


def clip_polylines_to_cube(
    xyz_um: np.ndarray,
    times_s: np.ndarray,
    half_size: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a polyline into the pieces that stay inside the cube [-h, h]^3.

    Each piece is cut exactly at the cube surface, so a track that leaves the
    displayed volume is truncated instead of being drawn outside the box.
    """
    h = float(half_size)
    points = np.asarray(xyz_um, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    runs: list[list[list]] = []
    previous_end = None
    for index in range(points.shape[0] - 1):
        start, stop = points[index], points[index + 1]
        delta = stop - start
        t0, t1 = _slab_interval(start, delta, h)
        if t1 <= t0 + 1e-12:
            previous_end = None
            continue
        entry = start + t0 * delta
        exit_ = start + t1 * delta
        entry_time = times[index] + t0 * (times[index + 1] - times[index])
        exit_time = times[index] + t1 * (times[index + 1] - times[index])
        if runs and previous_end == 1.0 and t0 <= 1e-12:
            runs[-1][0].append(exit_)
            runs[-1][1].append(exit_time)
        else:
            runs.append([[entry, exit_], [entry_time, exit_time]])
        previous_end = t1
    return [
        (np.asarray(points_, dtype=np.float64), np.asarray(times_, dtype=np.float64))
        for points_, times_ in runs
    ]


def panel_data_limits(panel: dict[str, object], padding_frac: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Tight 2-D limits around the projected box (which always contains the track)."""
    corners = np.asarray(panel["projected_corners"], dtype=np.float64)
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    span_x = max(x_max - x_min, 1e-9)
    span_y = max(y_max - y_min, 1e-9)
    pad = float(padding_frac) * max(span_x, span_y)
    x_centre, y_centre = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    half_x = 0.5 * span_x + pad
    half_y = 0.5 * span_y + pad
    return (x_centre - half_x, x_centre + half_x), (y_centre - half_y, y_centre + half_y)


def canonical_aspect(elev: float, azim: float, focal_length: float, padding_frac: float) -> float:
    """Width/height of the projected 1:1:1 box (identical for every panel)."""
    matrix = rl.make_projection_matrix(
        (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
        box_aspect=(1.0, 1.0, 1.0), elev=elev, azim=azim, focal_length=focal_length,
    )
    corners = rl.project_points(cube_corners(((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))), matrix)
    span_x = float(np.ptp(corners[:, 0]))
    span_y = float(np.ptp(corners[:, 1]))
    pad = float(padding_frac) * max(span_x, span_y)
    return (span_x + 2.0 * pad) / (span_y + 2.0 * pad)


def scale_grey(grey: float) -> tuple[float, float, float]:
    return (float(grey), float(grey), float(grey))


# Face vertices (corner ids) for the six cube faces, keyed by outward normal.
FACE_VERTICES: dict[tuple[float, float, float], tuple[int, int, int, int]] = {
    normal: vertex_ids for vertex_ids, normal in CUBE_FACES
}

# Candidate projection planes: (outward normal, axis index, which limit, inward sign).
PLANE_CANDIDATES: tuple[tuple[tuple[float, float, float], int, str, float], ...] = (
    ((0.0, 0.0, -1.0), 2, "lo", 1.0),   # floor       -> xy projection
    ((0.0, 0.0, 1.0), 2, "hi", -1.0),   # ceiling     -> xy projection
    ((0.0, -1.0, 0.0), 1, "lo", 1.0),   # front wall  -> xz projection
    ((0.0, 1.0, 0.0), 1, "hi", -1.0),   # back wall   -> xz projection
    ((-1.0, 0.0, 0.0), 0, "lo", 1.0),   # left wall   -> yz projection
    ((1.0, 0.0, 0.0), 0, "hi", -1.0),   # right wall  -> yz projection
)


def hidden_planes(limits3d, view_dir: np.ndarray) -> list[dict[str, object]]:
    """Box faces that point away from the camera (the planes we can project onto)."""
    planes: list[dict[str, object]] = []
    for normal, axis, which, inward in PLANE_CANDIDATES:
        normal_arr = np.asarray(normal, dtype=np.float64)
        if float(normal_arr @ view_dir) >= 0.0:
            continue
        planes.append(
            {
                "normal": normal_arr,
                "axis": int(axis),
                "value": float(limits3d[axis][0] if which == "lo" else limits3d[axis][1]),
                "inward": float(inward),
            }
        )
    return planes


def leftmost_wall(
    walls: list[dict[str, object]],
    corners2d: np.ndarray,
) -> list[dict[str, object]]:
    """Hidden wall whose projected face sits furthest to the left in the image."""
    best, best_x = None, np.inf
    for plane in walls:
        vertex_ids = FACE_VERTICES[tuple(np.asarray(plane["normal"], dtype=np.float64))]
        x_centre = float(np.mean(corners2d[list(vertex_ids), 0]))
        if x_centre < best_x:
            best, best_x = plane, x_centre
    return [best] if best is not None else []


def select_projection_planes(
    planes: list[dict[str, object]],
    mode: str,
    corners2d: np.ndarray,
) -> list[dict[str, object]]:
    """Pick the (hidden) box faces onto which the track is projected."""
    if mode == "none":
        return []
    floor = [plane for plane in planes if int(plane["axis"]) == 2]
    walls = [plane for plane in planes if int(plane["axis"]) != 2]
    if mode == "floor":
        return floor
    if mode == "walls":
        return walls
    if mode == "left-wall":
        return leftmost_wall(walls, corners2d)
    if mode == "floor+left-wall":
        return floor + leftmost_wall(walls, corners2d)
    return list(planes)  # "all"


def draw_projection(ax: plt.Axes, projected: np.ndarray, args: argparse.Namespace) -> None:
    ax.plot(
        projected[:, 0],
        projected[:, 1],
        color=scale_grey(float(args.projection_grey)),
        linewidth=float(args.line_width) * float(args.projection_scale),
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=1.6,
    )


def draw_box(
    ax: plt.Axes,
    panel: dict[str, object],
    *,
    args: argparse.Namespace,
) -> None:
    """Perspective 3-D box: uniform wireframe edges plus optional face shading."""
    limits3d = panel["limits3d"]
    matrix = panel["matrix"]
    corners3d = cube_corners(limits3d)
    corners2d = np.asarray(panel["projected_corners"], dtype=np.float64)
    view_dir = view_direction(float(args.elev), float(args.azim))

    # --- optional face shading -------------------------------------------
    if args.face_mode != "none":
        centre3d = 0.5 * (corners3d.min(axis=0) + corners3d.max(axis=0))
        patches: list[tuple[float, Polygon]] = []
        for vertex_ids, normal in CUBE_FACES:
            normal_arr = np.asarray(normal, dtype=np.float64)
            facing = float(normal_arr @ view_dir)
            hidden = facing <= 0.0
            if (args.face_mode == "hidden") != hidden or abs(facing) < 1e-6:
                continue
            quad3d = corners3d[list(vertex_ids)]
            depth = float(np.mean((quad3d - centre3d) @ view_dir))
            grey = float(np.clip(FACE_GREY[tuple(normal)] - 0.035 * depth, 0.85, 0.995))
            patches.append(
                (
                    depth,
                    Polygon(corners2d[list(vertex_ids)], closed=True, facecolor=scale_grey(grey),
                            edgecolor="none", zorder=1),
                )
            )
        for _, patch in sorted(patches, key=lambda item: item[0]):
            ax.add_patch(patch)

    # --- faint floor grid -------------------------------------------------
    if args.floor_grid:
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, _) = limits3d
        divisions = 6
        xs = np.linspace(x_lo, x_hi, divisions + 1)
        ys = np.linspace(y_lo, y_hi, divisions + 1)
        grid_segments: list[np.ndarray] = []
        for x in xs:
            grid_segments.append(
                np.column_stack(
                    [np.full(2, x), np.asarray([y_lo, y_hi], dtype=np.float64), np.full(2, z_lo)]
                )
            )
        for y in ys:
            grid_segments.append(
                np.column_stack(
                    [np.asarray([x_lo, x_hi], dtype=np.float64), np.full(2, y), np.full(2, z_lo)]
                )
            )
        projected_grid = [rl.project_points(seg, matrix) for seg in grid_segments]
        ax.add_collection(
            LineCollection(projected_grid, colors=scale_grey(float(args.grid_grey)),
                           linewidths=float(args.grid_line_width), zorder=1.5)
        )

    # --- orthographic projections of the track on the hidden faces ---------
    planes = select_projection_planes(
        hidden_planes(limits3d, view_dir), str(args.projection_mode), corners2d
    )
    if planes:
        runs = panel.get("runs") or [(np.asarray(panel["centred"], dtype=np.float64), None)]
        offset = 0.012 * float(panel["half_size"])
        for plane in planes:
            axis = int(plane["axis"])
            for points_, _ in runs:
                proj = np.asarray(points_, dtype=np.float64).copy()
                proj[:, axis] = float(plane["value"]) + float(plane["inward"]) * offset
                projected = rl.project_points(proj, matrix)
                draw_projection(ax, projected, args)
                color=scale_grey(float(args.projection_grey)),
                linewidth=float(args.line_width) * float(args.projection_scale),
    # --- uniform wireframe edges ------------------------------------------
    edge_segments = [corners2d[[a, b]] for a, b in CUBE_EDGES]
    ax.add_collection(
        LineCollection(
            edge_segments,
            colors=[scale_grey(float(args.box_grey))],
            linewidths=float(args.box_line_width),
            capstyle="round",
            joinstyle="round",
            zorder=2,
        )
    )


def draw_trajectory(
    ax: plt.Axes,
    panel: dict[str, object],
    *,
    time_cmap,
    norm: Normalize,
    line_width: float,
    outline_scale: float = 1.9,
) -> None:
    segments = panel["segments"]
    if segments:
        if outline_scale > 0.0:
            # Black halo stroked under the coloured track; --no-trajectory-outline
            # sets outline_scale to 0 and leaves only the coloured line.
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=scale_grey(0.0),
                    linewidths=line_width * float(outline_scale),
                    capstyle="round",
                    joinstyle="round",
                    zorder=3,
                )
            )
        trajectory = LineCollection(
            segments,
            cmap=time_cmap,
            norm=norm,
            linewidths=line_width,
            capstyle="round",
            joinstyle="round",
            zorder=4,
        )
        trajectory.set_array(np.asarray(panel["segment_times"], dtype=np.float64))
        ax.add_collection(trajectory)

    start_xy = panel.get("start_xy")
    end_xy = panel.get("end_xy")
    if start_xy is not None:
        ax.plot([start_xy[0]], [start_xy[1]], marker="o", markersize=2.6,
                markerfacecolor="black", markeredgecolor="white", markeredgewidth=0.35, zorder=6)
    if end_xy is not None:
        ax.plot([end_xy[0]], [end_xy[1]], marker="x", markersize=3.2,
                markeredgewidth=0.9, color="black", zorder=6)


def scale_bar_length_ndc(panel: dict[str, object], length_um: float) -> float:
    """2-D (NDC) length of a ``length_um``-long bar drawn at the box centre.

    ``panel["matrix"]`` is a perspective camera, so the um -> NDC factor is
    measured locally at the box centre and taken as the largest singular value
    of that local Jacobian: that is the factor for a displacement lying in the
    image plane, which is the direction a printed scale bar refers to.
    """
    matrix = np.asarray(panel["matrix"], dtype=np.float64)
    centre = np.zeros((1, 3), dtype=np.float64)
    step_um = 1e-4
    jacobian = np.zeros((2, 3), dtype=np.float64)
    for axis in range(3):
        step = np.zeros((1, 3), dtype=np.float64)
        step[0, axis] = step_um
        forward = rl.project_points(centre + step, matrix)[0]
        backward = rl.project_points(centre - step, matrix)[0]
        jacobian[:, axis] = (forward - backward) / (2.0 * step_um)
    return float(length_um) * float(np.linalg.svd(jacobian, compute_uv=False)[0])


def draw_panel_scale_bar(ax: plt.Axes, panel: dict[str, object], args: argparse.Namespace) -> None:
    """Exact ``--scale-bar-um`` bar in the lower-left corner of a single panel.

    The bar is drawn in axes (NDC) units, where ``scale_bar_length_ndc``
    measured the um -> NDC factor of this panel's own camera, so the bar is
    exactly the requested physical length on that panel - even though every
    panel frames a differently sized cube.
    """
    length_ndc = scale_bar_length_ndc(panel, float(args.scale_bar_um))
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    span_x = float(x_hi) - float(x_lo)
    span_y = float(y_hi) - float(y_lo)

    length_frac = length_ndc / span_x
    # A 1 um bar is a large part of a tightly framed panel, so slide it right
    # only as far as needed to keep the whole bar inside the panel.
    x_frac = min(float(args.scale_bar_x_frac), max(0.015, 0.975 - length_frac))
    x_start = float(x_lo) + x_frac * span_x
    y_bar = float(y_lo) + float(args.scale_bar_y_frac) * span_y
    ax.plot(
        [x_start, x_start + length_ndc],
        [y_bar, y_bar],
        color=str(args.scale_bar_color),
        linewidth=float(args.scale_bar_line_width),
        solid_capstyle="butt",
        zorder=9,
        clip_on=True,
    )


def warn_about_panel_scale(rows: list[tuple[str, list[dict[str, object]]]], args: argparse.Namespace) -> None:
    """Every panel is scaled to its own track unless --common-cube-um was given."""
    if getattr(args, "common_cube_um", None):
        return
    half_sizes = [float(panel["half_size"]) for _, panels in rows for panel in panels]
    if not half_sizes or max(half_sizes) - min(half_sizes) <= 1e-3 * max(half_sizes):
        return
    print(
        "[warn] panels do not share one physical scale "
        f"(cube half-size {min(half_sizes):.2f} - {max(half_sizes):.2f} um), so the "
        f"{float(args.scale_bar_um):g} um scale bar is exact for the bottom-left panel only. "
        "Pass --common-cube-um <um> to give all panels one physical scale."
    )


def draw_scale_bar(
    fig: plt.Figure,
    *,
    bottom_left: dict[str, object],
    args: argparse.Namespace,
    panel_w: float,
    panel_h: float,
    fig_w: float,
    fig_h: float,
    band_in: float = 0.0,
    bottom_margin_in: float = 0.0,
) -> None:
    """One bar for the whole figure, in the empty band below the panels."""
    ax = bottom_left["ax"]
    length_ndc = scale_bar_length_ndc(bottom_left["panel"], float(args.scale_bar_um))

    # Data -> figure fraction through the axes transform, so the bar keeps the
    # panel's um-per-inch scale whatever the aspect adjustment does.
    x_lo, _ = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    y_ref = 0.5 * (float(y_lo) + float(y_hi))
    start = ax.transData.transform((float(x_lo), y_ref))
    end = ax.transData.transform((float(x_lo) + length_ndc, y_ref))
    length_frac = abs(float(end[0]) - float(start[0])) / (float(fig.get_figwidth()) * float(fig.dpi))

    if length_frac > 0.95 * panel_w / fig_w:
        print(
            f"[warn] a {float(args.scale_bar_um):g} um bar covers "
            f"{100.0 * length_frac * fig_w / panel_w:.0f}% of the bottom-left panel width; "
            "consider --common-cube-um or a shorter --scale-bar-um."
        )

    x_frac = (float(bottom_left["x0_in"]) + float(args.scale_bar_x_frac) * panel_w) / fig_w
    y_frac = (float(bottom_margin_in) + 0.5 * float(band_in)) / fig_h
    fig.add_artist(
        Line2D(
            [x_frac, x_frac + length_frac],
            [y_frac, y_frac],
            transform=fig.transFigure,
            color=str(args.scale_bar_color),
            linewidth=float(args.scale_bar_line_width),
            solid_capstyle="butt",
            zorder=10,
        )
    )


def row_label(group_key: str, group_label: str, n: int) -> str:
    if "near1" in group_key:
        base = "\u03b1 \u2248 1"
    elif "gt" in group_key:
        threshold = group_key.split("gt", 1)[1].replace("p", ".")
        base = f"\u03b1 > {float(threshold):g}"
    else:
        base = group_label
    return f"{base}  ({n} tracks)"


def save_orientation_figure(args: argparse.Namespace, output_stem: Path) -> None:
    """Standalone x/y/z triad drawn with exactly the camera of the 2x5 panels."""
    half_size = 1.0
    limits3d = ((-half_size, half_size), (-half_size, half_size), (-half_size, half_size))
    matrix = rl.make_projection_matrix(
        *limits3d,
        box_aspect=(1.0, 1.0, 1.0),
        elev=float(args.elev),
        azim=float(args.azim),
        focal_length=float(args.focal_length),
    )
    corners2d = rl.project_points(cube_corners(limits3d), matrix)
    origin = rl.project_points(np.zeros((1, 3), dtype=np.float64), matrix)[0]

    fig = plt.figure(figsize=(float(args.xyz_size), float(args.xyz_size)), dpi=int(args.dpi))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0.03, 0.03, 0.94, 0.94))
    ax.set_facecolor("white")

    ax.add_collection(
        LineCollection(
            [corners2d[[a, b]] for a, b in CUBE_EDGES],
            colors=[scale_grey(float(args.box_grey) + 0.35)],
            linewidths=float(args.box_line_width),
            capstyle="round",
            joinstyle="round",
            zorder=1,
        )
    )

    font_size = float(args.font_size) * float(args.font_scale)
    label_gap = 0.035 * float(np.ptp(corners2d[:, 0]))
    tips: list[np.ndarray] = []
    for axis_index, axis_label in enumerate(("x", "y", "z")):
        tip3d = np.zeros((1, 3), dtype=np.float64)
        tip3d[0, axis_index] = 0.82 * half_size
        tip = rl.project_points(tip3d, matrix)[0]
        tips.append(tip)
        ax.annotate(
            "",
            xy=(float(tip[0]), float(tip[1])),
            xytext=(float(origin[0]), float(origin[1])),
            arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.1, shrinkA=0.0, shrinkB=0.0),
            zorder=3,
        )
        direction = tip - origin
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-9 else np.asarray([1.0, 0.0])
        ax.text(
            float(tip[0]) + label_gap * float(direction[0]),
            float(tip[1]) + label_gap * float(direction[1]),
            f"${axis_label}$",
            ha="center",
            va="center",
            fontsize=font_size * 1.2,
            color="black",
            zorder=4,
        )
    ax.plot([float(origin[0])], [float(origin[1])], marker="o", markersize=2.2, color="black", zorder=4)

    cloud = np.vstack([corners2d, np.asarray(tips), origin[None, :]])
    x_min, y_min = cloud.min(axis=0)
    x_max, y_max = cloud.max(axis=0)
    pad = 0.10 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        target = output_stem.with_suffix(f".{str(fmt).lower()}")
        fig.savefig(target, dpi=int(args.dpi), facecolor="white")
        print(f"Saved: {target}")
    plt.close(fig)


def resolve_layout(args: argparse.Namespace) -> dict[str, float]:
    """Fill in the per-mode layout defaults for anything the caller left unset."""
    layout = dict(MODE_LAYOUT[str(args.label_mode)])
    for key in ("caption_band", "footnote_band", "row_gap", "left_margin", "top_margin", "bottom_margin"):
        value = getattr(args, key)
        if value is not None:
            layout[key] = float(value)
    return layout


def speed_value_and_unit(speed_um_s: float, args: argparse.Namespace) -> tuple[float, str, str]:
    """Return (converted value, number format, LaTeX unit) for the requested unit."""
    if str(args.speed_unit) == "um_min":
        return float(speed_um_s) * 60.0, ".1f", r"\mu m\,min^{-1}"
    return float(speed_um_s), ".3f", r"\mu m\,s^{-1}"


def build_figure(
    rows: list[tuple[str, list[dict[str, object]]]],
    args: argparse.Namespace,
    output_stem: Path,
    *,
    with_text: bool,
    duration_max: float,
) -> None:
    time_cmap = time_colormap(args)
    norm = Normalize(vmin=0.0, vmax=float(duration_max))
    outline_scale = float(args.trajectory_outline_scale) if bool(args.trajectory_outline) else 0.0
    bottom_left: dict[str, object] | None = None
    panel_axes: list[tuple[plt.Axes, dict[str, object]]] = []

    aspect = canonical_aspect(float(args.elev), float(args.azim), float(args.focal_length), DATA_PAD)
    panel_w = float(args.panel_width)
    panel_h = panel_w / aspect
    n_cols = max(len(panels) for _, panels in rows)
    n_rows = len(rows)
    layout = resolve_layout(args)
    font_scale = float(args.font_scale)

    # The text and text-free versions share exactly the same panel geometry:
    # the text version fills the annotation bands, the text-free one leaves
    # them blank so the two files can be overlaid.
    top_band = layout["caption_band"]
    bottom_band = layout["footnote_band"]

    fig_w = (
        layout["left_margin"]
        + n_cols * panel_w
        + (n_cols - 1) * float(args.col_gap)
        + float(args.right_margin)
    )
    scale_bar_band = (
        float(args.scale_bar_band)
        if bool(args.scale_bar) and str(args.scale_bar_scope) == "figure"
        else 0.0
    )
    fig_h = (
        layout["top_margin"]
        + sum(panel_h + top_band + bottom_band for _ in range(n_rows))
        + (n_rows - 1) * layout["row_gap"]
        + scale_bar_band
        + layout["bottom_margin"]
    )
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(args.dpi))
    fig.patch.set_facecolor("white")

    for row_index, (label, panels) in enumerate(rows):
        y_cursor = fig_h - layout["top_margin"]
        for _ in range(row_index):
            y_cursor -= panel_h + top_band + bottom_band + layout["row_gap"]
        panel_top = y_cursor - top_band
        for col_index, panel in enumerate(panels):
            x0 = layout["left_margin"] + col_index * (panel_w + float(args.col_gap))
            rect = (
                x0 / fig_w,
                (panel_top - panel_h) / fig_h,
                panel_w / fig_w,
                panel_h / fig_h,
            )
            ax = fig.add_axes(rect)
            panel_axes.append((ax, panel))
            ax.set_facecolor("white")
            draw_box(ax, panel, args=args)
            draw_trajectory(ax, panel, time_cmap=time_cmap, norm=norm,
                            line_width=float(args.line_width), outline_scale=outline_scale)
            x_limits, y_limits = panel_data_limits(panel, DATA_PAD)
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()

            if row_index == n_rows - 1 and col_index == 0:
                bottom_left = {
                    "ax": ax,
                    "panel": panel,
                    "x0_in": x0,
                    "bottom_in": panel_top - panel_h,
                }

            if with_text and args.label_mode == "full":
                fig.text(
                    (x0 + 0.5 * panel_w) / fig_w,
                    (panel_top + float(args.speed_offset) * font_scale) / fig_h,
                    rf"$\bar{{v}}$ = {float(panel['mean_speed']):.3f} $\mathrm{{\mu m\,s^{{-1}}}}$",
                    ha="center",
                    va="center",
                    fontsize=float(args.font_size) * font_scale,
                    color="black",
                )
                fig.text(
                    (x0 + 0.5 * panel_w) / fig_w,
                    (panel_top + float(args.alpha_offset) * font_scale) / fig_h,
                    "\u03b1 = " f"{float(panel['alpha']):.3f}",
                    ha="center",
                    va="center",
                    fontsize=float(args.font_size) * font_scale,
                    color="black",
                )
                fig.text(
                    (x0 + 0.5 * panel_w) / fig_w,
                    (panel_top - panel_h - float(args.track_offset) * font_scale) / fig_h,
                    f"track {int(panel['track_id'])}",
                    ha="center",
                    va="center",
                    fontsize=float(args.font_size) * font_scale - 0.3,
                    color="black",
                )
                fig.text(
                    (x0 + 0.5 * panel_w) / fig_w,
                    (panel_top - panel_h - float(args.d_offset) * font_scale) / fig_h,
                    f"D = {float(panel['D']):.4g} \u00b5m\u00b2/s$^{{\\alpha}}$",
                    ha="center",
                    va="center",
                    fontsize=float(args.font_size) * font_scale - 0.3,
                    color="black",
                )
            elif with_text and args.label_mode == "speed":
                value, number_format, unit = speed_value_and_unit(float(panel['mean_speed']), args)
                text = format(value, number_format)
                if col_index == len(panels) - 1:
                    text = text + rf" ($\mathrm{{{unit}}}$)"
                fig.text(
                    (x0 + 0.5 * panel_w) / fig_w,
                    (panel_top + 0.5 * top_band) / fig_h,
                    text,
                    ha="center",
                    va="center",
                    fontsize=float(args.font_size) * font_scale,
                    color="black",
                )

        if with_text and args.label_mode == "full":
            fig.text(
                0.5 * layout["left_margin"] / fig_w,
                (panel_top - 0.5 * panel_h) / fig_h,
                label,
                ha="center",
                va="center",
                rotation=90,
                fontsize=float(args.font_size) * font_scale + 0.4,
                color="black",
            )

    if bool(args.scale_bar):
        # One draw so the axes transforms below already include any aspect fix.
        fig.canvas.draw()
        if str(args.scale_bar_scope) == "figure" and bottom_left is not None:
            warn_about_panel_scale(rows, args)
            draw_scale_bar(
                fig,
                bottom_left=bottom_left,
                args=args,
                panel_w=panel_w,
                panel_h=panel_h,
                fig_w=fig_w,
                fig_h=fig_h,
                band_in=scale_bar_band,
                bottom_margin_in=layout["bottom_margin"],
            )
        else:
            for ax, panel in panel_axes:
                draw_panel_scale_bar(ax, panel, args)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        target = output_stem.with_suffix(f".{str(fmt).lower()}")
        fig.savefig(target, dpi=int(args.dpi), facecolor="white")
        print(f"Saved: {target}")
    plt.close(fig)


def save_colorbar(duration_s: float, args: argparse.Namespace, output_stem: Path) -> None:
    """Standalone time colour bar: no tick values, only the two time endpoints."""
    time_cmap = time_colormap(args)
    norm = Normalize(vmin=0.0, vmax=float(duration_s))

    height = float(args.cbar_height)
    if bool(args.cbar_plain):
        # Bare strip: the canvas is trimmed to the gradient itself.
        width, bar_left, bar_width = 0.26, 0.02, 0.22
        bar_bottom, bar_height = 0.02, height - 0.04
    else:
        width, bar_left, bar_width = 0.62, 0.06, 0.20
        bar_bottom, bar_height = 0.055, 0.89
    fig = plt.figure(figsize=(width, height), dpi=int(args.dpi))
    fig.patch.set_facecolor("white")

    cax = fig.add_axes((bar_left / width, bar_bottom / height, bar_width / width, bar_height / height))
    mappable = ScalarMappable(norm=norm, cmap=time_cmap)
    mappable.set_array(np.asarray([0.0, float(duration_s)]))
    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.set_ticks([])
    colorbar.ax.tick_params(length=0.0)
    if bool(args.cbar_plain):
        # Bare gradient strip: no frame, no text at all.
        colorbar.outline.set_visible(False)
    else:
        colorbar.outline.set_edgecolor("black")
        colorbar.outline.set_linewidth(0.7)
        label_x = (bar_left + bar_width + 0.055) / width
        fig.text(label_x, bar_bottom / height, "0 s", ha="left", va="center",
                 fontsize=float(args.font_size), color="black")
        fig.text(label_x, (bar_bottom + bar_height) / height, f"{float(duration_s):.2f} s",
                 ha="left", va="center", fontsize=float(args.font_size), color="black")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        target = output_stem.with_suffix(f".{str(fmt).lower()}")
        fig.savefig(target, dpi=int(args.dpi), facecolor="white")
        print(f"Saved: {target}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    style_rcparams(args)

    tracks, groups = load_groups(args)

    if args.common_cube_from:
        try:
            row_text, col_text = str(args.common_cube_from).replace(" ", "").split(",")
            ref_row, ref_col = int(row_text), int(col_text)
        except ValueError as exc:
            raise SystemExit("--common-cube-from must look like '2,2' (row,column; 1-based)") from exc
        if not 1 <= ref_row <= len(args.groups):
            raise SystemExit(f"--common-cube-from row {ref_row} is out of range (1-{len(args.groups)})")
        reference = groups[args.groups[ref_row - 1]][1]
        if not 1 <= ref_col <= len(reference):
            raise SystemExit(
                f"--common-cube-from column {ref_col} is out of range (1-{len(reference)} "
                f"for row {ref_row})"
            )
        ref_track_row = reference.iloc[ref_col - 1]
        ref_points = tracks[tracks["track_id"] == int(ref_track_row["track_id"])]
        saved_cube_um = args.common_cube_um
        args.common_cube_um = None
        ref_panel = build_panel(ref_track_row, ref_points, args)
        args.common_cube_um = 2.0 * float(ref_panel["half_size"])
        print(
            f"[scale] row {ref_row}, column {ref_col} (track {int(ref_track_row['track_id'])}) sets the "
            f"common zoom for every panel: cube {float(args.common_cube_um):.3f} um "
            f"(half-size {float(ref_panel['half_size']):.3f} um)"
            + ("" if saved_cube_um is None else f"; overrides --common-cube-um {saved_cube_um:g}")
        )

    rows: list[tuple[str, list[dict[str, object]]]] = []
    durations: list[float] = []
    for row_index, group_key in enumerate(args.groups):
        group_label, group = groups[group_key]
        panels: list[dict[str, object]] = []
        for _, track_row in group.iterrows():
            track_id = int(track_row["track_id"])
            track_points = tracks[tracks["track_id"] == track_id]
            if track_points.empty:
                raise ValueError(f"Track {track_id} missing from {DEFAULT_TRACKS_CSV}")
            panel = build_panel(track_row, track_points, args)
            panels.append(panel)
            durations.append(float(panel["duration_s"]))
            print(
                f"row {row_index + 1} ({group_key}) track {track_id}: "
                f"alpha={float(panel['alpha']):.4f}, D={float(panel['D']):.6g}, "
                f"mean speed={float(panel['mean_speed']):.4f} um/s, "
                f"T={float(panel['duration_s']):.2f} s"
            )
        if args.row_labels and row_index < len(args.row_labels):
            label = str(args.row_labels[row_index])
        else:
            label = row_label(group_key, group_label, len(panels))
        rows.append((label, panels))

    duration_max = float(np.max(durations))
    print(f"Colour-bar range: 0 - {duration_max:.2f} s")

    tag = str(args.output_tag)
    if args.mode == "colorbar":
        save_colorbar(duration_max, args, args.output_dir / f"time_colorbar{tag}")
        return
    if args.mode in ("panels", "both"):
        stem = args.output_dir / ({
            "full": "trajectory_2x5_3dbox",
            "speed": "trajectory_2x5_3dbox_speed_only",
            "none": "trajectory_2x5_3dbox_no_text",
        }[str(args.label_mode)] + tag)

        variant = "" if bool(args.trajectory_outline) else "_nooutline"
        build_figure(rows, args, stem.with_name(stem.name + variant + "_with_text"),
                     with_text=True, duration_max=duration_max)
        save_colorbar(duration_max, args, args.output_dir / "time_colorbar")
    if args.mode in ("xyz", "both"):
        save_orientation_figure(args, args.output_dir / f"xyz_orientation{tag}")


if __name__ == "__main__":
    main()
