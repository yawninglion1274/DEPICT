"""Streamline visibility controlled by speed, path length, and boundary fitting."""
from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import CubicHermiteSpline

MINIMUM_XY_SPEED_UM_S = .25
MINIMUM_PATH_LENGTH = 26.
DISPLAY_SIDE = 431
SEED_DENSITY = 1.6
REFERENCE_WIDTH_INCHES = 7.1
STROKE_WIDTH = 2.*430/(REFERENCE_WIDTH_INCHES*72)
# One grid unit covers bilinear boundary interpolation, plus the visible stroke.
BOUNDARY_CLEARANCE = 1.+STROKE_WIDTH/2+.15
HEAD_GAP = STROKE_WIDTH/2+.15


def visibility_policy():
    return dict(rules=["XY speed threshold", "minimum path length", "boundary containment"],
                speed_quantity="XY planar speed", minimum_speed_um_s=MINIMUM_XY_SPEED_UM_S,
                minimum_path_length_display_units=MINIMUM_PATH_LENGTH,
                boundary="stroke and complete filled head inside the speed-qualified tissue domain",
                stroke_boundary_clearance_display_units=BOUNDARY_CLEARANCE,
                display_grid_shape=[DISPLAY_SIDE, DISPLAY_SIDE], seed_density=SEED_DENSITY,
                streamplot_minlength=0., streamplot_maxlength=1.05,
                component_area_filter=False, curvature_filter=False,
                path_extent_filter=False, head_turn_filter=False, head_spacing_filter=False)


def _sample(array, points):
    return ndi.map_coordinates(array, [points[:, 1], points[:, 0]], order=1,
                               mode="constant", cval=0., prefilter=False)


def _velocity(field, points):
    return np.column_stack([_sample(field[..., 1], points), _sample(field[..., 0], points)])


def _interpolate_path(path, field):
    path = np.asarray(path, dtype=float)
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-7]]
    if len(path) < 2:
        return None
    velocity = _velocity(field, path)
    if np.sum(np.diff(path, axis=0)*velocity[:-1]) < 0:
        path, velocity = path[::-1], velocity[::-1]
    tangent = velocity/np.maximum(np.linalg.norm(velocity, axis=1, keepdims=True), 1e-12)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    spline = CubicHermiteSpline(s, path, tangent, axis=0)
    points = spline(np.linspace(0., s[-1], max(2, int(np.ceil(s[-1]/.65)))+1))
    return points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7]]


def _interval(path, s, lo, hi):
    lo, hi = max(0., float(lo)), min(float(s[-1]), float(hi))
    if hi <= lo:
        return None
    endpoints = np.array([[np.interp(v, s, path[:, k]) for k in range(2)] for v in (lo, hi)])
    return np.vstack([endpoints[0], path[(s > lo) & (s < hi)], endpoints[1]])


def _triangle_samples(vertices):
    sides = vertices-np.roll(vertices, 1, axis=0)
    n = max(2, int(np.ceil(np.linalg.norm(sides, axis=1).max()/.65)))
    i, j = np.indices((n+1, n+1))
    inside = i+j <= n
    u, v = i[inside, None]/n, j[inside, None]/n
    return vertices[0]+u*(vertices[1]-vertices[0])+v*(vertices[2]-vertices[0])


def build_geometry(field, domain, *, triangle_length, minimum_path_length=MINIMUM_PATH_LENGTH):
    """Return complete paths and heads using only length and boundary tests.

    ``domain`` already encodes the speed rule. Field components are (row, column)
    on the display grid; returned coordinates are (column, row).
    """
    field, domain = np.asarray(field), np.asarray(domain, dtype=bool)
    if field.ndim != 3 or field.shape[-1] != 2 or domain.shape != field.shape[:2] or min(domain.shape) < 2:
        raise ValueError("Expected a rectangular field with two components and a matching domain")
    height, width = domain.shape
    if not np.isfinite(triangle_length) or triangle_length <= 0:
        raise ValueError("triangle_length must be positive and finite")
    if not np.isfinite(minimum_path_length) or minimum_path_length <= 0:
        raise ValueError("minimum_path_length must be positive and finite")
    domain = domain & np.isfinite(field).all(-1) & (np.linalg.norm(field, axis=-1) > 1e-6)
    interpolation_field = np.where(np.isfinite(field), field, 0.)
    clearance = ndi.distance_transform_edt(np.pad(domain, 1))[1:-1, 1:-1]
    valid = domain & (clearance >= BOUNDARY_CLEARANCE)
    valid_float = valid.astype(float)
    stats = dict(seed_paths=0, retained_paths=0, line_parts=0, filled_triangles=0,
                 paths_without_triangle=0, paths_rejected_by_length=0,
                 rejected_head_boundary_candidates=0,
                 minimum_path_length_display_units=float(minimum_path_length),
                 triangle_length_display_units=float(triangle_length),
                 minimum_retained_path_length=None, line_min_sampled_valid_mask=None,
                 arrow_min_sampled_valid_mask=None, arrow_field_min_cosine=None,
                 field_modified=False, vector_field_smoothing_applied=False,
                 component_area_filter=False, curvature_filter=False, path_extent_filter=False,
                 head_turn_filter=False, head_spacing_filter=False)
    if not valid.any():
        return dict(parts=[], triangles=[], paths=[], statistics=stats)
    temp, ax = plt.subplots(figsize=(6, 6))
    try:
        stream = ax.streamplot(np.arange(width), np.arange(height),
            np.ma.array(interpolation_field[..., 1], mask=~valid),
            np.ma.array(interpolation_field[..., 0], mask=~valid),
            density=(SEED_DENSITY if height == width else
                     (SEED_DENSITY, SEED_DENSITY*(height-1)/(width-1))),
            minlength=0., maxlength=1.05,
            broken_streamlines=True, arrowsize=.001, integration_direction="both")
        seeds = [np.asarray(p).copy() for p in stream.lines.get_segments()]
    finally:
        plt.close(temp)
    if seeds and max(map(len, seeds)) == 2:
        joined = []
        for segment in seeds:
            if joined and np.linalg.norm(joined[-1][-1]-segment[0]) < 1e-5:
                joined[-1].append(segment[-1])
            else:
                joined.append(list(segment))
        seeds = [np.asarray(p) for p in joined]
    stats["seed_paths"] = len(seeds)
    paths = []
    for seed in seeds:
        points = _interpolate_path(seed, interpolation_field)
        if points is None:
            continue
        inside = _sample(valid_float, points) >= .999
        breaks = np.flatnonzero(np.diff(np.r_[False, inside, False]))
        for begin, end in zip(breaks[::2], breaks[1::2]):
            path = points[begin:end]
            if len(path) < 2:
                continue
            if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < minimum_path_length:
                stats["paths_rejected_by_length"] += 1
                continue
            paths.append(path)
    parts, triangles, lengths, line_minima, head_minima, cosines = [], [], [], [], [], []
    for path in paths:
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        lengths.append(float(s[-1]))
        line_minima.append(float(_sample(valid_float, path).min()))
        candidates = np.flatnonzero((s > .4*triangle_length+HEAD_GAP)
                                   & (s < s[-1]-.6*triangle_length-HEAD_GAP))
        candidates = candidates[np.argsort(abs(s[candidates]/s[-1]-.55), kind="stable")]
        selected = None
        for k in candidates:
            vector = _velocity(interpolation_field, path[k:k+1])[0]
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm <= 1e-6:
                continue
            tangent = vector/norm
            normal = np.array([-tangent[1], tangent[0]])
            vertices = np.stack([path[k]+.6*triangle_length*tangent,
                                 path[k]-.4*triangle_length*tangent+.29*triangle_length*normal,
                                 path[k]-.4*triangle_length*tangent-.29*triangle_length*normal])
            mask_min = float(_sample(valid_float, _triangle_samples(vertices)).min())
            if mask_min < .999:
                stats["rejected_head_boundary_candidates"] += 1
                continue
            triangles.append(vertices)
            head_minima.append(mask_min)
            tip = vertices[0]-(vertices[1]+vertices[2])*.5
            cosines.append(float(np.dot(tip, vector)/(np.linalg.norm(tip)*norm)))
            selected = int(k)
            break
        if selected is None:
            parts.append(path)
            stats["paths_without_triangle"] += 1
        else:
            for lo, hi in ((0., s[selected]-.4*triangle_length-HEAD_GAP),
                           (s[selected]+.6*triangle_length+HEAD_GAP, s[-1])):
                piece = _interval(path, s, lo, hi)
                if piece is not None:
                    parts.append(piece)
    stats.update(retained_paths=len(paths), line_parts=len(parts), filled_triangles=len(triangles),
                 minimum_retained_path_length=min(lengths) if lengths else None,
                 line_min_sampled_valid_mask=min(line_minima) if line_minima else None,
                 arrow_min_sampled_valid_mask=min(head_minima) if head_minima else None,
                 arrow_field_min_cosine=min(cosines) if cosines else None)
    return dict(parts=parts, triangles=triangles, paths=paths, statistics=stats)


def prepare_flow_geometry(flow, tissue_mask, *, pixel_um, fps, triangle_length, components=(1, 2)):
    if flow.shape != (3, *tissue_mask.shape) or min(tissue_mask.shape) < 2:
        raise ValueError("Expected a rectangular mask and a matching (dz,dy,dx) flow")
    row_component, column_component = components
    height, width = tissue_mask.shape
    factor = (DISPLAY_SIDE-1)/(width-1)
    columns = np.linspace(0, width-1, DISPLAY_SIDE)
    rows = columns if height == width else np.arange(int(np.floor((height-1)*factor))+1)/factor
    rr, cc = np.meshgrid(rows, columns, indexing="ij")
    sample = lambda a: ndi.map_coordinates(a, [rr, cc], order=1, prefilter=False)
    visible = sample(tissue_mask.astype(float)) > .999
    planar_speed = sample(np.hypot(flow[row_component], flow[column_component])*pixel_um*fps)
    domain = visible & (planar_speed >= MINIMUM_XY_SPEED_UM_S)
    field = np.stack([sample(component) for component in (flow[row_component], flow[column_component])], axis=-1)*factor
    before = hashlib.sha256(field.tobytes()+domain.tobytes()).hexdigest()
    geometry = build_geometry(field, domain, triangle_length=triangle_length)
    assert before == hashlib.sha256(field.tobytes()+domain.tobytes()).hexdigest()
    for name in ("parts", "triangles", "paths"):
        geometry[name] = [p/factor for p in geometry[name]]
    geometry["display_field_mask_sha256"] = before
    return geometry
