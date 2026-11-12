"""Trajectory camera and projection geometry."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def make_projection_matrix(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
    *,
    box_aspect: tuple[float, float, float],
    elev: float,
    azim: float,
    focal_length: float,
) -> np.ndarray:
    """Return the 4x4 mplot3d camera matrix for an arbitrary display box."""
    if x_limits[1] <= x_limits[0] or y_limits[1] <= y_limits[0] or z_limits[1] <= z_limits[0]:
        raise ValueError(f"Invalid display limits: x={x_limits}, y={y_limits}, z={z_limits}")
    fig = plt.figure(figsize=(4.8, 4.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(*x_limits)
    # The XY projection uses an upper-left image origin.
    ax.set_ylim(y_limits[1], y_limits[0])
    ax.set_zlim(*z_limits)
    ax.set_box_aspect(box_aspect)
    ax.set_proj_type("persp", focal_length=focal_length)
    ax.view_init(elev=elev, azim=azim)
    matrix = ax.get_proj()
    plt.close(fig)
    return np.asarray(matrix, dtype=np.float64)

def project_points_ndc3(points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Project (N, 3) centred-um points to normalised device coordinates (x, y, z)."""
    points = np.asarray(points_xyz, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    clip = homogeneous @ matrix.T
    w = clip[:, 3]
    if np.any(np.abs(w) < 1e-12):
        raise FloatingPointError("A trajectory point projects to a zero homogeneous coordinate.")
    return clip[:, :3] / w[:, None]

def project_points(points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Project (N, 3) centred-um points to NDC (x, y)."""
    return project_points_ndc3(points_xyz, matrix)[:, :2]

def box_corners(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in x_limits
            for y in y_limits
            for z in z_limits
        ],
        dtype=np.float64,
    )
