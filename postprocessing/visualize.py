"""Clean RI and speed TIFFs with the shared three-rule streamline renderer."""
from __future__ import annotations

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "depict-matplotlib"))
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import matplotlib.patheffects as patheffects
import numpy as np
import tifffile

from . import _streamlines
from .optical_flow import speed_um_s

REFERENCE_PANEL_INCHES = 7.1
REFERENCE_HEAD_LENGTH_GRID = ((5.1*2/.95)*(5.8*72/380))*430/(REFERENCE_PANEL_INCHES*72)
DEFAULTS = dict(ri_range=[0., .06], speed_range=[0., 6.5], tile_width_px=1000, dpi=300, gap_px=12)


def visibility_policy():
    return _streamlines.visibility_policy()


def prepare_flow_geometry(flow, mask, *, voxel_um, fps, components=(1, 2)):
    return _streamlines.prepare_flow_geometry(flow, mask, pixel_um=voxel_um, fps=fps,
        triangle_length=REFERENCE_HEAD_LENGTH_GRID, components=components)


def draw_flow_geometry(ax, geometry):
    """Draw white streamlines with black outlines and filled arrowheads."""
    width_inches = ax.get_position().width*ax.figure.get_figwidth()
    scale = width_inches/REFERENCE_PANEL_INCHES
    lines = LineCollection(geometry["parts"], colors="white", linewidths=2*scale,
                           alpha=.97, capstyle="round", joinstyle="round", zorder=5)
    lines.set_path_effects([patheffects.Stroke(linewidth=3.2*scale, foreground="black",
        capstyle="round", joinstyle="round"), patheffects.Normal()])
    ax.add_collection(lines)
    for triangle in geometry["triangles"]:
        marker = Polygon(triangle, closed=True, facecolor="white", edgecolor="none",
                         linewidth=0, antialiased=True, zorder=6)
        marker.set_path_effects([patheffects.Stroke(linewidth=1.2*scale, foreground="black",
            joinstyle="round"), patheffects.Normal()])
        ax.add_patch(marker)


def save_clean_tiff(path, rgb, *, dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Clean TIFF exports must be uint8 RGB")
    tifffile.imwrite(path, rgb, photometric="rgb", compression="deflate", metadata=None,
                      resolution=(dpi, dpi), resolutionunit="INCH")


def render_clean(image, *, limits, mask=None, geometry=None, tile_width_px=1000, dpi=300):
    """Return a tightly cropped RGB image without labels, bars or axes."""
    image = np.asarray(image)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2D image")
    if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("Invalid display limits")
    width = int(tile_width_px)
    height = max(1, int(round(width*image.shape[0]/image.shape[1])))
    if width < 2 or dpi <= 0:
        raise ValueError("Invalid export size or DPI")
    fig = Figure(figsize=(width/dpi, height/dpi), dpi=dpi, facecolor="black")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    if mask is None:
        cmap, data = "gray", image
    else:
        if mask.shape != image.shape:
            raise ValueError("Mask and displayed image dimensions differ")
        cmap = matplotlib.colormaps["magma"].with_extremes(bad="black")
        data = np.ma.masked_where(~mask, image)
    ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="nearest")
    ax.set(xlim=(-.5, image.shape[1]-.5), ylim=(image.shape[0]-.5, -.5),
           aspect="equal", facecolor="black")
    ax.axis("off")
    if geometry is not None:
        draw_flow_geometry(ax, geometry)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    fig.clear()
    return rgb


def render_speed(flow, mask, *, voxel_um, fps, components=(1, 2), settings=None):
    settings = {**DEFAULTS, **(settings or {})}
    geometry = prepare_flow_geometry(flow, mask, voxel_um=voxel_um, fps=fps, components=components)
    speed = speed_um_s(flow, voxel_um=voxel_um, fps=fps)
    rgb = render_clean(speed, limits=settings["speed_range"], mask=mask, geometry=geometry,
                       tile_width_px=settings["tile_width_px"], dpi=settings["dpi"])
    return rgb, geometry, speed


def join_tiles(images, *, gap_px=12, axis=1):
    images = [np.asarray(image) for image in images]
    if not images or axis not in (0, 1) or int(gap_px) != gap_px or gap_px < 0:
        raise ValueError("Provide images, a nonnegative gap and axis 0 or 1")
    across = 1-axis
    if len({image.shape[across] for image in images}) != 1:
        raise ValueError("Tile dimensions differ along the joining edge")
    shape = list(images[0].shape)
    shape[axis] = int(gap_px)
    gap = np.full(shape, 255, dtype=np.uint8)
    pieces = []
    for index, image in enumerate(images):
        if index:
            pieces.append(gap)
        pieces.append(image)
    return np.concatenate(pieces, axis=axis)
