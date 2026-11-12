"""Render Figure 3 image panels and zoom regions from local volume inputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from . import sections as source

ROI_CENTERS_XZ = {'293t': (137, 371), 'u2os': (118, 263), 'hela': (135, 358), 'tcell': (105, 299), 'cos7': (153, 343), 'tcell_mitosis': (83, 301)}
ROI_SIZE_PX = 50
PANEL_GAP_PX = 12
ROI_COLOR = (255, 255, 0)
OUTPUT_DPI = 300
ZOOM_ORDER_TOP_TO_BOTTOM = ("rytov", "bpm", "tide")

def load_panel_slices(source, cell):
    cfg = source.CELL_CONFIGS[cell]
    slices = {}
    for method in ("tide", "rytov", "bpm"):
        orientation = {"rotate_xy": cfg["rotate_xy"]} if "rotate_xy" in cfg else {}
        xy, xz = source.load_slices(
            cfg[method + "_path"],
            cfg["z_idx"],
            cfg["y_idx"],
            include_xy=(method == "tide"),
            offset=cfg.get("offset_" + method, 0.0),
            swap_xy=cfg.get("swap_xy", False),
            **orientation,
        )
        if cfg.get("flip_ud", False):
            xz = np.flipud(xz)
            if xy is not None:
                xy = np.flipud(xy)
        if method == "tide":
            slices["xy"] = xy
        if xz.shape != (450, 256):
            raise ValueError("Unexpected XZ shape for {}: {}".format(cell, xz.shape))
        slices[method] = xz
    slices["comparison"] = source.combine_rytov_bpm(slices["rytov"], slices["bpm"])
    return slices

def roi_bounds(cell, shape):
    cx, cy = ROI_CENTERS_XZ[cell]
    if not isinstance(ROI_SIZE_PX, int) or ROI_SIZE_PX <= 0:
        raise ValueError("ROI_SIZE_PX must be a positive integer")
    if any(not isinstance(value, int) for value in (cx, cy)):
        raise ValueError("ROI centers must be integer (x, y) pixel coordinates")
    x0 = cx - ROI_SIZE_PX // 2
    y0 = cy - ROI_SIZE_PX // 2
    x1, y1 = x0 + ROI_SIZE_PX, y0 + ROI_SIZE_PX
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
        raise ValueError("ROI for {} is outside XZ shape {}".format(cell, shape))
    return (x0, y0, x1, y1)

def layout_boxes(size, single_size):
    width, height = size
    gap = max(1, round(PANEL_GAP_PX * height / single_size[1]))
    zoom_width = (height - 2 * gap) // 3
    available_width = width - zoom_width - 3 * gap
    if zoom_width < 1 or available_width < 3:
        raise ValueError("Output size is too small for the requested panel gap")

    # Use the same scale in both directions and crop equal widths from each view.
    scale = height / 450
    removed_width = ((450 + 256 + 256) * scale - available_width) / 3
    xz_width = round(256 * scale - removed_width)
    widths = (available_width - 2 * xz_width, xz_width, xz_width)
    if min(widths) < 1 or removed_width < 0:
        raise ValueError("Output size cannot fit the views using center crops")
    boxes = {}
    viewports = {}
    left = 0
    for name, native_width, panel_width in zip(
        ("xy", "tide", "comparison"), (450, 256, 256), widths
    ):
        boxes[name] = (left, 0, left + panel_width, height)
        source_left = (native_width - panel_width / scale) / 2
        viewports[name] = {
            "source_bounds_xyxy": (source_left, 0, native_width - source_left, 450),
            "pixels_per_source_pixel": scale,
        }
        left += panel_width + gap

    left = width - zoom_width
    zoom_top = (height - (3 * zoom_width + 2 * gap)) // 2
    for row, method in enumerate(ZOOM_ORDER_TOP_TO_BOTTOM):
        top = zoom_top + row * (zoom_width + gap)
        boxes["zoom_" + method] = (left, top, width, top + zoom_width)
    return boxes, gap, viewports

def grayscale_image(array, source):
    # Map the fixed RI range through Matplotlib's gray lookup table.
    rgba = source.plt.get_cmap("gray")(
        source.matplotlib.colors.Normalize(source.VMIN, source.VMAX)(array), bytes=True
    )
    return Image.fromarray(rgba[:, :, :3], mode="RGB")

def paste_cropped_view(canvas, array, box, viewport, source):
    image = grayscale_image(array, source)
    scale = viewport["pixels_per_source_pixel"]
    source_left = viewport["source_bounds_xyxy"][0]
    size = (box[2] - box[0], box[3] - box[1])
    # One scale sets the export resolution; layout only changes the crop.
    image = image.transform(
        size, Image.Transform.AFFINE,
        (1 / scale, 0, source_left, 0, 1 / scale, 0),
        resample=Image.Resampling.NEAREST,
    )
    canvas.paste(image, box[:2])

def paste_zoom(canvas, array, box, source):
    image = grayscale_image(array, source)
    size = (box[2] - box[0], box[3] - box[1])
    # Nearest-neighbor enlargement preserves measured pixels in the zoom views.
    image = image.resize(size, resample=Image.Resampling.NEAREST)
    canvas.paste(image, box[:2])

def dashed_line(draw, start, end, color, width, dash, gap):
    x0, y0 = start
    x1, y1 = end
    distance = float(np.hypot(x1 - x0, y1 - y0))
    if distance == 0:
        return
    for offset in np.arange(0, distance, dash + gap):
        stop = min(offset + dash, distance)
        a = (x0 + (x1 - x0) * offset / distance, y0 + (y1 - y0) * offset / distance)
        b = (x0 + (x1 - x0) * stop / distance, y0 + (y1 - y0) * stop / distance)
        draw.line((a, b), fill=color, width=width)

def draw_roi(draw, panel_box, crop, viewport):
    left, top, _, _ = panel_box
    x0, y0, x1, y1 = crop
    source_left, _, source_right, _ = viewport["source_bounds_xyxy"]
    if not (source_left <= x0 < x1 <= source_right):
        raise ValueError("ROI is outside the visible center-cropped XZ view")
    scale = viewport["pixels_per_source_pixel"]
    roi_left = round(left + (x0 - source_left) * scale)
    roi_top = round(top + y0 * scale)
    side = round((x1 - x0) * scale)
    if y1 - y0 != x1 - x0:
        raise ValueError("ROI must be square")
    points = [
        (roi_left, roi_top),
        (roi_left + side, roi_top),
        (roi_left + side, roi_top + side),
        (roi_left, roi_top + side),
    ]
    stroke = 2 * max(1, round(0.8 * scale))
    for index in range(4):
        dashed_line(
            draw, points[index], points[(index + 1) % 4],
            ROI_COLOR, stroke, max(2, 3 * scale), max(2, 2 * scale),
        )
    return (roi_left, roi_top, roi_left + side, roi_top + side)

def render_panel(source, cell, slices, size, single_size):
    boxes, gap, viewports = layout_boxes(size, single_size)
    canvas = Image.new("RGB", size, "white")
    crop = roi_bounds(cell, slices["tide"].shape)
    x0, y0, x1, y1 = crop
    for name in ("xy", "tide", "comparison"):
        paste_cropped_view(canvas, slices[name], boxes[name], viewports[name], source)
    for method in ZOOM_ORDER_TOP_TO_BOTTOM:
        zoom = slices[method][y0:y1, x0:x1]
        if zoom.shape != (ROI_SIZE_PX, ROI_SIZE_PX):
            raise ValueError("Incomplete ROI crop for " + cell)
        paste_zoom(canvas, zoom, boxes["zoom_" + method], source)

    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = boxes["comparison"]
    height = bottom - top
    scale = viewports["comparison"]["pixels_per_source_pixel"]
    source_left, _, source_right, _ = viewports["comparison"]["source_bounds_xyxy"]
    slope = 0.5 * 449 / 255
    y_left = top + (0.25 * 449 + slope * (source_left - 0.5) + 0.5) * scale
    y_right = top + (0.25 * 449 + slope * (source_right - 0.5) + 0.5) * scale
    draw.line(
        ((left, y_left), (right - 1, y_right)),
        fill="white", width=max(1, round(0.8 * OUTPUT_DPI / 72 * height / single_size[1])),
    )
    roi_boxes = {
        name: draw_roi(draw, boxes[name], crop, viewports[name])
        for name in ("tide", "comparison")
    }

    if cell == "293t":
        left, top, right, bottom = boxes["xy"]
        scale = viewports["xy"]["pixels_per_source_pixel"]
        length = round(5.0 / source.VOXEL_SIZE_UM)
        bar_right = right - 18 * scale
        draw.line(
            ((bar_right - length * scale, top + 28 * scale),
             (bar_right, top + 28 * scale)),
            fill="white", width=max(1, round(3 * OUTPUT_DPI / 72 * height / single_size[1])),
        )
    return canvas, {
        "size_px": list(size), "gap_px": gap, "boxes_xyxy": boxes,
        "viewports": viewports, "roi_boxes_xyxy": roi_boxes,
    }

def save_tiff(image, path):
    image.save(path, format="TIFF", compression="tiff_lzw", dpi=(OUTPUT_DPI, OUTPUT_DPI))
    with Image.open(path) as saved:
        if saved.size != image.size or saved.mode != "RGB" or saved.n_frames != 1:
            raise RuntimeError("Unexpected TIFF dimensions or mode: " + str(path))
    print("Saved: {} ({} x {})".format(path, *image.size), flush=True)

def build_images(root, config, selected):
    records = {}
    for panel in config["panels"]:
        if panel["kind"] != "image" or panel["label"] not in selected:
            continue
        cell = panel["cell"]
        cfg = dict(panel["display"])
        cfg.update({{"depict":"tide", "bpm":"bpm", "rytov":"rytov"}[method]+"_path":
                    str((root/path).resolve()) for method,path in config["volumes"][cell].items()})
        source.CELL_CONFIGS[cell] = cfg
        ROI_CENTERS_XZ[cell] = tuple(panel["roi_center_xy"])
        slices = load_panel_slices(source, cell)
        size = tuple(panel["size_px"])
        image, geometry = render_panel(source, cell, slices, size, size)
        output = root/("panel_"+panel["label"])
        output.mkdir(exist_ok=True)
        save_tiff(image, output/("panel_"+panel["label"]+".tif"))
        np.savez_compressed(output/"section_data.npz", **slices)
        records[panel["label"]] = dict(cell=cell, geometry=geometry)
    return records
