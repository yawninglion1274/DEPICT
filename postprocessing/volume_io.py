"""Volume-series input and native-coordinate section extraction."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


def resolve(path, base):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


def signature(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty table")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_array(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return tifffile.memmap(path, mode="r")
    except ValueError:
        return tifffile.imread(path)


def read_z(path, z):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(read_array(path)[z], dtype=np.float32)
    with tifffile.TiffFile(path) as handle:
        if len(handle.pages) > 1:
            return handle.pages[z].asarray().astype(np.float32, copy=False)
        array = handle.asarray()
    if array.ndim != 3:
        raise ValueError(f"Expected a ZYX volume: {path}, {array.shape}")
    return array[z].astype(np.float32, copy=False)


class VolumeSeries:
    """A contiguous native time series: CSV/explicit ZYX files or one TZYX array.

    CSV paths are relative to the CSV, while config paths are relative to the
    config. Optional CSV bytes/mtime_ns fields pin existing cache inputs.
    """

    def __init__(self, config, base):
        self.stack_path = None
        if "manifest" in config:
            manifest = resolve(config["manifest"], base)
            with manifest.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.frames = np.array([int(r["frame_idx"]) for r in rows])
            self.paths = tuple(resolve(r["path"], manifest.parent) for r in rows)
            for row, path in zip(rows, self.paths):
                actual = signature(path)
                for name in ("bytes", "mtime_ns"):
                    if row.get(name) and int(row[name]) != actual[name]:
                        raise ValueError(f"Input changed since manifest was written: {path}")
        elif "paths" in config:
            self.paths = tuple(resolve(p, base) for p in config["paths"])
            self.frames = np.array(config.get("frame_indices", range(len(self.paths))), dtype=int)
        elif "path" in config:
            self.stack_path = resolve(config["path"], base)
            array = read_array(self.stack_path)
            if array.ndim != 4:
                raise ValueError("A time-series array must have shape (T,Z,Y,X)")
            first = int(config.get("first_frame", 0))
            self.frames = np.arange(first, first + len(array))
            self.paths = ()
        else:
            raise ValueError("source needs manifest, paths, or a TZYX path")
        if len(self.frames) < 2 or np.any(np.diff(self.frames) != 1):
            raise ValueError("Provide at least two consecutive, ordered native frames")
        if self.stack_path is None and len(self.paths) != len(self.frames):
            raise ValueError("Number of frame indices and files differs")
        if self.stack_path:
            self.shape = tuple(read_array(self.stack_path).shape[1:])
        else:
            path = self.paths[0]
            if path.suffix.lower() == ".npy":
                self.shape = tuple(read_array(path).shape)
            else:
                with tifffile.TiffFile(path) as handle:
                    self.shape = tuple(handle.series[0].shape)
        if len(self.shape) != 3 or min(self.shape) < 2:
            raise ValueError(f"Expected native ZYX volumes, got {self.shape}")
        self.signatures = [signature(p) for p in ((self.stack_path,) if self.stack_path else self.paths)]

    def plane(self, z):
        if not 0 <= z < self.shape[0]:
            raise ValueError(f"z={z} outside volume shape {self.shape}")
        if self.stack_path:
            result = np.array(read_array(self.stack_path)[:, z], dtype=np.float32)
        else:
            result = np.stack([read_z(path, z) for path in self.paths])
        if result.shape != (len(self.frames), *self.shape[1:]) or not np.isfinite(result).all():
            raise ValueError("Inconsistent or nonfinite input volume planes")
        return result


def cut_scalar(volume, section):
    """Cut XY or displayed XZ; a 90-degree XY rotation also transforms XZ."""
    plane, index = section["plane"], int(section["index"])
    rotation = int(section.get("rotation_cw", 0))
    if rotation not in (0, 90):
        raise ValueError("Supported display rotations: 0 or 90 degrees clockwise")
    if plane == "xy":
        result = volume[..., index, :, :]
        if rotation:
            result = np.rot90(result, -1, axes=(-2, -1))
    elif plane == "xz":
        if rotation:
            # Copy in native storage order before reversing; backward mmap reads
            # otherwise turn a large YZ extraction into millions of random reads.
            result = np.array(volume[..., :, :, index])
            return result[..., ::-1].copy()
        result = volume[..., :, index, :]
    else:
        raise ValueError(f"Unknown section plane: {plane}")
    return np.array(result)


def cut_flow(flow, section):
    result = cut_scalar(flow, section)
    if section.get("rotation_cw", 0):
        dz, dy, dx = result
        result = np.stack([dz, dx, -dy])
    return result


def crop(array, bounds=None):
    if bounds is None:
        return array
    y0, y1, x0, x1 = map(int, bounds)
    if not (0 <= y0 < y1 <= array.shape[-2] and 0 <= x0 < x1 <= array.shape[-1]):
        raise ValueError(f"Crop {bounds} outside section shape {array.shape[-2:]}")
    return array[..., y0:y1, x0:x1]
