"""Full native-volume coarse-to-fine dense 3D TV-L1 displacement."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .volume_io import digest, signature, write_json

DEFAULTS = dict(attachment=15., tightness=.3, num_warp=5, num_iter=10,
                tol=1e-4, prefilter=False, ri_range=[0., .06])


def dense_flow(reference, moving, *, parameters=None, device="auto"):
    """Return float32 (dz,dy,dx), in native voxels/frame; never subsample."""
    import torch
    from ._tvl1 import tvl1_3d_pyramid

    settings = {**DEFAULTS, **(parameters or {})}
    lo, hi = settings.pop("ri_range")
    reference, moving = np.asarray(reference), np.asarray(moving)
    if reference.ndim != 3 or reference.shape != moving.shape or min(reference.shape) < 2:
        raise ValueError("Expected two matching native ZYX volumes")
    if not np.isfinite(reference).all() or not np.isfinite(moving).all() or hi <= lo:
        raise ValueError("Nonfinite volume or invalid flow normalization range")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reference = np.ascontiguousarray((np.clip(reference.astype(np.float32), lo, hi)-lo)/np.float32(hi-lo))
    moving = np.ascontiguousarray((np.clip(moving.astype(np.float32), lo, hi)-lo)/np.float32(hi-lo))
    with torch.no_grad():
        field = tvl1_3d_pyramid(torch.from_numpy(reference)[None].to(device),
                                torch.from_numpy(moving)[None].to(device), **settings)[0].cpu().numpy()
    if field.shape != (3, *reference.shape) or field.dtype != np.float32 or not np.isfinite(field).all():
        raise ValueError("Invalid dense TV-L1 output")
    return field


def speed_um_s(flow, *, voxel_um, fps):
    if not np.isfinite(voxel_um) or voxel_um <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Calibration and frame rate must be positive")
    return np.linalg.norm(flow, axis=0)*voxel_um*fps


class FlowStore:
    """Read explicit dense caches or compute missing pairs with the same recipe."""

    def __init__(self, filtered, output, *, parameters=None, device="auto", external_cache=None):
        self.filtered = filtered
        self.parameters = {**DEFAULTS, **(parameters or {})}
        self.device = device
        self.external = external_cache
        if external_cache and external_cache["parameters"] != self.parameters:
            raise ValueError("External flow recipe differs; use --no-external-cache")
        contract = dict(kalman=signature(filtered.path), parameters=self.parameters,
                        implementation=[signature(__file__), signature(Path(__file__).with_name("_tvl1.py"))])
        self.output = Path(output) / digest(contract)[:20]
        self.output.mkdir(parents=True, exist_ok=True)
        self.used = {}

    def get(self, frame):
        frame = int(frame)
        if frame not in self.filtered.frames or frame+1 not in self.filtered.frames:
            raise ValueError(f"Flow pair {frame}->{frame+1} is not covered by the filtered volume")
        path = self.output / f"flow_{frame:03d}_{frame+1:03d}.npy"
        reused = False
        if self.external:
            candidate = Path(self.external["pattern"].format(frame=frame, right=frame+1))
            if candidate.exists():
                path, reused = candidate, True
        if not path.exists():
            field = dense_flow(self.filtered.volume(frame), self.filtered.volume(frame+1),
                               parameters=self.parameters, device=self.device)
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, field)
            temporary.replace(path)
            write_json(path.with_suffix(".json"), dict(native_pair=[frame, frame+1],
                parameters=self.parameters, components=["dz", "dy", "dx"], units="native voxel/frame",
                finite=True, full_native_volume=True))
        field = np.load(path, mmap_mode="r")
        if field.shape != (3, *self.filtered.array.shape[1:]) or field.dtype != np.float32:
            raise ValueError(f"Dense flow shape/dtype mismatch: {path}")
        self.used[frame] = dict(**signature(path), explicit_external_cache=reused)
        return field
