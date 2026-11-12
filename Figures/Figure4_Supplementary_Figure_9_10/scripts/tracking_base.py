"""Baseline linking and joining of measured track fragments."""
from __future__ import annotations
import numpy as np
import pandas as pd
from laptrack_compat import LapTrack
FPS=16.25
VOXEL_UM=0.09
LINK_UM=5.0/FPS
LINK_PX=LINK_UM/VOXEL_UM
GAP_MAX_FRAME_DIFFERENCE=2
POSITION_COLUMNS=["z","y","x"]
FINAL_COLUMNS=["track_id","frame","detection_id","source_detection_ids","position_origin","z","y","x","z_um","y_um","x_um","voxel_count","max_value"]

def position_row(row: pd.Series, track_id: int, origin: str = "primary") -> dict:
    pos = row[POSITION_COLUMNS].to_numpy(dtype=float)
    detection_id = int(row["detection_id"])
    return {
        "track_id": track_id,
        "frame": int(row["frame"]),
        "detection_id": detection_id,
        "source_detection_ids": str(detection_id),
        "position_origin": origin,
        "z": pos[0], "y": pos[1], "x": pos[2],
        "z_um": pos[0] * VOXEL_UM,
        "y_um": pos[1] * VOXEL_UM,
        "x_um": pos[2] * VOXEL_UM,
        "voxel_count": int(row["voxel_count"]),
        "max_value": float(row["max_value"]),
    }

def link_detections(detections: pd.DataFrame) -> pd.DataFrame:
    tracker = LapTrack(
        metric="sqeuclidean",
        cutoff=LINK_PX**2,
        gap_closing_metric="sqeuclidean",
        gap_closing_cutoff=(GAP_MAX_FRAME_DIFFERENCE * LINK_PX) ** 2,
        gap_closing_max_frame_count=GAP_MAX_FRAME_DIFFERENCE,
        splitting_cutoff=False,
        merging_cutoff=False,
    )
    tracked, _, _ = tracker.predict_dataframe(
        detections, coordinate_cols=POSITION_COLUMNS, frame_col="frame",
        only_coordinate_cols=False,
    )
    return tracked.sort_values(["track_id", "frame"]).reset_index(drop=True)

def track_groups(tracked: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(track_id): group.sort_values("frame").reset_index(drop=True)
            for track_id, group in tracked.groupby("track_id")}

def find_overlap_joins(groups: dict[int, pd.DataFrame], last_frame: int) -> list[dict]:
    summaries = {
        track_id: (int(g.frame.iloc[0]), int(g.frame.iloc[-1]), len(g))
        for track_id, g in groups.items()
    }
    candidates = []
    for early_id, (start, end, n) in summaries.items():
        if not start < end < last_frame or n != end - start + 1:
            continue
        a = groups[early_id].iloc[-1]
        for late_id, (late_start, late_end, late_n) in summaries.items():
            if late_start != end or late_end <= late_start or late_n != late_end - late_start + 1:
                continue
            b = groups[late_id].iloc[0]
            distance_px = float(np.linalg.norm(
                a[POSITION_COLUMNS].to_numpy(dtype=float) - b[POSITION_COLUMNS].to_numpy(dtype=float)
            ))
            if distance_px <= LINK_PX:
                candidates.append({
                    "early_track_id": early_id,
                    "late_track_id": late_id,
                    "overlap_frame": end,
                    "overlap_distance_um": distance_px * VOXEL_UM,
                    "early_detection_id": int(a.detection_id),
                    "late_detection_id": int(b.detection_id),
                    "early_voxel_count": int(a.voxel_count),
                    "late_voxel_count": int(b.voxel_count),
                })
    early_counts = pd.Series([x["early_track_id"] for x in candidates]).value_counts()
    late_counts = pd.Series([x["late_track_id"] for x in candidates]).value_counts()
    return [x for x in candidates if early_counts[x["early_track_id"]] == 1
            and late_counts[x["late_track_id"]] == 1]

def join_pair(early: pd.DataFrame, late: pd.DataFrame, late_id: int) -> list[dict]:
    frame = int(early.frame.iloc[-1])
    a, b = early.iloc[-1], late.iloc[0]
    rows = [position_row(row, late_id) for _, row in early.iloc[:-1].iterrows()]
    weights = np.array([a.voxel_count, b.voxel_count], dtype=float)
    centers = np.stack([a[POSITION_COLUMNS].to_numpy(dtype=float), b[POSITION_COLUMNS].to_numpy(dtype=float)])
    center = np.average(centers, axis=0, weights=weights)
    union = position_row(b, late_id, "overlap_union")
    union.update({
        "source_detection_ids": f"{int(a.detection_id)};{int(b.detection_id)}",
        "z": center[0], "y": center[1], "x": center[2],
        "z_um": center[0] * VOXEL_UM,
        "y_um": center[1] * VOXEL_UM,
        "x_um": center[2] * VOXEL_UM,
        "voxel_count": int(weights.sum()),
        "max_value": max(float(a.max_value), float(b.max_value)),
    })
    assert union["frame"] == frame
    rows.append(union)
    rows.extend(position_row(row, late_id) for _, row in late.iloc[1:].iterrows())
    return rows
