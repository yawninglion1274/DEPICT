"""Reference identity association followed by identity-constrained LapTrack."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from laptrack_compat import LapTrack
import temporal_utils as pipeline
HERE=Path(__file__).resolve().parents[1]/"inputs"
RADIUS_UM=0.30

def associate(angle, reference, radius=RADIUS_UM, write=True,
              detections=None, native=None, destination=None,
              frame_selection="within_common"):
    folder = HERE / f"angle{angle}"
    destination = folder if destination is None else Path(destination)
    frames = pd.read_csv(folder / "frames.csv")
    frames = frames.loc[frames[frame_selection]]
    if detections is None:
        detections = pd.read_csv(folder / "localizations.csv")
    if native is None:
        native = pd.read_csv(folder / "laptrack_tracks.csv").set_index("detection_id").track_id
    predictions = pipeline.reference_predictions(reference, frames, angle, frame_selection=frame_selection)
    rows, audit_rows = [], []
    for global_frame, expected in predictions.groupby("global_frame"):
        expected = expected.sort_values("reference_track_id")
        observed = detections.loc[detections.global_frame.eq(global_frame)].sort_values("detection_id")
        distance = cdist(expected[pipeline.XYZ_UM], observed[pipeline.XYZ_UM])
        allowed = distance <= radius
        if "localization_priority" in observed.columns:
            choices = {}
            for priority in sorted(observed.localization_priority.unique()):
                remaining = np.array([i for i in range(len(expected)) if i not in choices],dtype=int)
                candidates = np.flatnonzero(observed.localization_priority.to_numpy()==priority)
                if not len(remaining) or not len(candidates):
                    continue
                selected = np.ix_(remaining,candidates)
                for i,j in pipeline.assign_with_rejection(distance[selected],allowed[selected]):
                    choices[int(remaining[i])] = int(candidates[j])
        else:
            choices = dict(pipeline.assign_with_rejection(distance, allowed))
        for i, item in enumerate(expected.itertuples()):
            nearest = np.sort(distance[i])
            chosen = choices.get(i)
            audit_rows.append(dict(reference_track_id=item.reference_track_id, global_frame=global_frame,
                n_detections_within_radius=int(allowed[i].sum()), accepted=chosen is not None,
                nearest_distance_um=float(nearest[0]), second_nearest_distance_um=float(nearest[1]),
                assigned_distance_um=float(distance[i,chosen]) if chosen is not None else np.nan,
                radius_um=radius))
            if chosen is None:
                continue
            row = observed.iloc[chosen].to_dict()
            detection_id = int(row["detection_id"])
            row.update(track_id=int(item.reference_track_id), reference_track_id=int(item.reference_track_id),
                native_track_id=int(native.loc[detection_id]), candidate_track_id=int(item.reference_track_id),
                source_detection_ids=str(detection_id), position_origin="reference_associated_detector_centroid",
                reference_distance_um=float(distance[i,chosen]),
                n_detections_within_radius=int(allowed[i].sum()),
                angle_group=angle, downsample_factor=angle//12, volume_rate_vps=195/angle,
                frame=int(global_frame-frames.global_frame.min()))
            rows.append(row)
    points = pd.DataFrame(rows).sort_values(["track_id","frame"])
    point_audit = pd.DataFrame(audit_rows)
    n_frames = len(frames)
    summaries = []
    complete = []
    for track_id in sorted(reference.track_id.unique()):
        g = points.loc[points.track_id.eq(track_id)]
        is_complete = len(g)==n_frames and g.frame.nunique()==n_frames
        linked_segments = 0
        if is_complete:
            cutoff = 5/(195/angle)/0.09
            tracker = LapTrack(metric="sqeuclidean",cutoff=cutoff**2,
                gap_closing_metric="sqeuclidean",gap_closing_cutoff=(2*cutoff)**2,
                gap_closing_max_frame_count=2,splitting_cutoff=False,merging_cutoff=False)
            linked,_,_ = tracker.predict_dataframe(g.drop(columns="track_id"),
                coordinate_cols=pipeline.XYZ,frame_col="frame",only_coordinate_cols=False)
            linked_segments = int(linked.track_id.nunique())
            is_complete = linked_segments==1
        summaries.append(dict(reference_track_id=track_id,n_associated_frames=len(g),
            n_expected_frames=n_frames,complete_after_linking=is_complete,
            n_native_track_ids=g.native_track_id.nunique(),linked_segments=linked_segments,
            median_distance_um=g.reference_distance_um.median(),
            max_distance_um=g.reference_distance_um.max(),
            n_frames_with_multiple_candidates=int(g.n_detections_within_radius.gt(1).sum())))
        if is_complete:
            complete.append(g)
    result = pd.concat(complete,ignore_index=True) if complete else points.iloc[:0].copy()
    if write:
        destination.mkdir(exist_ok=True,parents=True)
        frames.to_csv(destination / "frames.csv",index=False)
        predictions.to_csv(destination / "reference_window_mean_positions.csv",index=False)
        points.to_csv(destination / "reference_associated_detections.csv",index=False)
        point_audit.to_csv(destination / "reference_association_point_audit.csv",index=False)
        pd.DataFrame(summaries).to_csv(destination / "reference_association_track_audit.csv",index=False)
        result.to_csv(destination / "reference_assisted_tracks.csv",index=False)
    return result, pd.DataFrame(summaries)
