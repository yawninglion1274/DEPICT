"""Adapt LapTrack 0.16 parameter names and frame-column output."""
from laptrack import LapTrack as _LapTrack


class _FrameColumnAdapter:
    def __init__(self, tracker):
        self.tracker = tracker

    def predict_dataframe(self, *args, **kwargs):
        tracked, split, merge = self.tracker.predict_dataframe(*args, **kwargs)
        frame = kwargs.get("frame_col", "frame")
        if frame not in tracked.columns and frame in tracked.index.names:
            tracked = tracked.reset_index()
        return tracked, split, merge


def LapTrack(**kwargs):
    fields = getattr(_LapTrack, "model_fields", None)
    if fields is None:
        fields = _LapTrack.__fields__
    aliases = {
        "metric": "track_dist_metric",
        "cutoff": "track_cost_cutoff",
        "gap_closing_metric": "gap_closing_dist_metric",
        "gap_closing_cutoff": "gap_closing_cost_cutoff",
        "splitting_cutoff": "splitting_cost_cutoff",
        "merging_cutoff": "merging_cost_cutoff",
    }
    translated = {aliases.get(key, key) if key not in fields else key: value
                  for key, value in kwargs.items()}
    return _FrameColumnAdapter(_LapTrack(**translated))
