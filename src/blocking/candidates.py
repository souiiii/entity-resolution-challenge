from __future__ import annotations

import pandas as pd


def union_candidate_sets(*candidate_frames: pd.DataFrame) -> pd.DataFrame:
    frames = [
        frame for frame in candidate_frames
        if frame is not None and not frame.empty
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()
