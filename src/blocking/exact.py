from __future__ import annotations

import pandas as pd


def exact_block(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_id: str,
    right_id: str,
    key: str,
) -> pd.DataFrame:
    """Generate candidate pairs by exact equality on a normalized key."""
    pairs = left[[left_id, key]].merge(
        right[[right_id, key]],
        on=key,
        how="inner",
    )
    return pairs[[left_id, right_id]].drop_duplicates()
