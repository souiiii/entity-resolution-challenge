from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_submission(df: pd.DataFrame, output_path: str | Path) -> None:
    """Write a TSV; adapt schema after inspecting the official sample."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
