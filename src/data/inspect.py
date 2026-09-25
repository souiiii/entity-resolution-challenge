from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data.load import read_table


def summarize_table(path: str | Path) -> dict:
    df = read_table(path)
    return {
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_fraction": {
            str(col): float(df[col].isna().mean())
            for col in df.columns
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }


def preview_table(path: str | Path, n: int = 5) -> pd.DataFrame:
    return read_table(path).head(n)
