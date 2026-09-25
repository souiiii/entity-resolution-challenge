from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV/TSV file based on extension."""
    path = Path(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type: {path.suffix}")


def list_input_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}
    )
