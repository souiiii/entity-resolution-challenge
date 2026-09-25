"""Lossless string loaders for the official UTF-8 challenge tables."""
from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

LOGGER = logging.getLogger(__name__)
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


def _rows(path: str | Path, columns: tuple[str, ...]) -> Iterator[dict[str, str]]:
    path = Path(path)
    LOGGER.info("Reading %s", path)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", strict=True)
        if next(reader, None) != list(columns):
            raise ValueError(f"{path}: expected header {columns}")
        for values in reader:
            if len(values) != len(columns):
                raise ValueError(f"{path}:{reader.line_num}: expected {len(columns)} fields")
            yield dict(zip(columns, values))


def _check_id(value: str, prefixes: tuple[str, ...]) -> None:
    if not value.startswith(prefixes) or value != value.strip() or any(c.isspace() for c in value):
        raise ValueError(f"Invalid entity ID: {value!r}; expected prefixes {prefixes}")
    if len(value.split("-", 1)[1]) == 0:
        raise ValueError(f"Empty entity ID suffix: {value!r}")


def iter_source(path: str | Path, source: int) -> Iterator[dict[str, str]]:
    """Stream a source without coercion or normalization; empty fields stay ''.

    Country labels are unrestricted. Duplicate IDs and cross-file integrity are
    audited by inspection, rather than tracked in memory by this iterator.
    """
    if source not in (1, 2, 3):
        raise ValueError("source must be 1, 2, or 3")
    for row in _rows(path, SOURCE_COLUMNS):
        _check_id(row["entity_id"], (f"S{source}-",))
        yield row


def parse_matched_ids(value: str) -> tuple[str, ...]:
    """Parse the official comma list; only the empty string means no matches."""
    if value == "":
        return ()
    ids = tuple(value.split(","))
    for entity_id in ids:
        _check_id(entity_id, ("S2-", "S3-"))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Repeated target ID in {value!r}")
    return ids


def iter_ground_truth(path: str | Path) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Stream (Source 1 ID, ordered target tuple), including explicit singletons."""
    for row in _rows(path, GROUND_TRUTH_COLUMNS):
        _check_id(row["source1_entity_id"], ("S1-",))
        yield row["source1_entity_id"], parse_matched_ids(row["matched_entity_ids"])


def load_source(path: str | Path, source: int) -> pd.DataFrame:
    """Load a validated source as pandas strings; prefer iter_source at full scale."""
    import pandas as pd

    return pd.DataFrame(iter_source(path, source), columns=SOURCE_COLUMNS, dtype="string")


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Load validated truth, preserving the official comma-list string column."""
    import pandas as pd

    rows = ((entity_id, ",".join(ids)) for entity_id, ids in iter_ground_truth(path))
    return pd.DataFrame(rows, columns=GROUND_TRUTH_COLUMNS, dtype="string")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV/TSV as literal strings, preserving empty and NA-like values."""
    import pandas as pd

    path = Path(path)
    separators = {".tsv": "\t", ".csv": ","}
    if path.suffix.lower() not in separators:
        raise ValueError(f"Unsupported table type: {path.suffix}")
    return pd.read_csv(path, sep=separators[path.suffix.lower()], dtype="string",
                       keep_default_na=False, na_filter=False, encoding="utf-8")


def list_input_files(directory: str | Path) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*")
                  if p.is_file() and p.suffix.lower() in {".csv", ".tsv"})
