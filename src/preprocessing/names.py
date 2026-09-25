from __future__ import annotations

from src.preprocessing.normalization import canonicalize


def normalize_name(text: str) -> str:
    """Canonical business name; legal suffixes and business words are retained."""
    return canonicalize(text)
