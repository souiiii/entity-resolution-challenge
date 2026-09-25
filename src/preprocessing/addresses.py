from __future__ import annotations

from src.preprocessing.normalization import canonicalize


def normalize_address(text: str) -> str:
    """Canonical address; abbreviations and components are never rewritten."""
    return canonicalize(text)
