from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(value: object) -> str:
    """Conservative generic normalization shared by names and addresses."""
    if value is None:
        return ""

    text = str(value).strip().lower()
    if not text or text == "nan":
        return ""

    text = strip_accents(text)
    text = text.replace("&", " and ")
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
