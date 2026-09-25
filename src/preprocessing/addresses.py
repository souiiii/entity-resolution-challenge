from __future__ import annotations

from src.preprocessing.normalization import normalize_text

ADDRESS_REPLACEMENTS = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "highway": "hwy",
    "suite": "ste",
}


def normalize_address(value: object) -> str:
    address = normalize_text(value)
    tokens = [ADDRESS_REPLACEMENTS.get(token, token) for token in address.split()]
    return " ".join(tokens)
