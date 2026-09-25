from __future__ import annotations

from src.preprocessing.normalization import normalize_text

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "ltd", "limited", "pvt", "private",
}


def normalize_business_name(value: object, remove_legal_suffixes: bool = False) -> str:
    name = normalize_text(value)
    if not remove_legal_suffixes:
        return name
    tokens = [token for token in name.split() if token not in LEGAL_SUFFIXES]
    return " ".join(tokens)
