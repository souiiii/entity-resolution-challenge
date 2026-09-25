from __future__ import annotations

from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio


def name_similarity_features(left_name: str, right_name: str) -> dict[str, float]:
    return {
        "name_ratio": ratio(left_name, right_name) / 100.0,
        "name_token_set_ratio": token_set_ratio(left_name, right_name) / 100.0,
        "name_token_sort_ratio": token_sort_ratio(left_name, right_name) / 100.0,
        "name_exact": float(left_name == right_name and bool(left_name)),
    }
