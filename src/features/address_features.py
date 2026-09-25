from __future__ import annotations

from rapidfuzz.fuzz import ratio, token_set_ratio


def address_similarity_features(left_address: str, right_address: str) -> dict[str, float]:
    return {
        "address_ratio": ratio(left_address, right_address) / 100.0,
        "address_token_set_ratio": token_set_ratio(left_address, right_address) / 100.0,
        "address_exact": float(left_address == right_address and bool(left_address)),
    }
