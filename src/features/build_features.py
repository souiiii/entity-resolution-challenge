from __future__ import annotations

from src.features.address_features import address_similarity_features
from src.features.name_features import name_similarity_features


def build_pair_features(
    left_name: str,
    right_name: str,
    left_address: str,
    right_address: str,
) -> dict[str, float]:
    features = {}
    features.update(name_similarity_features(left_name, right_name))
    features.update(address_similarity_features(left_address, right_address))
    return features
