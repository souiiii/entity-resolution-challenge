"""Deterministic pairwise features for (Source 1 query, Source 2/3 candidate) pairs.

Every feature is a float32 column. ``NaN`` means the feature is undefined for the
pair (for example a similarity against an empty field); it never stands for zero.
Text features use the Phase 2 canonical representations, and blocking features
carry the Phase 3B evidence without reinterpreting it as a probability.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from src.blocking.channels import ChannelSpec
from src.blocking.keys import number_form
from src.preprocessing.normalization import canonicalize, compact, fold_latin_marks

FEATURES_VERSION = 'features-v1'
BLOCKING_FEATURES = ('blocking_rank', 'blocking_support_channels', 'blocking_support_families',
                     'blocking_exact_name', 'blocking_min_key_frequency', 'blocking_aggregate_rarity')
NAME_FEATURES = ('name_canonical_equal', 'name_folded_equal', 'name_folded_compact_equal', 'name_char_similarity',
                 'name_token_set_similarity', 'name_token_intersection', 'name_token_jaccard',
                 'name_token_containment', 'name_length_ratio', 'name_script_mismatch')
ADDRESS_FEATURES = ('address_canonical_equal', 'address_char_similarity', 'address_token_set_similarity',
                    'address_token_intersection', 'address_token_jaccard', 'address_token_containment',
                    'address_length_ratio')
NUMBER_FEATURES = ('query_number_count', 'target_number_count', 'shared_number_count', 'number_jaccard',
                   'query_numbers_in_target', 'target_numbers_in_query', 'number_set_equal')
MISSING_FEATURES = ('target_address_empty',)


@dataclass(frozen=True)
class RecordView:
    """Canonical text views of one record, computed once and shared by all its pairs."""
    name: str
    name_folded: str
    name_compact: str
    name_tokens: frozenset[str]
    name_non_latin: bool
    address: str
    address_tokens: frozenset[str]
    numbers: frozenset[str]


def _non_latin(text: str) -> bool:
    return any(c.isalpha() and not unicodedata.name(c, '').startswith('LATIN ') for c in text)


def record_view(name: str, address: str) -> RecordView:
    name_canonical = canonicalize(name)
    folded = fold_latin_marks(name_canonical)
    address_canonical = canonicalize(address)
    tokens = address_canonical.split()
    return RecordView(name_canonical, folded, compact(folded), frozenset(folded.split()), _non_latin(folded),
                      address_canonical, frozenset(tokens),
                      frozenset(number_form(t) for t in tokens if t.isdecimal()))


def family_of(spec: ChannelSpec) -> tuple[str, ...]:
    """Channel family used by Phase 3B ranking: the three exact-name views form one family."""
    if spec.kind == 'exact' and spec.parts[0].startswith('name_'):
        return ('exact_name',)
    return (spec.kind, *spec.parts)


def feature_names(channels: Sequence[ChannelSpec]) -> tuple[str, ...]:
    """Deterministic column order: source, blocking evidence, channel indicators, text, numbers, missingness."""
    return ('source_is_s3', *BLOCKING_FEATURES, *(f'channel_{c.name}' for c in channels), *NAME_FEATURES,
            *ADDRESS_FEATURES, *NUMBER_FEATURES, *MISSING_FEATURES)


@dataclass
class CandidateEvidence:
    """Phase 3B evidence for each candidate pair, aligned with the pair arrays."""
    source: int
    ranks: np.ndarray
    masks: np.ndarray
    min_frequency: np.ndarray
    aggregate_rarity: np.ndarray


def _blocking_columns(evidence: CandidateEvidence, channels: Sequence[ChannelSpec]) -> list[np.ndarray]:
    masks = np.asarray(evidence.masks, dtype=np.uint32)
    indicators = [((masks >> np.uint32(bit)) & np.uint32(1)).astype(np.float32) for bit in range(len(channels))]
    families: dict[tuple[str, ...], np.ndarray] = {}
    for spec, present in zip(channels, indicators):
        family = family_of(spec)
        families[family] = np.maximum(families.get(family, 0), present)
    exact_name = families.get(('exact_name',), np.zeros(len(masks), np.float32))
    return [np.full(len(masks), float(evidence.source == 3), np.float32),
            np.asarray(evidence.ranks, np.float32), sum(indicators, np.zeros(len(masks), np.float32)),
            sum(families.values(), np.zeros(len(masks), np.float32)), exact_name,
            np.asarray(evidence.min_frequency, np.float32), np.asarray(evidence.aggregate_rarity, np.float32),
            *indicators]


def _similarity(left: list[str], right: list[str], scorer, workers: int) -> np.ndarray:
    scores = cpdist(left, right, scorer=scorer, workers=workers, dtype=np.float32) / np.float32(100)
    empty = np.fromiter((not a or not b for a, b in zip(left, right)), bool, len(left))
    scores[empty] = np.nan
    return scores


def _overlap(left: list[frozenset], right: list[frozenset]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(left)
    shared = np.empty(n, np.float32)
    jaccard = np.full(n, np.nan, np.float32)
    containment = np.full(n, np.nan, np.float32)
    for i, (a, b) in enumerate(zip(left, right)):
        k = len(a & b)
        shared[i] = k
        if a and b:
            jaccard[i] = k / (len(a) + len(b) - k)
            containment[i] = k / min(len(a), len(b))
    return shared, jaccard, containment


def _length_ratio(left: list[str], right: list[str]) -> np.ndarray:
    a = np.fromiter(map(len, left), np.float32, len(left))
    b = np.fromiter(map(len, right), np.float32, len(right))
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.minimum(a, b) / np.maximum(a, b)
    return np.where((a > 0) & (b > 0), ratio, np.nan).astype(np.float32)


def _number_columns(left: list[frozenset], right: list[frozenset]) -> list[np.ndarray]:
    n = len(left)
    query_count = np.fromiter(map(len, left), np.float32, n)
    target_count = np.fromiter(map(len, right), np.float32, n)
    shared = np.fromiter((len(a & b) for a, b in zip(left, right)), np.float32, n)
    union = query_count + target_count - shared
    with np.errstate(invalid='ignore', divide='ignore'):
        jaccard = np.where(union > 0, shared / union, np.nan)
        query_in = np.where(query_count > 0, (shared == query_count).astype(np.float32), np.nan)
        target_in = np.where(target_count > 0, (shared == target_count).astype(np.float32), np.nan)
    equal = np.where(union > 0, ((shared == query_count) & (shared == target_count)).astype(np.float32), np.nan)
    return [query_count, target_count, shared, jaccard.astype(np.float32), query_in.astype(np.float32),
            target_in.astype(np.float32), equal.astype(np.float32)]


def extract_pair_features(queries: Sequence[RecordView], targets: Sequence[RecordView], query_index: np.ndarray,
                          target_index: np.ndarray, evidence: CandidateEvidence, channels: Sequence[ChannelSpec],
                          workers: int = 1) -> np.ndarray:
    """Feature matrix ``(pairs, len(feature_names(channels)))`` for pairs ``(queries[i], targets[j])``."""
    q = [queries[i] for i in np.asarray(query_index).tolist()]
    t = [targets[j] for j in np.asarray(target_index).tolist()]
    n = len(q)
    q_name, t_name = [v.name_folded for v in q], [v.name_folded for v in t]
    q_address, t_address = [v.address for v in q], [v.address for v in t]
    both_names = np.fromiter((bool(a.name) and bool(b.name) for a, b in zip(q, t)), bool, n)
    both_addresses = np.fromiter((bool(a.address) and bool(b.address) for a, b in zip(q, t)), bool, n)

    def equal(attribute: str, present: np.ndarray) -> np.ndarray:
        same = np.fromiter((getattr(a, attribute) == getattr(b, attribute) for a, b in zip(q, t)), bool, n)
        return (same & present).astype(np.float32)

    name_shared, name_jaccard, name_containment = _overlap([v.name_tokens for v in q], [v.name_tokens for v in t])
    address_shared, address_jaccard, address_containment = _overlap([v.address_tokens for v in q],
                                                                     [v.address_tokens for v in t])
    script = np.fromiter((a.name_non_latin != b.name_non_latin for a, b in zip(q, t)), np.float32, n)
    columns = [
        *_blocking_columns(evidence, channels),
        equal('name', both_names), equal('name_folded', both_names), equal('name_compact', both_names),
        _similarity(q_name, t_name, fuzz.ratio, workers), _similarity(q_name, t_name, fuzz.token_set_ratio, workers),
        name_shared, name_jaccard, name_containment, _length_ratio(q_name, t_name),
        np.where(both_names, script, np.nan).astype(np.float32),
        equal('address', both_addresses), _similarity(q_address, t_address, fuzz.ratio, workers),
        _similarity(q_address, t_address, fuzz.token_set_ratio, workers),
        address_shared, address_jaccard, address_containment, _length_ratio(q_address, t_address),
        *_number_columns([v.numbers for v in q], [v.numbers for v in t]),
        np.fromiter((not v.address for v in t), np.float32, n),
    ]
    matrix = np.column_stack(columns).astype(np.float32, copy=False) if n else np.zeros((0, len(columns)), np.float32)
    if matrix.shape[1] != len(feature_names(channels)):
        raise RuntimeError('Feature columns do not match the declared schema')
    return matrix
