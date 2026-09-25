"""Deterministic screening and matching rules over Phase 4 feature rows.

Evidence is reduced to five components: name similarity (unavailable when the
two names use different scripts), order-insensitive address similarity, a number
state (agree / partial / conflict / unavailable), blocking rarity and blocking
rank. ``NaN`` always means unavailable evidence, never zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np

UNAVAILABLE, AGREE, PARTIAL, CONFLICT = 0, 1, 2, 3


@dataclass(frozen=True)
class PrefilterConfig:
    """Keep a candidate if it has enough address or name similarity (and, optionally, no number
    conflict), or if its blocking rank is within ``keep_blocking_rank``. ``None`` thresholds keep all."""
    min_address_similarity: float | None
    min_name_similarity: float | None
    reject_number_conflict: bool
    keep_blocking_rank: int
    label: str = 'selected'


@dataclass(frozen=True)
class RuleConfig:
    name_address_min_name: float
    name_address_min_address: float
    script_mismatch_min_address: float
    numbers_unavailable_min_name: float
    numbers_unavailable_min_address: float
    relative_rarity: float

    @property
    def min_address(self) -> float:
        return min(self.name_address_min_address, self.script_mismatch_min_address,
                   self.numbers_unavailable_min_address)


@dataclass(frozen=True)
class MatcherConfig:
    version: str
    features_config: str
    prefilter: PrefilterConfig
    rules: RuleConfig
    pareto: tuple[PrefilterConfig, ...]
    raw: dict


def _prefilter(raw: dict, label: str = 'selected') -> PrefilterConfig:
    return PrefilterConfig(raw['min_address_similarity'], raw['min_name_similarity'], raw['reject_number_conflict'],
                           raw['keep_blocking_rank'], raw.get('label', label))


def load_matcher_config(path: str | Path) -> MatcherConfig:
    """Load the matcher config and check that the prefilter can never drop a rule-acceptable candidate."""
    raw = json.loads(Path(path).read_text())
    r = raw['rules']
    rules = RuleConfig(r['name_address']['min_name'], r['name_address']['min_address'],
                       r['script_mismatch']['min_address'], r['numbers_unavailable']['min_name'],
                       r['numbers_unavailable']['min_address'], r['relative_rarity'])
    prefilter = _prefilter(raw['prefilter'])
    if prefilter.min_address_similarity is not None and prefilter.min_address_similarity > rules.min_address:
        raise ValueError('Prefilter address threshold would remove candidates the rules can accept')
    if not 0 <= rules.relative_rarity <= 1:
        raise ValueError('relative_rarity must lie in [0, 1]')
    return MatcherConfig(raw['version'], raw['features_config'], prefilter, rules,
                         tuple(_prefilter(p) for p in raw.get('pareto_prefilters', [])), raw)


@dataclass
class Evidence:
    """Per-candidate evidence components; arrays are aligned with the candidate rows."""
    name: np.ndarray
    address: np.ndarray
    number_state: np.ndarray
    script_mismatch: np.ndarray
    rarity: np.ndarray
    rank: np.ndarray

    def take(self, mask: np.ndarray) -> Evidence:
        return Evidence(*(getattr(self, f)[mask] for f in ('name', 'address', 'number_state', 'script_mismatch',
                                                            'rarity', 'rank')))


def evidence_from_features(features: np.ndarray, names: tuple[str, ...]) -> Evidence:
    """Reduce Phase 4 feature rows to the matcher's evidence components."""
    col = {n: features[:, i] for i, n in enumerate(names)}
    mismatch = col['name_script_mismatch'] == 1
    query_numbers, target_numbers = col['query_number_count'], col['target_number_count']
    shared = col['shared_number_count']
    available = (query_numbers > 0) & (target_numbers > 0)
    agree = available & (shared == np.minimum(query_numbers, target_numbers))
    conflict = available & (shared == 0)
    state = np.select([agree, conflict, available], [AGREE, CONFLICT, PARTIAL], UNAVAILABLE).astype(np.int8)
    name = np.where(mismatch, np.nan, col['name_token_set_similarity']).astype(np.float32)
    return Evidence(name, col['address_token_set_similarity'].astype(np.float32), state, mismatch,
                    col['blocking_aggregate_rarity'].astype(np.float32), col['blocking_rank'].astype(np.int16))


def _at_least(values: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None:
        return np.ones(len(values), bool)
    return np.nan_to_num(values, nan=-np.inf) >= threshold


def prefilter_mask(evidence: Evidence, config: PrefilterConfig) -> np.ndarray:
    """Candidates kept by the screening stage; unavailable similarities never pass a threshold."""
    if config.min_address_similarity is None and config.min_name_similarity is None:
        similar = np.ones(len(evidence.rank), bool)
    else:
        similar = np.zeros(len(evidence.rank), bool)
        if config.min_address_similarity is not None:
            similar |= _at_least(evidence.address, config.min_address_similarity)
        if config.min_name_similarity is not None:
            similar |= _at_least(evidence.name, config.min_name_similarity)
    if config.reject_number_conflict:
        similar &= evidence.number_state != CONFLICT
    return similar | (evidence.rank <= config.keep_blocking_rank)


def rule_paths(evidence: Evidence, rules: RuleConfig) -> dict[str, np.ndarray]:
    """Absolute acceptance paths; a candidate is rule-acceptable if any path holds."""
    agree = evidence.number_state == AGREE
    return {
        'name_address': agree & _at_least(evidence.name, rules.name_address_min_name)
        & _at_least(evidence.address, rules.name_address_min_address),
        'script_mismatch': agree & evidence.script_mismatch
        & _at_least(evidence.address, rules.script_mismatch_min_address),
        'numbers_unavailable': (evidence.number_state == UNAVAILABLE)
        & _at_least(evidence.name, rules.numbers_unavailable_min_name)
        & _at_least(evidence.address, rules.numbers_unavailable_min_address),
    }


def rule_acceptable(evidence: Evidence, rules: RuleConfig) -> np.ndarray:
    paths = rule_paths(evidence, rules)
    return paths['name_address'] | paths['script_mismatch'] | paths['numbers_unavailable']


def relative_gate(owners: np.ndarray, acceptable: np.ndarray, rarity: np.ndarray, relative_rarity: float,
                  queries: int) -> np.ndarray:
    """Final matches: rule-acceptable candidates whose rarity is at least ``relative_rarity`` times the
    best rule-acceptable rarity of the same query (across both sources)."""
    best = np.zeros(queries, np.float64)
    np.maximum.at(best, owners[acceptable], rarity[acceptable])
    return acceptable & (rarity >= relative_rarity * best[owners])
