"""Lightweight candidate filtering, separate from the final entity matcher."""
from __future__ import annotations

from functools import lru_cache
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from src.blocking.evaluate import in_sample
from src.blocking.channels import ChannelSpec, BlockingConfig
from src.blocking.store import RecordStore
from src.blocking.recovery import ExpandedBatch
from src.blocking.recovery import cheap_score, families, hashed_evidence, rank_per_query
from src.features.text import SourceText
from src.preprocessing.normalization import canonicalize, fold_latin_marks

if TYPE_CHECKING:
    from lightgbm import Booster
    from src.matching.final_candidates import FinalPolicy, FinalBatch

FEATURE_VERSION = 'candidate-ranker-features-v1'
RANKER_RESCUE = 16
BASE_FEATURES = ('source_is_s3', 'name_shared', 'name_jaccard', 'name_containment', 'word_shared',
                 'word_jaccard', 'word_containment', 'number_state', 'exact_name', 'exact_address',
                 'min_frequency', 'rarity', 'key_hits', 'support_channels', 'support_families',
                 'overflow_only', 'frequency_band', 'cheap_score', 'cheap_rank', 'rarity_rank',
                 'name_char', 'name_token_set', 'address_char', 'address_token_set')


def feature_names(channels: tuple[ChannelSpec, ...]) -> tuple[str, ...]:
    return (*BASE_FEATURES, *(f'channel_{c.name}' for c in channels))


def training_queries(ids: np.ndarray, salt: str, fraction: float) -> np.ndarray:
    """A query's partition is independent of row order, source and candidate count."""
    return np.array([in_sample(i.decode() if isinstance(i, bytes) else str(i), salt, fraction) for i in ids], bool)


class CheapText:
    """Bounded canonical text cache; country and entity IDs are never returned as features."""

    def __init__(self, text: SourceText, cache_size: int = 50000,
                 normalized_root: Path | None = None, reserve_gib: float = 10.5):
        from src.matching.candidate_text import ensure_candidate_text

        self.text = text
        self.normalized = (ensure_candidate_text(text, normalized_root, reserve_gib)
                           if normalized_root is not None else None)
        read = self.normalized.read_one if self.normalized is not None else self._read_one
        self.read_one = lru_cache(maxsize=cache_size)(read)

    def _read_one(self, row: int) -> tuple[str, str]:
        name, address, _ = self.text.read(np.array([row]))[0]
        return fold_latin_marks(canonicalize(name)), canonicalize(address)

    def read(self, rows: np.ndarray) -> list[tuple[str, str]]:
        return [self.read_one(int(r)) for r in rows]


def retrieval_features(query_store: RecordStore, target_store: RecordStore, rows: np.ndarray,
                       broad: ExpandedBatch, deep: ExpandedBatch, channels: tuple[ChannelSpec, ...],
                       query_text: CheapText, target_text: CheapText,
                       selected: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cheap overlaps, provenance and four string scores; no Phase 4 matrix or matcher calls."""
    evidence = hashed_evidence(query_store, target_store, rows[broad.owners], broad.records)
    score = cheap_score(evidence, broad.aggregate_rarity)
    cheap_rank = rank_per_query(broad.owners, score, broad.min_frequency, target_store.rank[broad.records])
    rarity_rank = rank_per_query(broad.owners, broad.aggregate_rarity, broad.min_frequency,
                                target_store.rank[broad.records])
    deep_keys = np.sort(deep.owners * target_store.size + deep.records)
    keys = broad.owners * target_store.size + broad.records
    at = np.minimum(np.searchsorted(deep_keys, keys), max(0, len(deep_keys) - 1))
    overflow = deep_keys[at] != keys if len(deep_keys) else np.ones(len(keys), bool)
    indicators = [((broad.masks >> bit) & 1).astype(np.float32) for bit in range(len(channels))]
    columns = [np.full(len(keys), target_store.source == 3), evidence.name_shared, evidence.name_jaccard,
               evidence.name_containment, evidence.word_shared, evidence.word_jaccard, evidence.word_containment,
               evidence.number_state, evidence.exact_name, evidence.exact_address, broad.min_frequency,
               broad.aggregate_rarity, broad.hits, sum(indicators), families(broad.masks, channels), overflow,
               np.searchsorted([200, 300, 500, 750, 1000], broad.min_frequency, side='left'), score,
               cheap_rank, rarity_rank]
    selected = np.arange(len(keys)) if selected is None else selected
    columns = [c[selected] for c in columns]
    q_rows, qi = np.unique(rows[broad.owners[selected]], return_inverse=True)
    t_rows, ti = np.unique(broad.records[selected], return_inverse=True)
    q, t = query_text.read(q_rows), target_text.read(t_rows)
    for field in (0, 1):
        left = [q[i][field] for i in qi]
        right = [t[i][field] for i in ti]
        missing = np.fromiter((not a or not b for a, b in zip(left, right)), bool, len(left))
        for scorer in (fuzz.ratio, fuzz.token_set_ratio):
            values = cpdist(left, right, scorer=scorer, workers=1, dtype=np.float32) / 100
            values[missing] = np.nan
            columns.append(values)
    columns.extend(c[selected] for c in indicators)
    return np.column_stack(columns).astype(np.float32), cheap_rank


def sampled_pairs(broad: ExpandedBatch, labels: np.ndarray, target_size: int, seed: int, random_budget: int,
                  hard_budget: int, target_rank: np.ndarray) -> np.ndarray:
    """All positives plus reproducible uniform and rarity-hard negatives per query/source."""
    with np.errstate(over='ignore'):
        key = (broad.owners.astype(np.uint64) * np.uint64(target_size) + broad.records.astype(np.uint64)
               + np.uint64(seed))
        key = (key ^ (key >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        key = (key ^ (key >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        key ^= key >> np.uint64(31)
    random_rank = rank_per_query(broad.owners, (key >> np.uint64(11)).astype(np.float64),
                                broad.min_frequency, target_rank[broad.records])
    hard_rank = rank_per_query(broad.owners, broad.aggregate_rarity, broad.min_frequency, target_rank[broad.records])
    return np.flatnonzero(labels | (random_rank <= random_budget) | (hard_rank <= hard_budget))


def rescue_ranks(owners: np.ndarray, scores: np.ndarray, phase8: np.ndarray,
                 frequencies: np.ndarray, target_rank: np.ndarray) -> np.ndarray:
    """Rank only candidates outside the preserved Phase 8 set; ties use frequency and binary target ID."""
    result = np.zeros(len(owners), np.int32)
    rescue = ~phase8
    result[rescue] = rank_per_query(owners[rescue], scores[rescue], frequencies[rescue], target_rank[rescue])
    return result


def select_candidates(phase8: np.ndarray, rescue_rank: np.ndarray, budget: int) -> np.ndarray:
    """The exact set a later matcher and candidate_pairs.tsv must receive."""
    if budget < 0:
        raise ValueError('Rescue budget must be non-negative')
    return phase8 | ((rescue_rank > 0) & (rescue_rank <= budget))


def iter_ranked_candidates(phase8_policy: FinalPolicy, blocking: BlockingConfig, model: Booster,
                           query_store: RecordStore, query_text: SourceText, target_store: RecordStore,
                           target_text: SourceText, rows: np.ndarray, index_root: Path, overflow_root: Path,
                           maximum: int, budget: int, batch_size: int = 100,
                           reserve_gib: float = 10.5) -> Iterator[FinalBatch]:
    """Stream exactly Phase 8 plus ranker rescues, before any final-matcher scoring."""
    from src.blocking.candidates import SourceBlocker
    from src.blocking.overflow import BroadRetriever, disk_guard
    from src.matching.final_candidates import FinalBatch, iter_final_candidates

    if model.feature_name() != list(feature_names(blocking.channels)):
        raise ValueError('Candidate-ranker feature schema mismatch')
    blocker = SourceBlocker(blocking, target_store, index_root, min_free_bytes=int(reserve_gib * 2**30))
    retriever = BroadRetriever(blocker, overflow_root, maximum, reserve_gib)
    normalized = overflow_root.parent / 'candidate-text-v1'
    qtext = CheapText(query_text, normalized_root=normalized, reserve_gib=reserve_gib)
    ttext = CheapText(target_text, normalized_root=normalized, reserve_gib=reserve_gib)
    phase8_batches = iter_final_candidates(phase8_policy, blocking, query_store, query_text, target_store,
        target_text, rows, index_root, batch_size, workers=1)
    start = 0
    for base in phase8_batches:
        batch_rows = rows[start:start + len(base.union_counts)]
        deep, broad = retriever.retrieve(query_store, batch_rows)
        keys = broad.owners * target_store.size + broad.records
        base_keys = (base.owners - start) * target_store.size + base.records
        order = np.argsort(base_keys)
        at = np.minimum(np.searchsorted(base_keys[order], keys), max(len(base_keys) - 1, 0))
        present = base_keys[order][at] == keys if len(base_keys) else np.zeros(len(keys), bool)
        if present.sum() != len(base_keys):
            raise AssertionError('Broad retrieval must preserve every Phase 8 candidate')
        matrix, _ = retrieval_features(query_store, target_store, batch_rows, broad, deep,
                                       blocking.channels, qtext, ttext)
        scores = model.predict(matrix, num_threads=3)
        ranks = rescue_ranks(broad.owners, scores, present, broad.min_frequency, target_store.rank[broad.records])
        keep = select_candidates(present, ranks, budget)
        provenance = np.full(len(keys), RANKER_RESCUE, np.uint8)
        provenance[present] = base.provenance[order[at[present]]]
        deep_rank = np.zeros(len(keys), np.int64)
        deep_rank[present] = base.deep_rank[order[at[present]]]
        disk_guard(overflow_root, reserve_gib=reserve_gib)
        yield FinalBatch(target_store.source, broad.owners[keep] + start, broad.records[keep], provenance[keep],
                         deep_rank[keep], base.union_counts)
        start += len(batch_rows)
