"""Candidate-batch feature extraction on the frozen Phase 3B candidate pipeline."""
from __future__ import annotations

import resource
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.blocking.candidates import CandidateBatch, SourceBlocker
from src.blocking.channels import BlockingConfig, ChannelSpec
from src.blocking.ranking import candidate_ranks
from src.blocking.store import RecordStore
from src.features.pairwise import CandidateEvidence, extract_pair_features, feature_names, record_view
from src.features.text import SourceText
from src.utils.parallel import ordered_map


@dataclass
class FeatureBatch:
    """Selected candidates of one query batch and one target source, with their feature rows."""
    source: int
    owners: np.ndarray
    records: np.ndarray
    features: np.ndarray
    selection_seconds: float = 0.0
    feature_seconds: float = 0.0
    worker_peak_rss_kib: int = 0


def select_candidates(blocker: SourceBlocker, query_store: RecordStore,
                      rows: np.ndarray) -> tuple[CandidateBatch, np.ndarray]:
    """The configured Phase 3B selection (ranking strategy and top K) together with each kept rank."""
    ranking = blocker.config.raw['ranking']
    batch = blocker.candidates(query_store, rows)
    ranks = candidate_ranks(batch, blocker.store, blocker.config.channels, ranking['selected_strategy'])
    keep = ranks <= ranking['selected_k']
    return CandidateBatch(batch.owners[keep], batch.records[keep], batch.masks[keep], batch.channel_stats,
                          batch.min_frequency[keep], batch.aggregate_rarity[keep]), ranks[keep]


def compute_features(job: tuple) -> tuple[np.ndarray, float, int]:
    """Worker entry point: record views once per record, then the pair matrix, CPU seconds and peak RSS."""
    started = time.process_time()
    query_fields, target_fields, query_index, target_index, evidence, channels = job
    queries = [record_view(name, address) for name, address, _ in query_fields]
    targets = [record_view(name, address) for name, address, _ in target_fields]
    matrix = extract_pair_features(queries, targets, query_index, target_index, evidence, channels)
    return matrix, time.process_time() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def feature_jobs(blocker: SourceBlocker, query_store: RecordStore, query_text: SourceText, target_text: SourceText,
                 rows: np.ndarray, batch_size: int) -> Iterator[tuple[tuple, tuple]]:
    channels: tuple[ChannelSpec, ...] = blocker.config.channels
    for start in range(0, len(rows), batch_size):
        started = time.monotonic()
        batch_rows = rows[start:start + batch_size]
        batch, ranks = select_candidates(blocker, query_store, batch_rows)
        unique, target_index = np.unique(batch.records, return_inverse=True)
        evidence = CandidateEvidence(blocker.store.source, ranks, batch.masks, batch.min_frequency,
                                     batch.aggregate_rarity)
        job = (query_text.read(batch_rows), target_text.read(unique), batch.owners, target_index, evidence, channels)
        yield (batch.owners + start, batch.records, time.monotonic() - started), job


def iter_features(config: BlockingConfig, query_store: RecordStore, query_text: SourceText, target_store: RecordStore,
                  target_text: SourceText, rows: np.ndarray, index_root: str | Path, batch_size: int,
                  workers: int = 1) -> Iterator[FeatureBatch]:
    """Stream feature batches for ``rows`` (query store rows) against one target source, in query order."""
    if not config.raw.get('ranking', {}).get('selected_strategy'):
        raise ValueError('Feature extraction requires a blocking config with a selected ranking')
    blocker = SourceBlocker(config, target_store, index_root)
    jobs = feature_jobs(blocker, query_store, query_text, target_text, np.asarray(rows, np.int64), batch_size)
    for (owners, records, selection_seconds), (matrix, seconds, rss) in ordered_map(compute_features, jobs, workers):
        yield FeatureBatch(target_store.source, owners, records, matrix, selection_seconds, seconds, rss)


def schema(config: BlockingConfig) -> tuple[str, ...]:
    return feature_names(config.channels)
