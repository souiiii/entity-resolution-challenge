"""Phase 8 final candidate policy: the exact candidate set handed to the final matcher.

Per target source and query batch:

1. The unchanged Phase 3B union and aggregate ranking.
2. Candidates ranked within ``top_k`` receive Phase 4 features and are kept by the Phase 5
   prefilter or by the conflict rescue (number conflict with strong name and address).
3. Union candidates ranked beyond ``top_k`` are kept by a cheap hashed-evidence score
   (above a threshold, or the query's best few).
4. Deep retrieval (the same channels and indexes with caps lifted to the index keep limit)
   contributes each query's best few deep-only candidates by the same cheap score.

The union of 2-4 is the final candidate set; every candidate carries provenance flags.
No matching model is involved.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.blocking.candidates import SourceBlocker
from src.blocking.channels import BlockingConfig, ChannelSpec, load_config
from src.blocking.ranking import candidate_ranks
from src.blocking.recovery import cheap_score, deepen, expanded_union, hashed_evidence, rank_per_query
from src.blocking.store import RecordStore
from src.features.pairwise import CandidateEvidence, extract_pair_features, feature_names, record_view
from src.features.text import SourceText
from src.matching.rules import CONFLICT, PrefilterConfig, evidence_from_features, load_matcher_config, prefilter_mask
from src.utils.parallel import ordered_map

PHASE5, CONFLICT_RESCUE, RANK_RESCUE, DEEP_RESCUE = 1, 2, 4, 8
PROVENANCE = {'phase5': PHASE5, 'conflict_rescue': CONFLICT_RESCUE, 'rank_rescue': RANK_RESCUE,
              'deep_rescue': DEEP_RESCUE}


@dataclass(frozen=True)
class FinalPolicy:
    """Frozen parameters of the final candidate policy."""
    version: str
    top_k: int
    prefilter: PrefilterConfig
    conflict_min_name: float
    conflict_min_address: float
    rank_min_score: float
    rank_per_query: int
    deep_per_query: int
    raw: dict


def load_final_policy(path: str | Path, project_root: str | Path) -> tuple[FinalPolicy, BlockingConfig]:
    """Load the candidate policy and the blocking configuration it builds on."""
    raw = json.loads(Path(path).read_text())
    blocking = load_config(Path(project_root) / raw['blocking_config'])
    matcher = load_matcher_config(Path(project_root) / raw['prefilter_config'])
    policy = FinalPolicy(raw['version'], blocking.raw['ranking']['selected_k'], matcher.prefilter,
                         raw['conflict_rescue']['min_name'], raw['conflict_rescue']['min_address'],
                         raw['rank_rescue']['min_score'], raw['rank_rescue']['per_query_source'],
                         raw['deep_rescue']['per_query_source'], raw)
    if min(policy.rank_per_query, policy.deep_per_query) < 0 or policy.top_k < 1:
        raise ValueError('Budgets must be non-negative and top_k positive')
    return policy, blocking


@dataclass
class FinalBatch:
    """Final candidates of one query batch against one target source."""
    source: int
    owners: np.ndarray
    records: np.ndarray
    provenance: np.ndarray
    deep_rank: np.ndarray
    union_counts: np.ndarray


def screen_top(job: tuple) -> np.ndarray:
    """Worker: Phase 4 evidence for the top-ranked candidates and their Phase 5 / conflict-rescue flags."""
    query_fields, target_fields, query_index, target_index, evidence, channels, policy = job
    queries = [record_view(name, address) for name, address, _ in query_fields]
    targets = [record_view(name, address) for name, address, _ in target_fields]
    matrix = extract_pair_features(queries, targets, query_index, target_index, evidence, channels)
    rule_evidence = evidence_from_features(matrix, feature_names(channels))
    flags = np.where(prefilter_mask(rule_evidence, policy.prefilter), PHASE5, 0)
    name = np.nan_to_num(rule_evidence.name, nan=-np.inf)
    address = np.nan_to_num(rule_evidence.address, nan=-np.inf)
    rescue = ((rule_evidence.number_state == CONFLICT) & (name >= policy.conflict_min_name)
              & (address >= policy.conflict_min_address))
    return (flags | np.where(rescue, CONFLICT_RESCUE, 0)).astype(np.uint8)


def _rescue(store: RecordStore, query_store: RecordStore, batch_rows: np.ndarray, owners: np.ndarray,
            records: np.ndarray, rarity: np.ndarray, frequency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cheap score and one-based per-query position of rescue candidates."""
    if not len(owners):
        return np.zeros(0, np.float32), np.zeros(0, np.int64)
    score = cheap_score(hashed_evidence(query_store, store, batch_rows[owners], records), rarity)
    return score, rank_per_query(owners, score, frequency, store.rank[records])


def final_jobs(policy: FinalPolicy, blocker: SourceBlocker, deep_specs: tuple[ChannelSpec, ...],
               query_store: RecordStore, query_text: SourceText, target_text: SourceText, rows: np.ndarray,
               batch_size: int, deep_limit: int) -> Iterator[tuple[tuple, tuple]]:
    store = blocker.store
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        base = blocker.candidates(query_store, batch_rows)
        ranks = candidate_ranks(base, store, blocker.config.channels, 'aggregate')
        top = ranks <= policy.top_k
        beyond = ~top
        rank_score, rank_position = _rescue(store, query_store, batch_rows, base.owners[beyond], base.records[beyond],
                                            base.aggregate_rarity[beyond], base.min_frequency[beyond])
        rank_keep = (rank_score >= policy.rank_min_score) | (rank_position <= policy.rank_per_query)
        deep = expanded_union(blocker, deep_specs, query_store, batch_rows)
        base_keys = np.sort(base.owners * store.size + base.records)
        deep_keys = deep.owners * store.size + deep.records
        at = np.minimum(np.searchsorted(base_keys, deep_keys), max(len(base_keys) - 1, 0))
        deep_only = base_keys[at] != deep_keys if len(base_keys) else np.ones(len(deep_keys), bool)
        _, deep_position = _rescue(store, query_store, batch_rows, deep.owners[deep_only], deep.records[deep_only],
                                   deep.aggregate_rarity[deep_only], deep.min_frequency[deep_only])
        deep_keep = deep_position <= deep_limit
        unique, target_index = np.unique(base.records[top], return_inverse=True)
        evidence = CandidateEvidence(store.source, ranks[top], base.masks[top], base.min_frequency[top],
                                     base.aggregate_rarity[top])
        job = (query_text.read(batch_rows), target_text.read(unique), base.owners[top], target_index, evidence,
               blocker.config.channels, policy)
        context = (start, len(batch_rows), base.owners[top], base.records[top],
                   base.owners[beyond][rank_keep], base.records[beyond][rank_keep],
                   deep.owners[deep_only][deep_keep], deep.records[deep_only][deep_keep], deep_position[deep_keep],
                   np.bincount(base.owners, minlength=len(batch_rows)))
        yield context, job


def iter_final_candidates(policy: FinalPolicy, blocking: BlockingConfig, query_store: RecordStore,
                          query_text: SourceText, target_store: RecordStore, target_text: SourceText,
                          rows: np.ndarray, index_root: str | Path, batch_size: int, workers: int = 1,
                          deep_limit: int | None = None) -> Iterator[FinalBatch]:
    """Stream the final candidate set of ``rows`` against one target source, in query order.

    ``deep_limit`` above the policy budget returns extra deep-rescue candidates (with their
    per-query ``deep_rank``) for budget sweeps; the default returns exactly the final set.
    """
    blocker = SourceBlocker(blocking, target_store, index_root)
    deep_specs = tuple(deepen(spec) for spec in blocking.channels)
    limit = policy.deep_per_query if deep_limit is None else deep_limit
    jobs = final_jobs(policy, blocker, deep_specs, query_store, query_text, target_text,
                      np.asarray(rows, np.int64), batch_size, limit)
    for context, flags in ordered_map(screen_top, jobs, workers):
        (start, n, top_owners, top_records, rank_owners, rank_records, deep_owners, deep_records, deep_position,
         counts) = context
        kept = flags != 0
        owners = np.concatenate([top_owners[kept], rank_owners, deep_owners]) + start
        records = np.concatenate([top_records[kept], rank_records, deep_records])
        provenance = np.concatenate([flags[kept], np.full(len(rank_owners), RANK_RESCUE, np.uint8),
                                     np.full(len(deep_owners), DEEP_RESCUE, np.uint8)])
        deep_rank = np.concatenate([np.zeros(int(kept.sum()) + len(rank_owners), np.int64), deep_position])
        order = np.lexsort((target_store.rank[records], owners))
        yield FinalBatch(target_store.source, owners[order], records[order], provenance[order], deep_rank[order],
                         counts)
