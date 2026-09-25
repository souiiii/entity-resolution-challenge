"""Streaming screen-then-match pipeline over frozen Phase 3B candidates and Phase 4 features.

Stages per query batch and target source: Phase 3B candidates, Phase 4 features,
evidence reduction, prefilter. After both sources are collected, the matcher
receives exactly the screened candidates. Evaluation reuses the frozen per-entity
scorer; submission files go through the centralized official writer.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.blocking.channels import BlockingConfig
from src.blocking.evaluate import distribution
from src.blocking.ranking import oracle_ceiling
from src.blocking.store import RecordStore
from src.evaluation.evaluate import evaluate_sets
from src.features.extract import iter_features
from src.features.pairwise import feature_names
from src.features.text import SourceText
from src.matching.rules import (Evidence, MatcherConfig, PrefilterConfig, evidence_from_features, prefilter_mask,
                                relative_gate, rule_acceptable, rule_paths)
from src.submission.create_submission import CANDIDATE_HEADER, MATCHING_HEADER, write_id_list_file

LOGGER = logging.getLogger(__name__)


@dataclass
class Candidates:
    """Collected candidates of all queries: query position, source, target record, evidence and labels."""
    owners: np.ndarray
    sources: np.ndarray
    records: np.ndarray
    evidence: Evidence
    labels: np.ndarray | None
    queries: int
    unscreened_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    unscreened_found: np.ndarray | None = None

    def take(self, mask: np.ndarray) -> Candidates:
        return Candidates(self.owners[mask], self.sources[mask], self.records[mask], self.evidence.take(mask),
                          None if self.labels is None else self.labels[mask], self.queries,
                          self.unscreened_counts, self.unscreened_found)


def _concat_evidence(parts: list[Evidence]) -> Evidence:
    return Evidence(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                      ('name', 'address', 'number_state', 'script_mismatch', 'rarity', 'rank')))


def collect_candidates(blocking: BlockingConfig, prefilter: PrefilterConfig | None, query_store: RecordStore,
                       query_text: SourceText, targets: dict[int, tuple[RecordStore, SourceText]], rows: np.ndarray,
                       index_root: str | Path, batch_size: int, workers: int,
                       truth: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
                       feature_consumer: Callable[[np.ndarray], None] | None = None) -> Candidates:
    """Stream both target sources and keep candidates passing ``prefilter`` (all of them when ``None``)."""
    names = feature_names(blocking.channels)
    n = len(rows)
    parts: dict[str, list] = {k: [] for k in ('owners', 'sources', 'records', 'evidence', 'labels')}
    unscreened_counts = np.zeros(n, np.int64)
    unscreened_found = np.zeros(n, np.int64) if truth is not None else None
    for source, (store, text) in sorted(targets.items()):
        keys = None
        if truth is not None:
            owners, records = truth[source]
            keys = np.sort(owners * store.size + records)
        for batch in iter_features(blocking, query_store, query_text, store, text, rows, index_root, batch_size,
                                   workers):
            evidence = evidence_from_features(batch.features, names)
            unscreened_counts += np.bincount(batch.owners, minlength=n)
            labels = None
            if keys is not None:
                pair_keys = batch.owners * store.size + batch.records
                at = np.minimum(np.searchsorted(keys, pair_keys), max(len(keys) - 1, 0))
                labels = keys[at] == pair_keys if len(keys) else np.zeros(len(pair_keys), bool)
                unscreened_found += np.bincount(batch.owners[labels], minlength=n)
            keep = np.ones(len(batch.owners), bool) if prefilter is None else prefilter_mask(evidence, prefilter)
            if feature_consumer is not None:
                feature_consumer(batch.features[keep])
            parts['owners'].append(batch.owners[keep].astype(np.int32))
            parts['sources'].append(np.full(int(keep.sum()), source, np.int8))
            parts['records'].append(batch.records[keep].astype(np.int32))
            parts['evidence'].append(evidence.take(keep))
            if labels is not None:
                parts['labels'].append(labels[keep])
        LOGGER.info('S%s collected', source)
    return Candidates(np.concatenate(parts['owners']), np.concatenate(parts['sources']),
                      np.concatenate(parts['records']), _concat_evidence(parts['evidence']),
                      np.concatenate(parts['labels']) if truth is not None else None, n, unscreened_counts,
                      unscreened_found)


def screen(candidates: Candidates, prefilter: PrefilterConfig) -> Candidates:
    """The explicit prefilter stage: the candidate set handed to the matcher."""
    return candidates.take(prefilter_mask(candidates.evidence, prefilter))


def match(screened: Candidates, config: MatcherConfig) -> np.ndarray:
    """Matcher decisions for screened candidates only; may accept zero, one or several per query."""
    acceptable = rule_acceptable(screened.evidence, config.rules)
    return relative_gate(screened.owners.astype(np.int64), acceptable, screened.evidence.rarity,
                         config.rules.relative_rarity, screened.queries)


def evaluate(screened: Candidates, accepted: np.ndarray, truth_per_query: np.ndarray,
             query_mask: np.ndarray | None = None) -> dict:
    """Frozen per-entity Macro F0.5 plus candidate diagnostics for screened candidates and decisions."""
    if screened.labels is None:
        raise ValueError('Evaluation requires labelled candidates')
    n = screened.queries
    selected = np.ones(n, bool) if query_mask is None else query_mask
    order = np.lexsort((screened.records, screened.sources, screened.owners))
    owners, sources, records = screened.owners[order], screened.sources[order], screened.records[order]
    labels, decided = screened.labels[order], accepted[order]
    bounds = np.searchsorted(owners, np.arange(n + 1))
    found = np.bincount(owners[labels], minlength=n)

    def pairs():
        for q in np.flatnonzero(selected):
            lo, hi = bounds[q], bounds[q + 1]
            items = list(zip(sources[lo:hi].tolist(), records[lo:hi].tolist()))
            positive = {items[i] for i in np.flatnonzero(labels[lo:hi])}
            truth = positive | {('missing', i) for i in range(int(truth_per_query[q]) - int(found[q]))}
            yield truth, {items[i] for i in np.flatnonzero(decided[lo:hi])}

    result = evaluate_sets(pairs())
    counts = np.bincount(screened.owners, minlength=n)[selected]
    links = int(truth_per_query[selected].sum())
    result['candidates'] = distribution(counts)
    result['candidate_recall'] = float(found[selected].sum() / links) if links else None
    result['oracle_ceiling'] = oracle_ceiling(truth_per_query[selected], found[selected])
    if screened.unscreened_found is not None:
        result['unscreened_candidates'] = distribution(screened.unscreened_counts[selected])
        result['unscreened_recall'] = float(screened.unscreened_found[selected].sum() / links) if links else None
        result['unscreened_oracle_ceiling'] = oracle_ceiling(truth_per_query[selected],
                                                             screened.unscreened_found[selected])
    in_queries = selected[screened.owners]
    for source in np.unique(screened.sources).tolist():
        mask = in_queries & (screened.sources == source)
        tp = int((accepted & screened.labels & mask).sum())
        fp = int((accepted & ~screened.labels & mask).sum())
        result[f'source{source}'] = {'tp': tp, 'fp': fp, 'predicted': tp + fp,
                                     'precision': tp / (tp + fp) if tp + fp else None,
                                     'candidate_positives': int((screened.labels & mask).sum())}
    return result


def path_counts(screened: Candidates, accepted: np.ndarray, config: MatcherConfig) -> dict:
    """Accepted candidates per rule path (a candidate can satisfy several paths)."""
    paths = rule_paths(screened.evidence, config.rules)
    result = {}
    for name, mask in paths.items():
        chosen = mask & accepted
        entry = {'accepted': int(chosen.sum())}
        if screened.labels is not None:
            entry['true'] = int((chosen & screened.labels).sum())
        result[name] = entry
    return result


def submission_rows(ids: list[str], screened: Candidates, accepted: np.ndarray, stores: dict[int, RecordStore],
                    matches: bool):
    """Rows of one official file in query order; S2 before S3, each in binary target-ID order."""
    rank = np.zeros(len(screened.records), np.int64)
    for source, store in stores.items():
        mask = screened.sources == source
        rank[mask] = store.rank[screened.records[mask]]
    keep = accepted if matches else np.ones(len(accepted), bool)
    order = np.lexsort((rank[keep], screened.sources[keep], screened.owners[keep]))
    owners = screened.owners[keep][order]
    sources = screened.sources[keep][order]
    records = screened.records[keep][order]
    bounds = np.searchsorted(owners, np.arange(len(ids) + 1))
    id_arrays = {source: np.asarray(store.ids) for source, store in stores.items()}
    for q, source1 in enumerate(ids):
        lo, hi = bounds[q], bounds[q + 1]
        yield source1, [id_arrays[s][r].decode('utf-8') for s, r in zip(sources[lo:hi].tolist(),
                                                                          records[lo:hi].tolist())]


def write_submission(ids: list[str], screened: Candidates, accepted: np.ndarray, stores: dict[int, RecordStore],
                     output_dir: str | Path) -> dict[str, Path]:
    """Write candidate_pairs.tsv (the screened set) and matching_results.tsv (its accepted subset)."""
    output_dir = Path(output_dir)
    paths = {'candidate_pairs': output_dir / 'candidate_pairs.tsv',
             'matching_results': output_dir / 'matching_results.tsv'}
    write_id_list_file(paths['candidate_pairs'], CANDIDATE_HEADER, submission_rows(ids, screened, accepted, stores, False))
    write_id_list_file(paths['matching_results'], MATCHING_HEADER, submission_rows(ids, screened, accepted, stores, True))
    return paths
