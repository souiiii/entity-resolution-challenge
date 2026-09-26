"""Candidate-recall loss accounting and cheap hashed evidence for final-candidate policies.

Everything here works on the blocking stores (hashed, country-scoped tokens) and the
posting indexes; no raw text is read and no Phase 4 feature matrix is built.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.blocking.candidates import CandidateBatch, SourceBlocker
from src.blocking.channels import ChannelSpec, part_csr, query_keys, run_channel
from src.blocking.index import rows_of
from src.blocking.ranking import candidate_ranks
from src.blocking.store import RecordStore

LOSS_CATEGORIES = ('A_no_shared_key', 'B_index_truncated', 'C_frequency_cap', 'C_key_limit', 'D_rank_cut',
                   'E_prefilter', 'F_other')
NO_KEY = np.iinfo(np.int64).max


@dataclass
class LinkTrace:
    """Where each true link stands in the blocking union of its source (aligned with the truth arrays)."""
    in_union: np.ndarray
    rank: np.ndarray
    masks: np.ndarray
    min_frequency: np.ndarray
    aggregate_rarity: np.ndarray
    union_counts: np.ndarray


def trace_links(blocker: SourceBlocker, query_store: RecordStore, rows: np.ndarray, truth_owners: np.ndarray,
                truth_records: np.ndarray, batch_size: int, strategy: str = 'aggregate') -> LinkTrace:
    """Run the unchanged union and ranking per batch and record each true link's union evidence and rank."""
    target = blocker.store
    n, links = len(rows), len(truth_owners)
    in_union = np.zeros(links, bool)
    rank = np.full(links, np.iinfo(np.int32).max, np.int32)
    masks = np.zeros(links, np.uint32)
    min_frequency = np.zeros(links, np.int64)
    rarity = np.zeros(links, np.float64)
    counts = np.zeros(n, np.int64)
    truth_key = truth_owners * target.size + truth_records
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        batch = blocker.candidates(query_store, rows[start:stop])
        ranks = candidate_ranks(batch, target, blocker.config.channels, strategy)
        counts[start:stop] = np.bincount(batch.owners, minlength=stop - start)
        keys = (batch.owners + start) * target.size + batch.records
        order = np.argsort(keys)
        lo, hi = np.searchsorted(truth_owners, [start, stop])
        if not len(keys) or hi == lo:
            continue
        at = np.minimum(np.searchsorted(keys[order], truth_key[lo:hi]), len(keys) - 1)
        hit = keys[order][at] == truth_key[lo:hi]
        position = order[at][hit]
        index = np.arange(lo, hi)[hit]
        in_union[index] = True
        rank[index] = ranks[position]
        masks[index] = batch.masks[position]
        min_frequency[index] = batch.min_frequency[position]
        rarity[index] = batch.aggregate_rarity[position]
    return LinkTrace(in_union, rank, masks, min_frequency, rarity, counts)


def shared_key_profile(blocker: SourceBlocker, query_store: RecordStore, query_rows: np.ndarray,
                       target_rows: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """``(pairs, channels)`` smallest target frequency of a key shared by each pair; ``NO_KEY`` if none."""
    profile = np.full((len(query_rows), len(blocker.channels)), NO_KEY, np.int64)
    for start in range(0, len(query_rows), chunk):
        q, t = query_rows[start:start + chunk], target_rows[start:start + chunk]
        for column, (_, spec, index) in enumerate(blocker.channels):
            qo, qk = query_keys(spec, query_store, q)
            to, tk = query_keys(spec, blocker.store, t)
            owners = np.concatenate([qo, to])
            keys = np.concatenate([qk, tk])
            side = np.concatenate([np.zeros(len(qo), np.int8), np.ones(len(to), np.int8)])
            order = np.lexsort((side, keys, owners))
            owners, keys, side = owners[order], keys[order], side[order]
            shared = np.flatnonzero((owners[1:] == owners[:-1]) & (keys[1:] == keys[:-1]) & (side[1:] != side[:-1]))
            if len(shared):
                np.minimum.at(profile[start:start + len(q), column], owners[shared], index.frequency(keys[shared]))
    return profile


def classify_unretrieved(profile: np.ndarray, channels: list[ChannelSpec]) -> np.ndarray:
    """Loss category of links absent from the union, from their per-channel shared-key frequencies.

    A retrievable key within the channel cap that was not expanded was dropped by the
    channel's key limit; a retrievable key above the cap by the frequency cap; a key
    whose bucket exceeds the index keep limit is physically truncated.
    """
    caps = np.array([c.max_frequency for c in channels], np.int64)
    keeps = np.array([c.index_keep if c.index_keep is not None else NO_KEY - 1 for c in channels], np.int64)
    has = profile < NO_KEY
    key_limit = (has & (profile <= caps)).any(axis=1)
    frequency_cap = (has & (profile > caps) & (profile <= keeps)).any(axis=1)
    truncated = has.any(axis=1)
    return np.select([key_limit, frequency_cap, truncated], ['C_key_limit', 'C_frequency_cap', 'B_index_truncated'],
                     'A_no_shared_key')


_MIX = np.uint64(0x9E3779B97F4A7C15)


def _mix(owners: np.ndarray, values: np.ndarray) -> np.ndarray:
    with np.errstate(over='ignore'):
        return np.asarray(values, np.uint64) ^ (np.asarray(owners, np.uint64) * _MIX)


def distinct_counts(store: RecordStore, part: str, chunk: int = 500_000) -> np.ndarray:
    """Distinct hashed tokens of one part per record, computed once per store and cached on it."""
    cache = store.__dict__.setdefault('_distinct_counts', {})
    if part not in cache:
        counts = np.zeros(store.size, np.uint16)
        for start in range(0, store.size, chunk):
            rows = np.arange(start, min(store.size, start + chunk))
            offsets, values = part_csr(store, part, rows)
            counts[rows] = np.minimum(np.bincount(rows_of(offsets)[np.unique(_mix(rows_of(offsets), values),
                                                                                return_index=True)[1]],
                                                  minlength=len(rows)), np.iinfo(np.uint16).max)
        cache[part] = counts
    return cache[part]


def token_overlap(query_store: RecordStore, target_store: RecordStore, part: str, query_rows: np.ndarray,
                  target_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per aligned pair: distinct shared hashed tokens of one part and each side's distinct token count.

    Query tokens are gathered once per distinct query; each pair's target tokens are probed
    against them and only the hits are deduplicated.
    """
    n = len(query_rows)
    unique, local = np.unique(np.asarray(query_rows, np.int64), return_inverse=True)
    q_offsets, q_values = part_csr(query_store, part, unique)
    query_keys = np.unique(_mix(rows_of(q_offsets), q_values))
    t_offsets, t_values = part_csr(target_store, part, np.asarray(target_rows, np.int64))
    pair = rows_of(t_offsets)
    probe = _mix(local[pair], t_values)
    shared = np.zeros(n, np.int64)
    if len(query_keys) and len(probe):
        at = np.minimum(np.searchsorted(query_keys, probe), len(query_keys) - 1)
        hit = query_keys[at] == probe
        hits = np.unique(_mix(pair[hit], t_values[hit]), return_index=True)[1]
        shared = np.bincount(pair[hit][hits], minlength=n)
    return (shared, distinct_counts(query_store, part)[query_rows].astype(np.int64),
            distinct_counts(target_store, part)[target_rows].astype(np.int64))


@dataclass
class HashedEvidence:
    """Cheap pair evidence from hashed store tokens (no text, no Phase 4 features)."""
    name_shared: np.ndarray
    name_jaccard: np.ndarray
    name_containment: np.ndarray
    word_shared: np.ndarray
    word_jaccard: np.ndarray
    word_containment: np.ndarray
    number_state: np.ndarray
    exact_name: np.ndarray
    exact_address: np.ndarray


UNAVAILABLE, AGREE, PARTIAL, CONFLICT = 0, 1, 2, 3


def _ratios(shared: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    union = left + right - shared
    smaller = np.minimum(left, right)
    with np.errstate(invalid='ignore', divide='ignore'):
        jaccard = np.where(union > 0, shared / union, np.nan)
        containment = np.where(smaller > 0, shared / smaller, np.nan)
    return jaccard.astype(np.float32), containment.astype(np.float32)


def hashed_evidence(query_store: RecordStore, target_store: RecordStore, query_rows: np.ndarray,
                    target_rows: np.ndarray) -> HashedEvidence:
    """Name/address-word/number overlap and exact-key agreement for aligned (query row, target row) pairs."""
    name_shared, name_q, name_t = token_overlap(query_store, target_store, 'name', query_rows, target_rows)
    word_shared, word_q, word_t = token_overlap(query_store, target_store, 'address_word', query_rows, target_rows)
    number_shared, number_q, number_t = token_overlap(query_store, target_store, 'address_number', query_rows,
                                                      target_rows)
    name_jaccard, name_containment = _ratios(name_shared, name_q, name_t)
    word_jaccard, word_containment = _ratios(word_shared, word_q, word_t)
    available = (number_q > 0) & (number_t > 0)
    agree = available & (number_shared == np.minimum(number_q, number_t))
    conflict = available & (number_shared == 0)
    state = np.select([agree, conflict, available], [AGREE, CONFLICT, PARTIAL], UNAVAILABLE).astype(np.int8)

    def equal(field: str) -> np.ndarray:
        left = np.asarray(query_store.exact[field])[query_rows]
        return (left != 0) & (left == np.asarray(target_store.exact[field])[target_rows])

    return HashedEvidence(name_shared.astype(np.int16), name_jaccard, name_containment, word_shared.astype(np.int16),
                          word_jaccard, word_containment, state, equal('name_folded') | equal('name_folded_compact'),
                          equal('address_canonical'))


def families(masks: np.ndarray, channels: tuple[ChannelSpec, ...]) -> np.ndarray:
    """Number of distinct channel families (the three exact-name views count once) in each mask."""
    groups: dict[tuple, np.ndarray] = {}
    for bit, spec in enumerate(channels):
        present = (masks & np.uint32(1 << bit)) != 0
        family = ('exact_name',) if spec.kind == 'exact' and spec.parts[0].startswith('name_') else (spec.kind, *spec.parts)
        groups[family] = groups.get(family, np.zeros(len(masks), bool)) | present
    return sum((g.astype(np.int16) for g in groups.values()), np.zeros(len(masks), np.int16))


def union_batch(blocker: SourceBlocker, query_store: RecordStore, rows: np.ndarray,
                strategy: str = 'aggregate') -> tuple[CandidateBatch, np.ndarray]:
    batch = blocker.candidates(query_store, rows)
    return batch, candidate_ranks(batch, blocker.store, blocker.config.channels, strategy)


@dataclass
class ExpandedBatch:
    """Deduplicated deep-retrieval candidates of a query batch with blocking evidence and key-hit counts."""
    owners: np.ndarray
    records: np.ndarray
    masks: np.ndarray
    min_frequency: np.ndarray
    aggregate_rarity: np.ndarray
    hits: np.ndarray


def deepen(spec: ChannelSpec, depth: int | None = None, max_keys: int | None = None) -> ChannelSpec:
    """The same channel and posting index with its frequency cap lifted to ``depth`` (default: the index keep)."""
    limit = spec.index_keep if depth is None else (depth if spec.index_keep is None else min(depth, spec.index_keep))
    if limit is None:
        raise ValueError(f'{spec.name}: deep retrieval needs an index keep limit or an explicit depth')
    return replace(spec, max_frequency=max(spec.max_frequency, limit), max_keys=max_keys)


def expanded_union(blocker: SourceBlocker, specs: tuple[ChannelSpec, ...], query_store: RecordStore,
                   rows: np.ndarray) -> ExpandedBatch:
    """Union of ``specs`` (aligned with the blocker's channels) with the number of retrieved keys per pair."""
    owners, records, masks, frequencies = [], [], [], []
    for (bit, _, index), spec in zip(blocker.channels, specs):
        result = run_channel(spec, index, query_store, rows)
        owners.append(result.owners)
        records.append(result.records)
        frequencies.append(result.frequencies)
        masks.append(np.full(len(result.owners), 1 << bit, np.uint32))
    owners, records = np.concatenate(owners), np.concatenate(records)
    masks, frequencies = np.concatenate(masks), np.concatenate(frequencies)
    store = blocker.store
    combined = owners.astype(np.int64) * store.size + store.rank[records]
    order = np.argsort(combined, kind='stable')
    combined, masks, frequencies = combined[order], masks[order], frequencies[order]
    if not len(combined):
        empty = np.zeros(0, np.int64)
        return ExpandedBatch(empty, empty, np.zeros(0, np.uint32), empty, np.zeros(0), empty)
    starts = np.flatnonzero(np.r_[True, combined[1:] != combined[:-1]])
    hits = np.diff(np.r_[starts, len(combined)])
    rarity = np.add.reduceat(1.0 / frequencies, starts)
    owners, ranks = np.divmod(combined[starts], store.size)
    return ExpandedBatch(owners, store.sorted_order[ranks].astype(np.int64), np.bitwise_or.reduceat(masks, starts),
                         np.minimum.reduceat(frequencies, starts), rarity, hits)


def cheap_score(evidence: HashedEvidence, aggregate_rarity: np.ndarray) -> np.ndarray:
    """Candidate-generation score: name and address-word containment, number agreement, exact agreement, rarity."""
    number = np.select([evidence.number_state == AGREE, evidence.number_state == CONFLICT], [0.5, -0.5], 0.0)
    return (np.nan_to_num(evidence.name_containment, nan=0.0) + np.nan_to_num(evidence.word_containment, nan=0.0)
            + number + (evidence.exact_name | evidence.exact_address) + np.minimum(aggregate_rarity, 1.0)).astype(np.float32)


def rank_per_query(owners: np.ndarray, score: np.ndarray, tiebreak: np.ndarray, target_rank: np.ndarray) -> np.ndarray:
    """One-based position of each candidate within its query: score descending, ``tiebreak`` ascending, target ID."""
    order = np.lexsort((target_rank, tiebreak, -score, owners))
    sorted_owners = owners[order]
    starts = np.flatnonzero(np.r_[True, sorted_owners[1:] != sorted_owners[:-1]]) if len(order) else order
    position = np.empty(len(order), np.int64)
    position[order] = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)])) + 1
    return position


def top_per_query(owners: np.ndarray, score: np.ndarray, tiebreak: np.ndarray, target_rank: np.ndarray,
                  limit: int) -> np.ndarray:
    """Mask of each query's ``limit`` best candidates under :func:`rank_per_query`."""
    return rank_per_query(owners, score, tiebreak, target_rank) <= limit


def set_metrics(counts: np.ndarray, found: dict[int, np.ndarray], truth: dict[int, np.ndarray],
                countries: np.ndarray, country_names: list[str], reference: dict | None = None) -> dict:
    """Candidate-set recall, oracle Macro F0.5 ceiling and size diagnostics from per-query aggregates.

    ``found`` and ``truth`` map each target source to per-query true-link counts. ``reference``
    (another result of this function) adds incremental links, pairs and efficiency.
    """
    from src.blocking.ranking import oracle_ceiling

    total_found = sum(found.values())
    total_truth = sum(truth.values())
    links = int(total_truth.sum())
    linked = total_truth > 0
    result = {
        'recall': float(total_found.sum() / links), 'found_links': int(total_found.sum()), 'links': links,
        'oracle_macro_f05': oracle_ceiling(total_truth, total_found),
        'pairs': int(counts.sum()), 'mean': float(counts.mean()), 'median': float(np.median(counts)),
        **{f'p{q}': int(np.percentile(counts, q, method='inverted_cdf')) for q in (90, 95, 99)},
        'max': int(counts.max()), 'zero_candidate_queries': int((counts == 0).sum()),
        'entities_with_truth': int(linked.sum()),
        'entities_all_recovered': int((linked & (total_found == total_truth)).sum()),
        'entities_some_recovered': int((linked & (total_found > 0)).sum()),
        'entities_none_recovered': int((linked & (total_found == 0)).sum()),
        'source_recall': {f'S{s}': float(found[s].sum() / truth[s].sum()) for s in sorted(found)},
        'country_recall': {country_names[c]: float(total_found[countries == c].sum() / total_truth[countries == c].sum())
                           for c in np.unique(countries)}}
    if reference is not None:
        added_links = result['found_links'] - reference['found_links']
        added_pairs = result['pairs'] - reference['pairs']
        result.update({'incremental_true_links': added_links, 'incremental_pairs': added_pairs,
                       'recall_gain_pp': 100 * (result['recall'] - reference['recall']),
                       'true_links_per_1000_added_pairs': 1000 * added_links / added_pairs if added_pairs > 0 else None})
    return result
