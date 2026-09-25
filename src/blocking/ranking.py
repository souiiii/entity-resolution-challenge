from __future__ import annotations

import numpy as np

from src.blocking.candidates import CandidateBatch
from src.blocking.channels import ChannelSpec
from src.blocking.store import RecordStore
from src.evaluation.metrics import entity_f05

STRATEGIES = ('channels', 'families', 'rarity', 'aggregate')


def oracle_ceiling(truth: np.ndarray, found: np.ndarray) -> float:
    """Exact oracle blocking Macro F0.5 ceiling using the frozen entity scorer."""
    truth, found = np.asarray(truth), np.asarray(found)
    if (truth.ndim != 1 or truth.shape != found.shape or (found < 0).any() or (found > truth).any()
            or (truth != np.floor(truth)).any() or (found != np.floor(found)).any()):
        raise ValueError('Oracle recovery counts must lie between zero and truth')
    if not truth.size:
        return 0.0
    pairs, counts = np.unique(np.column_stack((truth, found)), axis=0, return_counts=True)
    scores = [entity_f05(set(range(int(t))), set(range(int(f)))) for t, f in pairs]
    return float(np.dot(scores, counts) / truth.size)


def candidate_ranks(batch: CandidateBatch, store: RecordStore, channels: tuple[ChannelSpec, ...],
                    strategy: str) -> np.ndarray:
    """One-based per-query ranks; final ties use target ID binary order.

    Exact address precedes exact name, then support and rarity. Families collapse
    the correlated exact-name views; rarity exchanges the last two priorities.
    Aggregate ranks by summed inverse frequencies before family support and
    minimum frequency. All evidence comes from the expanded blocking keys.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f'Unknown ranking strategy {strategy}')
    n = len(batch.owners)
    address, name = np.zeros(n, bool), np.zeros(n, bool)
    support = np.zeros(n, np.int16)
    groups: dict[tuple, np.ndarray] = {}
    for bit, spec in enumerate(channels):
        present = (batch.masks & (1 << bit)) != 0
        support += present
        if spec.kind == 'exact' and spec.parts == ('address_canonical',):
            address |= present
        if spec.kind == 'exact' and spec.parts[0].startswith('name_'):
            name |= present
            group = ('exact_name',)
        else:
            group = (spec.kind, *spec.parts)
        if group not in groups:
            groups[group] = present.copy()
        else:
            groups[group] |= present
    if strategy != 'channels':
        support = sum((v.astype(np.int16) for v in groups.values()), np.zeros(n, np.int16))
    frequency = batch.min_frequency
    if frequency is None:
        raise ValueError('Ranking requires shared-key bucket frequencies')
    secondary = (frequency, -support) if strategy != 'rarity' else (-support, frequency)
    if strategy == 'aggregate':
        if batch.aggregate_rarity is None:
            raise ValueError('Aggregate ranking requires summed inverse bucket frequencies')
        secondary = (frequency, -support, -batch.aggregate_rarity)
    order = np.lexsort((store.rank[batch.records], *secondary, -name.astype(np.int8),
                        -address.astype(np.int8), batch.owners))
    owners = batch.owners[order]
    starts = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]]) if n else np.zeros(0, np.int64)
    ranks = np.empty(n, np.int32)
    ranks[order] = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n])) + 1
    return ranks


def prune_candidates(batch: CandidateBatch, store: RecordStore, channels: tuple[ChannelSpec, ...],
                     strategy: str, top_k: int) -> CandidateBatch:
    """Prune one target source independently, retaining deterministic binary output order."""
    if top_k < 1:
        raise ValueError('top_k must be positive')
    keep = candidate_ranks(batch, store, channels, strategy) <= top_k
    return CandidateBatch(batch.owners[keep], batch.records[keep], batch.masks[keep],
                          batch.channel_stats, batch.min_frequency[keep],
                          batch.aggregate_rarity[keep] if batch.aggregate_rarity is not None else None)
