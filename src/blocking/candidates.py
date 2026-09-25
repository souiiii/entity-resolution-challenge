"""Union of channel candidates per target source, with per-candidate channel bitmasks."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.blocking.channels import BlockingConfig, ChannelSpec, build_index, run_channel
from src.blocking.index import PostingIndex
from src.blocking.store import RecordStore

LOGGER = logging.getLogger(__name__)


@dataclass
class CandidateBatch:
    """Deduplicated candidates of a query batch for one target source.

    Rows are sorted by query position, then by target ID in binary order. ``masks``
    has bit ``i`` set when configured channel ``i`` produced the candidate.
    """
    owners: np.ndarray
    records: np.ndarray
    masks: np.ndarray
    channel_stats: dict[str, dict[str, int]]


class SourceBlocker:
    """Channel indexes for one target store; channels not configured for it are skipped."""

    def __init__(self, config: BlockingConfig, store: RecordStore, index_root: str | Path, min_free_bytes: int = 0):
        self.store = store
        self.channels: list[tuple[int, ChannelSpec, PostingIndex]] = []
        self.build_seconds: dict[str, float] = {}
        for bit, spec in enumerate(config.channels):
            if store.source in spec.sources:
                started = time.monotonic()
                self.channels.append((bit, spec, build_index(spec, store, index_root, config.chunk_rows,
                                                             min_free_bytes)))
                self.build_seconds[spec.name] = round(time.monotonic() - started, 2)
                LOGGER.info('S%s index %s: %s postings', store.source, spec.name, len(self.channels[-1][2].keys))

    @property
    def index_bytes(self) -> dict[str, int]:
        return {spec.name: index.nbytes for _, spec, index in self.channels}

    def candidates(self, query_store: RecordStore, rows: np.ndarray) -> CandidateBatch:
        owners, records, masks, stats = [], [], [], {}
        for bit, spec, index in self.channels:
            result = run_channel(spec, index, query_store, rows)
            owners.append(result.owners)
            records.append(result.records)
            masks.append(np.full(len(result.owners), 1 << bit, dtype=np.uint32))
            stats[spec.name] = {'queries_capped': result.queries_capped, 'queries_limited': result.queries_limited,
                                'largest_bucket': result.largest_bucket}
        return union(np.concatenate(owners) if owners else np.zeros(0, dtype=np.int64),
                     np.concatenate(records) if records else np.zeros(0, dtype=np.int64),
                     np.concatenate(masks) if masks else np.zeros(0, dtype=np.uint32),
                     self.store, stats)


def union(owners: np.ndarray, records: np.ndarray, masks: np.ndarray, store: RecordStore,
          stats: dict[str, dict[str, int]] | None = None) -> CandidateBatch:
    """Deduplicate (query, record) pairs, OR-ing their channel masks, in deterministic order."""
    combined = owners.astype(np.int64) * store.size + store.rank[records]
    order = np.argsort(combined, kind='stable')
    combined, masks = combined[order], masks[order]
    if len(combined):
        starts = np.flatnonzero(np.r_[True, combined[1:] != combined[:-1]])
        masks = np.bitwise_or.reduceat(masks, starts)
        combined = combined[starts]
    owners, ranks = np.divmod(combined, store.size)
    return CandidateBatch(owners, store.sorted_order[ranks].astype(np.int64), masks, stats or {})
