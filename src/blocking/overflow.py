"""Compact supplemental postings for frequency-bounded oversized buckets."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from src.blocking.candidates import SourceBlocker
from src.blocking.channels import ChannelSpec, query_keys
from src.blocking.recovery import ExpandedBatch
from src.blocking.index import PostingIndex
from src.blocking.store import RecordStore, STORE_VERSION

OVERFLOW_VERSION = 'overflow-csr-v1'


def disk_guard(path: Path, required_bytes: int = 0, reserve_gib: float = 10.5) -> int:
    """Refuse writes that would consume the free-disk reserve."""
    free = shutil.disk_usage(path).free
    if free - required_bytes < int(reserve_gib * 2**30):
        raise OSError(f'Disk reserve {reserve_gib} GiB: free={free}, required={required_bytes}')
    return free


def estimate_overflow(index: PostingIndex, maximum: int) -> dict:
    selected = index.large_counts <= maximum
    buckets = int(selected.sum())
    postings = int(index.large_counts[selected].sum())
    return {'buckets': buckets, 'postings': postings, 'bytes': 16 * buckets + 8 + 4 * postings}


class OverflowIndex:
    """CSR key dictionary with int64 offsets and int32 target rows, disjoint from the base index."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / 'meta.json').read_text())
        if self.meta['version'] != OVERFLOW_VERSION:
            raise ValueError('Unsupported overflow version')
        self.keys = np.load(self.directory / 'keys.npy', mmap_mode='r')
        self.offsets = np.load(self.directory / 'offsets.npy', mmap_mode='r')
        self.records = np.load(self.directory / 'records.npy', mmap_mode='r')
        if self.keys.dtype != np.uint64 or self.offsets.dtype != np.int64 or self.records.dtype != np.int32:
            raise ValueError('Invalid overflow dtypes')
        if len(self.offsets) != len(self.keys) + 1 or self.offsets[-1] != len(self.records):
            raise ValueError('Invalid overflow dimensions')

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keys = np.asarray(keys, np.uint64)
        if not len(self.keys):
            empty = np.zeros(len(keys), np.int64)
            return empty, empty, empty
        at = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        present = self.keys[at] == keys
        lo = np.where(present, self.offsets[at], 0)
        hi = np.where(present, self.offsets[at + 1], 0)
        return lo, hi, hi - lo

    @property
    def nbytes(self) -> int:
        return self.keys.nbytes + self.offsets.nbytes + self.records.nbytes


def build_overflow(spec: ChannelSpec, store: RecordStore, base: PostingIndex, root: Path,
                   maximum: int, chunk_rows: int = 10000, reserve_gib: float = 10.5) -> OverflowIndex:
    """Fill preallocated CSR slots by ascending record batches; existing postings are never duplicated."""
    if store.source not in spec.sources or spec.index_keep is None or maximum <= spec.index_keep:
        raise ValueError('Overflow requires a supported source and maximum above the finite base keep')
    if store.size > np.iinfo(np.int32).max:
        raise ValueError('Target store exceeds int32 row capacity')
    directory = Path(root) / f'source{store.source}' / f'{spec.index_id}-to{maximum}'
    identity = {'version': OVERFLOW_VERSION, 'store_version': STORE_VERSION,
                'input_sha256': store.meta['input_sha256'], 'source': store.source,
                'base_index_id': spec.index_id, 'maximum': maximum}
    if (directory / 'meta.json').exists():
        result = OverflowIndex(directory)
        if all(result.meta.get(k) == v for k, v in identity.items()):
            return result
        raise ValueError(f'Stale overflow index: {directory}')
    directory.parent.mkdir(parents=True, exist_ok=True)
    estimate = estimate_overflow(base, maximum)
    lowest = disk_guard(directory.parent, estimate['bytes'] + 16384, reserve_gib)
    staging = directory.with_name(directory.name + '.building')
    if staging.exists():
        raise FileExistsError(f'Incomplete overflow build: {staging}')
    staging.mkdir()
    try:
        selected = base.large_counts <= maximum
        keys = base.large_keys[selected]
        counts = base.large_counts[selected]
        offsets = np.r_[0, np.cumsum(counts)]
        np.save(staging / 'keys.npy', keys)
        np.save(staging / 'offsets.npy', offsets)
        records = np.lib.format.open_memmap(staging / 'records.npy', mode='w+', dtype=np.int32,
                                            shape=(int(offsets[-1]),))
        cursor = offsets[:-1].copy()
        if len(keys):
            for start in range(0, store.size, chunk_rows):
                rows = np.arange(start, min(store.size, start + chunk_rows))
                owners, values = query_keys(spec, store, rows)
                at = np.minimum(np.searchsorted(keys, values), len(keys) - 1)
                hit = keys[at] == values
                at, owners = at[hit], owners[hit] + start
                order = np.lexsort((owners, at))
                at, owners = at[order], owners[order]
                if len(at):
                    starts = np.flatnonzero(np.r_[True, at[1:] != at[:-1]])
                    sizes = np.diff(np.r_[starts, len(at)])
                    within = np.arange(len(at)) - np.repeat(starts, sizes)
                    if (cursor[at[starts]] + sizes > offsets[at[starts] + 1]).any():
                        raise ValueError('Base bucket counts disagree with regenerated keys')
                    records[cursor[at] + within] = owners
                    cursor[at[starts]] += sizes
                lowest = min(lowest, disk_guard(staging, reserve_gib=reserve_gib))
        if not np.array_equal(cursor, offsets[1:]):
            raise ValueError('Incomplete overflow postings')
        records.flush()
        del records
        meta = {**identity, **estimate, 'min_free_disk_gib': lowest / 2**30}
        (staging / 'meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
        staging.rename(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return OverflowIndex(directory)


def merge_expanded(batches: list[ExpandedBatch], store: RecordStore) -> ExpandedBatch:
    """Merge retrieved key hits without duplicate query/target pairs, preserving evidence."""
    owners = np.concatenate([b.owners for b in batches])
    records = np.concatenate([b.records for b in batches])
    masks = np.concatenate([b.masks for b in batches])
    frequency = np.concatenate([b.min_frequency for b in batches])
    rarity = np.concatenate([b.aggregate_rarity for b in batches])
    hits = np.concatenate([b.hits for b in batches])
    key = owners * store.size + store.rank[records]
    order = np.argsort(key, kind='stable')
    key = key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.zeros(0, np.int64)
    owner, rank = np.divmod(key[starts], store.size)
    return ExpandedBatch(owner, store.sorted_order[rank], np.bitwise_or.reduceat(masks[order], starts),
                         np.minimum.reduceat(frequency[order], starts), np.add.reduceat(rarity[order], starts),
                         np.add.reduceat(hits[order], starts))


class BroadRetriever:
    """Existing deep pool plus supplemental overflow; this is never the official candidate output."""

    def __init__(self, blocker: SourceBlocker, root: Path, maximum: int = 1000, reserve_gib: float = 10.5) -> None:
        self.blocker = blocker
        self.maximum = maximum
        self.overflow = []
        for bit, spec, base in blocker.channels:
            index = build_overflow(spec, blocker.store, base, root, maximum, reserve_gib=reserve_gib)
            self.overflow.append((bit, spec, index))

    def retrieve(self, query_store: RecordStore, rows: np.ndarray) -> tuple[ExpandedBatch, ExpandedBatch]:
        from dataclasses import replace
        from src.blocking.channels import run_channel
        from src.blocking.recovery import ExpandedBatch, deepen, expanded_union

        deep = expanded_union(self.blocker, tuple(deepen(c) for _, c, _ in self.blocker.channels), query_store, rows)
        batches = [deep]
        for bit, spec, index in self.overflow:
            result = run_channel(replace(spec, index_keep=self.maximum, max_frequency=self.maximum, max_keys=None),
                                 index, query_store, rows)
            batches.append(ExpandedBatch(result.owners, result.records,
                np.full(len(result.owners), 1 << bit, np.uint32), result.frequencies,
                1.0 / result.frequencies, np.ones(len(result.owners), np.int64)))
        return deep, merge_expanded(batches, self.blocker.store)
