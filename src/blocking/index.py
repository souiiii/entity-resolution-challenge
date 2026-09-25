"""Sorted posting indexes and vectorized key combinations over CSR token arrays.

Large indexes are built on disk: keys are uniform hashes, so postings are split by
their top bits into partitions that are sorted independently and concatenated.
Peak memory is one partition plus one input chunk, and the finished index is
memory-mapped. Buckets larger than a keep limit are stored only as (key, count),
because no channel may expand them.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np

MASK64 = (1 << 64) - 1
_LEFT = 0x9E3779B97F4A7C15
_RIGHT = 0xBF58476D1CE4E5B9
INDEX_VERSION = 'posting-index-v2'


def combine_keys(left: np.ndarray, right: np.ndarray, symmetric: bool = False) -> np.ndarray:
    """Pair two uint64 key arrays into one key; ``symmetric`` ignores argument order."""
    left, right = np.asarray(left, dtype=np.uint64), np.asarray(right, dtype=np.uint64)
    if symmetric:
        left, right = np.minimum(left, right), np.maximum(left, right)
    with np.errstate(over='ignore'):
        key = (left * np.uint64(_LEFT)) ^ (right * np.uint64(_RIGHT))
    return np.where(key == 0, np.uint64(1), key)


def combine_key(left: int, right: int, symmetric: bool = False) -> int:
    """Scalar equivalent of :func:`combine_keys`."""
    if symmetric:
        left, right = min(left, right), max(left, right)
    return ((left * _LEFT) & MASK64) ^ ((right * _RIGHT) & MASK64) or 1


def rows_of(offsets: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))


def select(offsets: np.ndarray, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep masked CSR elements; returns new offsets and values."""
    counts = np.bincount(rows_of(offsets)[mask], minlength=len(offsets) - 1)
    new_offsets = np.zeros(len(offsets), dtype=np.int64)
    np.cumsum(counts, out=new_offsets[1:])
    return new_offsets, np.asarray(values)[mask]


def expand(lo: np.ndarray, hi: np.ndarray, owners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions covered by ``[lo, hi)`` ranges, with the owner of each position."""
    sizes = np.asarray(hi, dtype=np.int64) - np.asarray(lo, dtype=np.int64)
    total = int(sizes.sum())
    owner = np.repeat(np.asarray(owners), sizes)
    positions = (np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(sizes) - sizes, sizes)
                 + np.repeat(np.asarray(lo, dtype=np.int64), sizes))
    return owner, positions


def gather(offsets: np.ndarray, values: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batch-local CSR (offsets, values) of the given rows, in the given order."""
    rows = np.asarray(rows, dtype=np.int64)
    starts, stops = np.asarray(offsets[rows]), np.asarray(offsets[rows + 1])
    _, positions = expand(starts, stops, np.arange(len(rows)))
    local = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(stops - starts, out=local[1:])
    return local, np.asarray(values[positions]) if len(positions) else np.zeros(0, dtype=values.dtype)


def cross(offsets_a: np.ndarray, values_a: np.ndarray, offsets_b: np.ndarray, values_b: np.ndarray,
          symmetric: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per-row cross product of two CSR arrays as (row, combined key).

    With ``symmetric`` the same array is paired with itself and only ``i < j``
    pairs are produced.
    """
    count_a, count_b = np.diff(offsets_a), np.diff(offsets_b)
    per_row = count_a * count_b
    total = int(per_row.sum())
    if not total:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint64)
    rows = np.repeat(np.arange(len(per_row), dtype=np.int64), per_row)
    position = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(per_row) - per_row, per_row)
    width = count_b[rows]
    i, j = position // width, position % width
    if symmetric:
        keep = i < j
        rows, i, j = rows[keep], i[keep], j[keep]
    return rows, combine_keys(values_a[offsets_a[rows] + i], values_b[offsets_b[rows] + j], symmetric)


def sort_distinct(keys: np.ndarray, records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop zero keys and duplicate postings; sort by key, then record."""
    keys = np.asarray(keys, dtype=np.uint64)
    records = np.asarray(records, dtype=np.int32)
    present = keys != 0
    keys, records = keys[present], records[present]
    order = np.lexsort((records, keys))
    keys, records = keys[order], records[order]
    if len(keys):
        distinct = np.ones(len(keys), dtype=bool)
        distinct[1:] = (keys[1:] != keys[:-1]) | (records[1:] != records[:-1])
        keys, records = keys[distinct], records[distinct]
    return keys, records


def split_oversized(keys: np.ndarray, records: np.ndarray,
                    keep_limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sorted distinct postings into kept postings and (key, count) of larger buckets."""
    if keep_limit is None or not len(keys):
        return keys, records, np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    sizes = np.diff(np.r_[starts, len(keys)])
    large = sizes > keep_limit
    keep = np.repeat(~large, sizes)
    return keys[keep], records[keep], keys[starts[large]], sizes[large].astype(np.int64)


class PostingIndex:
    """Distinct (key, record) postings sorted by key, then record, plus counts of oversized buckets."""

    def __init__(self, keys: np.ndarray, records: np.ndarray, large_keys: np.ndarray | None = None,
                 large_counts: np.ndarray | None = None):
        self.keys, self.records = keys, records
        self.large_keys = np.zeros(0, dtype=np.uint64) if large_keys is None else large_keys
        self.large_counts = np.zeros(0, dtype=np.int64) if large_counts is None else large_counts

    @classmethod
    def from_pairs(cls, keys: np.ndarray, records: np.ndarray, keep_limit: int | None = None) -> PostingIndex:
        return cls(*split_oversized(*sort_distinct(keys, records), keep_limit))

    @classmethod
    def load(cls, directory: str | Path) -> PostingIndex:
        directory = Path(directory)
        meta = json.loads((directory / 'meta.json').read_text())
        length = meta['postings']
        large_keys = np.fromfile(directory / 'large_keys.bin', dtype=np.uint64)
        large_counts = np.fromfile(directory / 'large_counts.bin', dtype=np.int64)
        if not length:
            return cls(np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int32), large_keys, large_counts)
        return cls(np.memmap(directory / 'keys.bin', dtype=np.uint64, mode='r', shape=(length,)),
                   np.memmap(directory / 'records.bin', dtype=np.int32, mode='r', shape=(length,)),
                   large_keys, large_counts)

    @classmethod
    def build(cls, chunks: Iterable[tuple[np.ndarray, np.ndarray]], directory: str | Path, meta: dict,
              keep_limit: int | None = None, partition_bits: int = 4) -> PostingIndex:
        """Build from ``(records, keys)`` chunks into ``directory`` without holding all postings in memory."""
        directory = Path(directory)
        staging = directory.with_name(directory.name + '.building')
        shutil.rmtree(staging, ignore_errors=True)
        (staging / 'partitions').mkdir(parents=True)
        partitions = 1 << partition_bits
        shift = np.uint64(64 - partition_bits)
        try:
            handles = [((staging / 'partitions' / f'{p}.keys').open('wb'),
                        (staging / 'partitions' / f'{p}.records').open('wb')) for p in range(partitions)]
            try:
                for records, keys in chunks:
                    keys = np.asarray(keys, dtype=np.uint64)
                    records = np.asarray(records, dtype=np.int32)
                    part = (keys >> shift).astype(np.int64)
                    order = np.argsort(part, kind='stable')
                    bounds = np.searchsorted(part[order], np.arange(partitions + 1))
                    for p in range(partitions):
                        take = order[bounds[p]:bounds[p + 1]]
                        if len(take):
                            handles[p][0].write(keys[take].tobytes())
                            handles[p][1].write(records[take].tobytes())
            finally:
                for key_handle, record_handle in handles:
                    key_handle.close()
                    record_handle.close()
            postings = large = 0
            with (staging / 'keys.bin').open('wb') as key_out, (staging / 'records.bin').open('wb') as record_out, \
                    (staging / 'large_keys.bin').open('wb') as large_key_out, \
                    (staging / 'large_counts.bin').open('wb') as large_count_out:
                for p in range(partitions):
                    keys = np.fromfile(staging / 'partitions' / f'{p}.keys', dtype=np.uint64)
                    records = np.fromfile(staging / 'partitions' / f'{p}.records', dtype=np.int32)
                    (staging / 'partitions' / f'{p}.keys').unlink()
                    (staging / 'partitions' / f'{p}.records').unlink()
                    keys, records, large_keys, large_counts = split_oversized(*sort_distinct(keys, records), keep_limit)
                    key_out.write(keys.tobytes())
                    record_out.write(records.tobytes())
                    large_key_out.write(large_keys.tobytes())
                    large_count_out.write(large_counts.tobytes())
                    postings += len(keys)
                    large += len(large_keys)
            (staging / 'partitions').rmdir()
            meta = {**meta, 'index_version': INDEX_VERSION, 'keep_limit': keep_limit, 'postings': postings,
                    'oversized_keys': large, 'bytes': postings * 12 + large * 16}
            (staging / 'meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
            shutil.rmtree(directory, ignore_errors=True)
            staging.rename(directory)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return cls.load(directory)

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posting bounds of each key and its full target frequency (including oversized buckets)."""
        keys = np.asarray(keys, dtype=np.uint64)
        lo, hi = np.searchsorted(self.keys, keys, 'left'), np.searchsorted(self.keys, keys, 'right')
        frequency = hi - lo
        if len(self.large_keys) and len(keys):
            at = np.minimum(np.searchsorted(self.large_keys, keys), len(self.large_keys) - 1)
            frequency = np.where(self.large_keys[at] == keys, self.large_counts[at], frequency)
        return lo, hi, frequency

    def frequency(self, keys: np.ndarray) -> np.ndarray:
        return self.lookup(keys)[2]

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.records.nbytes + self.large_keys.nbytes + self.large_counts.nbytes)
