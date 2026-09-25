"""Config-driven candidate channels over a target :class:`RecordStore`."""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.blocking.index import INDEX_VERSION, PostingIndex, cross, expand, gather, rows_of, select
from src.blocking.keys import EXACT_FIELDS, NUMBER
from src.blocking.store import STORE_VERSION, RecordStore

TOKEN_PARTS = ('name', 'address_word', 'address_number')
KINDS = ('exact', 'token', 'combo')
INDEX_BYTES_PER_POSTING = 12


@dataclass(frozen=True)
class ChannelSpec:
    """One channel. ``max_frequency`` skips keys whose target bucket is larger;
    ``max_keys`` keeps only a query's rarest eligible keys (``None`` keeps all)."""
    name: str
    kind: str
    parts: tuple[str, ...]
    max_frequency: int
    max_keys: int | None = None
    symmetric: bool = False
    sources: tuple[int, ...] = (2, 3)
    index_keep: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f'{self.name}: unknown kind {self.kind!r}')
        allowed = EXACT_FIELDS if self.kind == 'exact' else TOKEN_PARTS + (EXACT_FIELDS if self.kind == 'combo' else ())
        expected = {'exact': 1, 'token': 1, 'combo': 2}[self.kind]
        if len(self.parts) != expected or any(p not in allowed for p in self.parts):
            raise ValueError(f'{self.name}: invalid parts {self.parts}')
        if self.symmetric and (self.kind != 'combo' or self.parts[0] != self.parts[1]):
            raise ValueError(f'{self.name}: symmetric combos need identical parts')
        if self.max_frequency < 1 or (self.max_keys is not None and self.max_keys < 1):
            raise ValueError(f'{self.name}: limits must be positive')
        if self.index_keep is not None and self.index_keep < self.max_frequency:
            raise ValueError(f'{self.name}: index_keep must be at least max_frequency')
        if not self.sources or not set(self.sources) <= {2, 3}:
            raise ValueError(f'{self.name}: sources must be a nonempty subset of (2, 3)')

    @property
    def index_id(self) -> str:
        """Identifies the posting index; query-time limits do not change it."""
        keep = 'all' if self.index_keep is None else f'le{self.index_keep}'
        return '-'.join((self.kind, *self.parts, 'sym' if self.symmetric else 'ord', keep))


@dataclass(frozen=True)
class BlockingConfig:
    version: str
    channels: tuple[ChannelSpec, ...]
    batch_size: int = 20_000
    workers: int = 6
    sample_salt: str = 'blocking-sample-v1'
    sample_fraction: float = 0.02
    min_free_disk_gib: float = 3.0
    chunk_rows: int = 200_000
    raw: dict = field(default_factory=dict, compare=False)


def load_config(path: str | Path) -> BlockingConfig:
    raw = json.loads(Path(path).read_text())
    channels = tuple(ChannelSpec(name=c['name'], kind=c['kind'], parts=tuple(c['parts']),
                                 max_frequency=c['max_frequency'], max_keys=c.get('max_keys'),
                                 symmetric=c.get('symmetric', False), sources=tuple(c.get('sources', (2, 3))),
                                 index_keep=c.get('index_keep', raw.get('index_keep')))
                     for c in raw['channels'])
    names = [c.name for c in channels]
    if len(names) != len(set(names)) or not channels or len(channels) > 31:
        raise ValueError('Channel names must be unique and number 1-31')
    return BlockingConfig(version=raw['version'], channels=channels, batch_size=raw.get('batch_size', 20_000),
                          workers=raw.get('workers', 6), sample_salt=raw['sample']['salt'],
                          sample_fraction=raw['sample']['fraction'],
                          min_free_disk_gib=raw.get('min_free_disk_gib', 3.0),
                          chunk_rows=raw.get('chunk_rows', 200_000), raw=raw)


def part_csr(store: RecordStore, part: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local CSR (offsets, keys) of one token part for the given store rows."""
    if part in EXACT_FIELDS:
        values = np.asarray(store.exact[part][rows])
        offsets = np.r_[0, np.cumsum(values != 0)]
        return offsets, values[values != 0]
    if part == 'name':
        return gather(store.name_offsets, store.name_tokens, rows)
    offsets, values = gather(store.address_offsets, store.address_tokens, rows)
    _, kinds = gather(store.address_offsets, store.address_kinds, rows)
    number = kinds == NUMBER
    return select(offsets, values, number if part == 'address_number' else ~number)


def batch_keys(spec: ChannelSpec, store: RecordStore, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position within ``rows``, key) pairs of one channel; keys may repeat within a position."""
    rows = np.asarray(rows, dtype=np.int64)
    if spec.kind == 'exact':
        keys = np.asarray(store.exact[spec.parts[0]][rows])
        present = keys != 0
        return np.flatnonzero(present), keys[present]
    if spec.kind == 'token':
        offsets, values = part_csr(store, spec.parts[0], rows)
        return rows_of(offsets), values
    left = part_csr(store, spec.parts[0], rows)
    right = left if spec.symmetric else part_csr(store, spec.parts[1], rows)
    return cross(*left, *right, symmetric=spec.symmetric)


def query_keys(spec: ChannelSpec, store: RecordStore, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct (position within ``rows``, key) pairs of one channel, sorted."""
    owners, keys = batch_keys(spec, store, rows)
    if len(keys):
        order = np.lexsort((keys, owners))
        owners, keys = owners[order], keys[order]
        distinct = np.ones(len(keys), dtype=bool)
        distinct[1:] = (owners[1:] != owners[:-1]) | (keys[1:] != keys[:-1])
        owners, keys = owners[distinct], keys[distinct]
    return owners, keys


def index_chunks(spec: ChannelSpec, store: RecordStore, chunk_rows: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for start in range(0, store.size, chunk_rows):
        rows = np.arange(start, min(store.size, start + chunk_rows), dtype=np.int64)
        owners, keys = batch_keys(spec, store, rows)
        yield owners + start, keys


def _part_counts(store: RecordStore, part: str) -> np.ndarray:
    if part in EXACT_FIELDS:
        return (np.asarray(store.exact[part]) != 0).astype(np.int64)
    if part == 'name':
        return np.diff(store.name_offsets)
    counts = np.diff(store.address_offsets)
    if not len(store.address_kinds):
        return counts
    number = np.asarray(store.address_kinds) == NUMBER
    starts = np.minimum(store.address_offsets[:-1], len(number) - 1)
    numbers = np.where(counts > 0, np.add.reduceat(number, starts, dtype=np.int64), 0)
    return numbers if part == 'address_number' else counts - numbers


def estimate_postings(spec: ChannelSpec, store: RecordStore) -> int:
    """Upper bound on the postings of a channel index, from per-row token counts."""
    if spec.kind == 'exact':
        return store.size
    left = _part_counts(store, spec.parts[0])
    if spec.kind == 'token':
        return int(left.sum())
    if spec.symmetric:
        return int((left * (left - 1) // 2).sum())
    return int((left * _part_counts(store, spec.parts[1])).sum())


def build_index(spec: ChannelSpec, store: RecordStore, index_root: str | Path, chunk_rows: int = 200_000,
                min_free_bytes: int = 0) -> PostingIndex:
    """Load the cached disk index of a channel for this store, building it if absent or stale."""
    directory = Path(index_root) / f'source{store.source}' / spec.index_id
    meta = {'store_version': STORE_VERSION, 'store_input_sha256': store.meta['input_sha256'],
            'source': store.source, 'index_id': spec.index_id}
    meta_path = directory / 'meta.json'
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing.get('index_version') == INDEX_VERSION and all(existing.get(k) == v for k, v in meta.items()):
            return PostingIndex.load(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory.parent).free
    estimate = 2 * INDEX_BYTES_PER_POSTING * estimate_postings(spec, store)
    if free - estimate < min_free_bytes:
        raise OSError(f'Refusing to build index {spec.index_id} for source {store.source}: '
                      f'needs ~{estimate / 2**30:.2f} GiB, {free / 2**30:.2f} GiB free')
    return PostingIndex.build(index_chunks(spec, store, chunk_rows), directory, meta, spec.index_keep)


@dataclass
class ChannelResult:
    owners: np.ndarray
    records: np.ndarray
    queries_capped: int
    queries_limited: int
    largest_bucket: int
    frequencies: np.ndarray


def run_channel(spec: ChannelSpec, index: PostingIndex, query_store: RecordStore,
                rows: np.ndarray) -> ChannelResult:
    """Candidates of one channel for a query batch, applying frequency and key-count limits."""
    owners, keys = query_keys(spec, query_store, rows)
    lo, hi, size = index.lookup(keys)
    capped = size > spec.max_frequency
    queries_capped = len(np.unique(owners[capped]))
    eligible = (size > 0) & ~capped
    owners, lo, hi, size, keys = owners[eligible], lo[eligible], hi[eligible], size[eligible], keys[eligible]
    queries_limited = 0
    if spec.max_keys is not None and len(owners):
        order = np.lexsort((keys, size, owners))
        owners, lo, hi, size = owners[order], lo[order], hi[order], size[order]
        starts = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])
        rank = np.arange(len(owners)) - np.repeat(starts, np.diff(np.r_[starts, len(owners)]))
        keep = rank < spec.max_keys
        queries_limited = len(np.unique(owners[~keep]))
        owners, lo, hi, size = owners[keep], lo[keep], hi[keep], size[keep]
    owner, positions = expand(lo, hi, owners)
    records = np.asarray(index.records[positions], dtype=np.int64) if len(positions) else np.zeros(0, np.int64)
    return ChannelResult(owner, records, queries_capped, queries_limited, int(size.max()) if len(size) else 0,
                         np.repeat(size, hi - lo))
