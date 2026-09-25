"""Compact, memory-mapped key store for one source file.

A store holds, per record in file order: the entity ID, a country code, the four
exact keys and CSR-encoded name/address token keys. It contains no text beyond IDs
and is rebuilt only when the input hash, normalization version or store version
changes.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

from src.blocking.keys import EXACT_FIELDS, record_keys
from src.data.inspect import _batches
from src.data.load import iter_source
from src.data.validation import file_hash
from src.preprocessing.normalization import NORMALIZATION_VERSION
from src.utils.parallel import ordered_map

LOGGER = logging.getLogger(__name__)
STORE_VERSION = 'blocking-store-v1'
ID_DTYPE = np.dtype('S16')
BYTES_PER_RAW_BYTE_ESTIMATE = 2.5
ARRAYS = {'ids': ID_DTYPE, 'country': np.dtype(np.int16), 'exact': np.dtype(np.uint64),
          'name_tokens': np.dtype(np.uint64), 'name_counts': np.dtype(np.uint8),
          'address_tokens': np.dtype(np.uint64), 'address_kinds': np.dtype(np.uint8),
          'address_counts': np.dtype(np.uint16)}


def encode_batch(rows: list[tuple[str, str, str]]) -> dict[str, np.ndarray]:
    """Key arrays for (name, address, country) rows, in input order."""
    exact = np.zeros((len(rows), len(EXACT_FIELDS)), dtype=np.uint64)
    name_tokens, name_counts, address_tokens, address_kinds, address_counts = [], [], [], [], []
    for i, (name, address, country) in enumerate(rows):
        keys = record_keys(name, address, country)
        exact[i] = keys.exact
        name_tokens.extend(keys.name_tokens)
        name_counts.append(len(keys.name_tokens))
        address_tokens.extend(keys.address_tokens)
        address_kinds.extend(keys.address_kinds)
        address_counts.append(len(keys.address_tokens))
    return {'exact': exact.reshape(-1), 'name_tokens': np.array(name_tokens, dtype=np.uint64),
            'name_counts': np.array(name_counts, dtype=np.uint8),
            'address_tokens': np.array(address_tokens, dtype=np.uint64),
            'address_kinds': np.array(address_kinds, dtype=np.uint8),
            'address_counts': np.array(address_counts, dtype=np.uint16)}


def _offsets(counts: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


class RecordStore:
    """Read-only view of a built store; arrays are memory-mapped."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / 'meta.json').read_text())
        arrays = {name: np.memmap(self.directory / f'{name}.bin', dtype=ARRAYS[name], mode='r',
                                  shape=(self.meta['lengths'][name],)) if self.meta['lengths'][name] else
                  np.zeros(0, dtype=ARRAYS[name]) for name in ARRAYS}
        self.source: int = self.meta['source']
        self.size: int = self.meta['records']
        self.countries: list[str] = self.meta['countries']
        self.ids = arrays['ids']
        self.country = arrays['country']
        exact = arrays['exact'].reshape(self.size, len(EXACT_FIELDS))
        self.exact = {field: exact[:, i] for i, field in enumerate(EXACT_FIELDS)}
        self.name_tokens = arrays['name_tokens']
        self.name_offsets = _offsets(arrays['name_counts'])
        self.address_tokens = arrays['address_tokens']
        self.address_kinds = arrays['address_kinds']
        self.address_offsets = _offsets(arrays['address_counts'])

    def index_of(self, ids: np.ndarray) -> np.ndarray:
        """Record positions of the given IDs; -1 where absent."""
        ids = np.asarray(ids, dtype=ID_DTYPE)
        if not self.size:
            return np.full(len(ids), -1, dtype=np.int64)
        found = np.minimum(np.searchsorted(self.sorted_ids, ids), self.size - 1)
        positions = self.sorted_order[found]
        return np.where(self.ids[positions] == ids, positions, -1)

    @property
    def sorted_order(self) -> np.ndarray:
        if not hasattr(self, '_sorted_order'):
            self._sorted_order = np.argsort(self.ids, kind='stable')
        return self._sorted_order

    @property
    def sorted_ids(self) -> np.ndarray:
        if not hasattr(self, '_sorted_ids'):
            self._sorted_ids = np.asarray(self.ids)[self.sorted_order]
        return self._sorted_ids

    @property
    def rank(self) -> np.ndarray:
        """Position of each record in binary ID order; used for deterministic output order."""
        if not hasattr(self, '_rank'):
            rank = np.empty(self.size, dtype=np.int64)
            rank[self.sorted_order] = np.arange(self.size)
            self._rank = rank
        return self._rank


def store_matches(directory: Path, input_hash: str, source: int) -> bool:
    meta_path = directory / 'meta.json'
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return (meta.get('store_version') == STORE_VERSION and meta.get('normalization_version') == NORMALIZATION_VERSION
            and meta.get('input_sha256') == input_hash and meta.get('source') == source)


def build_store(tsv_path: str | Path, source: int, directory: str | Path, workers: int = 1,
                batch_size: int = 20_000, input_hash: str | None = None) -> RecordStore:
    """Stream one source file into a store directory; a failed build leaves no partial store."""
    tsv_path, directory = Path(tsv_path), Path(directory)
    input_hash = input_hash or file_hash(tsv_path)
    staging = directory.with_name(directory.name + '.building')
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    countries: dict[str, int] = {}
    lengths = dict.fromkeys(ARRAYS, 0)
    try:
        handles = {name: (staging / f'{name}.bin').open('wb') for name in ARRAYS}
        try:
            jobs = ((batch, [(r['business_name'], r['business_address'], r['country']) for r in batch])
                    for batch in _batches(iter_source(tsv_path, source), batch_size))
            for batch, arrays in ordered_map(encode_batch, jobs, workers):
                ids = [r['entity_id'].encode('utf-8') for r in batch]
                if max(map(len, ids)) > ID_DTYPE.itemsize:
                    raise ValueError(f'Entity ID longer than {ID_DTYPE.itemsize} bytes in {tsv_path}')
                arrays['ids'] = np.array(ids, dtype=ID_DTYPE)
                arrays['country'] = np.array([countries.setdefault(r['country'], len(countries)) for r in batch],
                                             dtype=np.int16)
                for name, array in arrays.items():
                    handles[name].write(np.ascontiguousarray(array, dtype=ARRAYS[name]).tobytes())
                    lengths[name] += len(array)
        finally:
            for handle in handles.values():
                handle.close()
        meta = {'store_version': STORE_VERSION, 'normalization_version': NORMALIZATION_VERSION,
                'source': source, 'input': tsv_path.name, 'input_sha256': input_hash,
                'records': lengths['ids'], 'countries': list(countries), 'lengths': lengths,
                'bytes': sum((staging / f'{name}.bin').stat().st_size for name in ARRAYS)}
        (staging / 'meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
        shutil.rmtree(directory, ignore_errors=True)
        staging.rename(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    LOGGER.info('Built store for %s: %s records, %.1f MiB', tsv_path.name, meta['records'], meta['bytes'] / 2**20)
    return RecordStore(directory)


def ensure_store(tsv_path: str | Path, source: int, directory: str | Path, workers: int = 1,
                 batch_size: int = 20_000, min_free_bytes: int = 0) -> RecordStore:
    """Reuse a matching store or build one after checking free disk space."""
    tsv_path, directory = Path(tsv_path), Path(directory)
    input_hash = file_hash(tsv_path)
    if store_matches(directory, input_hash, source):
        return RecordStore(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    needed = int(tsv_path.stat().st_size * BYTES_PER_RAW_BYTE_ESTIMATE)
    free = shutil.disk_usage(directory.parent).free
    if free - needed < min_free_bytes:
        raise OSError(f'Refusing to build {directory.name}: needs ~{needed / 2**30:.2f} GiB, '
                      f'{free / 2**30:.2f} GiB free, reserve {min_free_bytes / 2**30:.2f} GiB')
    return build_store(tsv_path, source, directory, workers, batch_size, input_hash)
