"""Compact normalized text for repeated broad candidate-ranking inference."""
from __future__ import annotations

import json
import mmap
import shutil
from pathlib import Path

import numpy as np

from src.blocking.overflow import disk_guard
from src.features.text import SourceText
from src.preprocessing.normalization import NORMALIZATION_VERSION, canonicalize, fold_latin_marks

TEXT_VERSION = 'candidate-text-v1'


class CandidateText:
    """Memory-mapped UTF-8 name/address views, with one int64 offset per record."""

    def __init__(self, directory: Path, identity: dict):
        meta = json.loads((directory / 'meta.json').read_text())
        if any(meta.get(k) != v for k, v in identity.items()):
            raise ValueError(f'Stale candidate text: {directory}')
        self.offsets = np.load(directory / 'offsets.npy', mmap_mode='r')
        self._stream = (directory / 'text.bin').open('rb')
        self._data = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_READ) if meta['bytes'] else b''

    def read_one(self, row: int) -> tuple[str, str]:
        value = self._data[self.offsets[row]:self.offsets[row + 1]]
        name, address = value.split(b'\0', 1)
        return name.decode('utf-8'), address.decode('utf-8')

    def close(self) -> None:
        if hasattr(self._data, 'close'):
            self._data.close()
        self._stream.close()


def ensure_candidate_text(text: SourceText, root: Path, reserve_gib: float = 10.5,
                          batch_size: int = 10000) -> CandidateText:
    """Normalize once in bounded batches; guard every write and reuse input-bound artifacts."""
    directory = Path(root) / text.path.stem
    identity = {'version': TEXT_VERSION, 'normalization_version': NORMALIZATION_VERSION,
                'source': text.store.source, 'input_sha256': text.store.meta['input_sha256'],
                'records': text.store.size}
    if (directory / 'meta.json').exists():
        return CandidateText(directory, identity)
    directory.parent.mkdir(parents=True, exist_ok=True)
    estimate = 2 * text.path.stat().st_size + 8 * (text.store.size + 1) + 16384
    lowest = disk_guard(directory.parent, estimate, reserve_gib)
    staging = directory.with_name(directory.name + '.building')
    if staging.exists():
        raise FileExistsError(f'Incomplete candidate text build: {staging}')
    staging.mkdir()
    try:
        offsets = np.lib.format.open_memmap(staging / 'offsets.npy', mode='w+', dtype=np.int64,
                                           shape=(text.store.size + 1,))
        offsets[0] = 0
        position = 0
        with (staging / 'text.bin').open('wb') as stream:
            for start in range(0, text.store.size, batch_size):
                rows = np.arange(start, min(text.store.size, start + batch_size))
                values = [(fold_latin_marks(canonicalize(name)) + '\0' + canonicalize(address)).encode('utf-8')
                          for name, address, _ in text.read(rows)]
                chunk = b''.join(values)
                lowest = min(lowest, disk_guard(staging, len(chunk) + 16384, reserve_gib))
                stream.write(chunk)
                offsets[start + 1:start + len(rows) + 1] = position + np.cumsum([len(v) for v in values])
                position += len(chunk)
        offsets.flush()
        del offsets
        meta = {**identity, 'bytes': position, 'estimated_disk_bytes': estimate,
                'min_free_disk_gib': min(lowest, disk_guard(staging, 16384, reserve_gib)) / 2**30}
        (staging / 'meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
        staging.rename(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return CandidateText(directory, identity)
