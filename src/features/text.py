"""Bounded random access to raw source records through line byte offsets.

The only persistent artifact is one int64 offset per record (plus the end of the
file); names and addresses are read from the unchanged raw TSV and canonicalized
on demand. Every read is checked against the blocking store's entity ID.
"""
from __future__ import annotations

import csv
import json
import mmap
import shutil
from pathlib import Path

import numpy as np

from src.blocking.store import ID_DTYPE, RecordStore
from src.data.load import SOURCE_COLUMNS
from src.data.validation import file_hash

TEXT_VERSION = 'text-offsets-v1'
_CHUNK = 64 * 2**20


def _line_starts(path: Path) -> tuple[np.ndarray, int]:
    starts = [np.zeros(1, np.int64)]
    position = 0
    with path.open('rb') as stream:
        while chunk := stream.read(_CHUNK):
            newlines = np.flatnonzero(np.frombuffer(chunk, np.uint8) == 10)
            starts.append(newlines.astype(np.int64) + position + 1)
            position += len(chunk)
    starts = np.concatenate(starts)
    return starts[starts < position], position


def build_text_offsets(tsv_path: str | Path, store: RecordStore, directory: str | Path,
                       min_free_bytes: int = 0) -> Path:
    """Write ``offsets.bin`` (data-line starts plus file end) after checking every ID against the store."""
    tsv_path, directory = Path(tsv_path), Path(directory)
    needed = 8 * (store.size + 2)
    directory.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory.parent).free
    if free - needed < min_free_bytes:
        raise OSError(f'Refusing to build text offsets: needs {needed / 2**20:.1f} MiB, {free / 2**30:.2f} GiB free')
    staging = directory.with_name(directory.name + '.building')
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        starts, size = _line_starts(tsv_path)
        with tsv_path.open('rb') as stream:
            if stream.readline().rstrip(b'\r\n').decode('utf-8').split('\t') != list(SOURCE_COLUMNS):
                raise ValueError(f'{tsv_path}: unexpected header')
        offsets = np.append(starts[1:], size).astype(np.int64)
        if len(offsets) != store.size + 1:
            raise ValueError(f'{tsv_path}: {len(offsets) - 1} data lines, store has {store.size} records')
        raw = np.memmap(tsv_path, dtype=np.uint8, mode='r')
        width = ID_DTYPE.itemsize + 1
        for start in range(0, store.size, 1_000_000):
            begin = offsets[start:min(store.size, start + 1_000_000)]
            window = np.asarray(raw[np.minimum(begin[:, None] + np.arange(width), size - 1)])
            tab = np.argmax(window == 9, axis=1)
            if not (window[np.arange(len(begin)), tab] == 9).all():
                raise ValueError(f'{tsv_path}: entity ID without a following tab near row {start}')
            window = np.where(np.arange(width) < tab[:, None], window, 0)[:, :ID_DTYPE.itemsize]
            ids = np.ascontiguousarray(window, dtype=np.uint8).view(ID_DTYPE).reshape(-1)
            if not np.array_equal(ids, np.asarray(store.ids[start:start + len(begin)])):
                raise ValueError(f'{tsv_path}: offsets disagree with store IDs near row {start}')
        del raw
        offsets.tofile(staging / 'offsets.bin')
        meta = {'text_version': TEXT_VERSION, 'input': tsv_path.name, 'input_sha256': store.meta['input_sha256'],
                'records': store.size, 'bytes': int(offsets.nbytes)}
        (staging / 'meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
        shutil.rmtree(directory, ignore_errors=True)
        staging.rename(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return directory


class SourceText:
    """Read raw ``(name, address, country)`` fields of store rows."""

    def __init__(self, tsv_path: str | Path, store: RecordStore, directory: str | Path):
        self.path, self.store, self.directory = Path(tsv_path), store, Path(directory)
        meta = json.loads((self.directory / 'meta.json').read_text())
        if meta['text_version'] != TEXT_VERSION or meta['records'] != store.size or \
                meta['input_sha256'] != store.meta['input_sha256']:
            raise ValueError(f'Stale text offsets in {self.directory}')
        self.offsets = np.fromfile(self.directory / 'offsets.bin', dtype=np.int64)
        self._stream = self.path.open('rb')
        self._data = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_READ)

    def read(self, rows: np.ndarray) -> list[tuple[str, str, str]]:
        """Fields of each requested row, in request order; IDs are verified against the store."""
        result = []
        for row in np.asarray(rows, dtype=np.int64).tolist():
            line = self._data[self.offsets[row]:self.offsets[row + 1]].decode('utf-8').rstrip('\r\n')
            fields = next(csv.reader([line], delimiter='\t', strict=True))
            if len(fields) != len(SOURCE_COLUMNS) or fields[0].encode('utf-8') != self.store.ids[row]:
                raise ValueError(f'{self.path}: row {row} does not match store ID {self.store.ids[row]!r}')
            result.append((fields[1], fields[2], fields[3]))
        return result

    def close(self) -> None:
        self._data.close()
        self._stream.close()


def ensure_text(tsv_path: str | Path, store: RecordStore, directory: str | Path, min_free_bytes: int = 0) -> SourceText:
    directory = Path(directory)
    try:
        return SourceText(tsv_path, store, directory)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        build_text_offsets(tsv_path, store, directory, min_free_bytes)
        return SourceText(tsv_path, store, directory)
