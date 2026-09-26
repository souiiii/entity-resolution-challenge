from dataclasses import replace

import numpy as np
import pytest

from src.blocking.channels import ChannelSpec, build_index, run_channel
from src.blocking.overflow import BroadRetriever, build_overflow, disk_guard, estimate_overflow
from src.blocking.candidates import SourceBlocker
from tests.test_candidate_recovery import world, BLOCKING


def test_overflow_postings_bands_reuse_and_unicode(world):
    stores, _, root = world
    spec = ChannelSpec('name', 'token', ('name',), 1, index_keep=1)
    base = build_index(spec, stores[2], root / 'base')
    overflow = build_overflow(spec, stores[2], base, root / 'overflow', 6, chunk_rows=2, reserve_gib=0)
    full = build_index(replace(spec, index_keep=6, max_frequency=6), stores[2], root / 'full')
    assert estimate_overflow(base, 6)['bytes'] == overflow.nbytes
    for key, lo, hi in zip(overflow.keys, overflow.offsets[:-1], overflow.offsets[1:]):
        a, b, freq = full.lookup(np.array([key]))
        assert np.array_equal(overflow.records[lo:hi], full.records[a[0]:b[0]])
        assert 1 < freq[0] <= 6
        assert base.lookup(np.array([key]))[0][0] == base.lookup(np.array([key]))[1][0]
    path = overflow.directory / 'records.npy'
    timestamp = path.stat().st_mtime_ns
    reused = build_overflow(spec, stores[2], base, root / 'overflow', 6, reserve_gib=0)
    assert timestamp == path.stat().st_mtime_ns
    assert np.array_equal(reused.records, overflow.records)
    missing = reused.lookup(np.array([0], np.uint64))
    assert all(a.tolist() == [0] for a in missing)
    with pytest.raises(ValueError):
        build_overflow(replace(spec, sources=(3,)), stores[2], base, root / 'other', 6, reserve_gib=0)


def test_broad_union_is_unique_deterministic_and_preserves_deep(world):
    stores, _, root = world
    config = replace(BLOCKING, channels=tuple(replace(c, index_keep=2, max_frequency=1) for c in BLOCKING.channels))
    for source in (2, 3):
        blocker = SourceBlocker(config, stores[source], root / 'base')
        retriever = BroadRetriever(blocker, root / 'overflow', 6, reserve_gib=0)
        rows = stores[1].sorted_order
        deep, broad = retriever.retrieve(stores[1], rows)
        pairs = list(zip(broad.owners, broad.records))
        assert len(pairs) == len(set(pairs))
        assert set(zip(deep.owners, deep.records)) <= set(pairs)
        assert pairs == list(zip(*[getattr(retriever.retrieve(stores[1], rows)[1], k) for k in ('owners', 'records')]))
        assert pairs == sorted(pairs, key=lambda x:(x[0], stores[source].rank[x[1]]))
        assert all(stores[source].ids[r].startswith(f'S{source}-'.encode()) for _, r in pairs)


def test_disk_guard_reserves_future_allocation(monkeypatch, tmp_path):
    monkeypatch.setattr('src.blocking.overflow.shutil.disk_usage', lambda _: type('Usage', (), {'free': 12 * 2**30})())
    assert disk_guard(tmp_path, 2**30) == 12 * 2**30
    with pytest.raises(OSError, match='reserve'):
        disk_guard(tmp_path, 2 * 2**30)


def test_frequency_band_boundaries():
    from src.blocking.index import PostingIndex

    counts = np.array([200, 201, 300, 301, 500, 501, 750, 751, 1000, 1001])
    keys = np.repeat(np.arange(1, len(counts) + 1, dtype=np.uint64), counts)
    records = np.concatenate([np.arange(c, dtype=np.int32) for c in counts])
    base = PostingIndex.from_pairs(keys, records, keep_limit=200)
    for maximum in (300, 500, 750, 1000):
        selected = (counts > 200) & (counts <= maximum)
        result = estimate_overflow(base, maximum)
        assert result['buckets'] == selected.sum()
        assert result['postings'] == counts[selected].sum()


def test_stale_index_is_rejected(world):
    import json

    stores, _, root = world
    spec = ChannelSpec('name', 'token', ('name',), 1, index_keep=1)
    base = build_index(spec, stores[2], root / 'base')
    overflow = build_overflow(spec, stores[2], base, root / 'overflow', 6, reserve_gib=0)
    path = overflow.directory / 'meta.json'
    meta = json.loads(path.read_text())
    meta['input_sha256'] = 'stale'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='Stale'):
        build_overflow(spec, stores[2], base, root / 'overflow', 6, reserve_gib=0)
