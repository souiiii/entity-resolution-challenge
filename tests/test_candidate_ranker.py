from dataclasses import replace

import numpy as np
import pytest

from src.blocking.candidates import SourceBlocker
from src.blocking.overflow import BroadRetriever
from src.matching.candidate_ranker import (CheapText, feature_names, retrieval_features, training_queries,
                                          rescue_ranks, select_candidates, sampled_pairs)
from tests.test_candidate_recovery import world, BLOCKING


def test_query_split_is_deterministic_and_disjoint():
    ids = np.array([f'S1-{i}'.encode() for i in range(1000)])
    split = training_queries(ids, 'phase9', 0.7)
    assert 600 < split.sum() < 800
    assert np.array_equal(training_queries(ids[::-1], 'phase9', 0.7), split[::-1])
    assert not np.any(split & ~split)
    assert np.array_equal(training_queries(ids, 'phase9', 0.7), split)


def test_selection_preserves_phase8_and_stable_ties():
    owners = np.array([0, 0, 0, 1, 1])
    base = np.array([True, False, False, False, True])
    ranks = rescue_ranks(owners, np.ones(5), base, np.array([2, 2, 2, 3, 1]), np.array([0, 3, 1, 2, 4]))
    assert ranks.tolist() == [0, 2, 1, 1, 0]
    assert select_candidates(base, ranks, 0).tolist() == base.tolist()
    assert select_candidates(base, ranks, 1).tolist() == [True, False, True, True, True]
    with pytest.raises(ValueError):
        select_candidates(base, ranks, -1)


def test_feature_parity_subset_and_unicode(world):
    stores, texts, root = world
    blocking = replace(BLOCKING, channels=tuple(replace(c, index_keep=2, max_frequency=1) for c in BLOCKING.channels))
    blocker = SourceBlocker(blocking, stores[2], root / 'base')
    retriever = BroadRetriever(blocker, root / 'overflow', 6, reserve_gib=0)
    rows = stores[1].sorted_order
    deep, broad = retriever.retrieve(stores[1], rows)
    q, t = CheapText(texts[1]), CheapText(texts[2])
    matrix, ranks = retrieval_features(stores[1], stores[2], rows, broad, deep, blocking.channels, q, t)
    selected = np.arange(0, len(matrix), 2)
    subset, ranks2 = retrieval_features(stores[1], stores[2], rows, broad, deep, blocking.channels, q, t, selected)
    np.testing.assert_allclose(subset, matrix[selected], equal_nan=True)
    np.testing.assert_equal(ranks, ranks2)
    assert matrix.shape[1] == len(feature_names(blocking.channels))
    assert not any('country' in k or 'entity_id' in k for k in feature_names(blocking.channels))
    assert set(np.unique(matrix[:, 0])) == {0}
    labels = np.zeros(len(matrix), bool)
    labels[-1] = True
    sample = sampled_pairs(broad, labels, stores[2].size, 901, 1, 1, stores[2].rank)
    assert len(matrix) - 1 in sample
    np.testing.assert_equal(sample, sampled_pairs(broad, labels, stores[2].size, 901, 1, 1, stores[2].rank))


def test_official_candidate_semantics_and_phase8_preservation(world):
    from src.matching.candidate_ranker import iter_ranked_candidates
    from src.matching.final_candidates import iter_final_candidates
    from tests.test_candidate_recovery import policy

    stores, texts, root = world
    blocking = replace(BLOCKING, channels=tuple(replace(c, index_keep=2, max_frequency=1) for c in BLOCKING.channels))

    class Model:
        @staticmethod
        def feature_name():
            return list(feature_names(blocking.channels))

        @staticmethod
        def predict(matrix, num_threads):
            return np.nan_to_num(matrix[:, 21]) + np.nan_to_num(matrix[:, 23])

    rows = stores[1].sorted_order
    base = list(iter_final_candidates(policy(), blocking, stores[1], texts[1], stores[2], texts[2], rows,
                                     root / 'index', 1))
    args = (policy(), blocking, Model(), stores[1], texts[1], stores[2], texts[2], rows,
            root / 'index', root / 'overflow', 6)
    exact = list(iter_ranked_candidates(*args, budget=0, batch_size=1, reserve_gib=0))
    rescued = list(iter_ranked_candidates(*args, budget=1, batch_size=1, reserve_gib=0))
    for old, zero, added in zip(base, exact, rescued):
        np.testing.assert_equal(old.owners, zero.owners)
        np.testing.assert_equal(old.records, zero.records)
        np.testing.assert_equal(old.provenance, zero.provenance)
        assert set(old.records) <= set(added.records)
        assert len(added.records) - len(old.records) <= 1
        assert len(added.records) == len(set(added.records))
        assert set(added.provenance) <= set(old.provenance) | {16}


def test_normalized_text_cache_feature_parity_and_reuse(world):
    stores, texts, root = world
    raw = CheapText(texts[2])
    cached = CheapText(texts[2], normalized_root=root / 'normalized', reserve_gib=0)
    rows = np.arange(stores[2].size)
    assert raw.read(rows) == cached.read(rows)
    path = root / 'normalized' / texts[2].path.stem / 'text.bin'
    timestamp = path.stat().st_mtime_ns
    reused = CheapText(texts[2], normalized_root=root / 'normalized', reserve_gib=0)
    assert reused.read(rows) == raw.read(rows)
    assert path.stat().st_mtime_ns == timestamp
    cached.normalized.close()
    reused.normalized.close()


def test_empty_query_batch_has_no_candidates(world):
    stores, texts, root = world
    blocking = replace(BLOCKING, channels=tuple(replace(c, index_keep=2, max_frequency=1) for c in BLOCKING.channels))
    retriever = BroadRetriever(SourceBlocker(blocking, stores[2], root / 'base'), root / 'overflow', 6, reserve_gib=0)
    rows = np.zeros(0, np.int64)
    deep, broad = retriever.retrieve(stores[1], rows)
    matrix, ranks = retrieval_features(stores[1], stores[2], rows, broad, deep, blocking.channels,
                                       CheapText(texts[1]), CheapText(texts[2]))
    assert matrix.shape == (0, len(feature_names(blocking.channels)))
    assert ranks.size == 0


def test_shard_bounds_cover_rows_contiguously():
    from src.utils.parallel import shard_bounds

    for count, size in ((0, 3), (1, 5), (10, 3), (10, 10), (1986411, 20000)):
        bounds = shard_bounds(count, size)
        assert bounds == shard_bounds(count, size)
        assert all(0 < hi - lo <= size for lo, hi in bounds)
        assert [lo for lo, _ in bounds[1:]] == [hi for _, hi in bounds[:-1]]
        assert (bounds[0][0], bounds[-1][1]) == (0, count) if bounds else count == 0
    with pytest.raises(ValueError):
        shard_bounds(5, 0)


def test_tree_hash_tracks_content_and_names(tmp_path):
    from src.data.validation import tree_hash

    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'x.bin').write_bytes(b'one')
    (tmp_path / 'y.json').write_text('{}')
    first = tree_hash(tmp_path)
    assert tree_hash(tmp_path) == first
    (tmp_path / 'a' / 'x.bin').write_bytes(b'two')
    changed = tree_hash(tmp_path)
    assert changed != first
    (tmp_path / 'a' / 'x.bin').rename(tmp_path / 'a' / 'z.bin')
    assert tree_hash(tmp_path) not in (first, changed)


def test_ranked_candidates_do_not_depend_on_shards_or_batches(world):
    from src.matching.candidate_ranker import iter_ranked_candidates
    from tests.test_candidate_recovery import policy

    stores, texts, root = world
    blocking = replace(BLOCKING, channels=tuple(replace(c, index_keep=2, max_frequency=1) for c in BLOCKING.channels))

    class Model:
        @staticmethod
        def feature_name():
            return list(feature_names(blocking.channels))

        @staticmethod
        def predict(matrix, num_threads):
            return np.nan_to_num(matrix[:, 20]) - np.nan_to_num(matrix[:, 10]) / 10

    def run(rows, offset, batch_size):
        batches = iter_ranked_candidates(policy(), blocking, Model(), stores[1], texts[1], stores[2], texts[2], rows,
                                         root / 'index', root / 'overflow', 6, 1, batch_size, 0)
        return {(int(o) + offset, int(r), int(p)) for b in batches
                for o, r, p in zip(b.owners, b.records, b.provenance)}

    rows = stores[1].sorted_order
    whole = run(rows, 0, len(rows))
    assert {p for _, _, p in whole} & {16} and len({p for _, _, p in whole}) > 1
    split = len(rows) // 2
    assert run(rows[:split], 0, 1) | run(rows[split:], split, 2) == whole
