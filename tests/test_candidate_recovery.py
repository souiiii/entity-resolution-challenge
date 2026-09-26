from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.blocking.candidates import SourceBlocker
from src.blocking.channels import BlockingConfig, ChannelSpec, part_csr
from src.blocking.ranking import oracle_ceiling
from src.blocking.recovery import (AGREE, CONFLICT, NO_KEY, UNAVAILABLE, HashedEvidence, cheap_score,
                                   classify_unretrieved, deepen, expanded_union, hashed_evidence, rank_per_query,
                                   set_metrics, token_overlap, top_per_query)
from src.blocking.store import build_store
from src.data.load import SOURCE_COLUMNS
from src.features.text import ensure_text
from src.matching.final_candidates import (DEEP_RESCUE, PHASE5, FinalPolicy, iter_final_candidates)
from src.matching.pipeline import collect_candidates, screen
from src.matching.rules import PrefilterConfig

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = (ChannelSpec('address_exact', 'exact', ('address_canonical',), 50, index_keep=50),
            ChannelSpec('name_folded', 'exact', ('name_folded',), 50, index_keep=50),
            ChannelSpec('name_token_pair', 'combo', ('name', 'name'), 2, max_keys=4, symmetric=True, index_keep=10),
            ChannelSpec('address_number_x_word', 'combo', ('address_number', 'address_word'), 2, index_keep=10))
BLOCKING = BlockingConfig('test', CHANNELS, workers=1, min_free_disk_gib=0,
                          raw={'ranking': {'selected_strategy': 'aggregate', 'selected_k': 3}})
PREFILTER = PrefilterConfig(0.6, 0.75, True, 5)


def policy(deep: int = 1, rank: int = 1) -> FinalPolicy:
    return FinalPolicy('test', 3, PREFILTER, 0.8, 0.6, 1.5, rank, deep, {})


def write_source(path: Path, rows) -> Path:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerows([SOURCE_COLUMNS, *rows])
    return path


QUERIES = [('S1-a', 'Happy Hair Studio', '12 Main Street', 'France'),
           ('S1-b', 'राम ट्रेडर्स', '5 Gandhi Road', 'Atlantis')]
TARGETS2 = ([('S2-star', 'HAPPY HAIR STUDIOS INC', '12 Main Road', 'France')]
            + [(f'S2-h{i}', f'Happy Hair Salon {i}', f'{40 + i} Oak Lane', 'France') for i in range(5)]
            + [(f'S2-m{i}', f'Other Shop {i}', f'12 Main Avenue {i}', 'France') for i in range(3)]
            + [('S2-ram', 'राम ट्रेडर्स', '5 Gandhi Road', 'Atlantis'), ('S2-us', 'Happy Hair Studio', '12 Main Street', 'US')])
TARGETS3 = [('S3-1', 'Happy Hair Studio', '12, Main Street', 'France'), ('S3-2', 'Unrelated', '9 Elm', 'France')]


@pytest.fixture
def world(tmp_path):
    stores, texts = {}, {}
    for source, rows in ((1, QUERIES), (2, TARGETS2), (3, TARGETS3)):
        path = write_source(tmp_path / f'source{source}.tsv', rows)
        stores[source] = build_store(path, source, tmp_path / f'store{source}')
        texts[source] = ensure_text(path, stores[source], tmp_path / f'text{source}')
    yield stores, texts, tmp_path
    for text in texts.values():
        text.close()


def run_final(world, final_policy, source=2, deep_limit=None, rows=None):
    stores, texts, tmp = world
    rows = np.asarray(stores[1].sorted_order if rows is None else rows, np.int64)
    batches = list(iter_final_candidates(final_policy, BLOCKING, stores[1], texts[1], stores[source], texts[source],
                                         rows, tmp / 'index', 1, 1, deep_limit))
    owners = np.concatenate([b.owners for b in batches])
    records = np.concatenate([b.records for b in batches])
    provenance = np.concatenate([b.provenance for b in batches])
    deep_rank = np.concatenate([b.deep_rank for b in batches])
    ids = np.asarray(stores[source].ids)
    query_ids = np.asarray(stores[1].ids)[rows]
    return [(query_ids[o].decode(), ids[r].decode(), int(p), int(d))
            for o, r, p, d in zip(owners, records, provenance, deep_rank)]


def test_deep_rescue_recovers_target_reachable_only_through_capped_buckets(world):
    rescued = run_final(world, policy(deep=1))
    by_query = {}
    for query, target, flag, _ in rescued:
        by_query.setdefault(query, {})[target] = flag
    assert by_query['S1-a']['S2-star'] == DEEP_RESCUE
    without = run_final(world, policy(deep=0))
    assert 'S2-star' not in {t for q, t, _, _ in without if q == 'S1-a'}


def test_final_candidates_are_unique_sourced_and_country_isolated(world):
    for source in (2, 3):
        final = run_final(world, policy(deep=3), source)
        pairs = [(q, t) for q, t, _, _ in final]
        assert len(pairs) == len(set(pairs))
        assert all(t.startswith(f'S{source}-') for _, t in pairs)
        assert all(flag != 0 for _, _, flag, _ in final)
    final2 = {(q, t) for q, t, _, _ in run_final(world, policy(deep=3))}
    assert ('S1-b', 'S2-ram') in final2
    assert not any(t == 'S2-us' for _, t in final2)
    assert not any(q == 'S1-b' and t.startswith('S2-h') for q, t in final2)


def test_budget_sweep_is_a_superset_and_default_matches_policy(world):
    exact = run_final(world, policy(deep=1))
    swept = run_final(world, policy(deep=1), deep_limit=5)
    assert {(q, t) for q, t, _, _ in exact} <= {(q, t) for q, t, _, _ in swept}
    trimmed = {(q, t) for q, t, flag, rank in swept if flag != DEEP_RESCUE or rank <= 1}
    assert trimmed == {(q, t) for q, t, _, _ in exact}
    per_query = {}
    for q, _, flag, _ in exact:
        per_query[q] = per_query.get(q, 0) + (flag == DEEP_RESCUE)
    assert max(per_query.values()) <= 1


def test_query_isolation_and_determinism(world):
    stores, _, _ = world
    rows = np.asarray(stores[1].sorted_order, np.int64)
    together = run_final(world, policy(deep=2), rows=rows)
    alone = [c for i in range(len(rows)) for c in run_final(world, policy(deep=2), rows=rows[i:i + 1])]
    assert sorted(together) == sorted(alone)
    assert run_final(world, policy(deep=2)) == together


def test_phase5_provenance_regresses_to_existing_screen(world):
    stores, texts, tmp = world
    rows = np.asarray(stores[1].sorted_order, np.int64)
    for source in (2, 3):
        final = {(q, t) for q, t, flag, _ in run_final(world, policy(deep=2), source) if flag & PHASE5}
        candidates = collect_candidates(BLOCKING, PREFILTER, stores[1], texts[1], {source: (stores[source], texts[source])},
                                        rows, tmp / 'index', 1, 1)
        screened = screen(candidates, PREFILTER)
        query_ids, target_ids = np.asarray(stores[1].ids)[rows], np.asarray(stores[source].ids)
        expected = {(query_ids[o].decode(), target_ids[r].decode()) for o, r in zip(screened.owners, screened.records)}
        assert final == expected


def test_expanded_union_contains_base_union_and_counts_hits(world):
    stores, _, tmp = world
    blocker = SourceBlocker(BLOCKING, stores[2], tmp / 'index')
    rows = np.asarray(stores[1].sorted_order, np.int64)
    base = blocker.candidates(stores[1], rows)
    deep = expanded_union(blocker, tuple(deepen(c) for c in CHANNELS), stores[1], rows)
    base_pairs = set(zip(base.owners.tolist(), base.records.tolist()))
    deep_pairs = set(zip(deep.owners.tolist(), deep.records.tolist()))
    assert base_pairs < deep_pairs and (deep.hits >= 1).all()
    assert deepen(CHANNELS[2]).max_frequency == 10 and deepen(CHANNELS[2]).max_keys is None
    assert deepen(CHANNELS[2], depth=4).max_frequency == 4
    with pytest.raises(ValueError):
        deepen(ChannelSpec('x', 'exact', ('name_folded',), 5))


def test_hashed_overlap_matches_brute_force_and_handles_unicode(world):
    stores, _, _ = world
    rng = np.random.default_rng(3)
    query_rows = rng.integers(0, stores[1].size, 40)
    target_rows = rng.integers(0, stores[2].size, 40)
    for part in ('name', 'address_word', 'address_number'):
        shared, left, right = token_overlap(stores[1], stores[2], part, query_rows, target_rows)
        for i in range(40):
            _, q = part_csr(stores[1], part, query_rows[i:i + 1])
            _, t = part_csr(stores[2], part, target_rows[i:i + 1])
            assert (shared[i], left[i], right[i]) == (len(set(q.tolist()) & set(t.tolist())), len(set(q.tolist())),
                                                      len(set(t.tolist())))
    query = np.flatnonzero(np.asarray(stores[1].ids) == b'S1-b')
    target = np.flatnonzero(np.asarray(stores[2].ids) == b'S2-ram')
    evidence = hashed_evidence(stores[1], stores[2], query, target)
    assert evidence.name_containment[0] == 1 and evidence.exact_name[0] and evidence.number_state[0] == AGREE


def test_cheap_score_components():
    evidence = HashedEvidence(None, None, np.array([1.0, np.nan, 0.5], np.float32), None, None,
                              np.array([1.0, 0.5, np.nan], np.float32), np.array([AGREE, CONFLICT, UNAVAILABLE], np.int8),
                              np.array([True, False, False]), np.array([False, False, True]))
    assert cheap_score(evidence, np.array([3.0, 0.2, 0.0])).tolist() == pytest.approx([4.5, 0.2, 1.5])


def test_rescue_ordering_and_budgets_are_stable():
    owners = np.array([0, 0, 0, 1, 1])
    score = np.array([1.0, 2.0, 2.0, 0.5, 0.5], np.float32)
    tiebreak = np.array([5, 3, 3, 1, 1])
    target = np.array([10, 20, 11, 7, 3])
    assert rank_per_query(owners, score, tiebreak, target).tolist() == [3, 2, 1, 2, 1]
    order = np.random.default_rng(0).permutation(5)
    assert rank_per_query(owners[order], score[order], tiebreak[order], target[order]).tolist() == \
        rank_per_query(owners, score, tiebreak, target)[order].tolist()
    assert top_per_query(owners, score, tiebreak, target, 1).tolist() == [False, False, True, False, True]
    assert top_per_query(owners, score, tiebreak, target, 0).sum() == 0


def test_loss_classification():
    channels = [ChannelSpec('a', 'token', ('name',), 20, index_keep=200), ChannelSpec('b', 'token', ('address_word',), 10,
                                                                                      index_keep=50)]
    profile = np.array([[NO_KEY, NO_KEY], [15, NO_KEY], [100, NO_KEY], [500, 60], [NO_KEY, 5]])
    assert classify_unretrieved(profile, channels).tolist() == ['A_no_shared_key', 'C_key_limit', 'C_frequency_cap',
                                                                 'B_index_truncated', 'C_key_limit']


def test_set_metrics_oracle_and_entities():
    countries = np.array([0, 0, 1, 1])
    truth = {2: np.array([2, 0, 1, 0]), 3: np.array([1, 0, 0, 2])}
    found = {2: np.array([2, 0, 0, 0]), 3: np.array([0, 0, 0, 2])}
    counts = np.array([5, 0, 3, 4])
    result = set_metrics(counts, found, truth, countries, ['India', 'US'])
    assert result['recall'] == 4 / 6 and result['found_links'] == 4
    assert result['oracle_macro_f05'] == oracle_ceiling(np.array([3, 0, 1, 2]), np.array([2, 0, 0, 2]))
    assert (result['entities_all_recovered'], result['entities_some_recovered'], result['entities_none_recovered']) == (1, 2, 1)
    assert result['zero_candidate_queries'] == 1 and result['max'] == 5
    assert result['source_recall'] == {'S2': 2 / 3, 'S3': 2 / 3}
    assert result['country_recall'] == {'India': 2 / 3, 'US': 2 / 3}
    more = set_metrics(counts + 1, {2: np.array([2, 0, 1, 0]), 3: found[3]}, truth, countries, ['India', 'US'], result)
    assert more['incremental_true_links'] == 1 and more['incremental_pairs'] == 4
    assert more['true_links_per_1000_added_pairs'] == 250


def load_script():
    spec = importlib.util.spec_from_file_location('run_candidate_recovery', ROOT / 'scripts/run_candidate_recovery.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['run_candidate_recovery'] = module
    spec.loader.exec_module(module)
    return module


def test_resource_guard_refuses_low_disk(monkeypatch):
    script = load_script()
    guard = script.Guard(0, 0)
    guard.check()
    usage = type('Usage', (), {'free': 1 << 20})()
    monkeypatch.setattr(script.shutil, 'disk_usage', lambda _: usage)
    with pytest.raises(OSError, match='reserve'):
        script.Guard(1.0, 0).check()
