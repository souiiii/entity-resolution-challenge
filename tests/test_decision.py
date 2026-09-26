from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.matching.decision import (claim_validation, decisions, policy_grid, query_best, robustness,
                                   search_families, select_final, validation_gate)
from src.matching.pipeline import evaluate
from src.matching.supervised import fast_macro, membership_digest, threshold_predictions, tune_queries
from test_matching import AGREE, labelled

ROOT = Path(__file__).resolve().parents[1]


def fixture_candidates():
    candidates = labelled([(1, 1, AGREE, False, 1, 1)] * 5,
                          [0, 0, 1, 2, 2], [2, 3, 2, 2, 3], [1, 1, 2, 3, 3],
                          [True, True, False, True, False], 4)
    scores = np.array([.95, .75, .7, .8, .6], np.float32)
    counts = {2: np.array([1, 0, 1, 0]), 3: np.array([1, 0, 0, 0])}
    return candidates, scores, counts


def test_global_exactly_reproduces_phase6_threshold():
    scores = np.array([0, .5, np.nextafter(np.float32(.66), np.float32(0)), .66, 1, np.nan], np.float32)
    owners = np.array([0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(decisions(owners, scores, 4, {'family': 'global', 'threshold': .66}),
                                  threshold_predictions(scores, .66))


def test_query_best_handles_unsorted_owners_and_empty_queries():
    best = query_best(np.array([1, 0, 1, 0, 2]), np.array([.8, .5, .9, .75, np.nan]), 4)
    np.testing.assert_array_equal(best, [.75, .9, -np.inf, -np.inf])
    np.testing.assert_array_equal(query_best(np.array([], int), np.array([]), 2), [-np.inf, -np.inf])
    with pytest.raises(ValueError):
        query_best(np.array([2]), np.array([.5]), 2)


@pytest.mark.parametrize('policy', [
    {'family': 'relative', 'threshold': .6, 'alpha': .8},
    {'family': 'margin', 'threshold': .6, 'delta': .1},
])
def test_query_rules_do_not_mix_queries(policy):
    owners = np.array([0, 0, 1, 1])
    scores = np.array([.95, .7, .7, .69])
    np.testing.assert_array_equal(decisions(owners, scores, 3, policy), [True, False, True, True])
    order = np.array([3, 1, 0, 2])
    np.testing.assert_array_equal(decisions(owners[order], scores[order], 3, policy),
                                  decisions(owners, scores, 3, policy)[order])


def test_singleton_abstention_preserves_multiple_matches_across_sources():
    owners = np.array([0, 0, 1, 1, 2])
    scores = np.array([.6, .4, .6, .7, .8])
    policy = {'family': 'singleton', 'threshold': .5, 'single_threshold': .75}
    keep = decisions(owners, scores, 4, policy)
    assert keep.tolist() == [False, False, True, True, True]
    assert np.bincount(owners[keep], minlength=4).tolist() == [0, 2, 1, 0]


@pytest.mark.parametrize('policy', [
    {'family': 'relative', 'threshold': .5, 'alpha': 1.0},
    {'family': 'margin', 'threshold': .5, 'delta': 0.0},
])
def test_equal_best_scores_keep_all_ties_and_allow_empty_predictions(policy):
    owners = np.array([0, 0, 1, 1])
    keep = decisions(owners, np.array([.75, .75, .1, .2]), 3, policy)
    assert keep.tolist() == [True, True, False, False]
    assert set(np.flatnonzero(keep)) <= set(range(len(owners)))


@pytest.mark.parametrize('policy', [
    {'family': 'global', 'threshold': .66},
    {'family': 'relative', 'threshold': .6, 'alpha': .9},
    {'family': 'margin', 'threshold': .6, 'delta': .1},
    {'family': 'singleton', 'threshold': .6, 'single_threshold': .85},
])
def test_policy_metric_parity_and_empty_truth(policy):
    candidates, scores, counts = fixture_candidates()
    truth = sum(counts.values())
    keep = decisions(candidates.owners, scores, 4, policy)
    for selected in (np.ones(4, bool), np.array([False, True, False, True])):
        result = evaluate(candidates, keep, truth, selected)
        assert fast_macro(candidates.owners, candidates.labels, keep.astype(np.float32), truth, selected, .5) == result['macro_f05']
    empty = evaluate(candidates, np.zeros(len(scores), bool), truth)
    assert empty['macro_f05'] == .5


def test_search_confirm_hash_split_is_deterministic_and_query_disjoint():
    ids = np.array([f'S1-{i}'.encode() for i in range(1000)])
    confirm = tune_queries(ids, 'decision-v1-confirm-split', .4)
    assert 350 < confirm.sum() < 450
    np.testing.assert_array_equal(confirm[::-1], tune_queries(ids[::-1], 'decision-v1-confirm-split', .4))
    assert not set(ids[confirm]) & set(ids[~confirm])
    assert membership_digest(ids[confirm]) == membership_digest(ids[confirm][::-1])
    owners = np.repeat(np.arange(len(ids)), 3)
    assert (confirm[owners].reshape(-1, 3) == confirm[:, None]).all()


def test_fixed_grid_is_bounded_and_has_only_requested_families():
    config = json.loads((ROOT / 'experiments/configs/decision_v1.json').read_text())
    grid = policy_grid(config['grid'])
    assert {family: len(values) for family, values in grid.items()} == {
        'global': 26, 'relative': 36, 'margin': 30, 'singleton': 29}
    assert sum(map(len, grid.values())) == 121
    assert all(p['single_threshold'] > p['threshold'] for p in grid['singleton'])
    with pytest.raises(ValueError):
        policy_grid({**config['grid'], 'global_thresholds': [.5] * 200})


def test_search_winners_are_independent_of_confirm_labels_and_open_countries():
    candidates, scores, counts = fixture_candidates()
    search = np.array([True, False, True, False])
    grid = {'global': [{'family': 'global', 'threshold': t} for t in (.6, .8)],
            'relative': [{'family': 'relative', 'threshold': .6, 'alpha': .9}]}
    _, first = search_families(candidates, scores, counts, np.array(['India', 'US', 'US', 'India']), search, grid)
    labels = candidates.labels.copy()
    labels[candidates.owners == 1] = True
    changed = replace(candidates, labels=labels)
    new_counts = {2: np.array([1, 1, 1, 0]), 3: counts[3]}
    _, second = search_families(changed, scores, new_counts, np.array(['France', 'Atlantis', 'Atlantis', 'France']), search, grid)
    assert [entry['policy'] for entry in first] == [entry['policy'] for entry in second]
    assert [entry['search']['overall'] for entry in first] == [entry['search']['overall'] for entry in second]
    assert set(second[0]['search']['by_country']) == {'France', 'Atlantis'}


def test_robustness_rejects_gain_that_disappears_on_confirm():
    base = {'macro_f05': .9, 'micro_precision_diagnostic': .97, 'correct_singletons': 90, 'singleton_count': 100}
    limits = {'max_precision_drop': .005, 'max_singleton_accuracy_drop': .02}
    baselines = {name: {'overall': base} for name in ('search', 'confirm')}
    improved = {**base, 'macro_f05': .91}
    assert robustness(improved, improved, baselines, limits)['passed']
    assert not robustness(improved, base, baselines, limits)['passed']
    assert not robustness(improved, {**improved, 'correct_singletons': 80}, baselines, limits)['passed']
    assert not robustness(improved, {**improved, 'micro_precision_diagnostic': .95}, baselines, limits)['passed']


def test_selection_uses_confirm_fp_simplicity_and_conservative_ties():
    baseline = {'family': 'global', 'threshold': .66}
    metrics = {'macro_f05': .91, 'fp': 10, 'tp': 100}
    def entry(policy, result=metrics, passed=True):
        return {'policy': policy, 'confirm': {'overall': result}, 'robustness': {'passed': passed}}
    conservative = {'family': 'global', 'threshold': .75}
    finalists = [entry({'family': 'relative', 'threshold': .75, 'alpha': .95}),
                 entry(baseline), entry(conservative)]
    assert select_final(finalists, baseline) == conservative
    assert select_final(finalists[::-1], baseline) == conservative
    fewer_fp = entry({'family': 'margin', 'threshold': .6, 'delta': .1}, {**metrics, 'fp': 9})
    assert select_final([*finalists, fewer_fp], baseline) == fewer_fp['policy']
    assert select_final([entry(conservative, passed=False)], baseline) == baseline


def test_validation_gate_requires_material_gain_and_positive_confirm():
    base = {'macro_f05': .92, 'micro_precision_diagnostic': .97, 'correct_singletons': 90, 'singleton_count': 100}
    limits = {'min_macro_gain': .002, 'max_precision_drop': .005, 'max_singleton_accuracy_drop': .02}
    assert validation_gate(base, {**base, 'macro_f05': .923}, .001, limits)['passed']
    assert not validation_gate(base, {**base, 'macro_f05': .921}, .001, limits)['passed']
    assert not validation_gate(base, {**base, 'macro_f05': .923}, 0, limits)['passed']


def test_phase7_validation_refuses_failed_gate_and_second_attempt(tmp_path):
    path = tmp_path / 'validation.json'
    report = {'gate': {'passed': False}, 'frozen_policy_sha256': 'fixture'}
    with pytest.raises(ValueError, match='gate'):
        claim_validation(path, report)
    assert not path.with_suffix('.started.json').exists()
    report['gate']['passed'] = True
    claim_validation(path, report)
    with pytest.raises(FileExistsError):
        claim_validation(path, report)
    other = tmp_path / 'completed.json'
    other.write_text('{}')
    with pytest.raises(FileExistsError):
        claim_validation(other, report)


def test_policy_rejects_country_or_combined_family_fields():
    for extra in ({'country': 'US'}, {'alpha': .9}, {'source': 2}):
        with pytest.raises(ValueError, match='fields'):
            decisions(np.array([0]), np.array([.9]), 1, {'family': 'global', 'threshold': .6, **extra})
