from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from src.blocking.channels import load_config
from src.blocking.store import build_store
from src.features.pairwise import feature_names
from src.features.text import ensure_text
from src.matching.pipeline import collect_candidates, evaluate, match
from src.matching.rules import evidence_from_features, prefilter_mask
from src.matching.supervised import (claim_validation, development_gate, diagnostics, fast_macro, load_model,
                                     membership_digest, predict_scores, save_model, select_threshold,
                                     threshold_predictions, tune_queries, validate_schema)
from test_matching import AGREE, CONFIG, PREFILTER, labelled, write_source

ROOT = Path(__file__).resolve().parents[1]
NAMES = feature_names(load_config(ROOT / 'experiments/configs/blocking_v2.json').channels)


@pytest.fixture(scope='module')
def tiny_model():
    rng = np.random.default_rng(42)
    matrix = rng.random((100, len(NAMES))).astype(np.float32)
    labels = matrix[:, 29] > 0.5
    matrix[::5, 21] = np.nan
    model = lgb.LGBMClassifier(n_estimators=12, num_leaves=7, min_child_samples=5, n_jobs=1,
                               random_state=42, deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(matrix, labels, feature_name=list(NAMES))
    np.testing.assert_array_equal(predict_scores(model.booster_, matrix, NAMES),
                                  model.predict_proba(matrix)[:, 1].astype(np.float32))
    return model.booster_, matrix


def test_query_split_is_deterministic_disjoint_and_order_independent():
    ids = np.array([f'S1-{i}'.encode() for i in range(1000)])
    mask = tune_queries(ids, 'fixed', .2)
    assert np.array_equal(mask, tune_queries(ids, 'fixed', .2))
    assert np.array_equal(mask[::-1], tune_queries(ids[::-1], 'fixed', .2))
    assert 150 < mask.sum() < 250
    assert not set(ids[mask]) & set(ids[~mask])
    owners = np.repeat(np.arange(len(ids)), 3)
    assert np.array_equal(mask[owners].reshape(-1, 3), np.repeat(mask[:, None], 3, axis=1))
    assert membership_digest(ids[mask]) == membership_digest(ids[mask][::-1])


def test_feature_schema_and_nan_inference(tiny_model):
    model, matrix = tiny_model
    assert np.isfinite(predict_scores(model, matrix, NAMES)).all()
    with pytest.raises(ValueError, match='schema'):
        predict_scores(model, matrix, NAMES[::-1])
    with pytest.raises(ValueError, match='shape'):
        predict_scores(model, matrix[:, :-1], NAMES)
    with pytest.raises(ValueError):
        validate_schema(('id',), NAMES)
    assert predict_scores(model, matrix[:0], NAMES).shape == (0,)


def test_serialization_reproduces_probabilities_and_rejects_tampering(tmp_path, tiny_model):
    model, matrix = tiny_model
    saved = save_model(model, tmp_path, {'threshold': .7, 'gate': {'passed': True}})
    loaded, metadata = load_model(tmp_path, NAMES)
    assert saved == metadata
    np.testing.assert_array_equal(predict_scores(model, matrix, NAMES), predict_scores(loaded, matrix, NAMES))
    (tmp_path / 'model.txt').write_text((tmp_path / 'model.txt').read_text() + '\n')
    with pytest.raises(ValueError, match='checksum'):
        load_model(tmp_path, NAMES)


def test_threshold_zero_multiple_and_exact_empty_truth_semantics():
    candidates = labelled([(1, 1, AGREE, False, 1, 1)] * 4,
                          [0, 0, 1, 2], [2, 3, 2, 3], [1, 1, 2, 3], [True, True, False, True], 4)
    truth = np.array([2, 0, 1, 0])
    scores = np.array([.6, .6, .55, .1], np.float32)
    selected = np.ones(4, bool)
    for threshold in (0, .1, .55, .6, .9, 1.0000001):
        accepted = threshold_predictions(scores, threshold)
        assert fast_macro(candidates.owners, candidates.labels, scores, truth, selected, threshold) == \
            evaluate(candidates, accepted, truth)['macro_f05']
    accepted = threshold_predictions(scores, .6)
    assert np.bincount(candidates.owners[accepted], minlength=4).tolist() == [2, 0, 0, 0]
    assert set(candidates.records[accepted]) <= set(candidates.records)
    assert threshold_predictions(np.array([np.nan, 1.0, .5]), 1.0000001).tolist() == [False, False, False]
    sweep = select_threshold(candidates.owners, candidates.labels, scores, truth, selected, .05, .005)
    assert sweep['macro_f05'] == evaluate(candidates, threshold_predictions(scores, sweep['threshold']), truth)['macro_f05']


def test_threshold_ties_prefer_abstention_for_empty_truth():
    sweep = select_threshold(np.array([0]), np.array([False]), np.array([.8]), np.array([0]),
                             np.array([True]), .05, .005)
    assert sweep['macro_f05'] == 1.0 and sweep['threshold'] == 1.0000001


def test_validation_guard_blocks_gate_failure_and_second_attempt(tmp_path):
    path = tmp_path / 'validation.json'
    metadata = {'gate': {'passed': False}, 'threshold': .7, 'model_sha256': 'fixture'}
    with pytest.raises(ValueError, match='gate'):
        claim_validation(path, metadata)
    metadata['gate']['passed'] = True
    claim_validation(path, metadata)
    with pytest.raises(FileExistsError):
        claim_validation(path, metadata)
    other = tmp_path / 'existing.json'
    other.write_text('{}')
    with pytest.raises(FileExistsError):
        claim_validation(other, metadata)


def test_development_gate_requires_gain_precision_and_singletons():
    base = {'overall': {'macro_f05': .8, 'micro_precision_diagnostic': .94,
                        'correct_singletons': 90, 'singleton_count': 100}}
    config = {'min_macro_gain': .01, 'max_precision_drop': .02, 'max_singleton_accuracy_drop': .05}
    improved = {'overall': {**base['overall'], 'macro_f05': .9}}
    assert development_gate(base, improved, config)['passed']
    assert not development_gate(base, base, config)['passed']
    assert not development_gate(base, {'overall': {**improved['overall'], 'correct_singletons': 10}}, config)['passed']
    assert not development_gate(base, {'overall': {**improved['overall'], 'micro_precision_diagnostic': .7}}, config)['passed']


def test_callback_exactly_reuses_phase5_screen_and_open_countries(tmp_path, tiny_model):
    rows = {1: [('S1-a', 'Cafe Dupont', '12 Rue Paris', 'France'),
                ('S1-b', 'Acme', '55 Main Street', 'Atlantis')],
            2: [('S2-a', 'Cafe Dupont', '12 Rue Paris', 'France'),
                ('S2-b', 'Acme', '55 Main Street', 'Atlantis')],
            3: [('S3-a', 'Cafe Dupont', '12 Rue Paris', 'France'),
                ('S3-wrong', 'Acme', '55 Main Street', 'US')]}
    stores, texts = {}, {}
    for source, records in rows.items():
        path = write_source(tmp_path / f's{source}.tsv', records)
        stores[source] = build_store(path, source, tmp_path / f'store{source}')
        texts[source] = ensure_text(path, stores[source], tmp_path / f'text{source}')
    blocking = replace(load_config(ROOT / 'experiments/configs/blocking_v2.json'), min_free_disk_gib=0)
    truth = {2: (np.array([0, 1]), np.array([0, 1])), 3: (np.array([0]), np.array([0]))}
    args = (blocking, PREFILTER, stores[1], texts[1], {s: (stores[s], texts[s]) for s in (2, 3)},
            np.arange(2), tmp_path / 'index', 1, 1, truth)
    baseline = collect_candidates(*args)
    parts = []
    supervised = collect_candidates(*args, feature_consumer=parts.append)
    matrix = np.concatenate(parts)
    assert len(matrix) == len(supervised.records) == 3
    assert prefilter_mask(evidence_from_features(matrix, NAMES), PREFILTER).all()
    for attribute in ('owners', 'sources', 'records', 'labels'):
        np.testing.assert_array_equal(getattr(baseline, attribute), getattr(supervised, attribute))
    np.testing.assert_array_equal(match(baseline, CONFIG), match(supervised, CONFIG))
    accepted = threshold_predictions(predict_scores(tiny_model[0], matrix, NAMES), .5)
    result = diagnostics(supervised, accepted, {2: np.array([1, 1]), 3: np.array([1, 0])},
                         np.array(['France', 'Atlantis']), np.ones(2, bool))
    assert set(result['by_country']) == {'France', 'Atlantis'}
    assert result['overall']['source3']['truth_links'] == 1
    for text in texts.values():
        text.close()
