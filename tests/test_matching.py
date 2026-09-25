from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.blocking.channels import BlockingConfig, ChannelSpec
from src.blocking.store import build_store
from src.data.load import SOURCE_COLUMNS
from src.features.text import ensure_text
from src.matching.pipeline import Candidates, collect_candidates, evaluate, match, screen, write_submission
from src.matching.rules import (AGREE, CONFLICT, PARTIAL, UNAVAILABLE, Evidence, MatcherConfig, PrefilterConfig,
                                RuleConfig, load_matcher_config, prefilter_mask, relative_gate, rule_acceptable,
                                rule_paths)

ROOT = Path(__file__).resolve().parents[1]
RULES = RuleConfig(0.70, 0.60, 0.85, 0.85, 0.65, 0.10)
PREFILTER = PrefilterConfig(0.6, 0.75, True, 5)
CONFIG = MatcherConfig('test', 'unused', PREFILTER, RULES, (), {})
NAN = float('nan')


def evidence(rows: list[tuple]) -> Evidence:
    """Rows of (name, address, number_state, script_mismatch, rarity, rank)."""
    name, address, state, mismatch, rarity, rank = zip(*rows)
    return Evidence(np.array(name, np.float32), np.array(address, np.float32), np.array(state, np.int8),
                    np.array(mismatch, bool), np.array(rarity, np.float32), np.array(rank, np.int16))


def accept(rows: list[tuple]) -> list[bool]:
    return rule_acceptable(evidence(rows), RULES).tolist()


def test_rule_paths_and_exact_agreement():
    assert accept([(1.0, 1.0, AGREE, False, 5, 1)]) == [True]
    assert accept([(0.72, 0.61, AGREE, False, 1, 1)]) == [True]
    assert accept([(0.69, 0.99, AGREE, False, 1, 1), (0.99, 0.59, AGREE, False, 1, 1)]) == [False, False]
    paths = rule_paths(evidence([(1.0, 1.0, AGREE, False, 5, 1)]), RULES)
    assert paths['name_address'].tolist() == [True] and paths['script_mismatch'].tolist() == [False]


def test_numbers_gate_decisions():
    rows = [(1.0, 1.0, state, False, 5, 1) for state in (AGREE, PARTIAL, CONFLICT, UNAVAILABLE)]
    assert accept(rows) == [True, False, False, True]
    assert accept([(0.8, 0.9, UNAVAILABLE, False, 5, 1), (0.9, 0.6, UNAVAILABLE, False, 5, 1)]) == [False, False]


def test_nan_is_unavailable_not_zero_or_pass():
    assert accept([(NAN, 1.0, AGREE, False, 5, 1), (1.0, NAN, AGREE, False, 5, 1), (NAN, NAN, UNAVAILABLE, False, 5, 1)]) \
        == [False, False, False]
    kept = prefilter_mask(evidence([(NAN, NAN, AGREE, False, 1, 9), (NAN, 0.6, AGREE, False, 1, 9)]), PREFILTER)
    assert kept.tolist() == [False, True]


def test_script_mismatch_lets_address_and_numbers_decide():
    assert accept([(NAN, 0.9, AGREE, True, 5, 1), (NAN, 0.8, AGREE, True, 5, 1), (NAN, 0.95, CONFLICT, True, 5, 1)]) \
        == [True, False, False]


def test_relative_gate_zero_one_multiple_and_singletons():
    rows = [(1.0, 1.0, AGREE, False, 8.0, 1), (1.0, 0.9, AGREE, False, 4.0, 2), (1.0, 0.9, AGREE, False, 0.5, 3),
            (0.95, 0.95, AGREE, False, 2.0, 1),
            (0.3, 0.2, AGREE, False, 9.0, 1), (0.9, 0.9, CONFLICT, False, 9.0, 2)]
    owners = np.array([0, 0, 0, 1, 2, 2])
    ev = evidence(rows)
    decided = relative_gate(owners, rule_acceptable(ev, RULES), ev.rarity, RULES.relative_rarity, 3)
    assert decided.tolist() == [True, True, False, True, False, False]
    assert np.bincount(owners[decided], minlength=3).tolist() == [2, 1, 0]


def test_ties_are_deterministic_and_all_kept():
    ev = evidence([(1.0, 1.0, AGREE, False, 3.0, 1), (1.0, 1.0, AGREE, False, 3.0, 2)])
    first = relative_gate(np.array([0, 0]), rule_acceptable(ev, RULES), ev.rarity, 0.1, 1)
    second = relative_gate(np.array([0, 0]), rule_acceptable(ev, RULES), ev.rarity, 0.1, 1)
    assert first.tolist() == second.tolist() == [True, True]


def test_prefilter_keeps_blocking_top_rank_and_rejects_conflicts():
    ev = evidence([(0.9, 0.9, CONFLICT, False, 1, 9), (0.9, 0.9, CONFLICT, False, 1, 5), (0.1, 0.1, PARTIAL, False, 1, 7),
                   (0.8, NAN, UNAVAILABLE, False, 1, 30)])
    assert prefilter_mask(ev, PREFILTER).tolist() == [False, True, False, True]
    everything = PrefilterConfig(None, None, False, 50)
    assert prefilter_mask(ev, everything).all()


def test_prefilter_never_removes_rule_acceptable_candidates():
    rng = np.random.default_rng(1)
    n = 5000
    ev = Evidence(rng.random(n).astype(np.float32), rng.random(n).astype(np.float32),
                  rng.integers(0, 4, n).astype(np.int8), rng.random(n) < 0.2, rng.random(n).astype(np.float32),
                  rng.integers(1, 51, n).astype(np.int16))
    assert not (rule_acceptable(ev, RULES) & ~prefilter_mask(ev, PREFILTER)).any()


def test_config_rejects_prefilter_stricter_than_rules(tmp_path):
    raw = json.loads((ROOT / 'experiments/configs/matcher_v1.json').read_text())
    assert load_matcher_config(ROOT / 'experiments/configs/matcher_v1.json').prefilter.keep_blocking_rank == 5
    raw['prefilter']['min_address_similarity'] = 0.7
    (tmp_path / 'bad.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='Prefilter'):
        load_matcher_config(tmp_path / 'bad.json')


def labelled(rows, owners, sources, records, labels, queries):
    ev = evidence(rows)
    return Candidates(np.array(owners, np.int32), np.array(sources, np.int8), np.array(records, np.int32), ev,
                      np.array(labels, bool), queries, np.bincount(owners, minlength=queries),
                      np.bincount(np.array(owners)[np.array(labels, bool)], minlength=queries))


def test_screen_match_evaluate_accounting():
    rows = [(1.0, 1.0, AGREE, False, 5.0, 1), (0.2, 0.1, CONFLICT, False, 1.0, 8), (1.0, 0.95, AGREE, False, 4.0, 2),
            (0.2, 0.2, PARTIAL, False, 0.1, 20), (0.99, 0.99, AGREE, False, 6.0, 1)]
    candidates = labelled(rows, [0, 0, 0, 1, 2], [2, 2, 3, 2, 3], [10, 11, 12, 13, 14],
                          [True, False, True, False, False], 3)
    screened = screen(candidates, PREFILTER)
    assert screened.records.tolist() == [10, 12, 14]
    decided = match(screened, CONFIG)
    assert decided.tolist() == [True, True, True]
    truth = np.array([3, 0, 0])
    result = evaluate(screened, decided, truth)
    assert (result['tp'], result['fp'], result['fn']) == (2, 1, 1)
    assert result['candidate_recall'] == pytest.approx(2 / 3) and result['unscreened_recall'] == pytest.approx(2 / 3)
    assert result['correct_singletons'] == 1 and result['singleton_false_merges'] == 1
    assert result['source2']['tp'] == 1 and result['source3']['fp'] == 1
    assert result['macro_f05'] == pytest.approx((10 / 11 + 1 + 0) / 3)


def write_source(path: Path, rows) -> Path:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerows([SOURCE_COLUMNS, *rows])
    return path


def load_validator():
    spec = importlib.util.spec_from_file_location('validator', ROOT / 'official/utils/validate_submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_pipeline_tiny_fixture_open_countries_and_official_format(tmp_path):
    rows = {1: [('S1-a', 'Boulangerie Dupont', '12 Rue de la Paix, Paris', 'France'),
                ('S1-b', 'Acme Trading', '10 Market Street, Springfield', 'Atlantis'),
                ('S1-c', 'Lonely Shop', '99 Nowhere Lane', 'France')],
            2: [('S2-1', 'BOULANGERIE DUPONT', '12 RUE DE LA PAIX, PARIS', 'France'),
                ('S2-2', 'Boulangerie Dupont', '14 Rue de la Paix, Paris', 'France'),
                ('S2-3', 'Acme Trading', '10 Market Street', 'Atlantis'),
                ('S2-4', 'Lonely Shop Annex', '5 Other Road', 'US')],
            3: [('S3-1', 'Boulangerie Dupônt', 'Paris, 12 Rue de la Paix', 'France'),
                ('S3-2', 'Acme Trading', '10 Market Street, Springfield', 'France')]}
    stores, texts = {}, {}
    for source, source_rows in rows.items():
        path = write_source(tmp_path / f'source{source}.tsv', source_rows)
        stores[source] = build_store(path, source, tmp_path / f'store{source}')
        texts[source] = ensure_text(path, stores[source], tmp_path / f'text{source}')
    channels = (ChannelSpec('address_exact', 'exact', ('address_canonical',), 50),
                ChannelSpec('name_folded', 'exact', ('name_folded',), 50),
                ChannelSpec('name_token_pair', 'combo', ('name', 'name'), 50, symmetric=True),
                ChannelSpec('address_number_x_word', 'combo', ('address_number', 'address_word'), 50))
    blocking = BlockingConfig('test', channels, workers=1, min_free_disk_gib=0,
                              raw={'ranking': {'selected_strategy': 'aggregate', 'selected_k': 5}})
    query_rows = np.asarray(stores[1].sorted_order, np.int64)
    candidates = collect_candidates(blocking, PREFILTER, stores[1], texts[1], {s: (stores[s], texts[s]) for s in (2, 3)},
                                    query_rows, tmp_path / 'index', 2, 1)
    screened = screen(candidates, PREFILTER)
    decided = match(screened, CONFIG)
    ids = [i.decode() for i in np.asarray(stores[1].ids)[query_rows].tolist()]
    paths = write_submission(ids, screened, decided, {2: stores[2], 3: stores[3]}, tmp_path / 'output')
    matching = dict(line.rstrip('\n').split('\t') for line in paths['matching_results'].read_text().splitlines()[1:])
    candidate_rows = dict(line.rstrip('\n').split('\t') for line in paths['candidate_pairs'].read_text().splitlines()[1:])
    assert list(matching) == list(candidate_rows) == ['S1-a', 'S1-b', 'S1-c']
    assert matching == {'S1-a': 'S2-1,S3-1', 'S1-b': 'S2-3', 'S1-c': ''}
    for key, value in matching.items():
        assert set(filter(None, value.split(','))) <= set(filter(None, candidate_rows[key].split(',')))
    assert 'S3-2' not in candidate_rows['S1-b'] and 'S2-4' not in candidate_rows['S1-c']
    validator = load_validator()
    errors: list[str] = []
    required = {'S1-a', 'S1-b', 'S1-c'}
    valid = {'S2-1', 'S2-2', 'S2-3', 'S2-4', 'S3-1', 'S3-2'}
    for path, header, label in ((paths['matching_results'], validator.MATCHING_HEADER, 'matched_entity_ids'),
                                (paths['candidate_pairs'], validator.CANDIDATE_HEADER, 'candidate_entity_ids')):
        assert validator.validate_id_list_file(str(path), header, label, required, valid, errors) is not None
    assert errors == []
