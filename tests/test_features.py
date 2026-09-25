from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import pytest

from src.blocking.candidates import SourceBlocker
from src.blocking.channels import BlockingConfig, ChannelSpec, load_config
from src.blocking.ranking import prune_candidates
from src.blocking.store import RecordStore, build_store
from src.data.load import SOURCE_COLUMNS
from src.features.extract import iter_features, select_candidates
from src.features.pairwise import (ADDRESS_FEATURES, CandidateEvidence, MISSING_FEATURES, NAME_FEATURES,
                                   NUMBER_FEATURES, extract_pair_features, feature_names, record_view)
from src.features.text import SourceText, build_text_offsets, ensure_text

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = (ChannelSpec('address_exact', 'exact', ('address_canonical',), 50),
            ChannelSpec('name_exact', 'exact', ('name_canonical',), 50),
            ChannelSpec('name_folded', 'exact', ('name_folded',), 50),
            ChannelSpec('name_token_pair', 'combo', ('name', 'name'), 50, symmetric=True),
            ChannelSpec('address_number_x_word', 'combo', ('address_number', 'address_word'), 50))


def features(query: tuple[str, str], target: tuple[str, str], source: int = 2, mask: int = 0,
             rank: int = 1) -> dict[str, float]:
    evidence = CandidateEvidence(source, np.array([rank]), np.array([mask], np.uint32), np.array([3]),
                                 np.array([0.5]))
    row = extract_pair_features([record_view(*query)], [record_view(*target)], np.array([0]), np.array([0]),
                                evidence, CHANNELS)[0]
    return dict(zip(feature_names(CHANNELS), row.tolist()))


def test_exact_name_and_address_features():
    f = features(('Acme Trading', '10 Market Street, Springfield'), ('ACME  TRADING', '10 MARKET STREET SPRINGFIELD'))
    assert f['name_canonical_equal'] == f['name_folded_equal'] == f['name_folded_compact_equal'] == 1
    assert f['address_canonical_equal'] == 1 and f['name_char_similarity'] == 1 and f['address_token_jaccard'] == 1
    assert f['name_length_ratio'] == 1 and f['name_script_mismatch'] == 0


def test_folded_and_compact_name_behavior():
    f = features(('Prívate Ínc', 'x'), ('Private Inc', 'x'))
    assert f['name_canonical_equal'] == 0 and f['name_folded_equal'] == 1 and f['name_char_similarity'] == 1
    g = features(('Tri-State Center', 'x'), ('TriState Center', 'x'))
    assert g['name_folded_equal'] == 0 and g['name_folded_compact_equal'] == 1


def test_token_overlap_jaccard_containment():
    f = features(('Happy Hair Studio', 'a'), ('HAIR HAPPY LP', 'b'))
    assert f['name_token_intersection'] == 2
    assert f['name_token_jaccard'] == pytest.approx(2 / 4)
    assert f['name_token_containment'] == pytest.approx(2 / 3)
    assert f['name_token_set_similarity'] > f['name_char_similarity']


def test_empty_fields_are_undefined_not_zero():
    f = features(('Acme', '1 Main St'), ('Acme', ''))
    for name in ('address_char_similarity', 'address_token_set_similarity', 'address_token_jaccard',
                 'address_token_containment', 'address_length_ratio', 'target_numbers_in_query'):
        assert math.isnan(f[name]), name
    assert f['address_canonical_equal'] == 0 and f['target_address_empty'] == 1
    assert f['number_jaccard'] == 0 and f['query_numbers_in_target'] == 0
    g = features(('', ''), ('', ''))
    assert g['target_address_empty'] == 1 and g['name_canonical_equal'] == 0 and g['name_folded_equal'] == 0
    assert math.isnan(g['name_char_similarity']) and math.isnan(g['number_set_equal'])


def test_numeric_address_features_use_integer_forms():
    f = features(('A', '015115 Brookview Dr, Unit 2'), ('B', '15115 BROOKVIEW DRIVE'))
    assert (f['query_number_count'], f['target_number_count'], f['shared_number_count']) == (2, 1, 1)
    assert f['number_jaccard'] == 0.5 and f['query_numbers_in_target'] == 0 and f['target_numbers_in_query'] == 1
    assert f['number_set_equal'] == 0
    g = features(('A', '#1303 Montoya St'), ('B', 'Albuquerque, 1303 Montoya Street'))
    assert g['number_set_equal'] == 1 and g['query_numbers_in_target'] == 1


def test_completely_different_strings():
    f = features(('Zephyrine Bakery', '12 Ocean Road'), ('Quantum Logistics', '98 Hill Street'))
    assert f['name_token_intersection'] == 0 and f['name_token_jaccard'] == 0 and f['address_token_intersection'] == 0
    assert f['name_char_similarity'] < 0.35 and f['shared_number_count'] == 0 and f['number_set_equal'] == 0


def test_unicode_indic_and_accented_latin():
    f = features(('विजन ट्रेडिंग', 'मुंबई'), ('विजन  ट्रेडिंग', 'मुंबई'))
    assert f['name_canonical_equal'] == 1 and f['name_char_similarity'] == 1 and f['name_script_mismatch'] == 0
    g = features(('Vision Trading', 'Delhi'), ('विजन ट्रेडिंग', 'Delhi'))
    assert g['name_script_mismatch'] == 1 and g['name_token_intersection'] == 0 and g['address_canonical_equal'] == 1
    h = features(('École Normale', 'Paris'), ('ECOLE NORMALE', 'PARIS'))
    assert h['name_canonical_equal'] == 0 and h['name_folded_equal'] == 1 and h['name_script_mismatch'] == 0


def test_source_and_blocking_evidence_propagation():
    f = features(('A', 'x'), ('A', 'x'), source=3, mask=0b00111, rank=7)
    assert f['source_is_s3'] == 1 and f['blocking_rank'] == 7
    assert [f[f'channel_{c.name}'] for c in CHANNELS] == [1, 1, 1, 0, 0]
    assert f['blocking_support_channels'] == 3 and f['blocking_support_families'] == 2
    assert f['channel_address_exact'] == 1 and f['blocking_exact_name'] == 1
    assert f['blocking_min_key_frequency'] == 3 and f['blocking_aggregate_rarity'] == 0.5
    g = features(('A', 'x'), ('A', 'x'), source=2, mask=0b11000)
    assert g['source_is_s3'] == 0 and g['channel_address_exact'] == g['blocking_exact_name'] == 0
    assert g['blocking_support_families'] == 2


def test_feature_bounds_and_schema_order():
    names = feature_names(CHANNELS)
    assert len(names) == len(set(names)) and names[0] == 'source_is_s3'
    assert names[-len(MISSING_FEATURES):] == MISSING_FEATURES
    v2 = load_config(ROOT / 'experiments/configs/blocking_v2.json')
    assert feature_names(v2.channels) == feature_names(v2.channels) and len(feature_names(v2.channels)) == 42
    rows = [(('Acme Co', '1 A St'), ('ACME', '1 A Street')), (('X', ''), ('Y', '2 B')), (('राम', '5'), ('Ram', '5'))]
    bounded = [n for n in NAME_FEATURES + ADDRESS_FEATURES + NUMBER_FEATURES
               if 'similarity' in n or 'jaccard' in n or 'containment' in n or 'ratio' in n or 'equal' in n
               or n.endswith('_in_target') or n.endswith('_in_query')]
    for query, target in rows:
        f = features(query, target)
        for name in bounded:
            assert math.isnan(f[name]) or 0 <= f[name] <= 1, name


def write_source(path: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerows([SOURCE_COLUMNS, *rows])
    return path


def setup(tmp_path: Path, target_rows: list, query_rows: list, tag: str = ''):
    stores, texts = {}, {}
    for source, rows in ((1, query_rows), (2, target_rows)):
        path = write_source(tmp_path / f'{tag}source{source}.tsv', rows)
        stores[source] = build_store(path, source, tmp_path / f'{tag}store{source}')
        texts[source] = ensure_text(path, stores[source], tmp_path / f'{tag}text{source}')
    config = BlockingConfig('test', CHANNELS, batch_size=2, workers=1, min_free_disk_gib=0,
                            raw={'ranking': {'selected_strategy': 'aggregate', 'selected_k': 3}})
    return stores, texts, config


QUERIES = [('S1-1', 'Acme Trading', '10 Market Street', 'Atlantis'), ('S1-2', 'Zeta Care', '5 Hill Rd', 'France')]
TARGETS = [('S2-1', 'ACME TRADING', '10 MARKET STREET', 'Atlantis'), ('S2-2', 'Acme Trading', '11 Other', 'Atlantis'),
           ('S2-3', 'Acme Trading', '10 Market Street', 'France'), ('S2-4', 'ZETA CARE', '5 HILL RD', 'France'),
           ('S2-5', 'Zeta Care Clinic', '', 'France'), ('S2-6', 'Acme Trading Co', '10 Market St', 'Atlantis')]


def collect(tmp_path, stores, texts, config, workers=1):
    rows = np.arange(stores[1].size)
    result = {}
    for batch in iter_features(config, stores[1], texts[1], stores[2], texts[2], rows, tmp_path / 'index', 1, workers):
        for owner, record, row in zip(batch.owners, batch.records, batch.features):
            result[(stores[1].ids[owner].decode(), stores[2].ids[record].decode())] = row.tolist()
    return result


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(np.array(a[k]), np.array(b[k]), equal_nan=True) for k in a)


def test_end_to_end_open_countries_and_target_order_independence(tmp_path):
    stores, texts, config = setup(tmp_path, TARGETS, QUERIES)
    first = collect(tmp_path, stores, texts, config)
    assert {k for k in first} == {('S1-1', 'S2-1'), ('S1-1', 'S2-2'), ('S1-1', 'S2-6'), ('S1-2', 'S2-4'),
                                  ('S1-2', 'S2-5')}
    shuffled = TARGETS[:]
    random.Random(5).shuffle(shuffled)
    stores2, texts2, _ = setup(tmp_path, shuffled, QUERIES, 'b')
    second = collect(tmp_path / 'b', stores2, texts2, config)
    assert same(first, second)
    assert same(first, collect(tmp_path / 'c', stores, texts, config, workers=2))


def test_selection_matches_phase3b_pruning(tmp_path):
    stores, _, config = setup(tmp_path, TARGETS, QUERIES)
    blocker = SourceBlocker(config, stores[2], tmp_path / 'index')
    rows = np.arange(stores[1].size)
    selected, ranks = select_candidates(blocker, stores[1], rows)
    pruned = prune_candidates(blocker.candidates(stores[1], rows), stores[2], config.channels, 'aggregate', 3)
    assert np.array_equal(selected.records, pruned.records) and np.array_equal(selected.owners, pruned.owners)
    assert ranks.min() >= 1 and ranks.max() <= 3


def test_text_offsets_are_reproducible_verified_and_cleaned_up(tmp_path):
    path = write_source(tmp_path / 'source2.tsv', TARGETS)
    store = build_store(path, 2, tmp_path / 'store')
    build_text_offsets(path, store, tmp_path / 'a')
    build_text_offsets(path, store, tmp_path / 'b')
    assert (tmp_path / 'a' / 'offsets.bin').read_bytes() == (tmp_path / 'b' / 'offsets.bin').read_bytes()
    text = SourceText(path, store, tmp_path / 'a')
    assert text.read(np.array([3, 0])) == [('ZETA CARE', '5 HILL RD', 'France'), ('ACME TRADING', '10 MARKET STREET',
                                                                                 'Atlantis')]
    text.close()
    other = write_source(tmp_path / 'other.tsv', list(reversed(TARGETS)))
    with pytest.raises(ValueError, match='disagree'):
        build_text_offsets(other, store, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists() and not (tmp_path / 'bad.building').exists()
    with pytest.raises(OSError):
        build_text_offsets(path, store, tmp_path / 'full', min_free_bytes=2**62)
    (tmp_path / 'a' / 'meta.json').write_text('{"text_version": "old", "records": 0, "input_sha256": "x"}')
    rebuilt = ensure_text(path, store, tmp_path / 'a')
    assert rebuilt.read(np.array([1])) == [('Acme Trading', '11 Other', 'Atlantis')]
