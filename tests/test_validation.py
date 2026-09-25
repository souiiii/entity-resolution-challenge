from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from src.data.load import SOURCE_COLUMNS, iter_ground_truth
from src.data.validation import (file_hash, iter_manifest, partition_for, prepare_split, read_config, search_universe,
                                 training_target_pool, verify_manifests)
from src.evaluation.evaluate import run_sanity_checks
from src.utils.paths import INTERIM_DIR, PROJECT_ROOT


@pytest.fixture
def config():
    return {'seed': 42, 'validation_fraction': .1, 'split_version': 'source1-sha256-v1', 'expected_fingerprint': None}


@pytest.fixture
def training(tmp_path):
    root = tmp_path / 'train'
    root.mkdir()
    rows = {1: [], 2: [], 3: []}
    truth = []
    for i in range(300):
        entity_id = f'S1-{i:04}'
        country = 'US' if i % 2 else 'India'
        rows[1].append(f'{entity_id}\tName {i}\tStreet {i}\t{country}\n')
        targets = []
        for source in (2, 3):
            if i % 5 >= source - 1:
                target = f'S{source}-{i:04}'
                targets.append(target)
                rows[source].append(f'{target}\tName {i}\tStreet {i}\t{country}\n')
        truth.append(f"{entity_id}\t{','.join(targets)}\n")
    for source in (2, 3):
        rows[source].extend(f'S{source}-unused{i:04}\tOther\tElsewhere\tUS\n' for i in range(100))
    for source in (1, 2, 3):
        (root / f'train_source{source}.tsv').write_text('\t'.join(SOURCE_COLUMNS) + '\n' + ''.join(rows[source]))
    (root / 'train_ground_truth.tsv').write_text('source1_entity_id\tmatched_entity_ids\n' + ''.join(truth))
    return root


def membership(directory):
    return {(part, source): set(iter_manifest(directory / f'{part}_source{source}.tsv', source))
            for part in ('development', 'validation') for source in (1, 2, 3)}


def test_reproduce_reorder_and_freeze(training, config, tmp_path):
    first = tmp_path / 'first'
    metadata = prepare_split(training, first, config)
    assert metadata == prepare_split(training, first, config, rebuild=True)
    assert metadata == prepare_split(training, first, config)
    config['expected_fingerprint'] = metadata['fingerprint']
    assert verify_manifests(first, config) == metadata
    assert all(v == 0 for v in metadata['integrity_checks'].values())
    assert all(v['passed'] for v in metadata['representativeness'].values())
    config['seed'] += 1
    with pytest.raises(ValueError, match='configuration changed'):
        prepare_split(training, first, config)
    config['seed'] -= 1
    config['expected_fingerprint'] = None
    rng = random.Random(71)
    for path in training.glob('*.tsv'):
        header, *rows = path.read_text().splitlines(keepends=True)
        rng.shuffle(rows)
        path.write_text(header + ''.join(rows))
    second = tmp_path / 'second'
    reordered = prepare_split(training, second, config)
    assert membership(first) == membership(second)
    assert metadata['manifest_hashes'] == reordered['manifest_hashes']
    with pytest.raises(ValueError, match='input fingerprint changed'):
        prepare_split(training, first, config)
    manifest = second / 'validation_source1.tsv'
    manifest.write_text(manifest.read_text() + 'S1-tamper\n')
    with pytest.raises(ValueError, match='Manifest fingerprint mismatch'):
        verify_manifests(second)


def test_link_ownership_distractors_coverage_and_real_file_evaluation(training, config, tmp_path):
    output = tmp_path / 'split'
    prepare_split(training, output, config)
    ids = membership(output)
    for source in (1, 2, 3):
        assert ids['development', source].isdisjoint(ids['validation', source])
        raw = {line.split('\t')[0] for line in (training / f'train_source{source}.tsv').read_text().splitlines()[1:]}
        assert ids['development', source] | ids['validation', source] == raw
    for entity_id, targets in iter_ground_truth(training / 'train_ground_truth.tsv'):
        part = partition_for(entity_id, config)
        assert entity_id in ids[part, 1]
        assert all(target in ids[part, int(target[1])] for target in targets)
    for source in (2, 3):
        for i in range(100):
            target = f'S{source}-unused{i:04}'
            assert target in ids[partition_for(target, config, 'distractor'), source]
    universe = search_universe(output)
    for source in (2, 3):
        raw = {line.split('\t')[0] for line in (training / f'train_source{source}.tsv').read_text().splitlines()[1:]}
        assert {i for path in universe[source] for i in iter_manifest(path, source)} == raw
        for part in ('development', 'validation'):
            pool = {i for path in training_target_pool(output, part)[source] for i in iter_manifest(path, source)}
            assert pool == ids[part, source]
    with pytest.raises(ValueError, match='Unknown partition'):
        training_target_pool(output, 'test')
    sanity = run_sanity_checks(training / 'train_ground_truth.tsv', output, tmp_path / 'work')
    assert sanity['truth']['macro_f05'] == 1.0
    assert sanity['empty']['macro_f05'] == sanity['empty']['singleton_count'] / sanity['empty']['evaluated_entities']
    assert sanity['truth']['target_universe_counts'] == {'source2': 340, 'source3': 280}
    cross = sanity['cross']
    assert (cross['tp'], cross['fp'], cross['fn']) == (sanity['truth']['tp'], 1, 0)
    assert cross['development_owned_target'] in ids['development', int(cross['development_owned_target'][1])]
    assert cross['host_source1_id'] in ids['validation', 1]
    assert sanity['nonexistent']['rejected']


def test_missing_truth_and_unknown_target_rejected(training, config, tmp_path):
    truth = training / 'train_ground_truth.tsv'
    original = truth.read_text()
    truth.write_text(original.replace('S1-0001\tS2-0001\n', ''))
    with pytest.raises(ValueError, match='integrity checks failed'):
        prepare_split(training, tmp_path / 'missing', config)
    truth.write_text(original.replace('S2-0001', 'S2-unknown'))
    with pytest.raises(ValueError, match='integrity checks failed'):
        prepare_split(training, tmp_path / 'unknown', config)
    assert not (tmp_path / 'missing').exists()
    assert not (tmp_path / 'unknown').exists()


def test_shared_labelled_target_rejected(training, config, tmp_path):
    truth = training / 'train_ground_truth.tsv'
    truth.write_text(truth.read_text().replace('S2-0002', 'S2-0001'))
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        prepare_split(training, tmp_path / 'shared', config)


def test_configuration_validation(config, tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    assert read_config(path) == config
    config['validation_fraction'] = 1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        read_config(path)


def test_wrong_committed_fingerprint_rejected(training, config, tmp_path):
    output = tmp_path / 'frozen'
    prepare_split(training, output, config)
    config['expected_fingerprint'] = '0' * 64
    with pytest.raises(ValueError, match='committed fingerprint'):
        prepare_split(training, output, config)


def test_committed_real_split_is_frozen():
    split_dir = INTERIM_DIR / 'validation'
    if not (split_dir / 'metadata.json').exists():
        pytest.skip('Frozen real-data manifests are not prepared')
    config = read_config(PROJECT_ROOT / 'experiments/configs/validation.json')
    assert config['expected_fingerprint'] == '1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645'
    metadata = verify_manifests(split_dir, config)
    assert metadata['fingerprint'] == config['expected_fingerprint']
    assert sum(1 for _ in iter_manifest(split_dir / 'validation_source1.tsv', 1)) == 220_410
    universe = search_universe(split_dir)
    counts = {source: sum(1 for path in universe[source] for _ in iter_manifest(path, source)) for source in (2, 3)}
    assert counts == {2: 5_034_616, 3: 5_285_603}
