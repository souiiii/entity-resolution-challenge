from __future__ import annotations

import pytest

import src.preprocessing.analysis as analysis
from src.data.load import SOURCE_COLUMNS
from src.data.validation import partition_for, prepare_split
from src.preprocessing.analysis import analyze_normalization


@pytest.fixture
def config():
    return {'seed': 42, 'validation_fraction': .1, 'split_version': 'source1-sha256-v1', 'expected_fingerprint': None}


@pytest.fixture
def split(tmp_path, config):
    root = tmp_path / 'train'
    root.mkdir()
    rows = {1: [], 2: [], 3: []}
    truth = []
    for i in range(240):
        entity_id, country = f'S1-{i:04}', ('US' if i % 2 else 'India')
        rows[1].append((entity_id, f'Acme {i} Trading', f'{i} Main Street, Springfield', country))
        targets = []
        if i % 4:
            targets.append(f'S2-{i:04}')
            rows[2].append((targets[-1], f'ACME  {i} TRADING', f'{i} MAIN STREET, SPRINGFIELD', country))
        if i % 3:
            targets.append(f'S3-{i:04}')
            rows[3].append((targets[-1], f'Acme {i} Tráding', f'{i} Main Street, Springfield', country))
        truth.append(f"{entity_id}\t{','.join(targets)}\n")
    for source in (2, 3):
        rows[source].extend((f'S{source}-x{j:03}', 'विजन ट्रेडिंग', '', 'India') for j in range(60))
    for source in (1, 2, 3):
        (root / f'train_source{source}.tsv').write_text(
            '\t'.join(SOURCE_COLUMNS) + '\n' + ''.join('\t'.join(r) + '\n' for r in rows[source]))
    (root / 'train_ground_truth.tsv').write_text('source1_entity_id\tmatched_entity_ids\n' + ''.join(truth))
    metadata = prepare_split(root, tmp_path / 'split', config)
    return root, tmp_path / 'split', metadata


def expected_links(config):
    development = [i for i in range(240) if partition_for(f'S1-{i:04}', config) == 'development']
    return {'source2': sum(1 for i in development if i % 4), 'source3': sum(1 for i in development if i % 3)}


def test_development_only_equality_collisions_and_unicode(split, config, tmp_path):
    raw_dir, split_dir, metadata = split
    report = analyze_normalization(raw_dir, split_dir, config, tmp_path / 'work', workers=1)
    links = expected_links(config)
    assert report['development_link_counts'] == links
    assert links['source2'] == metadata['partitions']['development']['targets']['2']['links']
    assert report['validation_labels_used'] is False and report['partition'] == 'development'
    s2, s3 = (report['link_equality'][f'source{s}']['total'] for s in (2, 3))
    assert s2['links'] == links['source2'] and s3['links'] == links['source3']
    assert s2['name']['raw']['count'] == 0
    assert s2['name']['canonical']['count'] == s2['address']['canonical']['count'] == links['source2']
    assert s2['name_address']['canonical']['count'] == links['source2']
    assert s3['name']['canonical']['count'] == 0
    assert s3['name']['folded']['count'] == s3['name']['folded_compact']['count'] == links['source3']
    assert s3['address']['raw']['count'] == links['source3']
    assert s3['country_mismatches'] == 0
    development = {s: sum(1 for _ in open(split_dir / f'development_source{s}.tsv')) - 1 for s in (1, 2, 3)}
    assert {s: report['sources'][f'source{s}']['records'] for s in (1, 2, 3)} == development
    name = report['collisions']['name']
    assert name['canonical']['source1']['overall']['distinct_values'] == development[1]
    assert name['canonical']['source1']['overall']['largest_bucket'] == 1
    distractors = development[2] - links['source2']
    assert name['canonical']['source2']['overall']['largest_bucket'] == distractors
    assert report['collisions']['address']['raw']['source2']['overall']['nonempty_records'] == links['source2']
    cross = name['folded']['source1_to_source3_same_country']
    assert cross['equal_key_pairs'] == cross['true_links_among_them'] == links['source3']
    for source in (1, 2, 3):
        for flags in report['sources'][f'source{source}']['flags'].values():
            for bad in ('letters_or_digits_lost', 'canonical_not_idempotent', 'empty_after_normalization',
                        'devanagari_altered', 'folding_altered_indic'):
                assert flags.get(bad, 0) == 0
    assert report['sources']['source2']['flags']['name']['devanagari'] == distractors
    assert report['sources']['source3']['flags']['name']['accented_or_non_ascii_latin'] == links['source3']
    assert {e['kind'] for e in report['examples']} <= {'largest_bucket', 'most_merged'}
    assert not list((tmp_path / 'work').iterdir())


def test_parallel_workers_give_identical_results(split, config, tmp_path):
    raw_dir, split_dir, _ = split
    serial = analyze_normalization(raw_dir, split_dir, config, tmp_path / 'a', workers=1)
    parallel = analyze_normalization(raw_dir, split_dir, config, tmp_path / 'b', workers=2)
    for report in (serial, parallel):
        report.pop('runtime_seconds')
    assert serial == parallel


def test_link_count_mismatch_fails_loudly(split, config, tmp_path, monkeypatch):
    raw_dir, split_dir, _ = split
    monkeypatch.setattr(analysis, '_load_links', lambda db, path: {'source2': 0, 'source3': 0})
    with pytest.raises(ValueError, match='differ from frozen'):
        analyze_normalization(raw_dir, split_dir, config, tmp_path / 'work', include_examples=False, workers=1)
