from __future__ import annotations

import json

import numpy as np
import pytest

from src.blocking.channels import BlockingConfig, ChannelSpec
from src.blocking.evaluate import distribution, in_sample, run_blocking
from src.data.load import SOURCE_COLUMNS
from src.data.validation import partition_for, prepare_split

SPLIT = {'seed': 42, 'validation_fraction': .1, 'split_version': 'source1-sha256-v1', 'expected_fingerprint': None}
ADDRESS = ChannelSpec('address_exact', 'exact', ('address_canonical',), 50)
NAME = ChannelSpec('name_exact', 'exact', ('name_canonical',), 50)


@pytest.fixture
def split(tmp_path):
    root = tmp_path / 'train'
    root.mkdir()
    rows = {1: [], 2: [], 3: []}
    truth = []
    for i in range(200):
        entity_id, country = f'S1-{i:04}', ('US' if i % 2 else 'India')
        rows[1].append((entity_id, f'Acme {i} Trading', f'{i} Main Street', country))
        targets = []
        if i % 4:
            targets.append(f'S2-{i:04}')
            name = f'ACME {i} TRADING' if i % 3 else f'Acme {i} Tradng'
            rows[2].append((targets[-1], name, f'{i} MAIN STREET' if i % 5 else '', country))
        if i % 3:
            targets.append(f'S3-{i:04}')
            rows[3].append((targets[-1], f'Different {i}', f'Main Street {i}', country))
        truth.append(f"{entity_id}\t{','.join(targets)}\n")
    first = next(i for i in range(200) if i % 4 and i % 5 and partition_for(f'S1-{i:04}', SPLIT) == 'development')
    country = 'US' if first % 2 else 'India'
    rows[2].extend((f'S2-x{j}', f'Other {j}', f'{first} Main Street', country) for j in range(3))
    for source in (1, 2, 3):
        (root / f'train_source{source}.tsv').write_text(
            '\t'.join(SOURCE_COLUMNS) + '\n' + ''.join('\t'.join(r) + '\n' for r in rows[source]))
    (root / 'train_ground_truth.tsv').write_text('source1_entity_id\tmatched_entity_ids\n' + ''.join(truth))
    prepare_split(root, tmp_path / 'split', SPLIT)
    (tmp_path / 'validation.json').write_text(json.dumps(SPLIT))
    return root, tmp_path


def run(split, *channels, sample=False, candidates=None):
    root, tmp_path = split
    cfg = BlockingConfig(version='test', channels=channels, batch_size=7, workers=1, min_free_disk_gib=0,
                         sample_salt='s', sample_fraction=0.5)
    return run_blocking(cfg, root, tmp_path / 'split', tmp_path / 'validation.json', tmp_path / 'store',
                        tmp_path / 'index', sample=sample, sample_candidates_path=candidates)


def development():
    return [i for i in range(200) if partition_for(f'S1-{i:04}', SPLIT) == 'development']


def test_recall_entities_and_incremental_accounting(split):
    report = run(split, ADDRESS, NAME)
    dev = development()
    s2 = [i for i in dev if i % 4]
    s3 = [i for i in dev if i % 3]
    s2_found = [i for i in s2 if i % 5 or i % 3]
    assert report['links'] == {'source2': len(s2), 'source3': len(s3)}
    assert report['recall']['source2']['found'] == len(s2_found)
    assert report['recall']['source3']['found'] == 0
    assert report['recall']['overall']['recall'] == len(s2_found) / (len(s2) + len(s3))
    assert report['validation_labels_used'] is False and report['partition'] == 'development'
    by_country = report['recall']['by_country']
    assert by_country['US']['source2']['links'] + by_country['India']['source2']['links'] == len(s2)
    address, name = report['channels']
    assert address['source2']['standalone_true'] == len([i for i in s2 if i % 5])
    assert name['source2']['standalone_true'] == len([i for i in s2 if i % 3])
    assert address['source2']['incremental_true'] + name['source2']['incremental_true'] == len(s2_found)
    assert name['source2']['incremental_true'] == len([i for i in s2 if i % 3 and not i % 5])
    total_pairs = report['candidates']['source2']['total_pairs']
    assert address['source2']['incremental_candidates'] + name['source2']['incremental_candidates'] == total_pairs
    entities = report['entities']
    linked = [i for i in dev if i % 4 or i % 3]
    assert entities['with_links'] == len(linked) and entities['singletons'] == len(dev) - len(linked)
    assert entities['all_links_recovered'] == len([i for i in linked if not i % 3 and (i % 5 or i % 3)])
    assert entities['zero_recovered'] == len([i for i in linked if not (i % 4 and (i % 5 or i % 3))])
    assert report['country_isolation']['source2'] == {'cross_country_candidates': 0, 'true_links_crossing_country': 0}
    assert report['missed']['source3']['missed_links'] == len(s3)
    assert report['missed']['source3']['missed_links_sharing']['shares_address_word'] == len(s3)


def test_frequency_cap_is_reported_and_costs_the_capped_link(split):
    uncapped = run(split, ADDRESS)
    capped = run(split, ChannelSpec('address_exact', 'exact', ('address_canonical',), 3))
    assert uncapped['channels'][0]['source2']['queries_capped'] == 0
    assert capped['channels'][0]['source2']['queries_capped'] == 1
    assert capped['recall']['source2']['found'] == uncapped['recall']['source2']['found'] - 1
    assert capped['missed']['source2']['channels']['address_exact']['shared_over_frequency_cap'] == 1
    assert uncapped['channels'][0]['source2']['largest_bucket'] == 4


def test_sample_is_label_independent_and_writes_official_file(split):
    root, tmp_path = split
    path = tmp_path / 'sample_candidate_pairs.tsv'
    report = run(split, ADDRESS, NAME, sample=True, candidates=path)
    expected = [i for i in development() if in_sample(f'S1-{i:04}', 's', 0.5)]
    assert report['queries'] == len(expected) and report['mode'] == 'sample'
    lines = path.read_text().splitlines()
    assert lines[0] == 'source1_entity_id\tcandidate_entity_ids'
    assert [line.split('\t')[0] for line in lines[1:]] == sorted(f'S1-{i:04}' for i in expected)
    for line in lines[1:]:
        ids = line.split('\t')[1].split(',') if line.split('\t')[1] else []
        assert len(ids) == len(set(ids)) and all(t.startswith(('S2-', 'S3-')) for t in ids)
        assert ids == sorted(ids, key=lambda t: (t[:2], t.encode()))


def test_distribution_statistics():
    d = distribution(np.array([0, 1, 2, 3, 100, 600]))
    assert d['total_pairs'] == 706 and d['max'] == 600 and d['zero_candidates'] == 1
    assert d['p50'] == 2 and d['share_over_500'] == 1 / 6 and d['share_over_50'] == 2 / 6


def test_curve_accounting_and_oracle(split):
    root, tmp_path = split
    cfg = BlockingConfig(version='curves', channels=(ADDRESS, NAME), batch_size=7, workers=1,
                         min_free_disk_gib=0, raw={'ranking': {'strategies': ['families', 'rarity'],
                         'top_k': [1, 2, 5], 'baseline_channels': 1,
                         'selected_strategy': 'families', 'selected_k': 2}})
    report = run_blocking(cfg, root, tmp_path / 'split', tmp_path / 'validation.json', tmp_path / 'store',
                          tmp_path / 'index', sample=False, sample_candidates_path=tmp_path / 'candidates.tsv')
    curves = report['curves']
    assert report['selected'] == curves['families@2']
    for strategy in ('families', 'rarity'):
        points = [curves[f'{strategy}@{k}'] for k in (1, 2, 5)]
        assert [p['recall']['overall']['found'] for p in points] == sorted(p['recall']['overall']['found'] for p in points)
        assert [p['oracle_blocking_macro_f05_ceiling'] for p in points] == sorted(p['oracle_blocking_macro_f05_ceiling'] for p in points)
        for k, point in zip((1, 2, 5), points):
            c = point['candidates']
            assert c['combined']['total_pairs'] == c['source2']['total_pairs'] + c['source3']['total_pairs']
            assert c['combined']['max'] <= 2 * k
            assert point['recall']['overall']['found'] == sum(point['recall'][s]['found'] for s in ('source2', 'source3'))
        assert points[-1]['recall'] == curves['unpruned']['recall']
    assert curves['unpruned']['oracle_blocking_macro_f05_ceiling'] == report['oracle_blocking_macro_f05_ceiling']
    assert curves['phase3a_union']['oracle_delta_vs_phase3a'] == 0
    assert curves['unpruned']['recall']['overall']['found'] > curves['phase3a_union']['recall']['overall']['found']


def test_invalid_ranking_fails_before_data_access(tmp_path):
    for ranking in ({'strategies': ['unknown'], 'top_k': [10]},
                    {'strategies': ['aggregate'], 'top_k': [0]},
                    {'strategies': ['aggregate'], 'top_k': [10], 'selected_strategy': 'aggregate', 'selected_k': 5}):
        cfg = BlockingConfig(version='invalid', channels=(ADDRESS,), raw={'ranking': ranking})
        with pytest.raises(ValueError):
            run_blocking(cfg, tmp_path, tmp_path, tmp_path / 'missing.json', tmp_path, tmp_path, sample=True)
