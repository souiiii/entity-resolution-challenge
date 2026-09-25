from __future__ import annotations

import csv
from itertools import islice
from pathlib import Path

import pytest

from src.data.inspect import inspect_challenge, preview_table
from src.data.load import (
    GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, iter_ground_truth, iter_source,
    list_input_files, load_ground_truth, load_source, parse_matched_ids, read_table,
)
from src.utils.paths import RAW_DIR


def write_tsv(path: Path, columns: tuple[str, ...], rows: list[tuple]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def test_source_preserves_literal_strings_unicode_and_open_country(tmp_path):
    rows = [('S1-0001', ' Café राम, "NA" ', '', 'France'),
            ('S1-2', 'NA', '001 Road\tAnnex', 'Unseen country'),
            ('S1-3', '', 'NULL', '')]
    path = write_tsv(tmp_path / 'source.tsv', SOURCE_COLUMNS, rows)
    loaded = load_source(path, 1)
    assert list(loaded.columns) == list(SOURCE_COLUMNS)
    assert all(str(dtype) == 'string' for dtype in loaded.dtypes)
    assert list(loaded.itertuples(index=False, name=None)) == rows
    assert not loaded.isna().any().any()
    assert read_table(path).equals(loaded)
    assert preview_table(path, 1) == [dict(zip(SOURCE_COLUMNS, rows[0]))]


def test_ground_truth_zero_one_multiple_and_order(tmp_path):
    rows = [('S1-001', ''), ('S1-2', 'S3-9'), ('S1-3', 'S2-1,S3-2,S2-3')]
    path = write_tsv(tmp_path / 'truth.tsv', GROUND_TRUTH_COLUMNS, rows)
    expected = [('S1-001', ()), ('S1-2', ('S3-9',)), ('S1-3', ('S2-1', 'S3-2', 'S2-3'))]
    assert list(iter_ground_truth(path)) == expected
    assert list(load_ground_truth(path).itertuples(index=False, name=None)) == rows


@pytest.mark.parametrize('value', [' ', 'NA', 'S1-2', 'S2-', 'S2-1,', ',S3-1', 'S2-1, S3-2', 'S2-1,S2-1'])
def test_bad_truth_lists_rejected(value):
    with pytest.raises(ValueError):
        parse_matched_ids(value)


@pytest.mark.parametrize('columns,rows', [
    (SOURCE_COLUMNS[::-1], [('US', 'A', 'N', 'S1-1')]),
    (SOURCE_COLUMNS, [('S1-1', 'N', 'A')]),
    (SOURCE_COLUMNS, [('S1-1', 'N', 'A', 'US', 'extra')]),
    (SOURCE_COLUMNS, [('', 'N', 'A', 'US')]),
    (SOURCE_COLUMNS, [('S2-1', 'N', 'A', 'US')]),
])
def test_bad_source_schema_and_ids_rejected(tmp_path, columns, rows):
    path = write_tsv(tmp_path / 'bad.tsv', columns, rows)
    with pytest.raises(ValueError):
        list(iter_source(path, 1))


def test_preview_and_iterator_do_not_read_entire_file(tmp_path):
    path = write_tsv(tmp_path / 'source.tsv', SOURCE_COLUMNS, [('S1-1', 'Name', '', 'US')])
    with path.open('a') as stream:
        stream.write('malformed\n')
    assert len(preview_table(path, 1)) == 1
    assert len(list(islice(iter_source(path, 1), 1))) == 1
    with pytest.raises(ValueError):
        list(iter_source(path, 1))


@pytest.fixture
def small_challenge(tmp_path):
    root = tmp_path / 'raw'
    sources = {
        'train_source1': [('S1-a', 'Same', '1 St', 'US'), ('S1-b', 'Empty', '', 'India'), ('S1-c', 'Third', '2 St', 'US')],
        'train_source2': [('S2-a', 'Same', '1 St', 'US'), ('S2-b', 'Same', '1 St', 'US'), ('S2-u', 'Unused', 'X', 'US')],
        'train_source3': [('S3-a', 'Different', '1 St', 'US')],
        'test_source1': [('S1-t', 'École', 'Paris', 'France')],
        'test_source2': [('S2-t', 'École', 'Paris', 'France')],
        'test_source3': [('S3-t', 'राम', '', 'India')],
    }
    for name, rows in sources.items():
        write_tsv(root / name.split('_')[0] / f'{name}.tsv', SOURCE_COLUMNS, rows)
    write_tsv(root / 'train/train_ground_truth.tsv', GROUND_TRUTH_COLUMNS,
              [('S1-a', 'S2-a,S3-a'), ('S1-b', ''), ('S1-c', 'S2-b')])
    return root


def test_complete_inspector_exact_statistics_and_determinism(small_challenge, tmp_path):
    work = tmp_path / 'interim'
    report = inspect_challenge(small_challenge, work, work / 'stats.json')
    assert report['complete']
    assert len(list_input_files(small_challenge)) == 7
    assert report['sources']['train_source2']['duplicate_profiles'] == 1
    assert report['sources']['train_source2']['duplicate_rows'] == 0
    assert report['sources']['train_source2']['duplicate_ids'] == 0
    assert report['sources']['train_source1']['fields']['business_address']['missing_fraction'] == 1 / 3
    truth = report['ground_truth']
    assert truth['match_count_distribution'] == {0: 1, 1: 1, 2: 1}
    assert truth['sources']['2']['links'] == 2
    assert truth['sources']['2']['unlinked_records'] == 1
    assert truth['sources']['3']['links'] == 1
    assert truth['sources']['2']['raw_agreement']['equal_both'] == 1
    assert truth['missing_source1_ids'] == truth['unknown_source1_ids'] == 0
    assert truth['targets_linked_to_multiple_source1'] == 0
    assert report['train_test_id_overlap'] == {'1': 0, '2': 0, '3': 0}
    assert report == inspect_challenge(small_challenge, work)
    assert not list(work.glob('phase1-*'))


def test_inspector_reports_duplicates_unknowns_and_reused_targets(small_challenge, tmp_path):
    path = small_challenge / 'train/train_source2.tsv'
    with path.open('a') as stream:
        stream.write('S2-a\tSame\t1 St\tUS\n')
    write_tsv(small_challenge / 'train/train_ground_truth.tsv', GROUND_TRUTH_COLUMNS,
              [('S1-a', 'S2-a'), ('S1-a', 'S2-a'), ('S1-unknown', 'S2-missing'), ('S1-c', 'S2-a')])
    report = inspect_challenge(small_challenge, tmp_path / 'work')
    source = report['sources']['train_source2']
    assert source['duplicate_rows'] == source['duplicate_ids'] == 1
    truth = report['ground_truth']
    assert truth['duplicate_rows'] == truth['duplicate_ids'] == 1
    assert truth['unknown_source1_ids'] == truth['missing_source1_ids'] == 1
    assert truth['sources']['2']['unknown_targets'] == 1
    assert truth['targets_linked_to_multiple_source1'] == 1


@pytest.mark.parametrize('split,source', [(s, n) for s in ('train', 'test') for n in (1, 2, 3)])
def test_real_source_schema_sample(split, source):
    path = RAW_DIR / split / f'{split}_source{source}.tsv'
    if not path.exists():
        pytest.skip('Official data are not distributed with the repository')
    sample = list(islice(iter_source(path, source), 10))
    assert len(sample) == 10
    assert all(tuple(row) == SOURCE_COLUMNS for row in sample)
    assert len({row['entity_id'] for row in sample}) == 10


def test_real_ground_truth_schema_sample():
    path = RAW_DIR / 'train/train_ground_truth.tsv'
    if not path.exists():
        pytest.skip('Official data are not distributed with the repository')
    sample = list(islice(iter_ground_truth(path), 10))
    assert len(sample) == 10
    assert all(entity_id.startswith('S1-') for entity_id, _ in sample)
