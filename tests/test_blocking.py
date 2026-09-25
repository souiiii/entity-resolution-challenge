from __future__ import annotations

import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest

from src.blocking.candidates import SourceBlocker
from src.blocking.channels import BlockingConfig, ChannelSpec, estimate_postings, load_config, query_keys
from src.blocking.index import PostingIndex, combine_key, combine_keys
from src.blocking.keys import key_hash, number_form, record_keys
from src.blocking.store import RecordStore, build_store
from src.data.load import SOURCE_COLUMNS
from src.submission.create_submission import CANDIDATE_HEADER, MATCHING_HEADER, write_id_list_file

ROOT = Path(__file__).resolve().parents[1]


def write_source(path: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerows([SOURCE_COLUMNS, *rows])
    return path


def make_store(tmp_path: Path, source: int, rows: list[tuple[str, str, str, str]], name: str = '') -> RecordStore:
    path = write_source(tmp_path / f'{name}source{source}.tsv', rows)
    return build_store(path, source, tmp_path / f'{name}store{source}')


def config(*channels: ChannelSpec, batch_size: int = 2) -> BlockingConfig:
    return BlockingConfig(version='test', channels=channels, batch_size=batch_size, workers=1, min_free_disk_gib=0)


def candidates(tmp_path, cfg, queries, targets, source=2):
    blocker = SourceBlocker(cfg, targets, tmp_path / f'index-{cfg.version}-{id(cfg)}')
    batch = blocker.candidates(queries, np.arange(queries.size))
    ids = np.asarray(targets.ids)[batch.records]
    result: dict[str, list[str]] = {i.decode(): [] for i in np.asarray(queries.ids)}
    for owner, target in zip(batch.owners.tolist(), ids.tolist()):
        result[queries.ids[owner].decode()].append(target.decode())
    return result, batch


EXACT_NAME = ChannelSpec('name_exact', 'exact', ('name_canonical',), 50)
FOLDED = ChannelSpec('name_folded', 'exact', ('name_folded',), 50)
ADDRESS = ChannelSpec('address_exact', 'exact', ('address_canonical',), 50)


def test_keys_are_country_scoped_and_empty_values_have_no_keys():
    us, india = record_keys('Acme', '1 Main St', 'US'), record_keys('ACME', '1 MAIN ST', 'India')
    assert us.exact[0] != india.exact[0] and us.exact[3] != india.exact[3]
    assert record_keys('Acme', '1 Main St', 'US') == record_keys('ACME', '1  main st.', 'US')
    empty = record_keys('', '', 'US')
    assert empty.exact == (0, 0, 0, 0) and not empty.name_tokens and not empty.address_tokens
    assert key_hash('x', 'US', '') == 0 and number_form('015115') == '15115'
    assert combine_key(5, 9, True) == combine_key(9, 5, True)
    assert combine_key(9, 5) != combine_key(5, 9)
    assert int(combine_keys(np.array([5], np.uint64), np.array([9], np.uint64))[0]) == combine_key(5, 9)


def test_exact_name_folded_and_address_channels(tmp_path):
    queries = make_store(tmp_path, 1, [('S1-1', 'Acme Trading', '10 Market Street, Springfield', 'US')])
    targets = make_store(tmp_path, 2, [('S2-case', 'ACME  TRADING', 'elsewhere', 'US'),
                                       ('S2-accent', 'Acme Tráding', 'elsewhere', 'US'),
                                       ('S2-address', 'Other', '10 MARKET STREET, SPRINGFIELD', 'US'),
                                       ('S2-none', 'Unrelated', 'Nowhere', 'US')])
    assert candidates(tmp_path, config(EXACT_NAME), queries, targets)[0]['S1-1'] == ['S2-case']
    assert candidates(tmp_path, config(FOLDED), queries, targets)[0]['S1-1'] == ['S2-accent', 'S2-case']
    assert candidates(tmp_path, config(ADDRESS), queries, targets)[0]['S1-1'] == ['S2-address']


def test_country_isolation_and_open_country_labels(tmp_path):
    queries = make_store(tmp_path, 1, [('S1-fr', 'Boulangerie Dupont', 'Rue X', 'France'),
                                       ('S1-new', 'Boulangerie Dupont', 'Rue X', 'Atlantis')])
    targets = make_store(tmp_path, 2, [('S2-fr', 'BOULANGERIE DUPONT', 'rue x', 'France'),
                                       ('S2-us', 'Boulangerie Dupont', 'Rue X', 'US'),
                                       ('S2-new', 'boulangerie dupont', 'RUE X', 'Atlantis')])
    found, _ = candidates(tmp_path, config(EXACT_NAME, ADDRESS), queries, targets)
    assert found == {'S1-fr': ['S2-fr'], 'S1-new': ['S2-new']}


def test_dedup_across_channels_masks_and_prefixes(tmp_path):
    queries = make_store(tmp_path, 1, [('S1-1', 'Acme', '1 Main St', 'US')])
    targets = make_store(tmp_path, 3, [('S3-both', 'ACME', '1 MAIN ST', 'US'), ('S3-name', 'Acme', 'x', 'US')])
    found, batch = candidates(tmp_path, config(ADDRESS, EXACT_NAME), queries, targets)
    assert found['S1-1'] == ['S3-both', 'S3-name']
    assert batch.masks.tolist() == [0b11, 0b10]
    assert all(t.startswith('S3-') for t in found['S1-1'])


def test_deterministic_order_independent_of_target_file_order(tmp_path):
    rows = [(f'S2-{i}', 'Acme', f'{i} Main', 'US') for i in (5, 10, 1, 7, 42, 100)]
    queries = make_store(tmp_path, 1, [('S1-1', 'acme', 'x', 'US')])
    first = candidates(tmp_path, config(EXACT_NAME), queries, make_store(tmp_path, 2, rows, 'a'))[0]
    random.Random(3).shuffle(rows)
    second = candidates(tmp_path, config(EXACT_NAME), queries, make_store(tmp_path, 2, rows, 'b'))[0]
    assert first == second == {'S1-1': sorted(r[0] for r in rows)}


def test_generic_name_bucket_limit(tmp_path):
    queries = make_store(tmp_path, 1, [('S1-1', 'Primary Care', 'a', 'US'), ('S1-2', 'Zeta Care', 'b', 'US')])
    targets = make_store(tmp_path, 2, [(f'S2-{i}', 'Primary Care', f'{i}', 'US') for i in range(5)]
                         + [('S2-z', 'ZETA CARE', 'q', 'US')])
    capped = ChannelSpec('name_exact', 'exact', ('name_canonical',), 3)
    found, batch = candidates(tmp_path, config(capped), queries, targets)
    assert found == {'S1-1': [], 'S1-2': ['S2-z']}
    assert batch.channel_stats['name_exact']['queries_capped'] == 1


def test_rare_token_lookup_prefers_rarest_eligible_token(tmp_path):
    queries = make_store(tmp_path, 1, [('S1-1', 'Zephyrine Bakery', 'x', 'US')])
    targets = make_store(tmp_path, 2, [('S2-rare', 'Zephyrine Foods', 'y', 'US')]
                         + [(f'S2-b{i}', f'Bakery {i}', 'z', 'US') for i in range(3)])
    rare = ChannelSpec('name_rare_token', 'token', ('name',), 10, max_keys=1)
    assert candidates(tmp_path, config(rare), queries, targets)[0]['S1-1'] == ['S2-rare']
    strict = ChannelSpec('name_rare_token', 'token', ('name',), 2, max_keys=2)
    assert candidates(tmp_path, config(strict), queries, targets)[0]['S1-1'] == ['S2-rare']


def test_numeric_address_lookup_uses_integer_equivalent_numbers(tmp_path):
    queries = make_store(tmp_path, 1, [('S1-1', 'A', '015115 Brookview Drive, Urbandale, IA', 'US')])
    targets = make_store(tmp_path, 2, [('S2-hit', 'B', '15115 BROOKVIEW DR, URBANDALE', 'US'),
                                       ('S2-other', 'C', '15116 Brookview Dr', 'US')])
    combo = ChannelSpec('number_word', 'combo', ('address_number', 'address_word'), 10)
    assert candidates(tmp_path, config(combo), queries, targets)[0]['S1-1'] == ['S2-hit']
    assert estimate_postings(combo, targets) == 1 * 3 + 1 * 2


def test_name_token_pairs_and_empty_fields(tmp_path):
    queries = make_store(tmp_path, 1, [('S1-1', 'Happy Hair Studio', '', 'US'), ('S1-2', '', '', 'US')])
    targets = make_store(tmp_path, 2, [('S2-1', 'HAPPY HAIR STUDIO LP', '', 'US'), ('S2-2', 'Hair Happy', '', 'US'),
                                       ('S2-3', '', '', 'US')])
    pairs = ChannelSpec('name_pair', 'combo', ('name', 'name'), 10, symmetric=True)
    found, _ = candidates(tmp_path, config(pairs, ADDRESS), queries, targets)
    assert found == {'S1-1': ['S2-1', 'S2-2'], 'S1-2': []}
    owners, keys = query_keys(ADDRESS, queries, np.arange(2))
    assert len(keys) == 0


def test_disk_index_matches_in_memory_index(tmp_path):
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 2**63, 500, dtype=np.uint64) | np.uint64(1)
    keys = np.concatenate([keys, keys[:50], np.zeros(3, dtype=np.uint64)])
    records = np.concatenate([np.arange(500), np.arange(50), [1, 2, 3]]).astype(np.int32)
    chunks = [(records[i:i + 100], keys[i:i + 100]) for i in range(0, len(keys), 100)]
    disk = PostingIndex.build(chunks, tmp_path / 'index', {'test': True})
    memory = PostingIndex.from_pairs(keys, records)
    assert np.array_equal(disk.keys, memory.keys) and np.array_equal(disk.records, memory.records)
    assert len(disk.keys) == 500 and not (tmp_path / 'index.building').exists()
    bucket_keys = np.array([7] * 5 + [9] * 2 + [11], dtype=np.uint64)
    bucket_records = np.arange(8, dtype=np.int32)
    kept = PostingIndex.build([(bucket_records, bucket_keys)], tmp_path / 'kept', {}, keep_limit=2)
    assert kept.keys.tolist() == [9, 9, 11] and kept.large_keys.tolist() == [7]
    lo, hi, frequency = kept.lookup(np.array([7, 9, 11, 13], dtype=np.uint64))
    assert frequency.tolist() == [5, 2, 1, 0] and (hi - lo).tolist() == [0, 2, 1, 0]
    reloaded = PostingIndex.load(tmp_path / 'kept')
    assert reloaded.frequency(np.array([7], dtype=np.uint64)).tolist() == [5]


def test_index_keep_limit_does_not_change_candidates_or_cap_accounting(tmp_path):
    queries = make_store(tmp_path, 1, [('S1-1', 'Primary Care', 'a', 'US'), ('S1-2', 'Zeta Care', 'b', 'US')])
    targets = make_store(tmp_path, 2, [(f'S2-{i}', 'Primary Care', f'{i}', 'US') for i in range(5)]
                         + [('S2-z', 'ZETA CARE', 'q', 'US')])
    full = ChannelSpec('name_exact', 'exact', ('name_canonical',), 3)
    kept = ChannelSpec('name_exact', 'exact', ('name_canonical',), 3, index_keep=3)
    assert full.index_id != kept.index_id
    first, a = candidates(tmp_path, config(full), queries, targets)
    second, b = candidates(tmp_path, config(kept), queries, targets)
    assert first == second and a.channel_stats == b.channel_stats
    with pytest.raises(ValueError):
        ChannelSpec('bad', 'exact', ('name_canonical',), 10, index_keep=5)


def test_config_file_is_valid():
    cfg = load_config(ROOT / 'experiments/configs/blocking_v1.json')
    assert cfg.channels and all(c.max_frequency > 0 for c in cfg.channels)
    with pytest.raises(ValueError):
        ChannelSpec('bad', 'combo', ('name', 'address_word'), 5, symmetric=True)


def load_validator():
    spec = importlib.util.spec_from_file_location('validator', ROOT / 'official/utils/validate_submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_writer_format_and_official_validator(tmp_path):
    path = tmp_path / 'output' / 'candidate_pairs.tsv'
    assert write_id_list_file(path, CANDIDATE_HEADER, [('S1-1', ['S2-9', 'S3-1']), ('S1-2', [])]) == 2
    assert path.read_text() == 'source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-9,S3-1\nS1-2\t\n'
    errors: list[str] = []
    mapping = load_validator().validate_id_list_file(str(path), list(CANDIDATE_HEADER), 'candidate_entity_ids',
                                                     {'S1-1', 'S1-2'}, None, errors)
    assert errors == [] and mapping == {'S1-1': {'S2-9', 'S3-1'}, 'S1-2': set()}
    for rows in ([('S1-1', []), ('S1-1', [])], [('S1-1', ['S2-1', 'S2-1'])], [('S1-1', ['S1-2'])], [('S2-1', [])]):
        with pytest.raises(ValueError):
            write_id_list_file(tmp_path / 'bad.tsv', MATCHING_HEADER, rows)
        assert not (tmp_path / 'bad.tsv').exists()
