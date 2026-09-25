from __future__ import annotations

import numpy as np
import pytest

from src.blocking.candidates import SourceBlocker
from src.blocking.channels import ChannelSpec
from src.blocking.ranking import STRATEGIES, candidate_ranks, oracle_ceiling, prune_candidates
from test_blocking import ADDRESS, EXACT_NAME, FOLDED, config, make_store


@pytest.mark.parametrize('truth,found,expected', [([0], [0], 1), ([3], [0], 0), ([3], [2], 10 / 11),
                                                 ([0, 3, 3], [0, 0, 2], (1 + 10 / 11) / 3),
                                                 ([2, 4], [2, 4], 1), ([], [], 0)])
def test_oracle(truth, found, expected):
    assert oracle_ceiling(np.array(truth), np.array(found)) == pytest.approx(expected)


def test_oracle_rejects_invalid_counts():
    with pytest.raises(ValueError):
        oracle_ceiling(np.array([1]), np.array([2]))


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_ranking_exact_rarity_dedup_and_binary_ties(tmp_path, strategy):
    query = make_store(tmp_path, 1, [('S1-1', 'Acme Rare Common', '10 Main', 'France'),
                                    ('S1-2', 'Nothing', '', 'Atlantis')])
    rows = [('S2-z', 'Other', '10 Main', 'France'),
            ('S2-name', 'Acme Rare Common', '', 'France'),
            ('S2-rare', 'Rare', '', 'France'),
            ('S2-2', 'Common', '', 'France'), ('S2-10', 'Common', '', 'France'),
            ('S2-wrong', 'Acme Rare Common', '10 Main', 'US')]
    channels = (ADDRESS, EXACT_NAME, FOLDED, ChannelSpec('token', 'token', ('name',), 10))
    outputs = []
    for label, targets in [('a', rows), ('b', list(reversed(rows)))]:
        store = make_store(tmp_path, 2, targets, label)
        blocker = SourceBlocker(config(*channels), store, tmp_path / label)
        batch = blocker.candidates(query, np.arange(2))
        ranks = candidate_ranks(batch, store, channels, strategy)
        outputs.append(store.ids[batch.records[np.argsort(ranks)]].tolist())
        assert outputs[-1] == [b'S2-z', b'S2-name', b'S2-rare', b'S2-10', b'S2-2']
        pruned = prune_candidates(batch, store, channels, strategy, 4)
        assert len(pruned.records) == 4 and len(set(pruned.records)) == 4
        assert store.ids[pruned.records].tolist() == [b'S2-10', b'S2-name', b'S2-rare', b'S2-z']
        assert prune_candidates(batch, store, channels, strategy, 100).records.tolist() == batch.records.tolist()
    assert outputs[0] == outputs[1]


def test_top_k_is_independent_per_source(tmp_path):
    query = make_store(tmp_path, 1, [('S1-1', 'Acme', '', 'Atlantis')])
    selected = []
    for source in (2, 3):
        store = make_store(tmp_path, source, [(f'S{source}-{i}', 'Acme', '', 'Atlantis') for i in (2, 10, 3)])
        batch = SourceBlocker(config(EXACT_NAME), store, tmp_path / 'index').candidates(query, np.array([0]))
        keep = prune_candidates(batch, store, (EXACT_NAME,), 'families', 2)
        selected.extend(store.ids[keep.records].tolist())
    assert selected == [b'S2-10', b'S2-2', b'S3-10', b'S3-2']


def test_exact_token_conjunction_caps_limits_and_country(tmp_path):
    query = make_store(tmp_path, 1, [('S1-1', 'Café', '1 2 2', 'France'),
                                    ('S1-empty', '', '1', 'France')])
    target = make_store(tmp_path, 2, [('S2-a', 'Cafe', '1', 'France'),
                                     ('S2-b', 'Café', '1', 'France'),
                                     ('S2-c', 'Cafe', '2 2', 'France'),
                                     ('S2-us', 'Cafe', '2', 'US')])
    spec = ChannelSpec('recovery', 'combo', ('name_folded', 'address_number'), 2, max_keys=1)
    blocker = SourceBlocker(config(spec), target, tmp_path / 'index')
    batch = blocker.candidates(query, np.array([0, 1]))
    assert target.ids[batch.records].tolist() == [b'S2-c']
    assert batch.min_frequency.tolist() == [1]
    assert batch.channel_stats['recovery']['queries_limited'] == 1
    capped = ChannelSpec('recovery', 'combo', ('name_folded', 'address_number'), 1)
    other = SourceBlocker(config(capped), target, tmp_path / 'index').candidates(query, np.array([0, 1]))
    assert target.ids[other.records].tolist() == [b'S2-c']
    assert other.channel_stats['recovery']['queries_capped'] == 1


def test_aggregate_rarity_preserves_multiple_shared_keys(tmp_path):
    from src.blocking.candidates import union

    target = make_store(tmp_path, 2, [('S2-1', '', '', 'US'), ('S2-2', '', '', 'US')])
    batch = union(np.array([0, 0, 0]), np.array([0, 1, 1]), np.array([1, 1, 1], np.uint32), target,
                  frequencies=np.array([2, 3, 3]))
    spec = (ChannelSpec('word', 'token', ('address_word',), 10),)
    assert batch.min_frequency.tolist() == [2, 3]
    assert batch.aggregate_rarity.tolist() == [0.5, 2 / 3]
    assert candidate_ranks(batch, target, spec, 'rarity').tolist() == [1, 2]
    assert candidate_ranks(batch, target, spec, 'aggregate').tolist() == [2, 1]


def test_empty_candidate_ranking(tmp_path):
    query = make_store(tmp_path, 1, [('S1-1', '', '', 'US')])
    target = make_store(tmp_path, 2, [('S2-1', '', '', 'US')])
    batch = SourceBlocker(config(EXACT_NAME), target, tmp_path / 'index').candidates(query, np.array([0]))
    for strategy in STRATEGIES:
        assert candidate_ranks(batch, target, (EXACT_NAME,), strategy).tolist() == []
        assert prune_candidates(batch, target, (EXACT_NAME,), strategy, 1).records.tolist() == []


def test_configured_candidate_pipeline(tmp_path):
    from dataclasses import replace

    query = make_store(tmp_path, 1, [('S1-1', 'Acme', '', 'US')])
    target = make_store(tmp_path, 2, [(f'S2-{i}', 'Acme', '', 'US') for i in (1, 10, 2)])
    cfg = replace(config(EXACT_NAME), raw={'ranking': {'selected_strategy': 'aggregate', 'selected_k': 2}})
    blocker = SourceBlocker(cfg, target, tmp_path / 'index')
    selected = blocker.configured_candidates(query, np.array([0]))
    assert target.ids[selected.records].tolist() == [b'S2-1', b'S2-10']
