from __future__ import annotations

import pytest

from src.evaluation.evaluate import evaluate_predictions, evaluate_sets
from src.evaluation.metrics import entity_f05


@pytest.mark.parametrize('truth,predicted,expected', [
    (set(), set(), 1.0), (set(), {'S2-a'}, 0.0), ({'S2-a'}, set(), 0.0),
    ({'S2-a'}, {'S2-a'}, 1.0), ({'S2-a', 'S3-b'}, {'S2-a'}, 5 / 6),
    ({'S2-a'}, {'S2-a', 'S3-b'}, 5 / 9),
    ({'S2-a', 'S3-b'}, {'S2-a', 'S3-b', 'S3-c'}, 5 / 7),
])
def test_entity_metric(truth, predicted, expected):
    assert entity_f05(truth, predicted) == expected


def test_macro_is_not_micro_and_diagnostics():
    result = evaluate_sets([(set(), set()), ({'S2-a'}, set()), (set(), {'S3-b'})])
    assert result['macro_f05'] == 1 / 3
    assert result['micro_precision_diagnostic'] == result['micro_recall_diagnostic'] == 0.0
    assert result['singleton_count'] == 2
    assert result['correct_singletons'] == result['singleton_false_merges'] == 1
    assert result['exact_match_entities'] == 1
    assert (result['tp'], result['fp'], result['fn']) == (0, 1, 1)
    assert result['zero_prediction_entities'] == 2
    assert result['one_prediction_entities'] == 1
    assert result['multiple_prediction_entities'] == 0
    with pytest.raises(ValueError, match='empty'):
        evaluate_sets([])


@pytest.fixture
def files(tmp_path):
    paths = {name: tmp_path / f'{name}.tsv' for name in ('truth', 'predictions', 's1', 's2', 's3')}
    paths['truth'].write_text('source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a,S3-b\nS1-b\t\n')
    paths['predictions'].write_text(paths['truth'].read_text())
    for source, ids in [(1, ['S1-a', 'S1-b']), (2, ['S2-a', 'S2-other']), (3, ['S3-b'])]:
        paths[f's{source}'].write_text('entity_id\n' + '\n'.join(ids) + '\n')
    return paths


def score(files, tmp_path):
    return evaluate_predictions(files['truth'], files['predictions'], files['s1'],
                                (files['s2'], files['s3']), tmp_path / 'scratch')


def test_perfect_file_and_order_independence(files, tmp_path):
    assert score(files, tmp_path)['macro_f05'] == 1.0
    files['predictions'].write_text('source1_entity_id\tmatched_entity_ids\nS1-b\t\nS1-a\tS3-b,S2-a\n')
    result = score(files, tmp_path)
    assert result['macro_f05'] == 1.0
    assert result['exact_match_entities'] == 2
    assert result['tp'] == 2
    assert not list((tmp_path / 'scratch').iterdir())


@pytest.mark.parametrize('body,error', [
    ('S1-a\tS2-a,S2-a\nS1-b\t\n', 'Repeated target'),
    ('S1-a\t\nS1-a\t\nS1-b\t\n', 'Repeated Source 1'),
    ('S1-a\tS2-unknown\nS1-b\t\n', 'unknown or outside'),
    ('S1-a\tS1-b\nS1-b\t\n', 'Invalid entity ID'),
    ('S1-a\t\n', 'Missing prediction'),
    ('S1-a\t\nS1-b\t\nS1-x\t\n', 'Unknown or unselected'),
    ('S1-a\nS1-b\t\n', 'Malformed'),
    ('S1-a\tS2-a\textra\nS1-b\t\n', 'Malformed'),
    ('S1-a\tS2-a,\nS1-b\t\n', 'Invalid entity ID'),
    ('S1-a\t"S2-a"\nS1-b\t\n', 'Invalid entity ID'),
])
def test_broken_predictions_rejected(files, tmp_path, body, error):
    files['predictions'].write_text('source1_entity_id\tmatched_entity_ids\n' + body)
    with pytest.raises(ValueError, match=error):
        score(files, tmp_path)


def test_cross_entity_target_reuse_is_scored_not_rejected(files, tmp_path):
    files['predictions'].write_text('source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a,S3-b\nS1-b\tS2-a\n')
    result = score(files, tmp_path)
    assert result['macro_f05'] == .5
    assert result['singleton_false_merges'] == 1


def test_bad_truth_and_manifests(files, tmp_path):
    original = files['truth'].read_text()
    files['truth'].write_text(original + 'S1-a\t\n')
    with pytest.raises(ValueError, match='ground truth'):
        score(files, tmp_path)
    files['truth'].write_text(original)
    files['s3'].write_text('entity_id\n')
    files['predictions'].write_text('source1_entity_id\tmatched_entity_ids\nS1-a\t\nS1-b\t\n')
    with pytest.raises(ValueError, match='Positive truth target'):
        score(files, tmp_path)


def test_wrong_header_rejected(files, tmp_path):
    files['predictions'].write_text('source1_entity_id,matched_entity_ids\n')
    with pytest.raises(ValueError, match='header'):
        score(files, tmp_path)


def test_duplicate_manifest_ids_rejected(files, tmp_path):
    files['s1'].write_text('entity_id\nS1-a\nS1-a\n')
    with pytest.raises(ValueError, match='Repeated ID'):
        score(files, tmp_path)


def test_unknown_selected_truth_id_rejected(files, tmp_path):
    files['s1'].write_text('entity_id\nS1-absent\n')
    with pytest.raises(ValueError, match='missing from ground truth'):
        score(files, tmp_path)
