from __future__ import annotations

import logging
import math
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import closing
from pathlib import Path

from src.data.inspect import _batches
from src.data.load import GROUND_TRUTH_COLUMNS, _check_id, iter_ground_truth, parse_matched_ids
from src.data.validation import iter_manifest, search_universe
from src.evaluation.metrics import entity_f05

LOGGER = logging.getLogger(__name__)
EVALUATION_PROTOCOL = 'full-training-search-universe-v2'
NONEXISTENT_TARGET = 'S2-does-not-exist'


def iter_predictions(path: str | Path) -> Iterator[tuple[str, tuple[str, ...]]]:
    with Path(path).open(encoding='utf-8', newline='') as stream:
        if stream.readline().rstrip('\r\n').split('\t') != list(GROUND_TRUTH_COLUMNS):
            raise ValueError('Predictions must have the exact official matching-results header')
        for line_number, line in enumerate(stream, 2):
            fields = line.rstrip('\r\n').split('\t')
            if len(fields) != 2:
                raise ValueError(f'Malformed prediction row at line {line_number}')
            entity_id, raw_targets = fields
            _check_id(entity_id, ('S1-',))
            yield entity_id, parse_matched_ids(raw_targets)


def evaluate_sets(pairs: Iterable[tuple[set[str], set[str]]]) -> dict:
    """Aggregate aligned entity sets. File/ID validation belongs to evaluate_predictions."""
    result = dict.fromkeys(('evaluated_entities', 'exact_match_entities', 'singleton_count',
                            'correct_singletons', 'singleton_false_merges', 'tp', 'fp', 'fn',
                            'zero_prediction_entities', 'one_prediction_entities', 'multiple_prediction_entities'), 0)

    def scores() -> Iterator[float]:
        for truth, predicted in pairs:
            result['evaluated_entities'] += 1
            result['exact_match_entities'] += truth == predicted
            result['singleton_count'] += not truth
            result['correct_singletons'] += not truth and not predicted
            result['singleton_false_merges'] += not truth and bool(predicted)
            result['tp'] += len(truth & predicted)
            result['fp'] += len(predicted - truth)
            result['fn'] += len(truth - predicted)
            result['zero_prediction_entities'] += not predicted
            result['one_prediction_entities'] += len(predicted) == 1
            result['multiple_prediction_entities'] += len(predicted) > 1
            yield entity_f05(truth, predicted)

    total_score = math.fsum(scores())
    if not result['evaluated_entities']:
        raise ValueError('Cannot evaluate an empty Source 1 subset')
    result['macro_f05'] = total_score / result['evaluated_entities']
    tp, fp, fn = (result[key] for key in ('tp', 'fp', 'fn'))
    result['micro_precision_diagnostic'] = tp / (tp + fp) if tp + fp else 0.0
    result['micro_recall_diagnostic'] = tp / (tp + fn) if tp + fn else 0.0
    return result


def evaluate_predictions(truth_path: str | Path, prediction_path: str | Path,
                         source1_manifest: str | Path,
                         target_universe: Mapping[int, Sequence[str | Path]],
                         work_dir: str | Path) -> dict:
    """Score exactly the selected IDs, rejecting invalid IDs, duplicates and missing rows.

    ``target_universe`` maps Sources 2 and 3 to the ID manifests of every record a
    prediction may reference. Any listed target that is not in an entity's truth
    set is a false positive, whatever its ownership partition; an ID outside the
    universe is rejected. All lookup tables reside in temporary SQLite storage.
    """
    if set(target_universe) != {2, 3}:
        raise ValueError('target_universe must map exactly Sources 2 and 3')
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='evaluation-', dir=work_dir) as tmp:
        with closing(sqlite3.connect(Path(tmp) / 'evaluation.sqlite')) as db:
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('PRAGMA synchronous=OFF')
            db.execute('PRAGMA cache_size=-65536')
            for table in ('selected', 'allowed'):
                db.execute(f'CREATE TABLE {table} (id TEXT PRIMARY KEY) WITHOUT ROWID')
            try:
                for batch in _batches(iter_manifest(source1_manifest, 1)):
                    db.executemany('INSERT INTO selected VALUES (?)', ((value,) for value in batch))
                universe_counts = {'source2': 0, 'source3': 0}
                for source in (2, 3):
                    for path in target_universe[source]:
                        for batch in _batches(iter_manifest(path, source)):
                            db.executemany('INSERT INTO allowed VALUES (?)', ((value,) for value in batch))
                            universe_counts[f'source{source}'] += len(batch)
                db.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError('Repeated ID in selection/target manifests') from exc
            db.execute('CREATE TABLE truth (id TEXT, matches TEXT)')
            LOGGER.info('Reading ground truth for the selected subset')
            for batch in _batches(iter_ground_truth(truth_path)):
                db.executemany('INSERT INTO truth VALUES (?,?)', ((i, ','.join(ids)) for i, ids in batch))
                db.commit()
            try:
                db.execute('CREATE UNIQUE INDEX truth_ids ON truth(id)')
            except sqlite3.IntegrityError as exc:
                raise ValueError('Repeated Source 1 row in ground truth') from exc
            if db.execute('SELECT 1 FROM selected WHERE id NOT IN (SELECT id FROM truth) LIMIT 1').fetchone():
                raise ValueError('Selected Source 1 ID is missing from ground truth')
            db.execute('CREATE TABLE predictions (id TEXT PRIMARY KEY, matches TEXT) WITHOUT ROWID')
            db.execute('CREATE TABLE referenced (id TEXT)')
            LOGGER.info('Validating prediction rows and target IDs')
            try:
                for batch in _batches(iter_predictions(prediction_path)):
                    db.executemany('INSERT INTO predictions VALUES (?,?)', ((i, ','.join(ids)) for i, ids in batch))
                    db.executemany('INSERT INTO referenced VALUES (?)', ((target,) for _, ids in batch for target in ids))
                    db.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError('Repeated Source 1 row in predictions') from exc
            if db.execute('SELECT 1 FROM predictions WHERE id NOT IN (SELECT id FROM selected) LIMIT 1').fetchone():
                raise ValueError('Unknown or unselected prediction Source 1 ID')
            if db.execute('SELECT 1 FROM selected WHERE id NOT IN (SELECT id FROM predictions) LIMIT 1').fetchone():
                raise ValueError('Missing prediction row; an explicit empty list is required')
            if db.execute('SELECT 1 FROM referenced WHERE id NOT IN (SELECT id FROM allowed) LIMIT 1').fetchone():
                raise ValueError('Prediction target is not in the target search universe')
            db.execute('DELETE FROM referenced')
            for batch in _batches(db.execute('SELECT matches FROM truth JOIN selected USING(id)')):
                db.executemany('INSERT INTO referenced VALUES (?)', ((target,) for (value,) in batch for target in parse_matched_ids(value)))
            if db.execute('SELECT 1 FROM referenced WHERE id NOT IN (SELECT id FROM allowed) LIMIT 1').fetchone():
                raise ValueError('Positive truth target is absent from the target search universe')
            query = '''SELECT t.matches,p.matches FROM selected s
                JOIN truth t ON t.id=s.id JOIN predictions p ON p.id=s.id ORDER BY s.id'''
            result = evaluate_sets((set(parse_matched_ids(t)), set(parse_matched_ids(p))) for t, p in db.execute(query))
            return {**result, 'target_universe_counts': universe_counts}


def run_sanity_checks(truth_path: str | Path, split_dir: str | Path,
                      work_dir: str | Path) -> dict:
    """Exercise the file evaluator on validation fixtures derived from truth, never model outputs.

    All fixtures search the full training target universe. Besides the truth copy
    and all-empty fixtures, the first non-singleton validation entity receives one
    extra target labelled for the first non-singleton development entity (it must
    be accepted and cost exactly one false positive), then a nonexistent target
    (it must be rejected).
    """
    split_dir, work_dir = Path(split_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    universe = search_universe(split_dir)
    with tempfile.TemporaryDirectory(prefix='sanity-', dir=work_dir) as tmp:
        path = Path(tmp)
        host, cross = None, None
        with closing(sqlite3.connect(path / 'selection.sqlite')) as db:
            db.execute('CREATE TABLE selected (id TEXT PRIMARY KEY) WITHOUT ROWID')
            db.executemany('INSERT INTO selected VALUES (?)', ((i,) for i in iter_manifest(split_dir / 'validation_source1.tsv', 1)))
            with (path / 'truth_fixture.tsv').open('w') as perfect, (path / 'empty_fixture.tsv').open('w') as empty:
                for stream in (perfect, empty):
                    stream.write('\t'.join(GROUND_TRUTH_COLUMNS) + '\n')
                for entity_id, ids in iter_ground_truth(truth_path):
                    if db.execute('SELECT 1 FROM selected WHERE id=?', (entity_id,)).fetchone():
                        perfect.write(f"{entity_id}\t{','.join(ids)}\n")
                        empty.write(f'{entity_id}\t\n')
                        if host is None and ids:
                            host = entity_id, ids
                    elif cross is None and ids:
                        cross = ids[0]
        if host is None or cross is None:
            raise ValueError('Sanity checks need a non-singleton entity in each partition')
        header, *rows = (path / 'truth_fixture.tsv').read_text().splitlines(keepends=True)
        for name, extra in (('cross', cross), ('nonexistent', NONEXISTENT_TARGET)):
            (path / f'{name}_fixture.tsv').write_text(header + ''.join(
                f"{host[0]}\t{','.join(host[1] + (extra,))}\n" if row.startswith(f'{host[0]}\t') else row
                for row in rows))
        source1 = split_dir / 'validation_source1.tsv'
        results = {name: evaluate_predictions(truth_path, path / f'{name}_fixture.tsv', source1, universe, work_dir)
                   for name in ('truth', 'empty', 'cross')}
        try:
            evaluate_predictions(truth_path, path / 'nonexistent_fixture.tsv', source1, universe, work_dir)
        except ValueError as exc:
            if 'search universe' not in str(exc):
                raise
            results['nonexistent'] = {'target': NONEXISTENT_TARGET, 'rejected': True, 'error': str(exc)}
        else:
            raise ValueError('Nonexistent-target sanity check failed')
    if results['truth']['macro_f05'] != 1.0:
        raise ValueError('Truth-copy sanity check failed')
    expected = results['empty']['singleton_count'] / results['empty']['evaluated_entities']
    if results['empty']['macro_f05'] != expected:
        raise ValueError('Empty-prediction sanity check failed')
    truth, cross_result = results['truth'], results['cross']
    n = truth['evaluated_entities']
    expected = math.fsum([1.0] * (n - 1) + [entity_f05(set(host[1]), {*host[1], cross})]) / n
    if ((cross_result['tp'], cross_result['fp'], cross_result['fn']) != (truth['tp'], 1, 0)
            or cross_result['exact_match_entities'] != n - 1 or cross_result['macro_f05'] != expected):
        raise ValueError('Cross-partition target sanity check failed')
    cross_result.update(host_source1_id=host[0], development_owned_target=cross,
                        host_entity_f05=entity_f05(set(host[1]), {*host[1], cross}))
    return results
