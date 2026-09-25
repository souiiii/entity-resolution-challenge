from __future__ import annotations

import logging
import math
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import closing
from pathlib import Path

from src.data.inspect import _batches
from src.data.load import GROUND_TRUTH_COLUMNS, _check_id, iter_ground_truth, parse_matched_ids
from src.data.validation import iter_manifest
from src.evaluation.metrics import entity_f05

LOGGER = logging.getLogger(__name__)


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
                         target_manifests: tuple[str | Path, str | Path],
                         work_dir: str | Path) -> dict:
    """Score exactly the selected IDs, rejecting invalid IDs, duplicates and missing rows.

    All lookup tables reside in temporary SQLite storage. Target manifests define
    the permitted inference pool, so predictions into the other partition fail.
    """
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
                for source, path in zip((2, 3), target_manifests):
                    for batch in _batches(iter_manifest(path, source)):
                        db.executemany('INSERT INTO allowed VALUES (?)', ((value,) for value in batch))
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
                raise ValueError('Prediction target is unknown or outside the selected target pool')
            db.execute('DELETE FROM referenced')
            for batch in _batches(db.execute('SELECT matches FROM truth JOIN selected USING(id)')):
                db.executemany('INSERT INTO referenced VALUES (?)', ((target,) for (value,) in batch for target in parse_matched_ids(value)))
            if db.execute('SELECT 1 FROM referenced WHERE id NOT IN (SELECT id FROM allowed) LIMIT 1').fetchone():
                raise ValueError('Positive truth target is absent from the selected target pool')
            query = '''SELECT t.matches,p.matches FROM selected s
                JOIN truth t ON t.id=s.id JOIN predictions p ON p.id=s.id ORDER BY s.id'''
            return evaluate_sets((set(parse_matched_ids(t)), set(parse_matched_ids(p))) for t, p in db.execute(query))


def run_sanity_checks(truth_path: str | Path, split_dir: str | Path,
                      work_dir: str | Path) -> dict:
    """Exercise the file evaluator with truth-copy and empty fixtures, never model outputs."""
    split_dir, work_dir = Path(split_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sanity-', dir=work_dir) as tmp:
        path = Path(tmp)
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
        targets = (split_dir / 'validation_source2.tsv', split_dir / 'validation_source3.tsv')
        results = {name: evaluate_predictions(truth_path, path / f'{name}_fixture.tsv',
                   split_dir / 'validation_source1.tsv', targets, work_dir) for name in ('truth', 'empty')}
    if results['truth']['macro_f05'] != 1.0:
        raise ValueError('Truth-copy sanity check failed')
    expected = results['empty']['singleton_count'] / results['empty']['evaluated_entities']
    if results['empty']['macro_f05'] != expected:
        raise ValueError('Empty-prediction sanity check failed')
    return results
