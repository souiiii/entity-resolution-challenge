from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import tempfile
from contextlib import closing
from collections.abc import Iterator
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from src.data.inspect import _batches
from src.data.load import _check_id, _rows, iter_ground_truth, iter_source

LOGGER = logging.getLogger(__name__)
PARTITIONS = ('development', 'validation')


def file_hash(path: str | Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    if set(config) != {'seed', 'validation_fraction', 'split_version', 'expected_fingerprint'}:
        raise ValueError('Unexpected validation configuration fields')
    if type(config['seed']) is not int or not isinstance(config['split_version'], str) or not config['split_version']:
        raise ValueError('Invalid seed or split_version')
    fraction = Decimal(str(config['validation_fraction']))
    if not fraction.is_finite() or not 0 < fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    expected = config['expected_fingerprint']
    if expected is not None and (not isinstance(expected, str) or len(expected) != 64):
        raise ValueError('Invalid expected_fingerprint')
    return config


def partition_for(entity_id: str, config: dict, domain: str = 'source1') -> str:
    payload = f"{config['split_version']}\0{config['seed']}\0{domain}\0{entity_id}".encode('utf-8')
    fraction = Fraction(str(config['validation_fraction']))
    threshold = fraction.numerator * (1 << 256) // fraction.denominator
    return 'validation' if int.from_bytes(hashlib.sha256(payload).digest(), 'big') < threshold else 'development'


def iter_manifest(path: str | Path, source: int) -> Iterator[str]:
    for row in _rows(path, ('entity_id',)):
        _check_id(row['entity_id'], (f'S{source}-',))
        yield row['entity_id']


def search_universe(split_dir: str | Path) -> dict[int, tuple[Path, ...]]:
    """Validation retrieval universe: every training S2/S3 record, as both disjoint partition manifests."""
    return {source: tuple(Path(split_dir) / f'{part}_source{source}.tsv' for part in PARTITIONS)
            for source in (2, 3)}


def training_target_pool(split_dir: str | Path, partition: str) -> dict[int, tuple[Path, ...]]:
    """Ownership-isolated S2/S3 pool of one partition, for leakage-safe training and its diagnostics."""
    if partition not in PARTITIONS:
        raise ValueError(f'Unknown partition: {partition!r}')
    return {source: (Path(split_dir) / f'{partition}_source{source}.tsv',) for source in (2, 3)}


def verify_manifests(directory: str | Path, config: dict | None = None) -> dict:
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    fingerprint = metadata['fingerprint']
    payload = {k: v for k, v in metadata.items() if k != 'fingerprint'}
    if object_hash(payload) != fingerprint:
        raise ValueError('Split metadata fingerprint mismatch')
    if config is not None:
        algorithm = {k: v for k, v in config.items() if k != 'expected_fingerprint'}
        if metadata['configuration'] != algorithm:
            raise ValueError('Frozen split configuration changed')
        if config['expected_fingerprint'] not in (None, fingerprint):
            raise ValueError('Split differs from committed fingerprint')
    expected_names = {f'{part}_source{source}.tsv' for part in PARTITIONS for source in (1, 2, 3)}
    if set(metadata['manifest_hashes']) != expected_names:
        raise ValueError('Invalid manifest inventory')
    for name, expected in metadata['manifest_hashes'].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f'Manifest fingerprint mismatch: {name}')
    return metadata


def _count(db: sqlite3.Connection, sql: str) -> int:
    return db.execute(sql).fetchone()[0]


def _statistics(db: sqlite3.Connection) -> dict:
    result = {}
    for part in PARTITIONS:
        hist = dict(db.execute('SELECT n2+n3,COUNT(*) FROM truth WHERE part=? GROUP BY n2+n3', (part,)))
        n = sum(hist.values())
        if not n:
            raise ValueError('Both partitions must contain Source 1 entities')
        countries = dict(db.execute('SELECT country,COUNT(*) FROM s1 WHERE part=? GROUP BY country', (part,)))
        links = db.execute('SELECT SUM(n2),SUM(n3) FROM truth WHERE part=?', (part,)).fetchone()
        targets = {}
        for source in (2, 3):
            total = db.execute('SELECT COUNT(*) FROM targets WHERE part=? AND source=?', (part, source)).fetchone()[0]
            targets[str(source)] = {'total': total, 'links': links[source - 2], 'distractors': total - links[source - 2]}
        result[part] = {
            'source1_entities': n, 'countries': countries,
            'country_fractions': {c: v / n for c, v in countries.items()},
            'singletons': hist.get(0, 0), 'singleton_fraction': hist.get(0, 0) / n,
            'one_match_fraction': hist.get(1, 0) / n,
            'multiple_match_fraction': sum(v for k, v in hist.items() if k > 1) / n,
            'match_count_distribution': {str(k): v for k, v in hist.items()},
            'mean_matches': sum(k * v for k, v in hist.items()) / n, 'targets': targets,
        }
    a, b = (result[p] for p in PARTITIONS)
    na, nb = a['source1_entities'], b['source1_entities']
    comparisons = {}
    for field in ('countries', 'match_count_distribution'):
        for key in sorted(set(a[field]) | set(b[field])):
            ca, cb = a[field].get(key, 0), b[field].get(key, 0)
            pooled = (ca + cb) / (na + nb)
            delta = abs(ca / na - cb / nb)
            limit = max(.005, 5 * math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb)))
            comparisons[f'{field}:{key}'] = {'absolute_fraction_difference': delta, 'limit': limit, 'passed': delta <= limit}
    mean = (a['mean_matches'] * na + b['mean_matches'] * nb) / (na + nb)
    second = sum(int(k)**2 * v for p in (a, b) for k, v in p['match_count_distribution'].items()) / (na + nb)
    mean_limit = max(.02 * mean, 5 * math.sqrt(max(0, second - mean**2) * (1 / na + 1 / nb)))
    comparisons['mean_matches'] = {'absolute_difference': abs(a['mean_matches'] - b['mean_matches']), 'limit': mean_limit,
                                   'passed': abs(a['mean_matches'] - b['mean_matches']) <= mean_limit}
    if not all(c['passed'] for c in comparisons.values()):
        raise ValueError('Material split imbalance; review deterministic stratification before freezing')
    return {'partitions': result, 'representativeness': comparisons}


def prepare_split(raw_dir: str | Path, output_dir: str | Path, config: dict,
                  rebuild: bool = False) -> dict:
    """Create sorted ID-only manifests; reject changes to an already frozen split."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    inputs = {p.name: file_hash(p) for p in [raw_dir / f'train_source{s}.tsv' for s in (1, 2, 3)] + [raw_dir / 'train_ground_truth.tsv']}
    existing = None
    if output_dir.exists():
        existing = verify_manifests(output_dir, config)
        if inputs != existing['input_hashes']:
            raise ValueError('Raw training input fingerprint changed')
        if not rebuild:
            return existing
    with tempfile.TemporaryDirectory(prefix='validation-build-', dir=output_dir.parent) as tmp:
        stage = Path(tmp)
        with closing(sqlite3.connect(stage / 'scratch.sqlite')) as db:
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('PRAGMA synchronous=OFF')
            db.execute('PRAGMA cache_size=-65536')
            db.execute('PRAGMA temp_store=FILE')
            db.execute('CREATE TABLE s1 (id TEXT, country TEXT, part TEXT)')
            db.execute('CREATE TABLE truth (id TEXT, part TEXT, n2 INTEGER, n3 INTEGER)')
            db.execute('CREATE TABLE targets (id TEXT, part TEXT, owner TEXT, source INTEGER)')
            LOGGER.info('Reading Source 1 IDs and countries')
            for batch in _batches(iter_source(raw_dir / 'train_source1.tsv', 1)):
                db.executemany('INSERT INTO s1 VALUES (?,?,?)', ((r['entity_id'], r['country'], partition_for(r['entity_id'], config)) for r in batch))
                db.commit()
            db.execute('CREATE UNIQUE INDEX s1_ids ON s1(id)')
            LOGGER.info('Assigning labelled targets to their reference partition')
            for batch in _batches(iter_ground_truth(raw_dir / 'train_ground_truth.tsv')):
                truth, targets = [], []
                for entity_id, ids in batch:
                    part = partition_for(entity_id, config)
                    n2 = sum(i.startswith('S2-') for i in ids)
                    truth.append((entity_id, part, n2, len(ids) - n2))
                    targets.extend((i, part, entity_id, int(i[1])) for i in ids)
                db.executemany('INSERT INTO truth VALUES (?,?,?,?)', truth)
                db.executemany('INSERT INTO targets VALUES (?,?,?,?)', targets)
                db.commit()
            db.execute('CREATE UNIQUE INDEX truth_ids ON truth(id)')
            db.execute('CREATE UNIQUE INDEX target_ids ON targets(id)')
            checks = {
                'source1_missing_truth': _count(db, 'SELECT COUNT(*) FROM s1 WHERE id NOT IN (SELECT id FROM truth)'),
                'truth_unknown_source1': _count(db, 'SELECT COUNT(*) FROM truth WHERE id NOT IN (SELECT id FROM s1)'),
                'positive_partition_mismatches': _count(db, 'SELECT COUNT(*) FROM targets t JOIN s1 s ON s.id=t.owner WHERE t.part<>s.part'),
            }
            for source in (2, 3):
                LOGGER.info('Reading Source %s IDs and assigning unlinked distractors', source)
                db.execute('CREATE TABLE pool (id TEXT)')
                for batch in _batches(iter_source(raw_dir / f'train_source{source}.tsv', source)):
                    db.executemany('INSERT INTO pool VALUES (?)', ((r['entity_id'],) for r in batch))
                    db.commit()
                db.execute('CREATE UNIQUE INDEX pool_ids ON pool(id)')
                checks[f'source{source}_unknown_positive_targets'] = db.execute('SELECT COUNT(*) FROM targets WHERE source=? AND id NOT IN (SELECT id FROM pool)', (source,)).fetchone()[0]
                db.create_function('distractor_partition', 1, lambda value: partition_for(value, config, 'distractor'))
                db.execute('INSERT INTO targets SELECT id,distractor_partition(id),NULL,? FROM pool WHERE id NOT IN (SELECT id FROM targets)', (source,))
                checks[f'source{source}_unassigned_records'] = _count(db, 'SELECT COUNT(*) FROM pool WHERE id NOT IN (SELECT id FROM targets)')
                checks[f'source{source}_partition_count_difference'] = db.execute('SELECT COUNT(*) FROM targets WHERE source=?', (source,)).fetchone()[0] - _count(db, 'SELECT COUNT(*) FROM pool')
                db.execute('DROP TABLE pool')
                db.commit()
            for source in (1, 2, 3):
                table = 's1' if source == 1 else 'targets'
                restriction = '' if source == 1 else f'WHERE source={source}'
                checks[f'source{source}_overlap'] = _count(db, f'SELECT COUNT(*) FROM (SELECT id FROM {table} {restriction} GROUP BY id HAVING COUNT(DISTINCT part)>1)')
            if any(checks.values()):
                raise ValueError(f'Split integrity checks failed: {checks}')
            stats = _statistics(db)
            manifest_hashes = {}
            for part in PARTITIONS:
                for source in (1, 2, 3):
                    name = f'{part}_source{source}.tsv'
                    LOGGER.info('Writing %s', name)
                    query = 'SELECT id FROM s1 WHERE part=? ORDER BY id' if source == 1 else 'SELECT id FROM targets WHERE part=? AND source=? ORDER BY id'
                    args = (part,) if source == 1 else (part, source)
                    with (stage / name).open('w', encoding='utf-8', newline='') as stream:
                        stream.write('entity_id\n')
                        stream.writelines(f'{row[0]}\n' for row in db.execute(query, args))
                    manifest_hashes[name] = file_hash(stage / name)
            metadata = {'configuration': {k: v for k, v in config.items() if k != 'expected_fingerprint'},
                        'input_hashes': inputs, 'manifest_hashes': manifest_hashes,
                        'integrity_checks': checks, **stats}
            metadata['fingerprint'] = object_hash(metadata)
            if config['expected_fingerprint'] not in (None, metadata['fingerprint']):
                raise ValueError('Regenerated split differs from committed fingerprint')
            if existing is not None and existing != metadata:
                raise ValueError('Regenerated split differs from existing frozen metadata')
            (stage / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
        if existing is None:
            (stage / 'scratch.sqlite').unlink()
            stage.rename(output_dir)
    return metadata
