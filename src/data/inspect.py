"""Streaming, disk-backed data audit. No normalization or matching is performed."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from itertools import islice
from pathlib import Path
from typing import Any

from src.data.load import SOURCE_COLUMNS, GROUND_TRUTH_COLUMNS, iter_source, iter_ground_truth

LOGGER = logging.getLogger(__name__)
BATCH_SIZE = 20_000
DEVANAGARI = re.compile('[\u0900-\u097f]')
DIGIT = re.compile(r'\d')
NULL_LIKE = {'na', 'n/a', 'nan', 'null', 'none', '<na>'}


def _batches(rows: Any, size: int = BATCH_SIZE) -> Any:
    iterator = iter(rows)
    while batch := list(islice(iterator, size)):
        yield batch


def _distribution(histogram: Counter[int]) -> dict[str, int | float]:
    total = sum(histogram.values())
    if not total:
        return {}
    result: dict[str, int | float] = {
        'min': min(histogram), 'max': max(histogram),
        'mean': sum(k * v for k, v in histogram.items()) / total,
    }
    for label, q in [('median', .5), ('p95', .95)]:
        rank, count = math.ceil(q * total), 0
        for value, frequency in sorted(histogram.items()):
            count += frequency
            if count >= rank:
                result[label] = value
                break
    return result


def preview_table(path: str | Path, n: int = 5) -> list[dict[str, str]]:
    """Read at most n data rows, without first loading the entire table."""
    if n < 0:
        raise ValueError('n must be nonnegative')
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(islice(csv.DictReader(stream, delimiter='\t'), n))


def _scalar(db: sqlite3.Connection, sql: str) -> int:
    return int(db.execute(sql).fetchone()[0])


def _profile_source(db: sqlite3.Connection, path: Path, table: str, source: int) -> dict:
    # Table names are generated internally from split/source, never user SQL.
    db.execute(f'CREATE TABLE {table} (entity_id TEXT, business_name TEXT, business_address TEXT, country TEXT)')
    stats = {col: Counter() for col in SOURCE_COLUMNS}
    lengths = {col: Counter() for col in SOURCE_COLUMNS}
    countries: Counter[str] = Counter()
    examples: dict[str, dict] = {}
    count = 0
    for batch in _batches(iter_source(path, source)):
        db.executemany(f'INSERT INTO {table} VALUES (?,?,?,?)',
                       (tuple(row[col] for col in SOURCE_COLUMNS) for row in batch))
        for row in batch:
            count += 1
            countries[row['country']] += 1
            for col, value in row.items():
                lengths[col][len(value)] += 1
                stat = stats[col]
                stat['empty'] += value == ''
                stat['whitespace_only'] += value != '' and value.isspace()
                stat['boundary_whitespace'] += value != value.strip()
                stat['null_like_literal'] += value.lower() in NULL_LIKE
                stat['non_ascii'] += not value.isascii()
            for col in ('business_name', 'business_address'):
                value = row[col]
                stats[col]['devanagari'] += bool(DEVANAGARI.search(value))
                if not value.isascii():
                    examples.setdefault(f"{row['country']}_{col}_non_ascii", row)
            stats['business_address']['contains_digit'] += bool(DIGIT.search(row['business_address']))
        db.commit()
    LOGGER.info('%s: %s rows read; auditing exact duplicates', table, f'{count:,}')
    db.execute(f'CREATE INDEX {table}_ids ON {table}(entity_id)')
    unique_ids = _scalar(db, f'SELECT COUNT(DISTINCT entity_id) FROM {table}')
    duplicate_rows = 0 if unique_ids == count else _scalar(db, f'''SELECT COALESCE(SUM(n-1),0) FROM
        (SELECT COUNT(*) n FROM {table} GROUP BY entity_id,business_name,business_address,country)''')
    result = {'path': str(path), 'bytes': path.stat().st_size, 'rows': count,
              'columns': list(SOURCE_COLUMNS), 'types': {col: 'string' for col in SOURCE_COLUMNS},
              'unique_ids': unique_ids, 'duplicate_ids': count - unique_ids,
              'duplicate_rows': duplicate_rows, 'countries': dict(sorted(countries.items())),
              'fields': {}, 'examples': examples}
    for col in SOURCE_COLUMNS:
        field = dict(stats[col])
        field['missing_fraction'] = (field['empty'] + field['whitespace_only']) / count if count else 0
        field['length'] = _distribution(lengths[col])
        result['fields'][col] = field
    for col in ('business_name', 'business_address'):
        db.execute(f'CREATE TEMP TABLE counts AS SELECT {col} value, COUNT(*) n FROM {table} GROUP BY {col}')
        result['fields'][col]['unique_values'] = _scalar(db, 'SELECT COUNT(*) FROM counts')
        result['fields'][col]['duplicate_excess'] = count - result['fields'][col]['unique_values']
        result['fields'][col]['top_values'] = db.execute('SELECT value,n FROM counts ORDER BY n DESC,value LIMIT 5').fetchall()
        db.execute('DROP TABLE counts')
    result['duplicate_profiles'] = _scalar(db, f'''SELECT COALESCE(SUM(n-1),0) FROM
        (SELECT COUNT(*) n FROM {table} GROUP BY business_name,business_address,country)''')
    db.commit()
    return result


def _profile_truth(db: sqlite3.Connection, path: Path) -> dict:
    db.execute('CREATE TABLE truth (entity_id TEXT, matched_entity_ids TEXT, n2 INTEGER, n3 INTEGER)')
    db.execute('CREATE TABLE links (source1 TEXT, target TEXT)')
    counts: Counter[int] = Counter()
    joint: Counter[tuple[int, int]] = Counter()
    examples = {}
    rows = 0
    for batch in _batches(iter_ground_truth(path)):
        truth_rows, links = [], []
        for entity_id, targets in batch:
            n2 = sum(target.startswith('S2-') for target in targets)
            n3 = len(targets) - n2
            rows += 1
            counts[len(targets)] += 1
            joint[n2, n3] += 1
            category = 'zero' if not targets else 'one' if len(targets) == 1 else 'multiple'
            examples.setdefault(category, {'source1_entity_id': entity_id, 'matched_entity_ids': ','.join(targets)})
            truth_rows.append((entity_id, ','.join(targets), n2, n3))
            links.extend((entity_id, target) for target in targets)
        db.executemany('INSERT INTO truth VALUES (?,?,?,?)', truth_rows)
        db.executemany('INSERT INTO links VALUES (?,?)', links)
        db.commit()
    LOGGER.info('Ground truth: %s rows read; checking all reference IDs', f'{rows:,}')
    db.execute('CREATE INDEX truth_ids ON truth(entity_id)')
    db.execute('CREATE INDEX link_targets ON links(target)')
    result = {'path': str(path), 'bytes': path.stat().st_size, 'rows': rows,
              'columns': list(GROUND_TRUTH_COLUMNS), 'types': {col: 'string' for col in GROUND_TRUTH_COLUMNS},
              'match_count_distribution': dict(sorted(counts.items())),
              'match_count_summary': _distribution(counts),
              'joint_source_counts': {f'{a},{b}': n for (a, b), n in sorted(joint.items())},
              'examples': examples,
              'unique_ids': _scalar(db, 'SELECT COUNT(DISTINCT entity_id) FROM truth'),
              'duplicate_rows': _scalar(db, '''SELECT COALESCE(SUM(n-1),0) FROM
                  (SELECT COUNT(*) n FROM truth GROUP BY entity_id,matched_entity_ids)'''),
              'missing_source1_ids': _scalar(db, '''SELECT COUNT(*) FROM train_source1 s
                  WHERE NOT EXISTS (SELECT 1 FROM truth t WHERE t.entity_id=s.entity_id)'''),
              'unknown_source1_ids': _scalar(db, '''SELECT COUNT(*) FROM truth t
                  WHERE NOT EXISTS (SELECT 1 FROM train_source1 s WHERE t.entity_id=s.entity_id)'''),
              'targets_linked_to_multiple_source1': _scalar(db, '''SELECT COUNT(*) FROM
                  (SELECT target FROM links GROUP BY target HAVING COUNT(DISTINCT source1)>1)'''),
              'by_country': db.execute('''SELECT s.country,COUNT(*),SUM(t.n2+t.n3=0),SUM(t.n2+t.n3=1),
                  SUM(t.n2+t.n3>1),SUM(t.n2),SUM(t.n3) FROM truth t
                  JOIN train_source1 s ON s.entity_id=t.entity_id GROUP BY s.country ORDER BY s.country''').fetchall(),
              'sources': {}}
    result['duplicate_ids'] = rows - result['unique_ids']
    for source in (2, 3):
        table = f'train_source{source}'
        prefix = f'S{source}-%'
        links_count = _scalar(db, f"SELECT COUNT(*) FROM links WHERE target LIKE '{prefix}'")
        unknown = _scalar(db, f"""SELECT COUNT(*) FROM links l WHERE target LIKE '{prefix}'
            AND NOT EXISTS (SELECT 1 FROM {table} s WHERE s.entity_id=l.target)""")
        unique_targets = _scalar(db, f'''SELECT COUNT(DISTINCT target) FROM links l
            WHERE target LIKE '{prefix}' AND EXISTS
            (SELECT 1 FROM {table} s WHERE s.entity_id=l.target)''')
        # These are raw equality diagnostics on supplied positive labels, not matching features.
        raw_agreement = db.execute(f'''SELECT COUNT(*),
            COALESCE(SUM(a.country<>b.country),0),COALESCE(SUM(a.business_name=b.business_name),0),
            COALESCE(SUM(a.business_address=b.business_address),0),
            COALESCE(SUM(a.business_name=b.business_name AND a.business_address=b.business_address),0)
            FROM links l JOIN train_source1 a ON a.entity_id=l.source1
            JOIN {table} b ON b.entity_id=l.target''').fetchone()
        result['sources'][str(source)] = {
            'links': links_count, 'unknown_targets': unknown, 'unique_linked_targets': unique_targets,
            'unlinked_records': _scalar(db, f'''SELECT COUNT(*) FROM {table} s
                WHERE NOT EXISTS (SELECT 1 FROM links l WHERE l.target=s.entity_id)'''),
            'entities_with_matches': _scalar(db, f'SELECT COUNT(*) FROM truth WHERE n{source}>0'),
            'raw_agreement': dict(zip(('joined_links', 'country_mismatch', 'equal_name', 'equal_address', 'equal_both'), raw_agreement)),
        }
    db.commit()
    return result


def inspect_challenge(raw_dir: str | Path, work_dir: str | Path,
                      output: str | Path | None = None) -> dict:
    """Audit every row with bounded batches and temporary SQLite storage.

    Uses several GB of scratch disk for exact duplicate/reference checks. The raw
    TSVs are parsed once each. Additional passes operate on scratch tables, not
    the original files. A checkpoint JSON is written after each completed file.
    """
    raw_dir, work_dir = Path(raw_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {'complete': False, 'sources': {}, 'ground_truth': {}}

    def checkpoint() -> None:
        if output is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    checkpoint()
    with tempfile.TemporaryDirectory(prefix='phase1-', dir=work_dir) as directory:
        with closing(sqlite3.connect(Path(directory) / 'audit.sqlite')) as db:
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('PRAGMA synchronous=OFF')
            db.execute('PRAGMA cache_size=-65536')
            db.execute('PRAGMA temp_store=FILE')
            for split in ('train', 'test'):
                for source in (1, 2, 3):
                    table = f'{split}_source{source}'
                    report['sources'][table] = _profile_source(db, raw_dir / split / f'{table}.tsv', table, source)
                    checkpoint()
            report['train_test_id_overlap'] = {
                str(source): _scalar(db, f'''SELECT COUNT(*) FROM train_source{source} a
                    JOIN test_source{source} b ON a.entity_id=b.entity_id''') for source in (1, 2, 3)
            }
            report['ground_truth'] = _profile_truth(db, raw_dir / 'train/train_ground_truth.tsv')
            report['complete'] = True
            checkpoint()
    return report


def input_hashes(raw_dir: str | Path, official_dir: str | Path) -> dict[str, str]:
    """Fingerprint inputs for an explicit before/after immutability check."""
    paths = sorted(Path(raw_dir).rglob('*.tsv')) + sorted(p for p in Path(official_dir).rglob('*') if p.is_file())
    result = {}
    for path in paths:
        with path.open('rb') as stream:
            result[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result
