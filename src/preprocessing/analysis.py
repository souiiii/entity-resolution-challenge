"""Development-only, disk-backed diagnostics for the canonical normalizer.

Every measurement uses the frozen development partition: development Source 1
records, development-owned Source 2/3 records and labels of development Source 1
entities. Validation truth rows are discarded as they are read. Representations
are stored only as signed 64-bit BLAKE2b keys in a temporary SQLite database, so
no normalized copy of the raw data is written.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import tempfile
import time
import unicodedata
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from contextlib import closing, nullcontext
from pathlib import Path

from src.data.inspect import _batches
from src.data.load import iter_ground_truth, iter_source
from src.data.validation import file_hash, iter_manifest, verify_manifests
from src.preprocessing.normalization import (APOSTROPHES, KEPT_SYMBOLS, NORMALIZATION_VERSION, canonicalize, compact,
                                             fold_latin_marks)

LOGGER = logging.getLogger(__name__)
REPRESENTATIONS = ('raw', 'canonical', 'compact', 'folded', 'folded_compact')
FIELDS = {'name': 'business_name', 'address': 'business_address'}
KEYS = {'name': ('name',), 'address': ('address',), 'name_address': ('name', 'address')}
COLUMNS = tuple(f'{field}_{rep}' for field in FIELDS for rep in REPRESENTATIONS)
EXAMPLE_VARIANTS = 4
_INDIC = re.compile('[ऀ-෿]')
_DEVANAGARI = re.compile('[ऀ-ॿ]')
_MULTIPLE_WHITESPACE = re.compile(r'\s{2,}')
_CORRUPTION = re.compile('�|[ïÏ]¿½')


def _key(value: str) -> int | None:
    if not value:
        return None
    return int.from_bytes(hashlib.blake2b(value.encode('utf-8'), digest_size=8).digest(), 'big', signed=True)


def representations(raw: str) -> dict[str, str]:
    """All analysed views of one raw field; only ``canonical`` is the canonical form."""
    canonical = canonicalize(raw)
    folded = fold_latin_marks(canonical)
    return {'raw': raw, 'canonical': canonical, 'compact': compact(canonical),
            'folded': folded, 'folded_compact': compact(folded)}


def _significant(text: str) -> list[str]:
    return [c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c)[0] in 'LN' and c not in APOSTROPHES]


def field_flags(raw: str, views: dict[str, str]) -> list[str]:
    """Diagnostic flags for one raw field and its views."""
    canonical, folded = views['canonical'], views['folded']
    flags = []
    if raw:
        flags.append('nonempty')
        if not canonical:
            flags.append('empty_after_normalization')
    if canonical != raw:
        flags.append('canonical_changes')
    if folded != canonical:
        flags.append('folding_changes')
    if canonicalize(canonical) != canonical:
        flags.append('canonical_not_idempotent')
    if fold_latin_marks(folded) != folded:
        flags.append('folding_not_idempotent')
    if _INDIC.findall(unicodedata.normalize('NFD', folded)) != _INDIC.findall(unicodedata.normalize('NFD', canonical)):
        flags.append('folding_altered_indic')
    if raw.isascii():
        expected = [c.lower() for c in raw if c.isalnum()]
        if expected != [c for c in canonical if c != ' ' and c not in KEPT_SYMBOLS]:
            flags.append('letters_or_digits_lost')
    else:
        flags.append('non_ascii')
        if raw and not canonical:
            flags.append('non_ascii_empty_after_normalization')
        cleaned = _CORRUPTION.sub(' ', raw)
        expected = _significant(unicodedata.normalize('NFKC', unicodedata.normalize('NFKC', cleaned).casefold()))
        if expected != _significant(canonical):
            flags.append('letters_or_digits_lost')
        if _DEVANAGARI.search(raw):
            flags.append('devanagari')
            if _DEVANAGARI.findall(unicodedata.normalize('NFD', cleaned)) != _DEVANAGARI.findall(
                    unicodedata.normalize('NFD', canonical)):
                flags.append('devanagari_altered')
        elif _INDIC.search(raw):
            flags.append('other_indic_script')
        if any(not c.isascii() and unicodedata.name(c, '').startswith('LATIN ') for c in raw):
            flags.append('accented_or_non_ascii_latin')
        if any(unicodedata.category(c) == 'Cf' for c in raw):
            flags.append('format_character')
        if unicodedata.normalize('NFKC', raw) != raw:
            flags.append('nfkc_changes_raw')
        if _CORRUPTION.search(raw):
            flags.append('corruption_marker')
        if raw.casefold() != raw.lower():
            flags.append('casefold_differs_from_lower')
    if any(c in APOSTROPHES for c in raw):
        flags.append('apostrophe')
    if any(unicodedata.category(c) == 'Pd' for c in raw):
        flags.append('hyphen_or_dash')
    if '&' in raw:
        flags.append('ampersand')
    if _MULTIPLE_WHITESPACE.search(raw) or raw != raw.strip():
        flags.append('irregular_whitespace')
    if any(unicodedata.category(c) == 'Cc' for c in raw):
        flags.append('control_character')
    if raw != raw.lower():
        flags.append('has_uppercase')
    return flags


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    for pragma in ('journal_mode=OFF', 'synchronous=OFF', 'cache_size=-262144', 'temp_store=FILE'):
        db.execute(f'PRAGMA {pragma}')
    return db


def profile_batch(pairs: list[tuple[str, str]]) -> tuple[list[tuple], dict[str, Counter], dict[str, list[str]]]:
    """Representation keys and diagnostic flags for (name, address) pairs, in input order."""
    keys, flags = [], {field: Counter() for field in FIELDS}
    empties: dict[str, list[str]] = {field: [] for field in FIELDS}
    for values in pairs:
        row = []
        for field, raw in zip(FIELDS, values):
            views = representations(raw)
            row.extend(_key(views[rep]) for rep in REPRESENTATIONS)
            found = field_flags(raw, views)
            flags[field].update(found)
            if 'empty_after_normalization' in found and raw not in empties[field] and len(empties[field]) < 5:
                empties[field].append(raw)
        keys.append(tuple(row))
    return keys, flags, empties


class _Inline:
    def submit(self, function: Callable, *args: object) -> Future:
        future: Future = Future()
        future.set_result(function(*args))
        return future


def _ordered(pool: Executor | _Inline, function: Callable, jobs: Iterable[tuple], window: int) -> Iterator[tuple]:
    pending: deque = deque()
    for context, argument in jobs:
        pending.append((context, pool.submit(function, argument)))
        if len(pending) >= window:
            context, future = pending.popleft()
            yield context, future.result()
    while pending:
        context, future = pending.popleft()
        yield context, future.result()


def _development_records(db: sqlite3.Connection, raw_dir: Path, split_dir: Path, source: int) -> Iterator[list[dict]]:
    db.execute('CREATE TABLE member (id TEXT PRIMARY KEY) WITHOUT ROWID')
    for batch in _batches(iter_manifest(split_dir / f'development_source{source}.tsv', source)):
        db.executemany('INSERT INTO member VALUES (?)', ((i,) for i in batch))
    manifest = db.execute('SELECT COUNT(*) FROM member').fetchone()[0]
    records = 0
    for batch in _batches(iter_source(raw_dir / f'train_source{source}.tsv', source)):
        selected = [r for r in batch if db.execute('SELECT 1 FROM member WHERE id=?', (r['entity_id'],)).fetchone()]
        records += len(selected)
        yield selected
    db.execute('DROP TABLE member')
    if records != manifest:
        raise ValueError(f'Source {source}: {records} development records read, manifest has {manifest}')


def _load_source(db: sqlite3.Connection, raw_dir: Path, split_dir: Path, source: int,
                 pool: Executor | _Inline, window: int) -> dict:
    columns = ', '.join(f'{c} INTEGER' for c in COLUMNS)
    db.execute(f'CREATE TABLE r{source} (id TEXT PRIMARY KEY, country TEXT, {columns}) WITHOUT ROWID')
    insert = f'INSERT INTO r{source} VALUES ({", ".join("?" * (len(COLUMNS) + 2))})'
    flags = {field: Counter() for field in FIELDS}
    empties: dict[str, list[str]] = {field: [] for field in FIELDS}
    records = 0
    jobs = ((batch, [(r['business_name'], r['business_address']) for r in batch])
            for batch in _development_records(db, raw_dir, split_dir, source))
    for batch, (keys, batch_flags, batch_empties) in _ordered(pool, profile_batch, jobs, window):
        db.executemany(insert, ((r['entity_id'], r['country'], *k) for r, k in zip(batch, keys)))
        db.commit()
        records += len(batch)
        for field in FIELDS:
            flags[field].update(batch_flags[field])
            for raw in batch_empties[field]:
                if raw not in empties[field] and len(empties[field]) < 5:
                    empties[field].append(raw)
    LOGGER.info('Source %s: %s development records', source, records)
    return {'records': records, 'flags': {f: dict(sorted(c.items())) for f, c in flags.items()},
            'empty_after_normalization_examples': empties}


def _load_links(db: sqlite3.Connection, truth_path: Path) -> dict[str, int]:
    db.execute('CREATE TABLE links (s1 TEXT, target TEXT, source INTEGER)')
    for batch in _batches(iter_ground_truth(truth_path)):
        rows = [(entity_id, target, int(target[1])) for entity_id, targets in batch
                if targets and db.execute('SELECT 1 FROM r1 WHERE id=?', (entity_id,)).fetchone()
                for target in targets]
        db.executemany('INSERT INTO links VALUES (?,?,?)', rows)
        db.commit()
    return {f'source{s}': n for s, n in db.execute('SELECT source, COUNT(*) FROM links GROUP BY source')}


def _equal(rep: str, key: str) -> str:
    return ' AND '.join(f'a.{field}_{rep}=b.{field}_{rep}' for field in KEYS[key])


def _link_equality(db: sqlite3.Connection, source: int) -> dict:
    measures = [(key, rep) for key in KEYS for rep in REPRESENTATIONS]
    sums = ', '.join(f'COALESCE(SUM({_equal(rep, key)}), 0)' for key, rep in measures)
    query = f'''SELECT a.country, COUNT(*), SUM(b.address_raw IS NULL), SUM(a.country<>b.country), {sums}
        FROM links l JOIN r1 a ON a.id=l.s1 JOIN r{source} b ON b.id=l.target
        WHERE l.source=? GROUP BY a.country ORDER BY a.country'''
    result = {}
    for country, links, empty, mismatched, *counts in db.execute(query, (source,)):
        entry = {'links': links, 'target_address_empty': empty, 'country_mismatches': mismatched}
        for (key, rep), count in zip(measures, counts):
            entry.setdefault(key, {})[rep] = {'count': count, 'fraction': count / links}
        result[country] = entry
    total = {'links': sum(e['links'] for e in result.values()),
             'target_address_empty': sum(e['target_address_empty'] for e in result.values()),
             'country_mismatches': sum(e['country_mismatches'] for e in result.values())}
    for key, rep in measures:
        count = sum(e[key][rep]['count'] for e in result.values())
        total.setdefault(key, {})[rep] = {'count': count, 'fraction': count / total['links']}
    return {'total': total, 'by_country': result}


def _bucket_stats(rows: list[tuple]) -> dict:
    distinct, records, duplicated, largest, pairs = rows[0]
    return {'nonempty_records': records or 0, 'distinct_values': distinct,
            'records_in_duplicated_buckets': duplicated or 0,
            'duplicated_fraction': (duplicated or 0) / records if records else 0.0,
            'largest_bucket': largest or 0, 'same_key_record_pairs': pairs or 0}


_AGGREGATE = ('COUNT(*), SUM(c), SUM(CASE WHEN c>1 THEN c ELSE 0 END), MAX(c), '
              'SUM(CAST(c AS INTEGER)*(c-1)/2)')


def _collisions(db: sqlite3.Connection, link_equality: dict) -> dict:
    result = {}
    for key, fields in KEYS.items():
        for rep in REPRESENTATIONS:
            LOGGER.info('Collision statistics: %s / %s', key, rep)
            columns = [f'{field}_{rep}' for field in fields]
            k2 = columns[1] if len(columns) > 1 else '0'
            entry = {}
            for source in (1, 2, 3):
                db.execute(f'CREATE TABLE g{source} (country TEXT, k1 INTEGER, k2 INTEGER, c INTEGER, '
                           'PRIMARY KEY (country, k1, k2)) WITHOUT ROWID')
                present = ' AND '.join(f'{c} IS NOT NULL' for c in columns)
                db.execute(f'INSERT INTO g{source} SELECT country, {columns[0]}, {k2}, COUNT(*) FROM r{source} '
                           f'WHERE {present} GROUP BY 1, 2, 3')
                overall = _bucket_stats(db.execute(
                    f'SELECT {_AGGREGATE} FROM (SELECT SUM(c) c FROM g{source} GROUP BY k1, k2)').fetchall())
                countries = {country: _bucket_stats([row]) for country, *row in db.execute(
                    f'SELECT country, {_AGGREGATE} FROM g{source} GROUP BY country ORDER BY country')}
                entry[f'source{source}'] = {'overall': overall, 'by_country': countries}
            for source in (2, 3):
                pairs = db.execute(f'SELECT COALESCE(SUM(a.c*b.c), 0) FROM g1 a JOIN g{source} b '
                                   'USING (country, k1, k2)').fetchone()[0]
                true = link_equality[f'source{source}']['total'][key][rep]['count']
                entry[f'source1_to_source{source}_same_country'] = {
                    'equal_key_pairs': pairs, 'true_links_among_them': true,
                    'true_link_fraction': true / pairs if pairs else 0.0}
            for source in (1, 2, 3):
                db.execute(f'DROP TABLE g{source}')
            result.setdefault(key, {})[rep] = entry
    return result


def _example_keys(db: sqlite3.Connection) -> dict[tuple[int, str, str], dict[int, str]]:
    wanted: dict[tuple[int, str, str], dict[int, str]] = {}
    for source in (1, 2, 3):
        for field, rep, parent in (('name', 'canonical', 'raw'), ('address', 'canonical', 'raw'),
                                   ('name', 'folded', 'canonical')):
            column, merged = f'{field}_{rep}', f'{field}_{parent}'
            chosen = wanted.setdefault((source, field, rep), {})
            for (value,) in db.execute(f'SELECT {column} FROM r{source} WHERE {column} IS NOT NULL '
                                       f'GROUP BY {column} ORDER BY COUNT(*) DESC, {column} LIMIT 1'):
                chosen.setdefault(value, 'largest_bucket')
            for (value,) in db.execute(f'SELECT {column} FROM r{source} WHERE {column} IS NOT NULL GROUP BY '
                                       f'{column} ORDER BY COUNT(DISTINCT {merged}) DESC, {column} LIMIT 2'):
                chosen.setdefault(value, 'most_merged')
    return wanted


def example_batch(job: tuple[list[tuple[str, str]], dict[tuple[str, str], set[int]]]) -> list[tuple]:
    """Records of a batch whose selected view key is wanted: (field, rep, key, value, variant)."""
    pairs, targets = job
    hits = []
    for values in pairs:
        for field, raw in zip(FIELDS, values):
            canonical = canonicalize(raw)
            for rep in ('canonical', 'folded'):
                keys = targets.get((field, rep))
                if keys:
                    value = canonical if rep == 'canonical' else fold_latin_marks(canonical)
                    key = _key(value)
                    if key in keys:
                        hits.append((field, rep, key, value, raw if rep == 'canonical' else canonical))
    return hits


def _examples(db: sqlite3.Connection, raw_dir: Path, split_dir: Path, pool: Executor | _Inline, window: int) -> list[dict]:
    wanted = _example_keys(db)
    found = []
    for source in (1, 2, 3):
        kinds = {(field, rep): keys for (s, field, rep), keys in wanted.items() if s == source}
        targets = {view: set(keys) for view, keys in kinds.items()}
        collected: dict[tuple[str, str, int], dict] = {}
        jobs = ((None, ([(r['business_name'], r['business_address']) for r in batch], targets))
                for batch in _development_records(db, raw_dir, split_dir, source))
        for _, hits in _ordered(pool, example_batch, jobs, window):
            for field, rep, key, value, variant in hits:
                item = collected.setdefault((field, rep, key), {
                    'source': source, 'field': field, 'representation': rep, 'kind': kinds[field, rep][key],
                    'value': value, 'records': 0, 'variants': []})
                item['records'] += 1
                if variant not in item['variants'] and len(item['variants']) < EXAMPLE_VARIANTS:
                    item['variants'].append(variant)
        found.extend(collected.values())
    return found


def analyze_normalization(raw_dir: str | Path, split_dir: str | Path, config: dict,
                          work_dir: str | Path, include_examples: bool = True, workers: int | None = None) -> dict:
    """Measure development positive-link equality, collisions and Unicode effects."""
    started = time.monotonic()
    raw_dir, split_dir, work_dir = Path(raw_dir), Path(split_dir), Path(work_dir)
    metadata = verify_manifests(split_dir, config)
    for name, expected in metadata['input_hashes'].items():
        if file_hash(raw_dir / name) != expected:
            raise ValueError(f'Raw input differs from the frozen split: {name}')
    expected_links = {f'source{s}': metadata['partitions']['development']['targets'][str(s)]['links']
                      for s in (2, 3)}
    workers = max(1, min(6, (os.cpu_count() or 2) - 1)) if workers is None else workers
    work_dir.mkdir(parents=True, exist_ok=True)
    executor = ProcessPoolExecutor(workers) if workers > 1 else nullcontext(_Inline())
    with tempfile.TemporaryDirectory(prefix='normalization-', dir=work_dir) as tmp, executor as pool, \
            closing(_connect(Path(tmp) / 'analysis.sqlite')) as db:
        window = 2 * workers
        sources = {f'source{s}': _load_source(db, raw_dir, split_dir, s, pool, window) for s in (1, 2, 3)}
        links = _load_links(db, raw_dir / 'train_ground_truth.tsv')
        if links != expected_links:
            raise ValueError(f'Development link counts {links} differ from frozen counts {expected_links}')
        for source in (2, 3):
            joined = db.execute(f'SELECT COUNT(*) FROM links l JOIN r1 a ON a.id=l.s1 '
                                f'JOIN r{source} b ON b.id=l.target WHERE l.source=?', (source,)).fetchone()[0]
            if joined != expected_links[f'source{source}']:
                raise ValueError(f'Source {source}: {joined} links join development records')
        LOGGER.info('Development links verified: %s', links)
        equality = {f'source{s}': _link_equality(db, s) for s in (2, 3)}
        collisions = _collisions(db, equality)
        examples = _examples(db, raw_dir, split_dir, pool, window) if include_examples else []
    return {'normalization_version': NORMALIZATION_VERSION, 'split_fingerprint': metadata['fingerprint'],
            'partition': 'development', 'validation_labels_used': False,
            'development_link_counts': links, 'sources': sources, 'link_equality': equality,
            'collisions': collisions, 'examples': examples,
            'runtime_seconds': round(time.monotonic() - started, 1)}
