"""Development-only blocking evaluation, computed in streaming batches.

Queries are development Source 1 entities (optionally a label-independent hash
sample); targets are the complete training Source 2 and Source 3 files. Truth rows
of validation entities are discarded as they are read. Candidate pairs are
scored in memory per batch and never written, except for an optional sample
candidate file.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import resource
import shutil
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from src.blocking.candidates import SourceBlocker
from src.blocking.channels import BlockingConfig, ChannelSpec, query_keys
from src.blocking.store import STORE_VERSION, RecordStore, ensure_store
from src.data.inspect import _batches
from src.data.load import iter_ground_truth
from src.data.validation import iter_manifest, read_config, verify_manifests
from src.preprocessing.normalization import NORMALIZATION_VERSION
from src.submission.create_submission import CANDIDATE_HEADER, write_id_list_file

LOGGER = logging.getLogger(__name__)
PERCENTILES = (50, 90, 95, 99)
THRESHOLDS = (50, 100, 250, 500)
DIAGNOSTIC_SPECS = (ChannelSpec('shares_name_token', 'token', ('name',), 1),
                    ChannelSpec('shares_address_word', 'token', ('address_word',), 1),
                    ChannelSpec('shares_address_number', 'token', ('address_number',), 1))


def in_sample(entity_id: str, salt: str, fraction: float) -> bool:
    """Label-independent hash sample membership."""
    ratio = Fraction(str(fraction))
    threshold = ratio.numerator * (1 << 256) // ratio.denominator
    return int.from_bytes(hashlib.sha256(f'{salt}\0{entity_id}'.encode('utf-8')).digest(), 'big') < threshold


def select_queries(store: RecordStore, manifest: str | Path, salt: str | None = None,
                   fraction: float | None = None) -> np.ndarray:
    """Store rows of the manifest's Source 1 IDs (optionally sampled), in binary ID order."""
    ids = sorted(i for i in iter_manifest(manifest, 1) if salt is None or in_sample(i, salt, fraction))
    rows = store.index_of(np.array(ids, dtype=store.ids.dtype))
    if (rows < 0).any():
        raise ValueError('Query manifest contains IDs absent from the Source 1 store')
    return rows


def load_truth(truth_path: str | Path, query_store: RecordStore, rows: np.ndarray,
               targets: dict[int, RecordStore]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per target source, (query position, target record) links of the selected queries only."""
    position = np.full(query_store.size, -1, dtype=np.int64)
    position[rows] = np.arange(len(rows))
    parts: dict[int, list] = {source: [] for source in targets}
    for batch in _batches(iter_ground_truth(truth_path)):
        found = query_store.index_of(np.array([e for e, _ in batch], dtype=query_store.ids.dtype))
        owners = np.where(found >= 0, position[np.maximum(found, 0)], -1)
        pairs: dict[int, tuple[list, list]] = {source: ([], []) for source in targets}
        for (_, ids), owner in zip(batch, owners):
            if owner >= 0:
                for target in ids:
                    pairs[int(target[1])][0].append(owner)
                    pairs[int(target[1])][1].append(target)
        for source, (link_owners, link_ids) in pairs.items():
            if link_owners:
                records = targets[source].index_of(np.array(link_ids, dtype=targets[source].ids.dtype))
                if (records < 0).any():
                    raise ValueError(f'Truth target missing from Source {source} store')
                parts[source].append((np.array(link_owners, dtype=np.int64), records))
    result = {}
    for source, chunks in parts.items():
        owners = np.concatenate([o for o, _ in chunks]) if chunks else np.zeros(0, dtype=np.int64)
        records = np.concatenate([r for _, r in chunks]) if chunks else np.zeros(0, dtype=np.int64)
        order = np.lexsort((records, owners))
        result[source] = (owners[order], records[order])
    return result


def distribution(counts: np.ndarray) -> dict:
    counts = np.asarray(counts, dtype=np.int64)
    if not len(counts):
        return {'queries': 0}
    result = {'queries': int(len(counts)), 'total_pairs': int(counts.sum()), 'mean': float(counts.mean()),
              'zero_candidates': int((counts == 0).sum()), 'max': int(counts.max())}
    for p in PERCENTILES:
        result[f'p{p}'] = int(np.percentile(counts, p, method='inverted_cdf'))
    for t in THRESHOLDS:
        result[f'share_over_{t}'] = float((counts > t).mean())
    return result


def _recall(found: int, links: int) -> dict:
    return {'links': int(links), 'found': int(found), 'recall': found / links if links else None}


def _unified_country(query_store: RecordStore, target: RecordStore) -> tuple[np.ndarray, np.ndarray]:
    labels = sorted(set(query_store.countries) | set(target.countries))
    code = {label: i for i, label in enumerate(labels)}
    return (np.array([code[c] for c in query_store.countries], dtype=np.int64),
            np.array([code[c] for c in target.countries], dtype=np.int64))


def _shared_min_frequency(spec: ChannelSpec, query_store: RecordStore, target: RecordStore, query_rows: np.ndarray,
                          target_rows: np.ndarray, index=None) -> tuple[np.ndarray, np.ndarray]:
    """Per pair: whether any key is shared, and the smallest target frequency of a shared key."""
    n = len(query_rows)
    qo, qk = query_keys(spec, query_store, query_rows)
    to, tk = query_keys(spec, target, target_rows)
    owners = np.concatenate([qo, to])
    keys = np.concatenate([qk, tk])
    side = np.concatenate([np.zeros(len(qo), dtype=np.int8), np.ones(len(to), dtype=np.int8)])
    order = np.lexsort((side, keys, owners))
    owners, keys, side = owners[order], keys[order], side[order]
    shared = np.flatnonzero((owners[1:] == owners[:-1]) & (keys[1:] == keys[:-1]) & (side[1:] != side[:-1]))
    has = np.zeros(n, dtype=bool)
    minimum = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    if len(shared):
        has[owners[shared]] = True
        if index is not None:
            np.minimum.at(minimum, owners[shared], index.frequency(keys[shared]))
    return has, minimum


def diagnose_missed(blocker: SourceBlocker, query_store: RecordStore, query_rows: np.ndarray,
                    target_rows: np.ndarray, chunk: int = 100_000) -> dict:
    """Why true links were not retrieved: shared keys above caps, key limits, or no shared key."""
    n = len(query_rows)
    channels = {spec.name: {'shared_key': 0, 'shared_over_frequency_cap': 0, 'shared_within_cap_but_key_limit': 0}
                for _, spec, _ in blocker.channels}
    descriptors = {spec.name: 0 for spec in DIAGNOSTIC_SPECS}
    any_shared = np.zeros(n, dtype=bool)
    target = blocker.store
    for start in range(0, n, chunk):
        q, t = query_rows[start:start + chunk], target_rows[start:start + chunk]
        for _, spec, index in blocker.channels:
            has, minimum = _shared_min_frequency(spec, query_store, target, q, t, index)
            any_shared[start:start + len(q)] |= has
            channels[spec.name]['shared_key'] += int(has.sum())
            channels[spec.name]['shared_over_frequency_cap'] += int((has & (minimum > spec.max_frequency)).sum())
            channels[spec.name]['shared_within_cap_but_key_limit'] += int((has & (minimum <= spec.max_frequency)).sum())
        for spec in DIAGNOSTIC_SPECS:
            descriptors[spec.name] += int(_shared_min_frequency(spec, query_store, target, q, t)[0].sum())
    empty_address = int((np.diff(target.address_offsets)[target_rows] == 0).sum())
    return {'missed_links': int(n), 'no_shared_key_in_any_channel': int((~any_shared).sum()),
            'shared_key_only_above_limits': int(any_shared.sum()), 'channels': channels,
            'missed_links_sharing': descriptors, 'missed_links_with_empty_target_address': empty_address}


def evaluate_source(config: BlockingConfig, query_store: RecordStore, rows: np.ndarray, target: RecordStore,
                    truth_owners: np.ndarray, truth_records: np.ndarray, index_root: str | Path,
                    keep_candidates: int = 0) -> dict:
    """Stream all query batches against one target source and accumulate metrics."""
    started = time.monotonic()
    blocker = SourceBlocker(config, target, index_root, int(config.min_free_disk_gib * 2**30))
    index_seconds = time.monotonic() - started
    n = len(rows)
    counts = np.zeros(n, dtype=np.int64)
    truth_key = truth_owners * target.size + truth_records
    truth_mask = np.zeros(len(truth_key), dtype=np.uint32)
    names = [spec.name for _, spec, _ in blocker.channels]
    bits = {spec.name: bit for bit, spec, _ in blocker.channels}
    totals = {name: {'standalone_candidates': 0, 'incremental_candidates': 0, 'queries_capped': 0,
                     'queries_limited': 0, 'largest_bucket': 0} for name in names}
    query_codes, target_codes = _unified_country(query_store, target)
    query_country = query_codes[np.asarray(query_store.country)[rows]]
    target_country = target_codes[np.asarray(target.country)]
    cross_country = 0
    kept = []
    for start in range(0, n, config.batch_size):
        stop = min(n, start + config.batch_size)
        batch = blocker.candidates(query_store, rows[start:stop])
        owners = batch.owners + start
        counts[start:stop] = np.bincount(batch.owners, minlength=stop - start)
        cross_country += int((query_country[owners] != target_country[batch.records]).sum())
        keys = owners * target.size + batch.records
        order = np.argsort(keys)
        sorted_keys, sorted_masks = keys[order], batch.masks[order]
        lo, hi = np.searchsorted(truth_owners, [start, stop])
        wanted = truth_key[lo:hi]
        if len(sorted_keys):
            at = np.minimum(np.searchsorted(sorted_keys, wanted), len(sorted_keys) - 1)
            truth_mask[lo:hi] = np.where(sorted_keys[at] == wanted, sorted_masks[at], 0)
        lowest = batch.masks & (~batch.masks + np.uint32(1))
        for name in names:
            bit = np.uint32(1 << bits[name])
            totals[name]['standalone_candidates'] += int(((batch.masks & bit) != 0).sum())
            totals[name]['incremental_candidates'] += int((lowest == bit).sum())
            for key in ('queries_capped', 'queries_limited'):
                totals[name][key] += batch.channel_stats[name][key]
            totals[name]['largest_bucket'] = max(totals[name]['largest_bucket'],
                                                 batch.channel_stats[name]['largest_bucket'])
        if keep_candidates and start < keep_candidates:
            keep = owners < keep_candidates
            kept.append((owners[keep], batch.records[keep]))
        LOGGER.info('S%s queries %s/%s', target.source, stop, n)
    found = truth_mask != 0
    lowest = truth_mask & (~truth_mask + np.uint32(1))
    for name in names:
        bit = np.uint32(1 << bits[name])
        totals[name]['standalone_true'] = int(((truth_mask & bit) != 0).sum())
        totals[name]['incremental_true'] = int((lowest == bit).sum())
        totals[name]['standalone_recall'] = totals[name]['standalone_true'] / len(truth_mask) if len(truth_mask) else None
    missed = diagnose_missed(blocker, query_store, rows[truth_owners[~found]], truth_records[~found])
    link_country_mismatch = int((query_country[truth_owners] != target_country[truth_records]).sum())
    result = {'counts': counts, 'found_per_query': np.bincount(truth_owners[found], minlength=n),
              'truth_per_query': np.bincount(truth_owners, minlength=n), 'truth_found': found,
              'truth_owners': truth_owners, 'channels': totals, 'missed': missed,
              'cross_country_candidates': cross_country, 'links_crossing_country': link_country_mismatch,
              'index_bytes': blocker.index_bytes, 'index_postings': {spec.name: int(len(index.keys))
                                                                      for _, spec, index in blocker.channels},
              'index_build_seconds': blocker.build_seconds, 'index_seconds': round(index_seconds, 1),
              'seconds': round(time.monotonic() - started, 1), 'kept': kept}
    del blocker
    gc.collect()
    return result


def _entity_summary(per_source: dict[int, dict], countries: np.ndarray, labels: list[str]) -> dict:
    truth = sum(r['truth_per_query'] for r in per_source.values())
    found = sum(r['found_per_query'] for r in per_source.values())
    counts = sum(r['counts'] for r in per_source.values())
    linked = truth > 0
    summary = {'queries': int(len(truth)), 'with_links': int(linked.sum()),
               'all_links_recovered': int((linked & (found == truth)).sum()),
               'at_least_one_recovered': int((linked & (found > 0)).sum()),
               'zero_recovered': int((linked & (found == 0)).sum()),
               'singletons': int((~linked).sum()),
               'singleton_candidates': distribution(counts[~linked])}
    summary['by_country'] = {labels[c]: {'with_links': int((linked & (countries == c)).sum()),
                                         'all_links_recovered': int((linked & (found == truth) & (countries == c)).sum()),
                                         'zero_recovered': int((linked & (found == 0) & (countries == c)).sum())}
                             for c in np.unique(countries)}
    return summary


def run_blocking(config: BlockingConfig, raw_dir: str | Path, split_dir: str | Path, validation_config: str | Path,
                 store_dir: str | Path, index_root: str | Path, sample: bool,
                 sample_candidates_path: str | Path | None = None, sample_candidate_rows: int = 2_000) -> dict:
    """Evaluate the configured channels on development queries; returns a JSON-serializable report."""
    started = time.monotonic()
    raw_dir, split_dir, store_dir = Path(raw_dir), Path(split_dir), Path(store_dir)
    metadata = verify_manifests(split_dir, read_config(validation_config))
    reserve = int(config.min_free_disk_gib * 2**30)
    store_dir.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(store_dir).free
    if free_before < reserve:
        raise OSError(f'Only {free_before / 2**30:.2f} GiB free; configured reserve is {config.min_free_disk_gib} GiB')
    stores = {s: ensure_store(raw_dir / f'train_source{s}.tsv', s, store_dir / f'train_source{s}', config.workers,
                              min_free_bytes=reserve) for s in (1, 2, 3)}
    for s, store in stores.items():
        if store.meta['input_sha256'] != metadata['input_hashes'][f'train_source{s}.tsv']:
            raise ValueError(f'Source {s} store does not match the frozen split input')
    s1 = stores[1]
    rows = select_queries(s1, split_dir / 'development_source1.tsv', config.sample_salt if sample else None,
                          config.sample_fraction if sample else None)
    truth = load_truth(raw_dir / 'train_ground_truth.tsv', s1, rows, {2: stores[2], 3: stores[3]})
    links = {f'source{s}': int(len(truth[s][0])) for s in (2, 3)}
    if not sample:
        expected = {f'source{s}': metadata['partitions']['development']['targets'][str(s)]['links'] for s in (2, 3)}
        if links != expected:
            raise ValueError(f'Development link counts {links} differ from frozen counts {expected}')
    LOGGER.info('%s queries, links %s', len(rows), links)
    labels = s1.countries
    countries = np.asarray(s1.country)[rows]
    per_source = {}
    for s in (2, 3):
        per_source[s] = evaluate_source(config, s1, rows, stores[s], *truth[s], index_root,
                                        keep_candidates=sample_candidate_rows if sample_candidates_path else 0)
    ids = np.asarray(s1.ids)[rows]
    if sample_candidates_path is not None:
        write_sample_candidates(sample_candidates_path, ids, per_source, stores, sample_candidate_rows)
    report = _report(config, metadata, sample, rows, ids, countries, labels, per_source, stores, links)
    report['resources'].update({'disk_free_before_gib': round(free_before / 2**30, 2),
                                'disk_free_after_gib': round(shutil.disk_usage(store_dir).free / 2**30, 2),
                                'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                                'runtime_seconds': round(time.monotonic() - started, 1)})
    return report


def write_sample_candidates(path: str | Path, ids: np.ndarray, per_source: dict[int, dict],
                            stores: dict[int, RecordStore], limit: int) -> int:
    ids = ids[:limit]
    lists: list[list[str]] = [[] for _ in range(len(ids))]
    for s in (2, 3):
        for owners, records in per_source[s]['kept']:
            keep = owners < limit
            owners, records = owners[keep], records[keep]
            names = np.asarray(stores[s].ids)[records]
            for owner, name in zip(owners.tolist(), names.tolist()):
                lists[owner].append(name.decode('utf-8'))
    return write_id_list_file(path, CANDIDATE_HEADER, ((i.decode('utf-8'), row) for i, row in zip(ids.tolist(), lists)))


def _report(config: BlockingConfig, metadata: dict, sample: bool, rows: np.ndarray, ids: np.ndarray,
            countries: np.ndarray, labels: list[str], per_source: dict[int, dict], stores: dict[int, RecordStore],
            links: dict[str, int]) -> dict:
    recall = {}
    found_total = links_total = 0
    for s, r in per_source.items():
        recall[f'source{s}'] = _recall(r['truth_found'].sum(), len(r['truth_found']))
        found_total += int(r['truth_found'].sum())
        links_total += len(r['truth_found'])
    recall['overall'] = _recall(found_total, links_total)
    by_country = {}
    for code in np.unique(countries):
        entry, f_all, l_all = {}, 0, 0
        for s, r in per_source.items():
            mask = countries[r['truth_owners']] == code
            f, l = int(r['truth_found'][mask].sum()), int(mask.sum())
            entry[f'source{s}'] = _recall(f, l)
            f_all, l_all = f_all + f, l_all + l
        entry['overall'] = _recall(f_all, l_all)
        by_country[labels[code]] = entry
    recall['by_country'] = by_country
    combined = per_source[2]['counts'] + per_source[3]['counts']
    candidates = {'source2': distribution(per_source[2]['counts']), 'source3': distribution(per_source[3]['counts']),
                  'combined': distribution(combined),
                  'combined_by_country': {labels[c]: distribution(combined[countries == c]) for c in np.unique(countries)}}
    top = np.argsort(-combined, kind='stable')[:10]
    largest = [{'source1_entity_id': ids[i].decode('utf-8'), 'candidates': int(combined[i]),
                'source2': int(per_source[2]['counts'][i]), 'source3': int(per_source[3]['counts'][i])} for i in top]
    channels = []
    for spec in config.channels:
        entry = {'name': spec.name, 'kind': spec.kind, 'parts': list(spec.parts), 'symmetric': spec.symmetric,
                 'max_frequency': spec.max_frequency, 'max_keys': spec.max_keys, 'sources': list(spec.sources)}
        for s, r in per_source.items():
            if spec.name in r['channels']:
                entry[f'source{s}'] = {**r['channels'][spec.name], 'index_postings': r['index_postings'][spec.name],
                                       'index_bytes': r['index_bytes'][spec.name],
                                       'index_build_seconds': r['index_build_seconds'][spec.name]}
        channels.append(entry)
    return {'blocking_version': config.version, 'mode': 'sample' if sample else 'full',
            'partition': 'development', 'validation_labels_used': False,
            'split_fingerprint': metadata['fingerprint'], 'normalization_version': NORMALIZATION_VERSION,
            'store_version': STORE_VERSION, 'config': config.raw,
            'queries': int(len(rows)), 'query_countries': {labels[c]: int((countries == c).sum()) for c in np.unique(countries)},
            'links': links, 'recall': recall, 'entities': _entity_summary(per_source, countries, labels),
            'candidates': candidates, 'largest_candidate_sets': largest, 'channels': channels,
            'missed': {f'source{s}': r['missed'] for s, r in per_source.items()},
            'country_isolation': {f'source{s}': {'cross_country_candidates': r['cross_country_candidates'],
                                                 'true_links_crossing_country': r['links_crossing_country']}
                                  for s, r in per_source.items()},
            'resources': {'store_bytes': {f'source{s}': st.meta['bytes'] for s, st in stores.items()},
                          'index_postings': {f'source{s}': r['index_postings'] for s, r in per_source.items()},
                          'index_bytes': {f'source{s}': int(sum(r['index_bytes'].values())) for s, r in per_source.items()},
                          'source_seconds': {f'source{s}': r['seconds'] for s, r in per_source.items()},
                          'index_seconds': {f'source{s}': r['index_seconds'] for s, r in per_source.items()}}}
