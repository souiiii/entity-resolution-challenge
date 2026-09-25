"""Development-sample feature diagnostics: no model, no thresholds, development labels only."""
from __future__ import annotations

import json
import logging
import resource
import shutil
import time
from pathlib import Path

import numpy as np

from src.blocking.channels import load_config
from src.blocking.evaluate import load_truth, select_queries
from src.blocking.store import RecordStore
from src.data.validation import read_config, verify_manifests
from src.features.extract import iter_features
from src.features.pairwise import FEATURES_VERSION, feature_names
from src.features.text import ensure_text

LOGGER = logging.getLogger(__name__)


def auc(values: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUC over defined values, ties counted as one half."""
    defined = ~np.isnan(values)
    v, y = values[defined], labels[defined]
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average_rank = ends - (counts - 1) / 2
    rank_sum = float(average_rank[inverse][y].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def describe(values: np.ndarray, quantiles: list[int]) -> dict:
    defined = values[~np.isnan(values)]
    result = {'pairs': int(len(values)), 'coverage': float(len(defined) / len(values)) if len(values) else None}
    if len(defined):
        result.update({'mean': float(defined.mean()), 'nonzero_rate': float((defined != 0).mean()),
                       **{f'p{q}': float(v) for q, v in zip(quantiles, np.percentile(defined, quantiles))}})
    return result


def diagnostics(matrix: np.ndarray, labels: np.ndarray, names: tuple[str, ...], quantiles: list[int],
                full: bool = True) -> dict:
    result = {}
    for i, name in enumerate(names):
        column = matrix[:, i]
        entry = {'auc': auc(column, labels)}
        if full:
            entry['positive'] = describe(column[labels], quantiles)
            entry['negative'] = describe(column[~labels], quantiles)
        else:
            for side, mask in (('positive', labels), ('negative', ~labels)):
                defined = column[mask][~np.isnan(column[mask])]
                entry[f'{side}_mean'] = float(defined.mean()) if len(defined) else None
        result[name] = entry
    return result


def run_feature_analysis(features_config: dict, raw_dir: str | Path, split_dir: str | Path,
                         validation_config: str | Path, store_dir: str | Path, index_root: str | Path,
                         text_root: str | Path, project_root: str | Path, limit: int | None = None) -> dict:
    """Extract features for the development hash sample and summarize them against development labels."""
    started = time.monotonic()
    raw_dir, split_dir, store_dir, text_root = Path(raw_dir), Path(split_dir), Path(store_dir), Path(text_root)
    metadata = verify_manifests(split_dir, read_config(validation_config))
    blocking = load_config(Path(project_root) / features_config['blocking_config'])
    reserve = int(features_config['min_free_disk_gib'] * 2**30)
    text_root.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(text_root).free
    if free_before < reserve:
        raise OSError(f'{free_before / 2**30:.2f} GiB free is below the {reserve / 2**30:.1f} GiB reserve')
    stores = {s: RecordStore(store_dir / f'train_source{s}') for s in (1, 2, 3)}
    for s, store in stores.items():
        if store.meta['input_sha256'] != metadata['input_hashes'][f'train_source{s}.tsv']:
            raise ValueError(f'Source {s} store does not match the frozen split input')
    texts = {s: ensure_text(raw_dir / f'train_source{s}.tsv', stores[s], text_root / f'train_source{s}', reserve)
             for s in (1, 2, 3)}
    rows = select_queries(stores[1], split_dir / 'development_source1.tsv', blocking.sample_salt,
                          blocking.sample_fraction)
    if limit is not None:
        rows = rows[:limit]
    truth = load_truth(raw_dir / 'train_ground_truth.tsv', stores[1], rows, {2: stores[2], 3: stores[3]})
    names = feature_names(blocking.channels)
    labels_all = [label for label in stores[1].countries]
    query_country = np.asarray(stores[1].country)[rows]
    parts, timing, worker_rss = [], {}, 0
    setup_seconds = time.monotonic() - started
    for s in (2, 3):
        source_started = time.monotonic()
        truth_owners, truth_records = truth[s]
        truth_keys = np.sort(truth_owners * stores[s].size + truth_records)
        matrices, positives, owners_all, selection, cpu = [], [], [], 0.0, 0.0
        stream_started = time.monotonic()
        for batch in iter_features(blocking, stores[1], texts[1], stores[s], texts[s], rows, index_root,
                                   features_config['batch_size'], features_config['workers']):
            keys = batch.owners * stores[s].size + batch.records
            at = np.minimum(np.searchsorted(truth_keys, keys), max(len(truth_keys) - 1, 0))
            positives.append(truth_keys[at] == keys if len(truth_keys) else np.zeros(len(keys), bool))
            matrices.append(batch.features)
            owners_all.append(batch.owners)
            selection += batch.selection_seconds
            cpu += batch.feature_seconds
            worker_rss = max(worker_rss, batch.worker_peak_rss_kib)
        matrix = np.concatenate(matrices)
        label = np.concatenate(positives)
        owners = np.concatenate(owners_all)
        parts.append((s, matrix, label, owners))
        timing[f'source{s}'] = {'pairs': int(len(label)), 'wall_seconds': round(time.monotonic() - source_started, 1),
                                'streaming_wall_seconds': round(time.monotonic() - stream_started, 1),
                                'selection_and_text_seconds': round(selection, 1),
                                'feature_cpu_seconds': round(cpu, 1), 'truth_links': int(len(truth_owners)),
                                'positive_pairs': int(label.sum())}
        LOGGER.info('S%s: %s pairs, %s positives', s, len(label), int(label.sum()))
    for text in texts.values():
        text.close()
    extraction_seconds = time.monotonic() - started
    matrix = np.concatenate([p[1] for p in parts])
    label = np.concatenate([p[2] for p in parts])
    source = np.concatenate([np.full(len(p[2]), p[0], np.int8) for p in parts])
    country = query_country[np.concatenate([p[3] for p in parts])]
    del parts
    quantiles = features_config['quantiles']
    groups = {'all': np.ones(len(label), bool), 'source2': source == 2, 'source3': source == 3}
    groups.update({f'country:{labels_all[c]}': country == c for c in np.unique(query_country)})
    report_groups = {name: diagnostics(matrix if name == 'all' else matrix[mask], label if name == 'all' else label[mask],
                                       names, quantiles) for name, mask in groups.items()}
    for s in (2, 3):
        for c in np.unique(query_country):
            mask = (source == s) & (country == c)
            report_groups[f'source{s}:{labels_all[c]}'] = diagnostics(matrix[mask], label[mask], names, quantiles, False)
    counts = {name: {'pairs': int(mask.sum()), 'positive': int(label[mask].sum()),
                     'negative': int((~label[mask]).sum())} for name, mask in groups.items()}
    total_pairs = int(len(label))
    wall = sum(t['streaming_wall_seconds'] for t in timing.values())
    cpu = sum(t['feature_cpu_seconds'] for t in timing.values())
    selection = sum(t['selection_and_text_seconds'] for t in timing.values())
    full_pairs = features_config['full_development_candidates']
    offsets_bytes = sum((text_root / f'train_source{s}' / 'offsets.bin').stat().st_size for s in (1, 2, 3))
    return {
        'features_version': FEATURES_VERSION, 'config': features_config, 'blocking_version': blocking.version,
        'split_fingerprint': metadata['fingerprint'], 'partition': 'development', 'validation_labels_used': False,
        'queries': int(len(rows)), 'feature_names': list(names), 'counts': counts, 'groups': report_groups,
        'performance': {
            'pairs': total_pairs, 'setup_seconds': round(setup_seconds, 1),
            'extraction_wall_seconds': round(extraction_seconds, 1), 'streaming_wall_seconds': round(wall, 1),
            'pairs_per_streaming_second': round(total_pairs / wall, 1), 'feature_cpu_seconds': round(cpu, 1),
            'pairs_per_feature_cpu_second': round(total_pairs / cpu, 1) if cpu else None,
            'selection_and_text_seconds': round(selection, 1), 'per_source': timing,
            'workers': features_config['workers'],
            'projected_full_development_wall_hours': round(full_pairs / (total_pairs / wall) / 3600, 2),
            'projected_full_development_feature_cpu_hours': round(full_pairs * cpu / total_pairs / 3600, 2) if cpu else None,
            'projected_full_development_matrix_gib': round(full_pairs * len(names) * 4 / 2**30, 1),
            'peak_rss_mib_main': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            'peak_rss_mib_largest_worker': round(worker_rss / 1024),
            'sample_matrix_mib': round(matrix.nbytes / 2**20, 1)},
        'artifacts': {'text_offsets_bytes': int(offsets_bytes),
                      'disk_free_before_gib': round(free_before / 2**30, 2),
                      'disk_free_after_gib': round(shutil.disk_usage(text_root).free / 2**30, 2)},
        'runtime_seconds': round(time.monotonic() - started, 1)}


def write_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
