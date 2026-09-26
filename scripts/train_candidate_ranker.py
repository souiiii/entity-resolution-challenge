"""Development-query training and held-out candidate-budget evaluation for Phase 9."""
from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path
from dataclasses import replace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_large_bucket_recovery import ROOT, OVERFLOW, sample_context
from src.blocking.candidates import SourceBlocker
from src.blocking.channels import load_config
from src.blocking.overflow import BroadRetriever, disk_guard
from src.blocking.recovery import set_metrics, rank_per_query
from src.data.validation import file_hash
from src.features.text import SourceText
from src.matching.candidate_ranker import (CheapText, FEATURE_VERSION, feature_names, retrieval_features,
                                          sampled_pairs, training_queries, rescue_ranks, select_candidates)
from src.matching.final_candidates import iter_final_candidates, load_final_policy
from src.utils.paths import INTERIM_DIR, PROJECT_ROOT, RAW_DIR

MODEL = PROJECT_ROOT / 'output/candidate_ranker_v1.txt'


def membership(keys: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if not len(reference):
        return np.zeros(len(keys), bool)
    at = np.minimum(np.searchsorted(reference, keys), len(reference) - 1)
    return reference[at] == keys


def texts_for(stores):
    return {s: SourceText(RAW_DIR / 'train' / f'train_source{s}.tsv', stores[s],
                          INTERIM_DIR / 'features/text' / f'train_source{s}') for s in stores}


def collect(args, settings) -> None:
    started = time.monotonic()
    blocking, stores, rows, links = sample_context()
    train = training_queries(stores[1].ids[rows], settings['split_salt'], settings['training_fraction'])
    texts = texts_for(stores)
    cheap = {s: CheapText(t) for s, t in texts.items()}
    directory = ROOT / 'phase9_ranker_training'
    directory.mkdir(exist_ok=True)
    if list(directory.glob('*.npz')):
        raise FileExistsError('Training shards already exist')
    totals = {'pairs': 0, 'positives': 0, 'queries': int(train.sum())}
    lowest = disk_guard(ROOT)
    for source in (2, 3):
        blocker = SourceBlocker(blocking, stores[source], INTERIM_DIR / 'blocking/index')
        retriever = BroadRetriever(blocker, OVERFLOW, settings['overflow_maximum'])
        mask = links['source'] == source
        truth_keys = np.sort(links['owners'][mask] * stores[source].size + links['records'][mask])
        for start in range(0, len(rows), settings['batch_size']):
            positions = np.arange(start, min(len(rows), start + settings['batch_size']))
            positions = positions[train[positions]]
            if not len(positions):
                continue
            deep, broad = retriever.retrieve(stores[1], rows[positions])
            labels = membership(positions[broad.owners] * stores[source].size + broad.records, truth_keys)
            selected = sampled_pairs(broad, labels, stores[source].size, settings['negative_seed'] + start,
                                     settings['random_negatives_per_query_source'],
                                     settings['hard_negatives_per_query_source'], stores[source].rank)
            matrix, _ = retrieval_features(stores[1], stores[source], rows[positions], broad, deep, blocking.channels,
                                           cheap[1], cheap[source], selected)
            lowest = min(lowest, disk_guard(ROOT, matrix.nbytes + 5 * len(selected) + 16384))
            np.savez(directory / f's{source}-{start:06d}.npz', x=matrix, y=labels[selected],
                     owner=positions[broad.owners[selected]].astype(np.int32))
            totals['pairs'] += len(selected)
            totals['positives'] += int(labels[selected].sum())
            if start % 5000 == 0:
                logging.info('S%s training %s/%s', source, start, len(rows))
        del retriever, blocker
    report = {**totals, 'negatives': totals['pairs'] - totals['positives'], 'heldout_queries': int((~train).sum()),
              'feature_names': feature_names(blocking.channels), 'feature_version': FEATURE_VERSION,
              'config': settings, 'runtime_seconds': time.monotonic() - started,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              'min_free_disk_gib': lowest / 2**30}
    (ROOT / 'phase9_training_data.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def train_model(args, settings) -> None:
    import lightgbm as lgb

    started = time.monotonic()
    manifest = json.loads((ROOT / 'phase9_training_data.json').read_text())
    if manifest['config'] != settings or manifest['feature_version'] != FEATURE_VERSION:
        raise ValueError('Training shards do not match this configuration and feature schema')
    files = sorted((ROOT / 'phase9_ranker_training').glob('*.npz'))
    if not files:
        raise ValueError('Collect the training sample first')
    matrices, labels = [], []
    for path in files:
        with np.load(path) as data:
            matrices.append(data['x'])
            labels.append(data['y'])
    x, y = np.concatenate(matrices), np.concatenate(labels)
    del matrices, labels
    if len(y) != manifest['pairs'] or int(y.sum()) != manifest['positives']:
        raise ValueError('Training shard inventory is incomplete')
    model = lgb.LGBMClassifier(**settings['model'])
    blocking = load_config(
        PROJECT_ROOT / 'experiments/configs/blocking_v2.json')
    model.fit(x, y, feature_name=list(feature_names(blocking.channels)))
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    disk_guard(ROOT, 10 * 2**20)
    model.booster_.save_model(str(MODEL))
    report = {'rows': len(y), 'positive_rows': int(y.sum()), 'model_sha256': file_hash(MODEL),
              'runtime_seconds': time.monotonic() - started,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              'python_version': sys.version, 'lightgbm_version': lgb.__version__, 'numpy_version': np.__version__,
              'config_sha256': file_hash(args.config),
              'feature_importance': dict(zip(model.booster_.feature_name(),
                                             model.booster_.feature_importance(importance_type='gain').tolist()))}
    (ROOT / 'phase9_ranker_model.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def cache_phase8(stores, texts, rows) -> dict[int, np.ndarray]:
    policy, blocking = load_final_policy(PROJECT_ROOT / 'experiments/configs/blocking_v3.json', PROJECT_ROOT)
    blocking = replace(blocking, min_free_disk_gib=10.5)
    result = {}
    for s in (2, 3):
        path = ROOT / f'phase9_phase8_pairs_s{s}.npy'
        if not path.exists():
            arrays = []
            for batch in iter_final_candidates(policy, blocking, stores[1], texts[1], stores[s], texts[s], rows,
                    INTERIM_DIR / 'blocking/index', 500, workers=1):
                arrays.append(batch.owners * stores[s].size + batch.records)
            values = np.sort(np.concatenate(arrays))
            disk_guard(ROOT, values.nbytes + 1024)
            np.save(path, values)
        result[s] = np.load(path)
    return result


def evaluate(args, settings) -> None:
    import lightgbm as lgb

    started = time.monotonic()
    blocking, stores, rows, links = sample_context()
    texts = texts_for(stores)
    cheap = {s: CheapText(t, normalized_root=INTERIM_DIR / 'blocking/candidate-text-v1') for s, t in texts.items()}
    phase8 = cache_phase8(stores, texts, rows)
    model = lgb.Booster(model_file=str(MODEL))
    if model.feature_name() != list(feature_names(blocking.channels)):
        raise ValueError('Candidate-ranker feature mismatch')
    n = len(rows)
    train = training_queries(stores[1].ids[rows], settings['split_salt'], settings['training_fraction'])
    budgets = [0, *settings['top_k_curve']]
    counts = {k: np.zeros(n, np.int64) for k in budgets}
    found = {k: {s: np.zeros(n, np.int64) for s in (2, 3)} for k in budgets}
    ranker_found = {k: {s: np.zeros(n, np.int64) for s in (2, 3)} for k in budgets if k}
    truth = {s: np.bincount(links['owners'][links['source'] == s], minlength=n) for s in (2, 3)}
    lowest = disk_guard(ROOT)
    recovered = {k: np.zeros(len(links['owners']), bool) for k in budgets}
    rescue_rows = []
    for source in (2, 3):
        blocker = SourceBlocker(blocking, stores[source], INTERIM_DIR / 'blocking/index')
        retriever = BroadRetriever(blocker, OVERFLOW, settings['overflow_maximum'])
        link_index = np.flatnonzero(links['source'] == source)
        link_keys = links['owners'][link_index] * stores[source].size + links['records'][link_index]
        order = np.argsort(link_keys)
        truth_keys = link_keys[order]
        for start in range(0, n, settings['batch_size']):
            batch_rows = rows[start:start + settings['batch_size']]
            deep, broad = retriever.retrieve(stores[1], batch_rows)
            keys = (broad.owners + start) * stores[source].size + broad.records
            base = membership(keys, phase8[source])
            matrix, _ = retrieval_features(stores[1], stores[source], batch_rows, broad, deep,
                                           blocking.channels, cheap[1], cheap[source])
            scores = model.predict(matrix, num_threads=3)
            ranks = rescue_ranks(broad.owners, scores, base, broad.min_frequency, stores[source].rank[broad.records])
            labels = membership(keys, truth_keys)
            standalone_ranks = rank_per_query(broad.owners, scores, broad.min_frequency, stores[source].rank[broad.records])
            for k in ranker_found:
                hit = labels & (standalone_ranks <= k)
                ranker_found[k][source][start:start + len(batch_rows)] = np.bincount(broad.owners[hit], minlength=len(batch_rows))
            for k in budgets:
                keep = select_candidates(base, ranks, k)
                counts[k][start:start + len(batch_rows)] = np.bincount(broad.owners[keep], minlength=len(batch_rows)) + counts[k][start:start + len(batch_rows)]
                hit = keep & labels
                found[k][source][start:start + len(batch_rows)] = np.bincount(broad.owners[hit], minlength=len(batch_rows))
                recovered[k][link_index[order[np.searchsorted(truth_keys, keys[hit])]]] = True
            chosen = (ranks > 0) & (ranks <= 10)
            rescue_rows.append(np.column_stack((np.full(chosen.sum(), source), broad.owners[chosen] + start,
                                                broad.records[chosen], ranks[chosen], labels[chosen])))
            lowest = min(lowest, disk_guard(ROOT))
            if start % 5000 == 0:
                logging.info('S%s evaluation %s/%s', source, start, n)
        del retriever, blocker
    countries = np.asarray(stores[1].country)[rows]
    reachable = np.load(ROOT / 'phase9_broad_links.npz')['recovered']
    results = {}
    for subset, mask in [('sample', np.ones(n, bool)), ('heldout', ~train), ('training', train)]:
        results[subset] = {}
        reference = None
        for k in budgets:
            metrics = set_metrics(counts[k][mask], {s: found[k][s][mask] for s in (2, 3)},
                {s: truth[s][mask] for s in (2, 3)}, countries[mask], stores[1].countries, reference)
            if k == 0:
                reference = metrics
            else:
                metrics['rescued_precision'] = metrics['incremental_true_links'] / metrics['incremental_pairs']
            broad_positives = int((reachable & mask[links['owners']]).sum())
            metrics['recall_among_reachable_positives'] = metrics['found_links'] / broad_positives
            if k:
                metrics['rescue_recall_among_reachable_misses'] = metrics['incremental_true_links'] / (broad_positives - reference['found_links'])
            results[subset][str(k)] = metrics
    baseline = json.loads((ROOT / 'blocking-v3_sample_report.json').read_text())['stages']['deep8']
    if results['sample']['0']['pairs'] != baseline['pairs'] or results['sample']['0']['found_links'] != baseline['found_links']:
        raise AssertionError('Phase 8 candidate regression')
    standalone = {}
    for subset, mask in [('sample', np.ones(n, bool)), ('heldout', ~train)]:
        standalone[subset] = {}
        for k, found_by_source in ranker_found.items():
            standalone[subset][str(k)] = {}
            for source in (2, 3):
                denominator = int((reachable & mask[links['owners']] & (links['source'] == source)).sum())
                standalone[subset][str(k)][f'S{source}'] = {
                    'found_links': int(found_by_source[source][mask].sum()),
                    'reachable_links': denominator,
                    'recall_among_reachable': float(found_by_source[source][mask].sum() / denominator)}
    report = {'standalone_top_k': standalone, 'config': settings, 'feature_version': FEATURE_VERSION, 'model_sha256': file_hash(MODEL),
              'queries': n, 'training_queries': int(train.sum()), 'heldout_queries': int((~train).sum()),
              'results': results, 'runtime_seconds': time.monotonic() - started,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              'min_free_disk_gib': lowest / 2**30, 'validation_labels_used': False}
    (ROOT / 'phase9_ranker_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    disk_guard(ROOT, sum(v.nbytes for v in recovered.values()) + sum(v.nbytes for v in rescue_rows) + 16384)
    np.savez(ROOT / 'phase9_ranker_recovered.npz', **{f'k{k}': r for k, r in recovered.items()})
    np.save(ROOT / 'phase9_ranker_rescues.npy', np.concatenate(rescue_rows).astype(np.int32))
    for subset in ('sample', 'heldout'):
        for k, m in results[subset].items():
            print(subset, k, m['recall'], m['oracle_macro_f05'], m['mean'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('collect', 'train', 'evaluate'))
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'experiments/configs/candidate_ranker_v1.json')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    {'collect': collect, 'train': train_model, 'evaluate': evaluate}[args.mode](args, settings)


if __name__ == '__main__':
    main()
