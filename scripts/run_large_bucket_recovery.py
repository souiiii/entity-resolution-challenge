"""Phase 9 bounded overflow building and sample-first candidate evaluation."""
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

from src.blocking.candidates import SourceBlocker
from src.blocking.channels import load_config
from src.blocking.overflow import BroadRetriever, disk_guard
from src.blocking.recovery import set_metrics
from src.blocking.store import RecordStore
from src.utils.paths import INTERIM_DIR, PROJECT_ROOT

ROOT = INTERIM_DIR / 'candidates'
OVERFLOW = INTERIM_DIR / 'blocking/overflow-v1'


def sample_context():
    """Reuse the frozen development sample and its aligned labels without reading validation labels."""
    from src.blocking.evaluate import select_queries
    from src.data.validation import read_config, verify_manifests

    metadata = verify_manifests(INTERIM_DIR / 'validation', read_config(PROJECT_ROOT / 'experiments/configs/validation.json'))
    config = replace(load_config(PROJECT_ROOT / 'experiments/configs/blocking_v2.json'), min_free_disk_gib=10.5)
    stores = {s: RecordStore(INTERIM_DIR / 'blocking/store' / f'train_source{s}') for s in (1, 2, 3)}
    for s, store in stores.items():
        if store.meta['input_sha256'] != metadata['input_hashes'][f'train_source{s}.tsv']:
            raise ValueError('Store does not match the frozen inputs')
    links = np.load(ROOT / 'phase8_sample_links.npz')
    rows = select_queries(stores[1], INTERIM_DIR / 'validation/development_source1.tsv',
                          config.sample_salt, config.sample_fraction)
    if not np.array_equal(rows, links['rows']):
        raise ValueError('Cached development sample changed')
    return config, stores, rows, links


def ceiling(args) -> None:
    started = time.monotonic()
    config, stores, rows, links = sample_context()
    n = len(rows)
    counts = {k: np.zeros(n, np.int64) for k in ('deep', 'broad')}
    found = {k: {s: np.zeros(n, np.int64) for s in (2, 3)} for k in counts}
    truth = {s: np.bincount(links['owners'][links['source'] == s], minlength=n) for s in (2, 3)}
    lowest = disk_guard(ROOT)
    recovered = np.zeros(len(links['owners']), bool)
    build_seconds = 0.0
    for s in (2, 3):
        blocker = SourceBlocker(config, stores[s], INTERIM_DIR / 'blocking/index')
        build_start = time.monotonic()
        retriever = BroadRetriever(blocker, OVERFLOW, args.maximum)
        build_seconds += time.monotonic() - build_start
        selected = np.flatnonzero(links['source'] == s)
        truth_keys = links['owners'][selected] * stores[s].size + links['records'][selected]
        for start in range(0, n, args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            deep, broad = retriever.retrieve(stores[1], batch_rows)
            for name, batch in (('deep', deep), ('broad', broad)):
                counts[name][start:start + len(batch_rows)] += np.bincount(batch.owners, minlength=len(batch_rows))
                keys = np.sort((batch.owners + start) * stores[s].size + batch.records)
                lo, hi = np.searchsorted(links['owners'][selected], [start, start + len(batch_rows)])
                probe = truth_keys[lo:hi]
                at = np.minimum(np.searchsorted(keys, probe), max(len(keys) - 1, 0))
                hit = keys[at] == probe if len(keys) else np.zeros(len(probe), bool)
                found[name][s] += np.bincount(links['owners'][selected[lo:hi]][hit], minlength=n)
                if name == 'broad':
                    recovered[selected[lo:hi]] = hit
            lowest = min(lowest, disk_guard(ROOT))
            if start % 5000 == 0:
                logging.info('S%s ceiling %s/%s', s, start, n)
        del retriever, blocker
    country = np.asarray(stores[1].country)[rows]
    names = stores[1].countries
    results = {k: set_metrics(counts[k], found[k], truth, country, names) for k in counts}
    historic_deep = json.loads((ROOT / 'phase8_pool_designs.json').read_text())['deep_pool']
    if results['deep']['pairs'] != historic_deep['pairs'] or results['deep']['found_links'] != historic_deep['found_links']:
        raise AssertionError('Existing deep pool differs from the Phase 8 reference')
    baseline = json.loads((ROOT / 'blocking-v3_sample_report.json').read_text())['stages']['deep8']
    results['broad'] = set_metrics(counts['broad'], found['broad'], truth, country, names, baseline)
    report = {'maximum': args.maximum, 'queries': n, 'validation_labels_used': False, 'results': results,
              'incremental_links_over_deep': results['broad']['found_links'] - results['deep']['found_links'],
              'runtime_seconds': time.monotonic() - started, 'build_seconds': build_seconds,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              'min_free_disk_gib': lowest / 2**30,
              'overflow_disk_bytes': sum(p.stat().st_size for p in OVERFLOW.rglob('*') if p.is_file())}
    (ROOT / 'phase9_broad_ceiling.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    np.savez(ROOT / 'phase9_broad_links.npz', recovered=recovered)
    print(json.dumps(report, indent=2))


FROZEN = PROJECT_ROOT / 'experiments/configs/blocking_v4.json'
COUNTS = tuple(f'{kind}_{s}' for kind in ('pairs', 'found', 'phase8_pairs', 'phase8_found') for s in (2, 3))
_WORKER: dict = {}


def load_frozen(path: Path = FROZEN) -> dict:
    """The frozen blocking-v4 policy after verifying every recorded file and directory hash."""
    from src.data.validation import file_hash, tree_hash

    policy = json.loads(path.read_text())
    if policy['status'] != 'frozen':
        raise ValueError('Only a frozen sample-selected policy may run on development or validation')
    for name, expected in policy['artifact_hashes'].items():
        if file_hash(PROJECT_ROOT / name) != expected:
            raise ValueError(f'Frozen artifact changed: {name}')
    for name, expected in policy['directory_hashes'].items():
        if tree_hash(PROJECT_ROOT / name) != expected:
            raise ValueError(f'Frozen directory changed: {name}')
    return policy


def available_mib() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return -1


def init_worker() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(process)d %(message)s')


def worker_context(policy: dict) -> dict:
    """Stores, texts, Phase 8 policy and ranker loaded once per worker process."""
    if not _WORKER:
        import lightgbm as lgb
        from src.features.text import SourceText
        from src.matching.final_candidates import load_final_policy
        from src.utils.paths import RAW_DIR

        stores = {s: RecordStore(INTERIM_DIR / 'blocking/store' / f'train_source{s}') for s in (1, 2, 3)}
        _WORKER.update(
            stores=stores,
            texts={s: SourceText(RAW_DIR / 'train' / f'train_source{s}.tsv', stores[s],
                                 INTERIM_DIR / 'features/text' / f'train_source{s}') for s in stores},
            blocking=replace(load_config(PROJECT_ROOT / policy['blocking_config']),
                             min_free_disk_gib=policy['min_free_disk_gib']),
            phase8=load_final_policy(PROJECT_ROOT / policy['phase8_config'], PROJECT_ROOT)[0],
            model=lgb.Booster(model_file=str(PROJECT_ROOT / policy['ranker_model'])))
    return _WORKER


def frozen_shard(job: tuple) -> dict:
    """Per-query blocking-v4 and Phase 8 pair and true-link counts of one contiguous query shard."""
    from src.matching.candidate_ranker import RANKER_RESCUE, iter_ranked_candidates

    policy, path, start, rows, truth_keys = job
    started = time.monotonic()
    context = worker_context(policy)
    stores, texts, n = context['stores'], context['texts'], len(rows)
    counts = {name: np.zeros(n, np.int64) for name in COUNTS}
    reserve = policy['min_free_disk_gib']
    lowest, memory = disk_guard(ROOT, reserve_gib=reserve), available_mib()
    for source in (2, 3):
        size, reference = stores[source].size, truth_keys[source]
        for batch in iter_ranked_candidates(context['phase8'], context['blocking'], context['model'], stores[1],
                texts[1], stores[source], texts[source], rows, INTERIM_DIR / 'blocking/index',
                PROJECT_ROOT / policy['overflow_root'], policy['overflow_maximum'],
                policy['rescue_per_query_source'], policy['batch_size'], reserve):
            keys = batch.owners * size + batch.records
            at = np.minimum(np.searchsorted(reference, keys), max(len(reference) - 1, 0))
            hit = reference[at] == keys if len(reference) else np.zeros(len(keys), bool)
            phase8 = batch.provenance != RANKER_RESCUE
            for name, mask in (('pairs', slice(None)), ('found', hit), ('phase8_pairs', phase8),
                               ('phase8_found', hit & phase8)):
                counts[f'{name}_{source}'] += np.bincount(batch.owners[mask], minlength=n)
            lowest, memory = min(lowest, disk_guard(ROOT, reserve_gib=reserve)), min(memory, available_mib())
    stats = {'runtime_seconds': time.monotonic() - started, 'min_free_disk_gib': lowest / 2**30,
             'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
             'min_available_memory_mib': memory}
    partial = path.with_name(path.stem + '.partial.npz')
    disk_guard(ROOT, sum(v.nbytes for v in counts.values()) + 16384, reserve)
    np.savez(partial, start=start, stop=start + n, **counts, **{k: np.float64(v) for k, v in stats.items()})
    partial.rename(path)
    return {'shard': path.name, 'queries': n, **stats}


def evaluate_frozen(args) -> None:
    """One sharded full-development confirmation and, only after it passes, one frozen validation.

    Query shards run in separate processes through the unchanged ``iter_ranked_candidates``;
    each query's candidates do not depend on its batch or shard. Completed shards persist so
    that an interrupted run of the same frozen policy resumes instead of restarting.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    from src.blocking.evaluate import load_truth, select_queries
    from src.data.validation import file_hash
    from src.matching.candidate_ranker import training_queries
    from src.utils.parallel import shard_bounds
    from src.utils.paths import RAW_DIR

    policy = load_frozen()
    policy_hash = file_hash(FROZEN)
    output = ROOT / f'blocking-v4_{args.partition}_report.json'
    marker = output.with_suffix('.started')
    shards = ROOT / f'blocking-v4_{args.partition}_shards'
    if output.exists():
        raise FileExistsError(f'Only one {args.partition} run is allowed')
    if args.partition == 'validation':
        confirmation = json.loads((ROOT / 'blocking-v4_development_report.json').read_text())
        if not confirmation['confirms_sample'] or confirmation['policy_sha256'] != policy_hash:
            raise ValueError('Full-development confirmation is required before validation')
    if marker.exists():
        state = json.loads(marker.read_text())
        if not args.resume or state['policy_sha256'] != policy_hash:
            raise FileExistsError(f'Only one {args.partition} run is allowed; --resume continues it')
    else:
        if shards.exists():
            raise FileExistsError(f'Unexpected shard directory: {shards}')
        state = {'policy_sha256': policy_hash, 'shard_queries': args.shard_queries}
        with marker.open('x') as stream:
            json.dump(state, stream)
    shards.mkdir(exist_ok=True)
    for stale in shards.glob('*.partial.npz'):
        stale.unlink()
    started = time.monotonic()
    _, stores, sample_rows, _ = sample_context()
    rows = select_queries(stores[1], INTERIM_DIR / 'validation' / f'{args.partition}_source1.tsv')
    truth = load_truth(RAW_DIR / 'train/train_ground_truth.tsv', stores[1], rows, {s: stores[s] for s in (2, 3)})
    n = len(rows)
    bounds = shard_bounds(n, state['shard_queries'])
    jobs = []
    for index, (lo, hi) in enumerate(bounds):
        path = shards / f'shard-{index:04d}.npz'
        if path.exists():
            continue
        truth_keys = {}
        for s in (2, 3):
            a, b = np.searchsorted(truth[s][0], [lo, hi])
            truth_keys[s] = np.sort((truth[s][0][a:b] - lo) * stores[s].size + truth[s][1][a:b])
        jobs.append((policy, path, lo, rows[lo:hi], truth_keys))
    logging.info('%s queries, %s shards, %s pending, %s workers', n, len(bounds), len(jobs), args.workers)
    lowest = disk_guard(ROOT)
    with ProcessPoolExecutor(args.workers, mp_context=get_context('spawn'), initializer=init_worker) as pool:
        futures = [pool.submit(frozen_shard, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            lowest = min(lowest, disk_guard(ROOT))
            elapsed = time.monotonic() - started
            logging.info('%s done (%s/%s) %.0fs rss %.0f MiB available %s MiB; elapsed %.0fs, eta %.0fs',
                         result['shard'], done, len(jobs), result['runtime_seconds'], result['peak_rss_mib'],
                         result['min_available_memory_mib'], elapsed, elapsed / done * (len(jobs) - done))
    counts = {name: np.zeros(n, np.int64) for name in COUNTS}
    shard_stats = []
    for index, (lo, hi) in enumerate(bounds):
        with np.load(shards / f'shard-{index:04d}.npz') as data:
            if (int(data['start']), int(data['stop'])) != (lo, hi):
                raise ValueError(f'Shard {index} bounds changed')
            for name in COUNTS:
                counts[name][lo:hi] = data[name]
            shard_stats.append({k: float(data[k]) for k in ('runtime_seconds', 'min_free_disk_gib', 'peak_rss_mib',
                                                              'min_available_memory_mib')})
    truth_counts = {s: np.bincount(truth[s][0], minlength=n) for s in (2, 3)}
    countries = np.asarray(stores[1].country)[rows]

    def metrics(mask: np.ndarray, prefix: str = '', reference: dict | None = None) -> dict:
        return set_metrics(counts[f'{prefix}pairs_2'][mask] + counts[f'{prefix}pairs_3'][mask],
                           {s: counts[f'{prefix}found_{s}'][mask] for s in (2, 3)},
                           {s: truth_counts[s][mask] for s in (2, 3)}, countries[mask], stores[1].countries,
                           reference)

    everything = np.ones(n, bool)
    phase8 = metrics(everything, 'phase8_')
    result = metrics(everything, '', phase8)
    historic = json.loads((ROOT / f'blocking-v3_{args.partition}_report.json').read_text())['stages']['deep8']
    reproduced = phase8['pairs'] == historic['pairs'] and phase8['found_links'] == historic['found_links']
    ranker = json.loads((PROJECT_ROOT / policy['ranker_config']).read_text())
    training = sample_rows[training_queries(stores[1].ids[sample_rows], ranker['split_salt'],
                                            ranker['training_fraction'])]
    outside = ~np.isin(rows, training)
    heldout, tolerance, gates = policy['heldout_reference'], policy['confirmation'], policy['gates']
    confirms = bool(reproduced and result['recall'] >= heldout['recall'] - tolerance['max_recall_drop']
                    and result['oracle_macro_f05'] >= heldout['oracle_macro_f05'] - tolerance['max_oracle_drop']
                    and abs(result['mean'] - heldout['mean']) <= tolerance['max_mean_difference'])
    report = {'partition': args.partition, 'queries': n, 'metrics': result, 'phase8_metrics': phase8,
              'phase8_reproduced': reproduced, 'confirms_sample': confirms,
              'gates': {'recall': result['recall'] >= gates['min_recall'],
                        'oracle_macro_f05': result['oracle_macro_f05'] >= gates['min_oracle_macro_f05'],
                        'mean': result['mean'] <= gates['max_mean']},
              'ranker_training_queries_included': int(n - outside.sum()),
              'metrics_excluding_ranker_training_queries': (metrics(outside, '', metrics(outside, 'phase8_'))
                                                            if not outside.all() else None),
              'policy_sha256': policy_hash, 'validation_labels_used': args.partition == 'validation',
              'workers': args.workers, 'shard_queries': state['shard_queries'], 'shards': len(bounds),
              'resumed_shards': len(bounds) - len(jobs),
              'runtime_seconds': time.monotonic() - started,
              'shard_seconds_total': sum(s['runtime_seconds'] for s in shard_stats),
              'peak_rss_mib': max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                  resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
                                  *(s['peak_rss_mib'] for s in shard_stats)),
              'min_free_disk_gib': min(lowest / 2**30, *(s['min_free_disk_gib'] for s in shard_stats)),
              'min_available_memory_mib': min(s['min_available_memory_mib'] for s in shard_stats),
              'overflow_disk_bytes': sum(p.stat().st_size for p in OVERFLOW.rglob('*') if p.is_file())}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--ceiling', action='store_true')
    mode.add_argument('--evaluate', action='store_true')
    parser.add_argument('--partition', choices=('development', 'validation'), default='development')
    parser.add_argument('--maximum', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--shard-queries', type=int, default=20000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    ceiling(args) if args.ceiling else evaluate_frozen(args)


if __name__ == '__main__':
    main()
