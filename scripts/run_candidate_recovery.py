"""Phase 8 candidate recovery: sample pool collection, design evaluation, full-development and validation runs."""
from __future__ import annotations

import argparse
import json
import logging
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking.candidates import SourceBlocker
from src.blocking.channels import load_config
from src.blocking.evaluate import load_truth, select_queries
from src.blocking.ranking import candidate_ranks
from src.blocking.recovery import (HashedEvidence, cheap_score, deepen, expanded_union, hashed_evidence, set_metrics,
                                   top_per_query)
from src.blocking.store import RecordStore
from src.data.validation import file_hash, read_config, verify_manifests
from src.features.text import SourceText
from src.matching.final_candidates import (CONFLICT_RESCUE, DEEP_RESCUE, PHASE5, PROVENANCE, RANK_RESCUE,
                                           iter_final_candidates, load_final_policy)
from src.utils.paths import INTERIM_DIR, PROJECT_ROOT, RAW_DIR

OUTPUT_DIR = INTERIM_DIR / 'candidates'
POOL = OUTPUT_DIR / 'phase8_sample_pool.npz'
ABSENT = np.iinfo(np.int16).max
PHASE5_SAMPLE_PAIRS = 993_372


class Guard:
    """Free-disk and available-memory checks with the minimum observed values."""

    def __init__(self, min_free_gib: float, min_available_mib: int) -> None:
        self.min_free_bytes = int(min_free_gib * 2**30)
        self.min_available = min_available_mib
        self.lowest_free = shutil.disk_usage(INTERIM_DIR).free
        self.lowest_available = self.available()

    @staticmethod
    def available() -> int:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
        return 1 << 30

    def check(self) -> None:
        free, available = shutil.disk_usage(INTERIM_DIR).free, self.available()
        self.lowest_free, self.lowest_available = min(self.lowest_free, free), min(self.lowest_available, available)
        if free < self.min_free_bytes:
            raise OSError(f'Free disk {free / 2**30:.2f} GiB below the {self.min_free_bytes / 2**30:.1f} GiB reserve')
        if available < self.min_available:
            raise MemoryError(f'Available memory {available} MiB below {self.min_available} MiB')

    def report(self) -> dict:
        return {'min_free_disk_gib': round(self.lowest_free / 2**30, 2), 'min_available_memory_mib': self.lowest_available,
                'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}


def setup(args, sample: bool, manifest: str = 'development_source1.tsv'):
    metadata = verify_manifests(INTERIM_DIR / 'validation', read_config(args.validation_config))
    blocking = load_config(args.blocking_config)
    stores = {s: RecordStore(INTERIM_DIR / 'blocking' / 'store' / f'train_source{s}') for s in (1, 2, 3)}
    for s, store in stores.items():
        if store.meta['input_sha256'] != metadata['input_hashes'][f'train_source{s}.tsv']:
            raise ValueError(f'Source {s} store does not match the frozen split input')
    rows = select_queries(stores[1], INTERIM_DIR / 'validation' / manifest, blocking.sample_salt if sample else None,
                          blocking.sample_fraction if sample else None)
    truth = load_truth(RAW_DIR / 'train' / 'train_ground_truth.tsv', stores[1], rows, {2: stores[2], 3: stores[3]})
    return metadata, blocking, stores, rows, truth


def label(owners: np.ndarray, records: np.ndarray, truth_keys: np.ndarray, size: int) -> np.ndarray:
    keys = owners * size + records
    if not len(truth_keys):
        return np.zeros(len(keys), bool)
    at = np.minimum(np.searchsorted(truth_keys, keys), len(truth_keys) - 1)
    return truth_keys[at] == keys


def collect_sample(args) -> None:
    """Store the deep-retrieval pool of the development sample with base-union rank and hashed evidence."""
    guard = Guard(args.min_free_gib, args.min_available_mib)
    started = time.monotonic()
    metadata, blocking, stores, rows, truth = setup(args, sample=True)
    deep_specs = tuple(deepen(spec) for spec in blocking.channels)
    columns: dict[str, list] = {k: [] for k in ('owner', 'source', 'record', 'label', 'base_rank', 'base_rarity', 'mask',
                                                 'hits', 'rarity', 'name_containment', 'word_containment',
                                                 'number_state', 'exact_name', 'exact_address')}
    for source in (2, 3):
        store = stores[source]
        blocker = SourceBlocker(blocking, store, INTERIM_DIR / 'blocking' / 'index')
        owners, records = truth[source]
        truth_keys = np.sort(owners * store.size + records)
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            base = blocker.candidates(stores[1], batch_rows)
            ranks = candidate_ranks(base, store, blocking.channels, 'aggregate')
            deep = expanded_union(blocker, deep_specs, stores[1], batch_rows)
            base_keys = base.owners * store.size + base.records
            deep_keys = deep.owners * store.size + deep.records
            order = np.argsort(base_keys)
            at = np.minimum(np.searchsorted(base_keys[order], deep_keys), max(len(base_keys) - 1, 0))
            in_base = base_keys[order][at] == deep_keys
            if int(in_base.sum()) != len(base_keys):
                raise AssertionError('Deep retrieval must contain the unchanged base union')
            base_rank = np.full(len(deep_keys), ABSENT, np.int16)
            base_rank[in_base] = np.minimum(ranks[order][at][in_base], ABSENT - 1)
            base_rarity = np.zeros(len(deep_keys), np.float32)
            base_rarity[in_base] = base.aggregate_rarity[order][at][in_base]
            evidence = hashed_evidence(stores[1], store, batch_rows[deep.owners], deep.records)
            global_owners = deep.owners + start
            for key, value in (('owner', global_owners.astype(np.int32)), ('source', np.full(len(deep_keys), source, np.int8)),
                               ('record', deep.records.astype(np.int32)),
                               ('label', label(global_owners, deep.records, truth_keys, store.size)),
                               ('base_rank', base_rank), ('base_rarity', base_rarity), ('mask', deep.masks.astype(np.uint16)),
                               ('hits', np.minimum(deep.hits, ABSENT).astype(np.int16)),
                               ('rarity', deep.aggregate_rarity.astype(np.float32)),
                               ('name_containment', evidence.name_containment.astype(np.float16)),
                               ('word_containment', evidence.word_containment.astype(np.float16)),
                               ('number_state', evidence.number_state), ('exact_name', evidence.exact_name),
                               ('exact_address', evidence.exact_address)):
                columns[key].append(value)
            guard.check()
            logging.info('S%s sample queries %s/%s', source, min(len(rows), start + args.batch_size), len(rows))
        del blocker
    truth_per_query = sum(np.bincount(truth[s][0], minlength=len(rows)) for s in (2, 3))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key in list(columns):
        arrays[key] = np.concatenate(columns.pop(key))
    np.savez(POOL, rows=rows, truth_per_query=truth_per_query, country=np.asarray(stores[1].country)[rows],
             countries=np.array(stores[1].countries), **arrays)
    summary = {'pairs': int(len(arrays['owner'])), 'runtime_seconds': round(time.monotonic() - started, 1),
               'pool_bytes': POOL.stat().st_size, **guard.report(), 'split_fingerprint': metadata['fingerprint']}
    (OUTPUT_DIR / 'phase8_sample_pool.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=1))


def pool_reference(n: int) -> tuple[dict, dict]:
    """Phase 5 per-query found links and truth per source on the sample, from the loss-accounting table."""
    links = np.load(OUTPUT_DIR / 'phase8_sample_links.npz')
    kept = links['in_union'] & (links['rank'] <= 50) & links['prefilter_pass']
    found = {s: np.bincount(links['owners'][kept & (links['source'] == s)], minlength=n) for s in (2, 3)}
    truth = {s: np.bincount(links['owners'][links['source'] == s], minlength=n) for s in (2, 3)}
    return found, truth


def load_pool() -> dict:
    data = np.load(POOL)
    return {k: data[k] for k in data.files}


def evidence_of(pool: dict) -> HashedEvidence:
    return HashedEvidence(None, None, pool['name_containment'].astype(np.float32), None, None,
                          pool['word_containment'].astype(np.float32), pool['number_state'], pool['exact_name'],
                          pool['exact_address'])


def pool_metrics(pool: dict, selected: np.ndarray, truth: dict, reference: dict | None) -> dict:
    n = len(pool['truth_per_query'])
    counts = np.bincount(pool['owner'][selected], minlength=n)
    hit = selected & pool['label']
    found = {s: np.bincount(pool['owner'][hit & (pool['source'] == s)], minlength=n) for s in (2, 3)}
    return set_metrics(counts, found, truth, pool['country'], list(pool['countries']), reference)


def evaluate_pool(args) -> None:
    """Designs 1 and 2 on the stored sample pool: cheap-score top M per query and source."""
    pool = load_pool()
    n = len(pool['truth_per_query'])
    reference_found, truth = pool_reference(n)
    reference = set_metrics(np.full(n, PHASE5_SAMPLE_PAIRS // n), reference_found, truth, pool['country'],
                            list(pool['countries']))
    reference = {k: v for k, v in reference.items() if k in ('recall', 'found_links', 'links', 'oracle_macro_f05',
                                                              'source_recall', 'country_recall')}
    reference['pairs'] = PHASE5_SAMPLE_PAIRS
    group = pool['owner'].astype(np.int64) * 4 + pool['source']
    in_base = pool['base_rank'] < ABSENT
    tiebreak = pool['base_rank'].astype(np.int32)
    evidence = evidence_of(pool)
    base_score, deep_score = cheap_score(evidence, pool['base_rarity']), cheap_score(evidence, pool['rarity'])
    table = {'phase5_reference': reference, 'phase3b_union': pool_metrics(pool, in_base, truth, reference),
             'deep_pool': pool_metrics(pool, np.ones(len(group), bool), truth, reference)}
    for m in args.budgets:
        base_top = top_per_query(np.where(in_base, group, -1), np.where(in_base, base_score, -9), tiebreak,
                                 pool['record'], m)
        table[f'design1_top{m}'] = pool_metrics(pool, in_base & base_top, truth, reference)
        table[f'design2_top{m}'] = pool_metrics(pool, top_per_query(group, deep_score, tiebreak, pool['record'], m),
                                                truth, reference)
    (OUTPUT_DIR / 'phase8_pool_designs.json').write_text(json.dumps(table, indent=2, sort_keys=True) + '\n')
    for name, m in table.items():
        print(f"{name:20s} recall {m['recall']:.4f} gain {m.get('recall_gain_pp', 0.0):+.2f}pp "
              f"oracle {m['oracle_macro_f05']:.5f} mean {m.get('mean', float('nan')):.2f} p99 {m.get('p99')} "
              f"per1k {m.get('true_links_per_1000_added_pairs') or 0:.2f}")


def evaluate_final(args) -> None:
    """Production final-candidate policy with its stages and deep-budget sweep; metrics only."""
    config = json.loads(args.config.read_text())
    policy, blocking = load_final_policy(args.config, PROJECT_ROOT)
    guard = Guard(config['min_free_disk_gib'], config['min_available_memory_mib'])
    output = OUTPUT_DIR / f'{policy.version}_{args.partition}_report.json'
    marker = output.with_name(output.name + '.started')
    if args.partition == 'validation':
        if output.exists() or marker.exists():
            raise SystemExit(f'{output.name} or its start marker exists: validation was already evaluated once')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({'config_sha256': file_hash(args.config)}) + '\n')
    started = time.monotonic()
    manifest = 'validation_source1.tsv' if args.partition == 'validation' else 'development_source1.tsv'
    metadata, _, stores, rows, truth = setup(args, sample=args.partition == 'sample', manifest=manifest)
    texts = {s: SourceText(RAW_DIR / 'train' / f'train_source{s}.tsv', stores[s],
                           INTERIM_DIR / 'features' / 'text' / f'train_source{s}') for s in (1, 2, 3)}
    n = len(rows)
    sweep = sorted(set(config.get('sweep', {}).get('deep_per_query_source', [])) | {policy.deep_per_query})
    stages = ['phase5', 'phase5+conflict', 'phase5+conflict+rank'] + [f'deep{m}' for m in sweep if m]
    counts = {k: np.zeros(n, np.int64) for k in stages}
    found = {k: {s: np.zeros(n, np.int64) for s in (2, 3)} for k in stages}
    truth_counts = {s: np.bincount(truth[s][0], minlength=n) for s in (2, 3)}
    union_counts = np.zeros(n, np.int64)
    provenance = {name: {'pairs': 0, 'true': 0} for name in PROVENANCE}
    recovered_links = {}
    for source in (2, 3):
        store = stores[source]
        link_keys = truth[source][0] * store.size + truth[source][1]
        link_order = np.argsort(link_keys)
        truth_keys = link_keys[link_order]
        recovered_sorted = np.zeros(len(truth_keys), bool)
        start = 0
        for batch in iter_final_candidates(policy, blocking, stores[1], texts[1], store, texts[source], rows,
                                           INTERIM_DIR / 'blocking' / 'index', config['batch_size'],
                                           config['workers'], deep_limit=max(sweep)):
            union_counts[start:start + len(batch.union_counts)] += batch.union_counts
            start += len(batch.union_counts)
            labels = label(batch.owners, batch.records, truth_keys, store.size)
            deep = batch.provenance == DEEP_RESCUE
            final = ~deep | (batch.deep_rank <= policy.deep_per_query)
            if len(truth_keys):
                hit_keys = batch.owners[final & labels] * store.size + batch.records[final & labels]
                recovered_sorted[np.searchsorted(truth_keys, hit_keys)] = True
            for name, flag in PROVENANCE.items():
                chosen = final & ((batch.provenance & flag) != 0)
                provenance[name]['pairs'] += int(chosen.sum())
                provenance[name]['true'] += int((chosen & labels).sum())
            masks = {'phase5': (batch.provenance & PHASE5) != 0,
                     'phase5+conflict': (batch.provenance & (PHASE5 | CONFLICT_RESCUE)) != 0,
                     'phase5+conflict+rank': (batch.provenance & (PHASE5 | CONFLICT_RESCUE | RANK_RESCUE)) != 0}
            for m in sweep:
                if m:
                    masks[f'deep{m}'] = masks['phase5+conflict+rank'] | (deep & (batch.deep_rank <= m))
            for name, mask in masks.items():
                counts[name] += np.bincount(batch.owners[mask], minlength=n)
                found[name][source] += np.bincount(batch.owners[mask & labels], minlength=n)
            guard.check()
            logging.info('S%s queries %s/%s', source, start, n)
        recovered = np.zeros(len(link_keys), bool)
        recovered[link_order] = recovered_sorted
        recovered_links[source] = recovered
    countries, names = np.asarray(stores[1].country)[rows], stores[1].countries
    reference = set_metrics(counts['phase5'], found['phase5'], truth_counts, countries, names)
    results = {name: set_metrics(counts[name], found[name], truth_counts, countries, names,
                                 None if name == 'phase5' else reference) for name in stages}
    report = {'version': policy.version, 'partition': args.partition, 'queries': n,
              'validation_labels_used': args.partition == 'validation', 'split_fingerprint': metadata['fingerprint'],
              'config': config, 'config_sha256': file_hash(args.config),
              'selected_stage': f'deep{policy.deep_per_query}', 'stages': results, 'final_provenance': provenance,
              'phase3b_union_mean': float(union_counts.mean()), 'runtime_seconds': round(time.monotonic() - started, 1),
              **guard.report()}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    if args.partition == 'sample':
        np.savez(OUTPUT_DIR / f'{policy.version}_sample_links.npz',
                 **{f'recovered_S{s}': v for s, v in recovered_links.items()})
    for name, m in results.items():
        print(f"{name:22s} recall {m['recall']:.5f} gain {m.get('recall_gain_pp', 0.0):+.2f}pp "
              f"oracle {m['oracle_macro_f05']:.5f} mean {m['mean']:.2f} p99 {m['p99']} max {m['max']} "
              f"S2 {m['source_recall']['S2']:.4f} S3 {m['source_recall']['S3']:.4f} "
              + ' '.join(f'{k} {v:.4f}' for k, v in m['country_recall'].items()))
    print(json.dumps({k: report[k] for k in ('runtime_seconds', 'peak_rss_mib', 'min_free_disk_gib',
                                             'min_available_memory_mib', 'final_provenance')}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 8 candidate recovery (candidate generation only)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--collect-sample', action='store_true', help='store the development-sample deep pool')
    mode.add_argument('--evaluate-pool', action='store_true', help='designs 1 and 2 on the stored sample pool')
    mode.add_argument('--evaluate', action='store_true', help='run the final candidate policy on a partition')
    parser.add_argument('--partition', choices=('sample', 'development', 'validation'), default='sample')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'experiments/configs/blocking_v3.json')
    parser.add_argument('--blocking-config', type=Path, default=PROJECT_ROOT / 'experiments/configs/blocking_v2.json')
    parser.add_argument('--validation-config', type=Path, default=PROJECT_ROOT / 'experiments/configs/validation.json')
    parser.add_argument('--budgets', type=int, nargs='+', default=[8, 10, 12, 15, 20, 25])
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--min-free-gib', type=float, default=10.5)
    parser.add_argument('--min-available-mib', type=int, default=1024)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.collect_sample:
        collect_sample(args)
    elif args.evaluate_pool:
        evaluate_pool(args)
    else:
        evaluate_final(args)


if __name__ == '__main__':
    main()
