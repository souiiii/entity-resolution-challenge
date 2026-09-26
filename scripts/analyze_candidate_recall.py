"""Phase 8 loss accounting: where development true links leave the Phase 5 final candidate set."""
from __future__ import annotations

import argparse
import collections
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
from src.blocking.recovery import (LOSS_CATEGORIES, NO_KEY, classify_unretrieved, families, hashed_evidence,
                                   shared_key_profile, trace_links)
from src.blocking.store import RecordStore
from src.data.validation import read_config, verify_manifests
from src.features.pairwise import CandidateEvidence, extract_pair_features, feature_names, record_view
from src.features.text import SourceText
from src.matching.rules import evidence_from_features, load_matcher_config, prefilter_mask
from src.utils.paths import INTERIM_DIR, PROJECT_ROOT, RAW_DIR

OUTPUT_DIR = INTERIM_DIR / 'candidates'
STATES = {0: 'unavailable', 1: 'agree', 2: 'partial', 3: 'conflict'}


def phase4_evidence(blocking, texts, stores, source, query_rows, target_rows, trace, index):
    """Phase 4 evidence for arbitrary (query, target) link pairs, reading text only for those rows."""
    q_unique, q_index = np.unique(query_rows, return_inverse=True)
    t_unique, t_index = np.unique(target_rows, return_inverse=True)
    queries = [record_view(n, a) for n, a, _ in texts[1].read(q_unique)]
    targets = [record_view(n, a) for n, a, _ in texts[source].read(t_unique)]
    evidence = CandidateEvidence(source, np.minimum(trace.rank[index], 10_000), trace.masks[index],
                                 trace.min_frequency[index], trace.aggregate_rarity[index])
    matrix = extract_pair_features(queries, targets, q_index, t_index, evidence, blocking.channels)
    return evidence_from_features(matrix, feature_names(blocking.channels))


def quantiles(values: np.ndarray) -> dict:
    values = values[~np.isnan(values)]
    if not len(values):
        return {'defined': 0}
    return {'defined': int(len(values)), **{f'p{q}': round(float(v), 3)
                                             for q, v in zip((10, 25, 50, 75, 90), np.percentile(values, (10, 25, 50, 75, 90)))}}


def describe(mask, table, channel_names):
    """Compact diagnostics for the links selected by ``mask``."""
    n = int(mask.sum())
    if not n:
        return {'links': 0}
    capable = table['capable'][mask]
    combos = collections.Counter(tuple(channel_names[j] for j in np.flatnonzero(row)) for row in capable)
    min_frequency = table['best_frequency'][mask]
    bands = {'<=10': int((min_frequency <= 10).sum()), '11-50': int(((min_frequency > 10) & (min_frequency <= 50)).sum()),
             '51-200': int(((min_frequency > 50) & (min_frequency <= 200)).sum()),
             '>200': int(((min_frequency > 200) & (min_frequency < NO_KEY)).sum()), 'none': int((min_frequency == NO_KEY).sum())}
    return {
        'links': n,
        'source': {f'S{s}': int((table['source'][mask] == s).sum()) for s in (2, 3)},
        'country': dict(collections.Counter(table['country'][mask].tolist())),
        'script_mismatch': int(table['script_mismatch'][mask].sum()),
        'target_address_empty': int(table['target_address_empty'][mask].sum()),
        'number_state': {STATES[k]: int(v) for k, v in collections.Counter(table['number_state'][mask].tolist()).items()},
        'name_token_set_similarity': quantiles(table['name'][mask]),
        'address_token_set_similarity': quantiles(table['address'][mask]),
        'hashed_name_shared_tokens': dict(collections.Counter(np.minimum(table['name_shared'][mask], 3).tolist())),
        'best_shared_key_frequency_band': bands,
        'channels_capable': {name: int(capable[:, j].sum()) for j, name in enumerate(channel_names)},
        'top_channel_combinations': [[list(k), v] for k, v in combos.most_common(8)],
        'blocking_rank': quantiles(table['rank'][mask].astype(np.float64)) if 'rank' in table else None,
        'support_families': dict(collections.Counter(table['families'][mask].tolist())),
    }


def residual(args) -> None:
    """Original loss categories of sample true links still missing from the Phase 8 final candidate set."""
    blocking = load_config(args.blocking_config)
    links = np.load(OUTPUT_DIR / 'phase8_sample_links.npz')
    recovered_file = np.load(OUTPUT_DIR / f'{args.final_version}_sample_links.npz')
    recovered = np.concatenate([recovered_file[f'recovered_S{s}'] for s in (2, 3)])
    top_k = blocking.raw['ranking']['selected_k']
    category = np.full(len(recovered), 'kept', dtype=object)
    missing = ~links['in_union']
    category[missing] = classify_unretrieved(links['profile'][missing], list(blocking.channels))
    category[links['in_union'] & (links['rank'] > top_k)] = 'D_rank_cut'
    category[links['in_union'] & (links['rank'] <= top_k) & ~links['prefilter_pass']] = 'E_prefilter'
    before = collections.Counter(category[category != 'kept'].tolist())
    after = collections.Counter(category[~recovered].tolist())
    report = {'final_version': args.final_version, 'links': int(len(recovered)),
              'phase5_missed': int(sum(before.values())), 'phase8_missed': int((~recovered).sum()),
              'phase5_kept_now_missed': after.pop('kept', 0),
              'categories': {c: {'phase5_missed': before.get(c, 0), 'phase8_missed': after.get(c, 0),
                                 'recovered': before.get(c, 0) - after.get(c, 0)} for c in LOSS_CATEGORIES}}
    (OUTPUT_DIR / f'{args.final_version}_residual_sample.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description='Development-sample loss accounting for the Phase 5 candidate set')
    parser.add_argument('--residual', action='store_true', help='categories still missed by the Phase 8 final set')
    parser.add_argument('--final-version', default='blocking-v3')
    parser.add_argument('--blocking-config', type=Path, default=PROJECT_ROOT / 'experiments/configs/blocking_v2.json')
    parser.add_argument('--matcher-config', type=Path, default=PROJECT_ROOT / 'experiments/configs/matcher_v1.json')
    parser.add_argument('--validation-config', type=Path, default=PROJECT_ROOT / 'experiments/configs/validation.json')
    parser.add_argument('--batch-size', type=int, default=2500)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.residual:
        residual(args)
        return
    started = time.monotonic()
    metadata = verify_manifests(INTERIM_DIR / 'validation', read_config(args.validation_config))
    blocking = load_config(args.blocking_config)
    matcher = load_matcher_config(args.matcher_config)
    stores = {s: RecordStore(INTERIM_DIR / 'blocking' / 'store' / f'train_source{s}') for s in (1, 2, 3)}
    texts = {s: SourceText(RAW_DIR / 'train' / f'train_source{s}.tsv', stores[s],
                           INTERIM_DIR / 'features' / 'text' / f'train_source{s}') for s in (1, 2, 3)}
    rows = select_queries(stores[1], INTERIM_DIR / 'validation' / 'development_source1.tsv', blocking.sample_salt,
                          blocking.sample_fraction)
    truth = load_truth(RAW_DIR / 'train' / 'train_ground_truth.tsv', stores[1], rows, {2: stores[2], 3: stores[3]})
    top_k = blocking.raw['ranking']['selected_k']
    channel_names = [c.name for c in blocking.channels]
    labels = stores[1].countries
    columns = collections.defaultdict(list)
    cache = collections.defaultdict(list)
    union_counts = np.zeros(len(rows), np.int64)
    min_free = shutil.disk_usage(INTERIM_DIR).free
    for source in (2, 3):
        blocker = SourceBlocker(blocking, stores[source], INTERIM_DIR / 'blocking' / 'index')
        owners, records = truth[source]
        trace = trace_links(blocker, stores[1], rows, owners, records, args.batch_size)
        union_counts += trace.union_counts
        query_rows = rows[owners]
        in_top = trace.in_union & (trace.rank <= top_k)
        evidence = phase4_evidence(blocking, texts, stores, source, query_rows, records, trace, np.arange(len(owners)))
        passes = prefilter_mask(evidence, matcher.prefilter)
        profile = np.full((len(owners), len(blocking.channels)), NO_KEY, np.int64)
        missing = ~trace.in_union
        profile[missing] = shared_key_profile(blocker, stores[1], query_rows[missing], records[missing])
        category = np.full(len(owners), 'kept', dtype=object)
        category[missing] = classify_unretrieved(profile[missing], [spec for _, spec, _ in blocker.channels])
        category[trace.in_union & ~in_top] = 'D_rank_cut'
        category[in_top & ~passes] = 'E_prefilter'
        in_union_capable = ((trace.masks[:, None] >> np.arange(len(channel_names), dtype=np.uint32)) & 1).astype(bool)
        keeps = np.array([c.index_keep or NO_KEY - 1 for c in blocking.channels])
        capable = np.where(missing[:, None], profile <= keeps, in_union_capable)
        hashed = hashed_evidence(stores[1], stores[source], query_rows, records)
        best = np.where(missing, profile.min(axis=1), trace.min_frequency)
        columns['category'].append(category)
        columns['source'].append(np.full(len(owners), source))
        columns['country'].append(np.array(labels)[np.asarray(stores[1].country)[query_rows]])
        columns['script_mismatch'].append(evidence.script_mismatch)
        columns['target_address_empty'].append(np.isnan(evidence.address))
        columns['number_state'].append(evidence.number_state)
        columns['name'].append(evidence.name)
        columns['address'].append(evidence.address)
        columns['name_shared'].append(hashed.name_shared)
        columns['capable'].append(capable)
        columns['best_frequency'].append(best)
        columns['rank'].append(np.where(trace.in_union, trace.rank, -1))
        columns['families'].append(families(trace.masks, blocking.channels))
        for key, value in (('owners', owners), ('records', records), ('source', np.full(len(owners), source, np.int8)),
                           ('in_union', trace.in_union), ('rank', trace.rank), ('prefilter_pass', passes),
                           ('profile', profile)):
            cache[key].append(value)
        min_free = min(min_free, shutil.disk_usage(INTERIM_DIR).free)
        del blocker
    table = {k: np.concatenate(v) for k, v in columns.items()}
    total = len(table['category'])
    lost = table['category'] != 'kept'
    counts = collections.Counter(table['category'][lost].tolist())
    kept = int((~lost).sum())
    report = {
        'phase': '8-loss-accounting', 'partition': 'development-sample', 'validation_labels_used': False,
        'split_fingerprint': metadata['fingerprint'], 'queries': int(len(rows)), 'true_links': total,
        'phase5_final_kept': kept, 'phase5_final_recall': kept / total,
        'union_recall': float(np.concatenate(cache['in_union']).mean()),
        'top_k_recall': float((np.concatenate(cache['in_union']) & (np.concatenate(cache['rank']) <= top_k)).mean()),
        'union_mean_candidates': float(union_counts.mean()),
        'missed_links': int(lost.sum()),
        'categories': {c: {'links': counts.get(c, 0), 'share_of_missed': counts.get(c, 0) / max(int(lost.sum()), 1),
                           'share_of_all_links': counts.get(c, 0) / total} for c in LOSS_CATEGORIES},
        'diagnostics': {c: describe(table['category'] == c, table, channel_names) for c in LOSS_CATEGORIES},
        'kept_reference': describe(~lost, table, channel_names),
        'runtime_seconds': round(time.monotonic() - started, 1),
        'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        'min_free_disk_gib': round(min_free / 2**30, 2)}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_DIR / 'phase8_sample_links.npz', rows=rows, **{k: np.concatenate(v) for k, v in cache.items()})
    (OUTPUT_DIR / 'phase8_loss_sample.json').write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps({k: report[k] for k in ('true_links', 'phase5_final_kept', 'phase5_final_recall', 'union_recall',
                                             'top_k_recall', 'union_mean_candidates', 'missed_links', 'categories',
                                             'runtime_seconds', 'peak_rss_mib', 'min_free_disk_gib')}, indent=1))


if __name__ == '__main__':
    main()
