"""Development-only overlap-corrected overflow utility and storage estimates."""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking.channels import load_config
from src.blocking.index import PostingIndex
from src.blocking.overflow import disk_guard, estimate_overflow
from src.blocking.recovery import NO_KEY, classify_unretrieved, hashed_evidence
from src.blocking.ranking import oracle_ceiling
from src.utils.paths import INTERIM_DIR, PROJECT_ROOT


def main() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    started = time.monotonic()
    root = INTERIM_DIR / 'candidates'
    config = load_config(PROJECT_ROOT / 'experiments/configs/blocking_v2.json')
    links = np.load(root / 'phase8_sample_links.npz')
    recovered = np.load(root / 'blocking-v3_sample_links.npz')
    found = np.concatenate([recovered[f'recovered_S{s}'] for s in (2, 3)])
    profile, source = links['profile'], links['source']
    truncated = ~links['in_union'] & (classify_unretrieved(profile, list(config.channels)) == 'B_index_truncated')
    best = profile.min(axis=1)
    keeps = np.array([c.index_keep for c in config.channels])
    deep = links['in_union'] | (profile <= keeps).any(axis=1)
    n = len(links['rows'])
    truth = np.bincount(links['owners'], minlength=n)
    bands = [(51, 200), (201, 300), (301, 500), (501, 750), (751, 1000), (1001, NO_KEY - 1)]
    report = {'queries': n, 'links': len(found), 'phase8_missing': int((~found).sum()),
              'category_b': int(truncated.sum()), 'validation_labels_used': False,
              'attribution': 'Each link assigned to the minimum-frequency channel, ties by configured channel order.',
              'bands': {}, 'cumulative': {}}
    from src.blocking.store import RecordStore
    from src.features.text import SourceText
    from src.features.pairwise import record_view
    from src.utils.paths import RAW_DIR
    from rapidfuzz import fuzz

    evidence = {k: np.full(len(found), np.nan) for k in ('name', 'address', 'script_mismatch', 'number_state')}
    query_store = RecordStore(INTERIM_DIR / 'blocking/store/train_source1')
    query_text = SourceText(RAW_DIR / 'train/train_source1.tsv', query_store, INTERIM_DIR / 'features/text/train_source1')
    for source_id in (2, 3):
        store = RecordStore(INTERIM_DIR / 'blocking/store' / f'train_source{source_id}')
        text = SourceText(RAW_DIR / 'train' / f'train_source{source_id}.tsv', store,
                          INTERIM_DIR / 'features/text' / f'train_source{source_id}')
        selected = np.flatnonzero(truncated & (source == source_id))
        qrows, trows = links['rows'][links['owners'][selected]], links['records'][selected]
        qviews = [record_view(a, b) for a, b, _ in query_text.read(qrows)]
        tviews = [record_view(a, b) for a, b, _ in text.read(trows)]
        evidence['name'][selected] = [fuzz.token_set_ratio(q.name_folded, t.name_folded) / 100
                                      if q.name_folded and t.name_folded else np.nan for q, t in zip(qviews, tviews)]
        evidence['address'][selected] = [fuzz.token_set_ratio(q.address, t.address) / 100
                                         if q.address and t.address else np.nan for q, t in zip(qviews, tviews)]
        evidence['script_mismatch'][selected] = [q.name_non_latin != t.name_non_latin for q, t in zip(qviews, tviews)]
        evidence['number_state'][selected] = hashed_evidence(query_store, store, qrows, trows).number_state
        text.close()
    query_text.close()
    winner = np.argmin(profile, axis=1)
    for lo, hi in bands:
        mask = truncated & (best >= lo) & (best <= hi)
        report['bands'][f'{lo}-{hi}'] = {'unique_links': int(mask.sum()), 'attributed_channel_source': {
            f'S{s}/{c.name}': int((mask & (source == s) & (winner == j)).sum())
            for s in (2, 3) for j, c in enumerate(config.channels)},
            'evidence': {'script_mismatch': int(np.nansum(evidence['script_mismatch'][mask])),
                'median_name_similarity': float(np.nanmedian(evidence['name'][mask])) if mask.any() else None,
                'median_address_similarity': float(np.nanmedian(evidence['address'][mask])) if mask.any() else None,
                'number_state': {str(k): int((evidence['number_state'][mask] == k).sum()) for k in range(4)}}}
    for maximum in (300, 500, 750, 1000):
        reachable = truncated & (best <= maximum)
        broad = deep | (best <= maximum)
        storage = {}
        for s in (2, 3):
            for j, spec in enumerate(config.channels):
                index = PostingIndex.load(INTERIM_DIR / 'blocking/index' / f'source{s}' / spec.index_id)
                storage[f'S{s}/{spec.name}'] = {**estimate_overflow(index, maximum),
                    'independent_links': int((truncated & (source == s) & (profile[:, j] <= maximum)).sum()),
                    'attributed_links': int((reachable & (source == s) & (winner == j)).sum())}
        needed = sum(v['bytes'] for v in storage.values())
        report['cumulative'][str(maximum)] = {
            'additional_unique_links': int(reachable.sum()), 'fraction_remaining': float(reachable.sum() / (~found).sum()),
            'postings': sum(v['postings'] for v in storage.values()), 'estimated_array_bytes': needed,
            'estimated_disk_bytes': needed + len(storage) * 16384,
            'broad_theoretical_recall': float(broad.mean()),
            'broad_theoretical_oracle': oracle_ceiling(truth, np.bincount(links['owners'][broad], minlength=n)),
            'channel_source': storage}
    report.update(runtime_seconds=time.monotonic() - started,
                  min_free_disk_gib=disk_guard(root) / 2**30,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (root / 'phase9_large_bucket_analysis.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: {a:b for a,b in v.items() if a!='channel_source'} for k,v in report['cumulative'].items()}, indent=2))


if __name__ == '__main__':
    main()
