from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking.channels import load_config
from src.blocking.evaluate import distribution, load_truth, select_queries
from src.blocking.store import RecordStore, ensure_store
from src.data.validation import read_config, verify_manifests
from src.features.text import ensure_text
from src.matching.pipeline import collect_candidates, evaluate, match, path_counts, screen, write_submission
from src.matching.rules import load_matcher_config
from src.utils.paths import INTERIM_DIR, OUTPUT_DIR, PROJECT_ROOT, RAW_DIR

MATCHING_DIR = INTERIM_DIR / 'matching'


def file_digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resources(started: float) -> dict:
    return {'runtime_seconds': round(time.monotonic() - started, 1),
            'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            'disk_free_gib': round(shutil.disk_usage(INTERIM_DIR).free / 2**30, 2)}


def headline(name: str, result: dict) -> str:
    c = result['candidates']
    return (f"{name}: macro_f05 {result['macro_f05']:.5f} tp {result['tp']:,} fp {result['fp']:,} fn {result['fn']:,} "
            f"singletons {result['correct_singletons']:,}/{result['singleton_count']:,} "
            f"cands/S1 {c['mean']:.2f} recall {result['candidate_recall']:.4f} oracle {result['oracle_ceiling']:.5f}")


def labelled_run(args, config, blocking, features, rows_manifest: str, sample: bool) -> dict:
    started = time.monotonic()
    metadata = verify_manifests(INTERIM_DIR / 'validation', read_config(args.validation_config))
    stores = {s: RecordStore(INTERIM_DIR / 'blocking' / 'store' / f'train_source{s}') for s in (1, 2, 3)}
    for s, store in stores.items():
        if store.meta['input_sha256'] != metadata['input_hashes'][f'train_source{s}.tsv']:
            raise ValueError(f'Source {s} store does not match the frozen split input')
    reserve = int(features['min_free_disk_gib'] * 2**30)
    texts = {s: ensure_text(RAW_DIR / 'train' / f'train_source{s}.tsv', stores[s],
                            INTERIM_DIR / 'features' / 'text' / f'train_source{s}', reserve) for s in (1, 2, 3)}
    rows = select_queries(stores[1], INTERIM_DIR / 'validation' / rows_manifest,
                          blocking.sample_salt if sample else None, blocking.sample_fraction if sample else None)
    truth = load_truth(RAW_DIR / 'train' / 'train_ground_truth.tsv', stores[1], rows, {2: stores[2], 3: stores[3]})
    truth_per_query = sum(np.bincount(truth[s][0], minlength=len(rows)) for s in (2, 3))
    candidates = collect_candidates(blocking, None if sample else config.prefilter, stores[1], texts[1],
                                    {s: (stores[s], texts[s]) for s in (2, 3)}, rows,
                                    INTERIM_DIR / 'blocking' / 'index', features['batch_size'], features['workers'],
                                    truth)
    extraction_seconds = time.monotonic() - started
    countries = np.asarray(stores[1].country)[rows]
    labels = stores[1].countries
    report = {'matcher_version': config.version, 'mode': 'sample' if sample else 'validation',
              'partition': 'development' if sample else 'validation', 'split_fingerprint': metadata['fingerprint'],
              'config': config.raw, 'queries': int(len(rows)), 'links': int(truth_per_query.sum()),
              'extraction_seconds': round(extraction_seconds, 1)}
    if sample:
        pareto = {}
        for prefilter in config.pareto:
            screened = screen(candidates, prefilter)
            accepted = match(screened, config)
            result = evaluate(screened, accepted, truth_per_query)
            pareto[prefilter.label] = result
            print(headline(prefilter.label, result))
        report['pareto'] = pareto
    screened = screen(candidates, config.prefilter)
    accepted = match(screened, config)
    report['selected'] = evaluate(screened, accepted, truth_per_query)
    report['selected']['paths'] = path_counts(screened, accepted, config)
    report['by_country'] = {labels[c]: evaluate(screened, accepted, truth_per_query, countries == c)
                            for c in np.unique(countries)}
    report['resources'] = resources(started)
    print(headline('selected', report['selected']))
    for country, result in report['by_country'].items():
        print(headline(country, result))
    return report


def test_run(args, config, blocking, features) -> dict:
    started = time.monotonic()
    reserve = int(features['min_free_disk_gib'] * 2**30)
    raw = RAW_DIR / 'test'
    store_root, text_root = INTERIM_DIR / 'blocking' / 'store_test', INTERIM_DIR / 'features' / 'text_test'
    stores = {s: ensure_store(raw / f'test_source{s}.tsv', s, store_root / f'test_source{s}', blocking.workers,
                              min_free_bytes=reserve) for s in (1, 2, 3)}
    texts = {s: ensure_text(raw / f'test_source{s}.tsv', stores[s], text_root / f'test_source{s}', reserve)
             for s in (1, 2, 3)}
    rows = np.asarray(stores[1].sorted_order, np.int64)
    candidates = collect_candidates(blocking, config.prefilter, stores[1], texts[1],
                                    {s: (stores[s], texts[s]) for s in (2, 3)}, rows,
                                    INTERIM_DIR / 'blocking' / 'index_test', features['batch_size'],
                                    features['workers'])
    screened = screen(candidates, config.prefilter)
    accepted = match(screened, config)
    ids = [i.decode('utf-8') for i in np.asarray(stores[1].ids)[rows].tolist()]
    paths = write_submission(ids, screened, accepted, {2: stores[2], 3: stores[3]}, args.output_dir)
    countries = np.asarray(stores[1].country)[rows]
    matched = np.bincount(screened.owners[accepted], minlength=len(rows))
    candidate_counts = np.bincount(screened.owners, minlength=len(rows))
    report = {'matcher_version': config.version, 'mode': 'test', 'config': config.raw, 'queries': len(ids),
              'unscreened_candidates': distribution(candidates.unscreened_counts),
              'candidates': distribution(candidate_counts), 'matches': distribution(matched),
              'candidates_by_country': {stores[1].countries[c]: distribution(candidate_counts[countries == c])
                                        for c in np.unique(countries)},
              'matches_by_country': {stores[1].countries[c]: distribution(matched[countries == c])
                                     for c in np.unique(countries)},
              'files': {name: {'path': str(path.relative_to(PROJECT_ROOT)), 'bytes': path.stat().st_size,
                               'sha256': file_digest(path), 'rows': sum(1 for _ in path.open()) - 1}
                        for name, path in paths.items()},
              'store_bytes': {f'source{s}': st.meta['bytes'] for s, st in stores.items()},
              'resources': resources(started)}
    print(json.dumps({k: report[k] for k in ('queries', 'candidates', 'matches', 'files')}, indent=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Deterministic screen-then-match baseline (no learned model)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--sample', action='store_true', help='development hash sample: tuning report and Pareto')
    mode.add_argument('--validation', action='store_true', help='single frozen confirmation on validation')
    mode.add_argument('--test', action='store_true', help='write output/candidate_pairs.tsv and matching_results.tsv')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'experiments/configs/matcher_v1.json')
    parser.add_argument('--validation-config', type=Path, default=PROJECT_ROOT / 'experiments/configs/validation.json')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--force', action='store_true', help='allow overwriting an existing validation report')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = load_matcher_config(args.config)
    features = json.loads((PROJECT_ROOT / config.features_config).read_text())
    blocking = load_config(PROJECT_ROOT / features['blocking_config'])
    name = 'sample' if args.sample else 'validation' if args.validation else 'test'
    output = MATCHING_DIR / f'{config.version}_{name}_report.json'
    if args.validation and output.exists() and not args.force:
        parser.error(f'{output} exists: the frozen validation confirmation has already been run')
    if args.test:
        report = test_run(args, config, blocking, features)
    else:
        report = labelled_run(args, config, blocking, features,
                              'development_source1.tsv' if args.sample else 'validation_source1.tsv', args.sample)
    MATCHING_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'report: {output}')


if __name__ == '__main__':
    main()
