from __future__ import annotations

import argparse
import json
import logging
import resource
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking.channels import load_config
from src.blocking.evaluate import load_truth, select_queries
from src.blocking.store import RecordStore
from src.data.validation import file_hash, read_config, verify_manifests
from src.features.pairwise import feature_names
from src.features.text import ensure_text
from src.matching.pipeline import collect_candidates, evaluate, match
from src.matching.rules import load_matcher_config
from src.matching.supervised import (claim_validation, development_gate, diagnostics, load_model, membership_digest,
                                     predict_scores, save_model, select_threshold, threshold_predictions, tune_queries)
from src.utils.paths import INTERIM_DIR, PROJECT_ROOT, RAW_DIR

LOGGER = logging.getLogger(__name__)


class ResourceChecks:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.started = time.monotonic()
        self.minimum_disk_gib = float('inf')
        self.minimum_available_mib = float('inf')
        self.disk_checkpoints = []
        self.check('start')

    def check(self, stage: str | None = None) -> None:
        free = shutil.disk_usage(INTERIM_DIR).free / 2**30
        self.minimum_disk_gib = min(self.minimum_disk_gib, free)
        if stage is not None:
            self.disk_checkpoints.append({'stage': stage, 'free_disk_gib': free})
        if free < self.config['min_free_disk_gib']:
            raise OSError(f'Stopping at {free:.2f} GiB free disk; reserve would be threatened')
        path = Path('/proc/meminfo')
        if path.exists():
            available = next(float(line.split()[1]) / 1024 for line in path.read_text().splitlines()
                             if line.startswith('MemAvailable:'))
            self.minimum_available_mib = min(self.minimum_available_mib, available)
            if available < self.config['min_available_memory_mib']:
                raise MemoryError(f'Stopping at {available:.0f} MiB available system memory')

    def report(self) -> dict:
        self.check('finish')
        return {'runtime_seconds': round(time.monotonic() - self.started, 2),
                'main_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                'largest_child_peak_rss_mib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
                'minimum_free_disk_gib': self.minimum_disk_gib,
                'final_free_disk_gib': shutil.disk_usage(INTERIM_DIR).free / 2**30,
                'disk_checkpoints': self.disk_checkpoints,
                'minimum_observed_system_available_mib': self.minimum_available_mib}


def pipeline_configs(config: dict) -> tuple:
    matcher = load_matcher_config(PROJECT_ROOT / config['matcher_config'])
    features = json.loads((PROJECT_ROOT / matcher.features_config).read_text())
    blocking = load_config(PROJECT_ROOT / features['blocking_config'])
    names = feature_names(blocking.channels)
    if len(names) != 42:
        raise ValueError('Phase 6 requires the existing 42-feature schema')
    paths = [config['matcher_config'], matcher.features_config, features['blocking_config'], config['validation_config']]
    hashes = {p: file_hash(PROJECT_ROOT / p) for p in paths}
    return matcher, blocking, names, hashes


def collect(config: dict, matcher, blocking, sample: bool, consumer) -> tuple:
    metadata = verify_manifests(INTERIM_DIR / 'validation', read_config(PROJECT_ROOT / config['validation_config']))
    stores = {s: RecordStore(INTERIM_DIR / 'blocking' / 'store' / f'train_source{s}') for s in (1, 2, 3)}
    for s, store in stores.items():
        if store.meta['input_sha256'] != metadata['input_hashes'][f'train_source{s}.tsv']:
            raise ValueError('Blocking store does not match frozen source data')
    if file_hash(RAW_DIR / 'train' / 'train_ground_truth.tsv') != metadata['input_hashes']['train_ground_truth.tsv']:
        raise ValueError('Ground truth does not match frozen split input')
    manifest = 'development_source1.tsv' if sample else 'validation_source1.tsv'
    rows = select_queries(stores[1], INTERIM_DIR / 'validation' / manifest,
                          blocking.sample_salt if sample else None, blocking.sample_fraction if sample else None)
    truth = load_truth(RAW_DIR / 'train' / 'train_ground_truth.tsv', stores[1], rows,
                       {s: stores[s] for s in (2, 3)})
    counts = {s: np.bincount(truth[s][0], minlength=len(rows)) for s in (2, 3)}
    if not sample:
        expected = metadata['partitions']['validation']
        if len(rows) != expected['source1_entities'] or any(
                int(counts[s].sum()) != expected['targets'][str(s)]['links'] for s in (2, 3)):
            raise ValueError('Validation query/link counts disagree with frozen metadata')
    reserve = int(config['min_free_disk_gib'] * 2**30)
    texts = {s: ensure_text(RAW_DIR / 'train' / f'train_source{s}.tsv', stores[s],
                            INTERIM_DIR / 'features' / 'text' / f'train_source{s}', reserve) for s in (1, 2, 3)}
    try:
        candidates = collect_candidates(blocking, matcher.prefilter, stores[1], texts[1],
                                         {s: (stores[s], texts[s]) for s in (2, 3)}, rows,
                                         INTERIM_DIR / 'blocking' / 'index', config['feature_batch_size'],
                                         config['feature_workers'], truth, feature_consumer=consumer)
    finally:
        for text in texts.values():
            text.close()
    ids = np.asarray(stores[1].ids)[rows].copy()
    countries = np.array(stores[1].countries)[np.asarray(stores[1].country)[rows]]
    return candidates, counts, ids, countries, metadata['fingerprint']


def candidate_regression(actual: dict, baseline: dict) -> None:
    for key in ('candidates', 'unscreened_candidates'):
        if actual[key] != baseline[key]:
            raise ValueError(f'Phase 5 candidate statistics changed: {key}')
    for key in ('candidate_recall', 'oracle_ceiling', 'unscreened_recall', 'unscreened_oracle_ceiling'):
        if abs(actual[key] - baseline[key]) > 1e-12:
            raise ValueError(f'Phase 5 candidate statistics changed: {key}')
    for source in ('source2', 'source3'):
        if actual[source]['candidate_positives'] != baseline[source]['candidate_positives']:
            raise ValueError(f'Phase 5 positive candidate count changed: {source}')


def development(config: dict, output: Path, monitor: ResourceChecks) -> dict:
    if output.exists():
        raise FileExistsError('Phase 6 development report already exists; no automatic repeat experiments')
    if not 1 <= len(config['models']) <= 3:
        raise ValueError('Phase 6 allows at most three model configurations')
    matcher, blocking, names, hashes = pipeline_configs(config)
    reference = json.loads((PROJECT_ROOT / config['sample_baseline_report']).read_text())['selected']
    matrix = np.empty((reference['candidates']['total_pairs'], len(names)), np.float32)
    cursor = 0

    def consume(features: np.ndarray) -> None:
        nonlocal cursor
        monitor.check()
        stop = cursor + len(features)
        if stop > len(matrix):
            raise ValueError('Screened candidate count exceeds Phase 5 baseline; stop before training')
        matrix[cursor:stop] = features
        cursor = stop

    candidates, counts, ids, countries, fingerprint = collect(config, matcher, blocking, True, consume)
    monitor.check('candidate_collection_complete')
    if cursor != len(matrix):
        raise ValueError('Screened candidate count differs from Phase 5 baseline; stop before training')
    truth = sum(counts.values())
    baseline_decisions = match(candidates, matcher)
    sample_baseline = evaluate(candidates, baseline_decisions, truth)
    candidate_regression(sample_baseline, reference)
    if sample_baseline['macro_f05'] != reference['macro_f05']:
        raise ValueError('Deterministic Phase 5 sample score changed')
    LOGGER.info('Phase 5 regression passed: %s screened candidates, recall %.8f, oracle %.8f',
                len(matrix), sample_baseline['candidate_recall'], sample_baseline['oracle_ceiling'])
    tune = tune_queries(ids, config['split']['salt'], config['split']['tune_fraction'])
    train = ~tune
    if not train.any() or not tune.any():
        raise ValueError('Train/tune split must contain both partitions')
    train_pairs, tune_pairs = train[candidates.owners], tune[candidates.owners]
    x_train, x_tune = matrix[train_pairs], matrix[tune_pairs]
    del matrix
    y_train, y_tune = candidates.labels[train_pairs], candidates.labels[tune_pairs]
    baseline = diagnostics(candidates, baseline_decisions, counts, countries, tune)
    sizes = {name: {'queries': int(mask.sum()), 'pairs': int(pair_mask.sum()),
                    'positives': int(candidates.labels[pair_mask].sum()),
                    'negatives': int((~candidates.labels[pair_mask]).sum()),
                    'positive_rate': float(candidates.labels[pair_mask].mean()),
                    'membership_sha256': membership_digest(ids[mask])}
             for name, mask, pair_mask in (('train', train, train_pairs), ('tune', tune, tune_pairs))}
    trials = []
    best_model = best_trial = None
    for trial_config in config['models']:
        monitor.check(f"before_training_{trial_config['name']}")
        started = time.monotonic()
        parameters = {**config['common_parameters'], **{k: v for k, v in trial_config.items() if k != 'name'},
                      'random_state': config['seed'], 'data_random_seed': config['seed'],
                      'feature_fraction_seed': config['seed'], 'bagging_seed': config['seed']}
        model = lgb.LGBMClassifier(**parameters)
        LOGGER.info('Training %s', trial_config['name'])
        model.fit(x_train, y_train, eval_set=[(x_tune, y_tune)], eval_metric='binary_logloss',
                  feature_name=list(names), callbacks=[lgb.early_stopping(config['early_stopping_rounds'],
                                                                         first_metric_only=True, verbose=False)])
        monitor.check(f"after_training_{trial_config['name']}")
        scores = np.full(len(candidates.owners), np.nan, np.float32)
        scores[tune_pairs] = predict_scores(model.booster_, x_tune, names)
        sweep = select_threshold(candidates.owners, candidates.labels, scores, truth, tune, **config['threshold'])
        result = diagnostics(candidates, threshold_predictions(scores, sweep['threshold']), counts, countries, tune)
        if result['overall']['macro_f05'] != sweep['macro_f05']:
            raise ValueError('Fast threshold sweep disagrees with canonical frozen scorer')
        trial = {'name': trial_config['name'], 'parameters': parameters, 'best_iteration': model.best_iteration_,
                 'best_tune_logloss': model.best_score_['valid_0']['binary_logloss'], 'threshold_search': sweep,
                 'results': result, 'runtime_seconds': time.monotonic() - started,
                 'gain_importance': dict(zip(names, model.booster_.feature_importance(importance_type='gain').tolist()))}
        trials.append(trial)
        LOGGER.info('%s: Macro F0.5 %.8f, threshold %.3f, best iteration %s', trial_config['name'],
                    result['overall']['macro_f05'], sweep['threshold'], model.best_iteration_)
        if best_trial is None or result['overall']['macro_f05'] > best_trial['results']['overall']['macro_f05']:
            best_trial, best_model = trial, model.booster_
    gate = development_gate(baseline, best_trial['results'], config['gate'])
    monitor.check('before_model_save')
    frozen = save_model(best_model, PROJECT_ROOT / config['artifact_dir'],
                        {'config': config, 'dependency_config_sha256': hashes, 'split_fingerprint': fingerprint,
                         'selected_model': best_trial['name'], 'parameters': best_trial['parameters'],
                         'best_iteration': best_trial['best_iteration'], 'threshold': best_trial['threshold_search']['threshold'],
                         'training_seed': config['seed'], 'split': config['split'], 'sizes': sizes, 'gate': gate})
    loaded, _ = load_model(PROJECT_ROOT / config['artifact_dir'], names)
    if not np.array_equal(predict_scores(best_model, x_tune, names), predict_scores(loaded, x_tune, names)):
        raise ValueError('Serialized model does not reproduce tune probabilities')
    return {'mode': 'development', 'config': config, 'split_fingerprint': fingerprint, 'sizes': sizes,
            'candidate_regression': sample_baseline, 'baseline': baseline, 'trials': trials,
            'selected_model': best_trial['name'], 'selected': best_trial['results'], 'gate': gate,
            'frozen_model': frozen, 'resources': monitor.report()}


def validation(config: dict, output: Path, monitor: ResourceChecks) -> dict:
    matcher, blocking, names, hashes = pipeline_configs(config)
    model, frozen = load_model(PROJECT_ROOT / config['artifact_dir'], names)
    if frozen['config'] != config or frozen['dependency_config_sha256'] != hashes:
        raise ValueError('Configuration changed after freezing the supervised model')
    claim_validation(output, frozen)
    parts = []

    def consume(features: np.ndarray) -> None:
        monitor.check()
        parts.append(predict_scores(model, features, names))

    candidates, counts, ids, countries, fingerprint = collect(config, matcher, blocking, False, consume)
    monitor.check('streaming_inference_complete')
    if fingerprint != frozen['split_fingerprint']:
        raise ValueError('Frozen validation split fingerprint changed')
    scores = np.concatenate(parts)
    decisions = threshold_predictions(scores, frozen['threshold'])
    result = diagnostics(candidates, decisions, counts, countries, np.ones(len(ids), bool))
    baseline = json.loads((PROJECT_ROOT / config['validation_baseline_report']).read_text())
    candidate_regression(result['overall'], baseline['selected'])
    return {'mode': 'validation', 'config': config, 'split_fingerprint': fingerprint,
            'frozen_model': frozen, 'selected': result, 'baseline': baseline['selected'],
            'baseline_by_country': baseline['by_country'],
            'macro_gain': result['overall']['macro_f05'] - baseline['selected']['macro_f05'],
            'resources': monitor.report()}


def main() -> None:
    parser = argparse.ArgumentParser(description='Bounded Phase 6 supervised experiment; no test inference')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--development', action='store_true')
    mode.add_argument('--validation', action='store_true')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'experiments/configs/supervised_v1.json')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = json.loads(args.config.read_text())
    monitor = ResourceChecks(config)
    name = 'development' if args.development else 'validation'
    output = PROJECT_ROOT / config['report_dir'] / f"{config['version']}_{name}_report.json"
    report = development(config, output, monitor) if args.development else validation(config, output, monitor)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'mode': name, 'macro_f05': report['selected']['overall']['macro_f05'],
                      'gate': report.get('gate'), 'resources': report['resources'], 'report': str(output)}, indent=2))


if __name__ == '__main__':
    main()
