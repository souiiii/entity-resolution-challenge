from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_supervised import ResourceChecks, candidate_regression, collect, pipeline_configs
from src.data.validation import file_hash
from src.matching.decision import (claim_validation, decisions, policy_grid, robustness, search_families,
                                   select_final, validation_gate)
from src.matching.supervised import diagnostics, fast_macro, load_model, membership_digest, predict_scores, tune_queries
from src.utils.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)


def report_path(config: dict, mode: str) -> Path:
    return PROJECT_ROOT / config['report_dir'] / f"{config['version']}_{mode}_report.json"


def load_upstream(config: dict) -> tuple:
    upstream = json.loads((PROJECT_ROOT / config['supervised_config']).read_text())
    matcher, blocking, names, hashes = pipeline_configs(upstream)
    directory = PROJECT_ROOT / upstream['artifact_dir']
    if file_hash(directory / 'metadata.json') != config['model_metadata_sha256']:
        raise ValueError('Frozen Phase 6 metadata changed')
    model, metadata = load_model(directory, names)
    if (metadata['model_sha256'] != config['model_sha256'] or metadata['config'] != upstream
            or metadata['dependency_config_sha256'] != hashes or metadata['threshold'] != config['baseline_threshold']):
        raise ValueError('Frozen Phase 6 model, configuration or threshold changed')
    return upstream, matcher, blocking, names, model, metadata


def score_candidates(config: dict, sample: bool, monitor: ResourceChecks) -> tuple:
    upstream, matcher, blocking, names, model, frozen = load_upstream(config)
    parts = []

    def consume(features: np.ndarray) -> None:
        monitor.check()
        parts.append(predict_scores(model, features, names))

    def tune_only(ids: np.ndarray) -> np.ndarray:
        return tune_queries(ids, frozen['split']['salt'], frozen['split']['tune_fraction'])

    candidates, counts, ids, countries, fingerprint = collect(
        upstream, matcher, blocking, sample, consume, query_filter=tune_only if sample else None)
    scores = np.concatenate(parts)
    if len(scores) != len(candidates.owners) or fingerprint != frozen['split_fingerprint']:
        raise ValueError('Scores or split fingerprint do not match frozen candidate collection')
    if sample:
        expected = frozen['sizes']['tune']
        if (len(ids) != expected['queries'] or len(scores) != expected['pairs']
                or membership_digest(ids) != expected['membership_sha256']):
            raise ValueError('Phase 6 tune membership or candidate count changed; stop before search')
    monitor.check('streaming_scores_complete')
    return candidates, counts, ids, countries, scores, frozen


def baseline_policy(config: dict) -> dict:
    return {'family': 'global', 'threshold': config['baseline_threshold']}


def development(config: dict, monitor: ResourceChecks) -> dict:
    artifact = PROJECT_ROOT / config['policy_artifact']
    if report_path(config, 'development').exists() or artifact.exists():
        raise FileExistsError('Phase 7 decision development has already completed or frozen a policy')
    grid = policy_grid(config['grid'])
    candidates, counts, ids, countries, scores, model_metadata = score_candidates(config, True, monitor)
    complete = np.ones(len(ids), bool)
    original_policy = baseline_policy(config)
    baseline_decisions = decisions(candidates.owners, scores, len(ids), original_policy)
    baseline = diagnostics(candidates, baseline_decisions, counts, countries, complete)
    reference = json.loads((PROJECT_ROOT / config['development_baseline_report']).read_text())
    candidate_regression(baseline['overall'], reference['selected']['overall'])
    if baseline != reference['selected'] or baseline['overall']['macro_f05'] != config['expected_tune_macro_f05']:
        raise ValueError('Phase 6 full-tune baseline reproduction failed; no search allowed')
    LOGGER.info('Exact Phase 6 reproduction passed: %s queries, %s pairs, Macro F0.5 %.16f',
                len(ids), len(scores), baseline['overall']['macro_f05'])
    confirm = tune_queries(ids, config['split']['salt'], config['split']['confirm_fraction'])
    search = ~confirm
    if not search.any() or not confirm.any():
        raise ValueError('Decision search and confirm must both contain queries')
    masks = {'search': search, 'confirm': confirm}
    sizes = {name: {'queries': int(mask.sum()), 'pairs': int(mask[candidates.owners].sum()),
                    'positives': int((mask[candidates.owners] & candidates.labels).sum()),
                    'membership_sha256': membership_digest(ids[mask])} for name, mask in masks.items()}
    partition_baselines = {name: diagnostics(candidates, baseline_decisions, counts, countries, mask)
                          for name, mask in masks.items()}
    monitor.check('before_decision_search')
    trials, winners = search_families(candidates, scores, counts, countries, search, grid)
    monitor.check('after_decision_search')
    finalists = []
    for winner in winners:
        policy = winner['policy']
        accepted = decisions(candidates.owners, scores, len(ids), policy)
        confirmation = diagnostics(candidates, accepted, counts, countries, confirm)
        checked = robustness(winner['search']['overall'], confirmation['overall'], partition_baselines, config['gate'])
        finalists.append({**winner, 'confirm': confirmation, 'robustness': checked})
        LOGGER.info('%s: search gain %+.8f; confirm gain %+.8f; robust=%s',
                    policy, checked['search_gain'], checked['confirm_gain'], checked['passed'])
    selected_policy = select_final(finalists, original_policy)
    frozen = {'policy': selected_policy, 'config': config, 'sizes': sizes,
              'model_sha256': model_metadata['model_sha256'], 'feature_order': model_metadata['feature_order'],
              'split_fingerprint': model_metadata['split_fingerprint']}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open('x') as stream:
        json.dump(frozen, stream, indent=2, sort_keys=True)
        stream.write('\n')
    frozen_sha256 = file_hash(artifact)
    accepted = decisions(candidates.owners, scores, len(ids), selected_policy)
    selected = diagnostics(candidates, accepted, counts, countries, complete)
    score = fast_macro(candidates.owners, candidates.labels, accepted.astype(np.float32), sum(counts.values()), complete, .5)
    if score != selected['overall']['macro_f05']:
        raise ValueError('Selected decision score disagrees with canonical evaluator')
    selected_entry = next((entry for entry in finalists if entry['policy'] == selected_policy), None)
    confirm_gain = selected_entry['robustness']['confirm_gain'] if selected_entry else 0.0
    gate = validation_gate(baseline['overall'], selected['overall'], confirm_gain, config['gate'])
    monitor.check('decision_gate_complete')
    return {'mode': 'development', 'config': config, 'sizes': sizes, 'baseline': baseline,
            'partition_baselines': partition_baselines, 'trials': trials, 'finalists': finalists,
            'configurations_evaluated': len(trials), 'selected_policy': selected_policy, 'selected': selected,
            'production_policy': selected_policy if gate['passed'] else original_policy, 'gate': gate,
            'frozen_policy_sha256': frozen_sha256, 'frozen_policy': frozen}


def validation(config: dict, monitor: ResourceChecks) -> dict:
    development_report = json.loads(report_path(config, 'development').read_text())
    artifact = PROJECT_ROOT / config['policy_artifact']
    frozen = json.loads(artifact.read_text())
    if (file_hash(artifact) != development_report['frozen_policy_sha256'] or frozen['config'] != config
            or frozen['policy'] != development_report['selected_policy']):
        raise ValueError('Frozen Phase 7 policy or configuration changed')
    load_upstream(config)
    claim_validation(report_path(config, 'validation'), development_report)
    candidates, counts, ids, countries, scores, _ = score_candidates(config, False, monitor)
    mask = np.ones(len(ids), bool)
    baseline = diagnostics(candidates, decisions(candidates.owners, scores, len(ids), baseline_policy(config)),
                           counts, countries, mask)
    reference = json.loads((PROJECT_ROOT / config['validation_baseline_report']).read_text())
    candidate_regression(baseline['overall'], reference['selected']['overall'])
    if baseline != reference['selected']:
        raise ValueError('Frozen Phase 6 validation baseline reproduction failed')
    accepted = decisions(candidates.owners, scores, len(ids), frozen['policy'])
    selected = diagnostics(candidates, accepted, counts, countries, mask)
    return {'mode': 'validation', 'config': config, 'selected_policy': frozen['policy'], 'selected': selected,
            'baseline': baseline, 'macro_gain': selected['overall']['macro_f05'] - baseline['overall']['macro_f05'],
            'frozen_policy_sha256': development_report['frozen_policy_sha256']}


def protected_hashes() -> dict[str, str]:
    paths = ('output/candidate_pairs.tsv', 'output/matching_results.tsv',
             'output/supervised_v1/model.txt', 'output/supervised_v1/metadata.json')
    return {path: file_hash(PROJECT_ROOT / path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description='Bounded decisions over frozen Phase 6 scores; no test inference')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--development', action='store_true')
    modes.add_argument('--validation', action='store_true')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'experiments/configs/decision_v1.json')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = json.loads(args.config.read_text())
    monitor = ResourceChecks(config)
    before = protected_hashes()
    mode = 'development' if args.development else 'validation'
    result = development(config, monitor) if args.development else validation(config, monitor)
    if protected_hashes() != before:
        raise ValueError('A protected Phase 5/6 artifact changed')
    result.update(protected_sha256=before, resources=monitor.report())
    path = report_path(config, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'mode': mode, 'policy': result['selected_policy'],
                      'macro_f05': result['selected']['overall']['macro_f05'], 'gate': result.get('gate'),
                      'resources': result['resources'], 'report': str(path)}, indent=2))


if __name__ == '__main__':
    main()
