"""Small, country-independent decisions over frozen candidate probabilities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.matching.pipeline import Candidates
from src.matching.supervised import diagnostics, threshold_predictions


def query_best(owners: np.ndarray, scores: np.ndarray, queries: int) -> np.ndarray:
    """Best screened score across both target sources; empty queries have -infinity."""
    if owners.shape != scores.shape or owners.ndim != 1 or owners.dtype.kind not in 'iu':
        raise ValueError('Owners and scores must be aligned one-dimensional arrays')
    if queries < 0 or (owners < 0).any() or (owners >= queries).any():
        raise ValueError('Candidate owner is outside the query universe')
    best = np.full(queries, -np.inf, np.float64)
    np.maximum.at(best, owners, np.where(np.isfinite(scores), scores, -np.inf))
    return best


def decisions(owners: np.ndarray, scores: np.ndarray, queries: int, policy: dict,
              best: np.ndarray | None = None) -> np.ndarray:
    """Return a subset of supplied candidates; never force or break ties between matches."""
    fields = {'global': None, 'relative': 'alpha', 'margin': 'delta', 'singleton': 'single_threshold'}
    family = policy.get('family')
    if family not in fields:
        raise ValueError('Unknown decision family')
    parameter = fields[family]
    expected = {'family', 'threshold'} | ({parameter} if parameter else set())
    if set(policy) != expected:
        raise ValueError('Unexpected policy fields; only one decision family is allowed')
    if parameter and not 0 <= policy[parameter] <= 1:
        raise ValueError('Decision parameter must be in [0, 1]')
    if best is None:
        best = query_best(owners, scores, queries)
    elif best.shape != (queries,) or owners.shape != scores.shape:
        raise ValueError('Query best scores do not match the candidate arrays')
    keep = threshold_predictions(scores, policy['threshold'])
    values = scores.astype(np.float64)
    if family == 'relative':
        keep &= values >= policy['alpha'] * best[owners]
    elif family == 'margin':
        keep &= values >= best[owners] - policy['delta']
    elif family == 'singleton':
        counts = np.bincount(owners[keep], minlength=queries)
        keep &= (counts[owners] != 1) | (values >= policy['single_threshold'])
    return keep


def policy_grid(config: dict) -> dict[str, list[dict]]:
    result = {'global': [{'family': 'global', 'threshold': t} for t in config['global_thresholds']]}
    for family, key, values in (('relative', 'alpha', config['alphas']),
                                ('margin', 'delta', config['deltas']),
                                ('singleton', 'single_threshold', config['singleton_thresholds'])):
        result[family] = [{'family': family, 'threshold': t, key: p}
                          for t in config['family_thresholds'] for p in values
                          if family != 'singleton' or p > t]
    if not 0 < sum(map(len, result.values())) < 200 or any(not values for values in result.values()):
        raise ValueError('Decision search must have all four families and fewer than 200 configurations')
    return result


def selection_key(policy: dict, result: dict) -> tuple:
    """Macro F0.5, fewer FP, simplicity, then fewer predictions and stricter parameters."""
    complexity = {'global': 0, 'relative': 1, 'margin': 1, 'singleton': 2}[policy['family']]
    tightness = policy.get('alpha', -policy.get('delta', 0)) if policy['family'] != 'singleton' else policy['single_threshold']
    return (result['macro_f05'], -result['fp'], -complexity, -(result['tp'] + result['fp']),
            policy['threshold'], tightness)


def safety_checks(baseline: dict, result: dict, limits: dict) -> dict[str, bool]:
    def singleton_rate(values: dict) -> float:
        return values['correct_singletons'] / values['singleton_count'] if values['singleton_count'] else 1.0

    return {'precision': result['micro_precision_diagnostic'] >= baseline['micro_precision_diagnostic'] - limits['max_precision_drop'],
            'singletons': singleton_rate(result) >= singleton_rate(baseline) - limits['max_singleton_accuracy_drop']}


def robustness(search: dict, confirm: dict, baseline: dict, limits: dict) -> dict:
    search_gain = search['macro_f05'] - baseline['search']['overall']['macro_f05']
    confirm_gain = confirm['macro_f05'] - baseline['confirm']['overall']['macro_f05']
    checks = {'search_gain_positive': search_gain > 0, 'confirm_gain_positive': confirm_gain > 0}
    for name, result in (('search', search), ('confirm', confirm)):
        checks.update({f'{name}_{key}': value for key, value in safety_checks(baseline[name]['overall'], result, limits).items()})
    return {'passed': all(checks.values()), 'search_gain': search_gain, 'confirm_gain': confirm_gain, 'checks': checks}


def select_final(finalists: list[dict], baseline_policy: dict) -> dict:
    eligible = [entry for entry in finalists if entry['robustness']['passed']]
    if not eligible:
        return baseline_policy.copy()
    return max(eligible, key=lambda entry: selection_key(entry['policy'], entry['confirm']['overall']))['policy'].copy()


def search_families(candidates: Candidates, scores: np.ndarray, truth_by_source: dict[int, np.ndarray],
                    countries: np.ndarray, search: np.ndarray, grid: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    """Freeze one winner per family using search queries only, before any confirm scoring."""
    trials, winners = [], []
    best = query_best(candidates.owners, scores, candidates.queries)
    for family, policies in grid.items():
        entries = []
        for policy in policies:
            accepted = decisions(candidates.owners, scores, candidates.queries, policy, best)
            entry = {'policy': policy, 'search': diagnostics(candidates, accepted, truth_by_source, countries, search)}
            entries.append(entry)
        winners.append(max(entries, key=lambda entry: selection_key(entry['policy'], entry['search']['overall'])))
        trials.extend(entries)
    return trials, winners


def validation_gate(baseline: dict, selected: dict, confirm_gain: float, limits: dict) -> dict:
    gain = selected['macro_f05'] - baseline['macro_f05']
    checks = {'complete_tune_gain': gain >= limits['min_macro_gain'], 'confirm_gain_positive': confirm_gain > 0,
              **safety_checks(baseline, selected, limits)}
    return {'passed': all(checks.values()), 'macro_gain': gain, 'confirm_gain': confirm_gain, 'checks': checks}


def claim_validation(report_path: Path, development: dict) -> None:
    """Claim the single Phase 7 confirmation before loading any validation labels."""
    if not development['gate']['passed']:
        raise ValueError('Phase 7 development gate was not met')
    started = report_path.with_suffix('.started.json')
    if report_path.exists() or started.exists():
        raise FileExistsError('Phase 7 frozen validation was already started or completed')
    started.parent.mkdir(parents=True, exist_ok=True)
    with started.open('x') as stream:
        json.dump({'frozen_policy_sha256': development['frozen_policy_sha256']}, stream)
