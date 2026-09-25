from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.blocking.evaluate import in_sample
from src.data.validation import file_hash
from src.matching.pipeline import Candidates, evaluate


def tune_queries(ids: np.ndarray, salt: str, fraction: float) -> np.ndarray:
    """Label-independent query split; all pairs inherit their query's membership."""
    if not 0 < fraction < 1:
        raise ValueError('Tune fraction must be between zero and one')
    return np.array([in_sample(i.decode() if isinstance(i, bytes) else str(i), salt, fraction) for i in ids], bool)


def membership_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(b'\n'.join(sorted(i.encode() if isinstance(i, str) else bytes(i) for i in ids))).hexdigest()


def validate_schema(names: tuple[str, ...], expected: tuple[str, ...]) -> None:
    if names != expected or len(names) != len(set(names)):
        raise ValueError('Feature order/schema does not match frozen model metadata')


def predict_scores(model: lgb.Booster, matrix: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Binary model probabilities, with exact schema checks and unchanged NaN semantics."""
    validate_schema(names, tuple(model.feature_name()))
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError('Feature matrix shape does not match schema')
    if not len(matrix):
        return np.zeros(0, np.float32)
    scores = np.asarray(model.predict(matrix, num_threads=6), dtype=np.float32)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Model produced invalid probabilities')
    return scores


def threshold_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    if not np.isfinite(threshold) or threshold < 0 or threshold > 1.0000001:
        raise ValueError('Invalid global probability threshold')
    return np.isfinite(scores) & (scores.astype(np.float64) >= threshold)


def fast_macro(owners: np.ndarray, labels: np.ndarray, scores: np.ndarray, truth: np.ndarray,
               selected: np.ndarray, threshold: float) -> float:
    accepted = threshold_predictions(scores, threshold) & selected[owners]
    tp = np.bincount(owners[accepted & labels], minlength=len(truth))
    fp = np.bincount(owners[accepted & ~labels], minlength=len(truth))
    denominator = 5 * tp + 4 * fp + truth - tp
    values = np.divide(5 * tp, denominator, out=np.zeros(len(truth), np.float64), where=denominator != 0)
    values[(truth == 0) & (fp == 0)] = 1.0
    if not selected.any():
        raise ValueError('Empty tune partition')
    return math.fsum(values[selected]) / int(selected.sum())


def select_threshold(owners: np.ndarray, labels: np.ndarray, scores: np.ndarray, truth: np.ndarray,
                     selected: np.ndarray, coarse_step: float, fine_step: float) -> dict:
    """One global coarse-to-fine sweep; equal Macro F0.5 prefers the larger threshold."""
    if not 0 < fine_step < coarse_step < 1:
        raise ValueError('Threshold steps must satisfy 0 < fine < coarse < 1')
    measured: dict[float, float] = {}

    def score(thresholds: np.ndarray) -> float:
        for t in thresholds:
            threshold = float(t)
            if threshold not in measured:
                measured[threshold] = fast_macro(owners, labels, scores, truth, selected, threshold)
        return max(measured, key=lambda t: (measured[t], t))

    best = score(np.r_[np.round(np.arange(0, 1 + coarse_step / 2, coarse_step), 10), 1.0000001])
    fine = np.arange(max(0, best - coarse_step), min(1, best + coarse_step) + fine_step / 2, fine_step)
    best = score(np.round(fine, 10))
    return {'threshold': best, 'macro_f05': measured[best],
            'curve': [{'threshold': t, 'macro_f05': measured[t]} for t in sorted(measured)]}


def diagnostics(candidates: Candidates, accepted: np.ndarray, truth_by_source: dict[int, np.ndarray],
                countries: np.ndarray, selected: np.ndarray) -> dict:
    truth = sum(truth_by_source.values())

    def summary(mask: np.ndarray) -> dict:
        result = evaluate(candidates, accepted, truth, mask)
        for source, counts in truth_by_source.items():
            entry = result.get(f'source{source}', {'tp': 0, 'fp': 0, 'predicted': 0,
                                                 'precision': None, 'candidate_positives': 0})
            total = int(counts[mask].sum())
            entry.update(truth_links=total, fn=total - entry['tp'],
                         recall=entry['tp'] / total if total else 0.0)
            result[f'source{source}'] = entry
        return result

    return {'overall': summary(selected),
            'by_country': {country: summary(selected & (countries == country))
                           for country in sorted(set(countries[selected].tolist()))}}


def development_gate(baseline: dict, supervised: dict, config: dict) -> dict:
    b, s = baseline['overall'], supervised['overall']
    singleton_rate = lambda r: r['correct_singletons'] / r['singleton_count'] if r['singleton_count'] else 1.0
    delta = s['macro_f05'] - b['macro_f05']
    checks = {'macro_gain': delta >= config['min_macro_gain'],
              'precision': s['micro_precision_diagnostic'] >= b['micro_precision_diagnostic'] - config['max_precision_drop'],
              'singleton_handling': singleton_rate(s) >= singleton_rate(b) - config['max_singleton_accuracy_drop']}
    return {'passed': all(checks.values()), 'macro_gain': delta, 'checks': checks}


def save_model(model: lgb.Booster, directory: Path, metadata: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'model.txt'
    model.save_model(str(path))
    result = {**metadata, 'model_file': path.name, 'model_sha256': file_hash(path),
              'lightgbm_version': lgb.__version__, 'feature_order': model.feature_name()}
    (directory / 'metadata.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def load_model(directory: Path, names: tuple[str, ...]) -> tuple[lgb.Booster, dict]:
    metadata = json.loads((directory / 'metadata.json').read_text())
    validate_schema(names, tuple(metadata['feature_order']))
    path = directory / metadata['model_file']
    if file_hash(path) != metadata['model_sha256']:
        raise ValueError('Frozen model checksum mismatch')
    if metadata['lightgbm_version'] != lgb.__version__:
        raise ValueError('LightGBM version differs from frozen model environment')
    model = lgb.Booster(model_file=str(path))
    validate_schema(names, tuple(model.feature_name()))
    return model, metadata


def claim_validation(report_path: Path, metadata: dict) -> None:
    """Claim the one confirmation before reading validation labels; failures retain the claim."""
    if not metadata['gate']['passed']:
        raise ValueError('Development gate was not met')
    started = report_path.with_suffix('.started.json')
    if report_path.exists() or started.exists():
        raise FileExistsError('Frozen Phase 6 validation confirmation has already been started or completed')
    started.parent.mkdir(parents=True, exist_ok=True)
    with started.open('x') as stream:
        json.dump({'model_sha256': metadata['model_sha256'], 'threshold': metadata['threshold']}, stream)
