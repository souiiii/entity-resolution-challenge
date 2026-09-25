from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.metrics import f05


@dataclass(frozen=True)
class EvaluationResult:
    precision: float
    recall: float
    f05: float


def evaluate_binary_counts(tp: int, fp: int, fn: int) -> EvaluationResult:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return EvaluationResult(
        precision=precision,
        recall=recall,
        f05=f05(precision, recall),
    )
