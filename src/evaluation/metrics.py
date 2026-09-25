from __future__ import annotations


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denominator


def f05(precision: float, recall: float) -> float:
    return f_beta(precision, recall, beta=0.5)
