from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import random


@dataclass(frozen=True)
class FairnessAuditResult:
    mean_score_difference: float
    selection_rate_difference: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    original_mean: float
    perturbed_mean: float
    selection_threshold: float
    variant_count: int


def _bootstrap_ci(values: list[float], seed: int = 42, iterations: int = 500) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return lower, upper


def audit_fairness(
    original_scores: list[float],
    perturbed_scores: list[float],
    selection_threshold: float = 60.0,
) -> FairnessAuditResult:
    if not original_scores or not perturbed_scores:
        return FairnessAuditResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, selection_threshold, 0)

    min_len = min(len(original_scores), len(perturbed_scores))
    original_scores = original_scores[:min_len]
    perturbed_scores = perturbed_scores[:min_len]

    deltas = [perturbed - original for original, perturbed in zip(original_scores, perturbed_scores)]
    mean_delta = sum(deltas) / len(deltas)
    original_selection_rate = sum(score >= selection_threshold for score in original_scores) / len(original_scores)
    perturbed_selection_rate = sum(score >= selection_threshold for score in perturbed_scores) / len(perturbed_scores)
    selection_rate_difference = perturbed_selection_rate - original_selection_rate
    low, high = _bootstrap_ci(deltas)

    return FairnessAuditResult(
        mean_score_difference=round(mean_delta, 4),
        selection_rate_difference=round(selection_rate_difference, 4),
        bootstrap_ci_low=round(low, 4),
        bootstrap_ci_high=round(high, 4),
        original_mean=round(sum(original_scores) / len(original_scores), 4),
        perturbed_mean=round(sum(perturbed_scores) / len(perturbed_scores), 4),
        selection_threshold=selection_threshold,
        variant_count=len(original_scores),
    )


def fairness_result_to_dict(result: FairnessAuditResult) -> dict[str, float | int]:
    return asdict(result)
