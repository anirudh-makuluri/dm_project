from __future__ import annotations

from dataclasses import dataclass, asdict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    support: int


def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        precision_macro=round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        recall_macro=round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        f1_macro=round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        f1_weighted=round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        support=len(y_true),
    )


def metrics_to_dict(metrics: ClassificationMetrics) -> dict[str, float | int]:
    return asdict(metrics)
