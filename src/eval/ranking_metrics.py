"""Ranking metrics for candidate-job fit (nDCG@k, MAP@k, Precision/Recall@k).

Setup
-----
For each occupational category C (treated as a "job"), every test resume
is a candidate and is scored by ``fit_score(resume, job=C)``. Relevance is
binary: 1 if the resume's true category equals C, else 0.

We rank candidates by their fit score for each job and compute:

* Precision@k, Recall@k
* Mean Average Precision @k (MAP@k)
* Normalised Discounted Cumulative Gain @k (nDCG@k)

Macro averages across jobs are reported alongside per-job breakdowns. The
ranking source is the ``scores`` JSON column from the multi-agent
predictions CSV, so this module does not need to re-run the agents.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class RankingMetricsAtK:
    k: int
    precision: float
    recall: float
    map: float
    ndcg: float


@dataclass
class RankingReport:
    per_k: Dict[int, RankingMetricsAtK]
    per_job: Dict[str, Dict[int, RankingMetricsAtK]]
    n_jobs: int
    n_candidates: int


def _dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def _ndcg_at_k(relevances: Sequence[int], k: int) -> float:
    actual = _dcg(relevances[:k])
    ideal_relevances = sorted(relevances, reverse=True)[:k]
    ideal = _dcg(ideal_relevances)
    return actual / ideal if ideal > 0 else 0.0


def _average_precision_at_k(relevances: Sequence[int], k: int) -> float:
    cut = relevances[:k]
    total_relevant = sum(relevances)
    if total_relevant == 0:
        return 0.0
    hits = 0
    score_sum = 0.0
    for index, rel in enumerate(cut, start=1):
        if rel == 1:
            hits += 1
            score_sum += hits / index
    return score_sum / min(total_relevant, k)


def _precision_recall_at_k(relevances: Sequence[int], k: int) -> tuple[float, float]:
    cut = relevances[:k]
    hits = sum(cut)
    precision = hits / max(1, k)
    total_relevant = sum(relevances)
    recall = hits / total_relevant if total_relevant else 0.0
    return precision, recall


def _ranked_relevances(
    candidates: Sequence[tuple[float, int]],
    job_label: str,
    label_lookup: Mapping[int, str],
) -> List[int]:
    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    return [1 if str(label_lookup.get(resume_id, "")).strip() == job_label else 0 for _, resume_id in ranked]


def evaluate_ranking(
    predictions_df: pd.DataFrame,
    ks: Iterable[int] = (1, 3, 5, 10),
    score_column: str = "scores",
    id_column: str = "resume_id",
    label_column: str = "y_true",
) -> RankingReport:
    """Compute ranking metrics from a predictions DataFrame.

    The DataFrame must contain ``resume_id``, ``y_true`` (the true label) and
    a ``scores`` column whose value is JSON of ``{job_label: fit_score}``.
    """

    required = {id_column, label_column, score_column}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"Predictions DataFrame missing columns: {missing}")

    rows: list[tuple[int, str, dict[str, float]]] = []
    for _, row in predictions_df.iterrows():
        raw_scores = row[score_column]
        if isinstance(raw_scores, str):
            try:
                parsed = json.loads(raw_scores)
            except json.JSONDecodeError:
                continue
        elif isinstance(raw_scores, dict):
            parsed = raw_scores
        else:
            continue
        rows.append((int(row[id_column]), str(row[label_column]), {str(k): float(v) for k, v in parsed.items()}))

    if not rows:
        empty = {k: RankingMetricsAtK(k=k, precision=0.0, recall=0.0, map=0.0, ndcg=0.0) for k in ks}
        return RankingReport(per_k=empty, per_job={}, n_jobs=0, n_candidates=0)

    label_lookup = {resume_id: label for resume_id, label, _ in rows}
    job_labels: set[str] = set()
    for _, _, scores in rows:
        job_labels.update(scores.keys())
    job_labels.update(label_lookup.values())

    ks_list = sorted(set(int(k) for k in ks))
    per_job: dict[str, dict[int, RankingMetricsAtK]] = {}
    macro_accumulator: dict[int, dict[str, list[float]]] = {
        k: {"precision": [], "recall": [], "map": [], "ndcg": []} for k in ks_list
    }

    for job_label in sorted(job_labels):
        candidates = [
            (scores.get(job_label, float("-inf")), resume_id) for resume_id, _, scores in rows
        ]
        relevances = _ranked_relevances(candidates, job_label, label_lookup)
        if sum(relevances) == 0:
            # Skip jobs that have zero positive examples (no resume in the test set
            # carries that category label) to avoid biasing macro averages.
            continue
        per_k_for_job: dict[int, RankingMetricsAtK] = {}
        for k in ks_list:
            precision, recall = _precision_recall_at_k(relevances, k)
            ap = _average_precision_at_k(relevances, k)
            ndcg = _ndcg_at_k(relevances, k)
            per_k_for_job[k] = RankingMetricsAtK(
                k=k,
                precision=round(precision, 4),
                recall=round(recall, 4),
                map=round(ap, 4),
                ndcg=round(ndcg, 4),
            )
            macro_accumulator[k]["precision"].append(precision)
            macro_accumulator[k]["recall"].append(recall)
            macro_accumulator[k]["map"].append(ap)
            macro_accumulator[k]["ndcg"].append(ndcg)
        per_job[job_label] = per_k_for_job

    per_k: dict[int, RankingMetricsAtK] = {}
    for k, bucket in macro_accumulator.items():
        precisions = bucket["precision"]
        if not precisions:
            per_k[k] = RankingMetricsAtK(k=k, precision=0.0, recall=0.0, map=0.0, ndcg=0.0)
            continue
        per_k[k] = RankingMetricsAtK(
            k=k,
            precision=round(sum(precisions) / len(precisions), 4),
            recall=round(sum(bucket["recall"]) / len(bucket["recall"]), 4),
            map=round(sum(bucket["map"]) / len(bucket["map"]), 4),
            ndcg=round(sum(bucket["ndcg"]) / len(bucket["ndcg"]), 4),
        )

    return RankingReport(per_k=per_k, per_job=per_job, n_jobs=len(per_job), n_candidates=len(rows))


def report_to_dict(report: RankingReport) -> dict[str, object]:
    return {
        "macro_per_k": {str(k): asdict(metrics) for k, metrics in sorted(report.per_k.items())},
        "per_job": {
            job: {str(k): asdict(metrics) for k, metrics in sorted(per_k.items())}
            for job, per_k in report.per_job.items()
        },
        "n_jobs": report.n_jobs,
        "n_candidates": report.n_candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute candidate-job ranking metrics from a predictions CSV.")
    parser.add_argument("--predictions_csv", required=True, help="Path to multi_agent_predictions.csv.")
    parser.add_argument("--output", default="artifacts/ranking_metrics.json")
    parser.add_argument("--ks", default="1,3,5,10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions_csv)
    ks = [int(part.strip()) for part in args.ks.split(",") if part.strip()]
    report = evaluate_ranking(df, ks=ks)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
    macro = {str(k): asdict(metrics) for k, metrics in sorted(report.per_k.items())}
    print(json.dumps({"macro_per_k": macro, "n_jobs": report.n_jobs, "n_candidates": report.n_candidates}, indent=2))
    print(f"Saved ranking metrics to: {output_path}")


if __name__ == "__main__":
    main()
