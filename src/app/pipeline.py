"""End-to-end pipeline driver (milestone 4).

Runs the multi-agent evaluation, then layers downstream mining (clustering
and association rules) and writes an explainability report bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..eval.evaluation_runner import run_multi_agent_evaluation
from ..mining.association_rules import mine_association_rules, top_rules
from ..mining.clustering import cluster_candidates, cluster_summary_to_dict
from ..app.report_generator import write_report_bundle


def _load_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(Path(path), lines=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the milestone 4 analysis pipeline.")
    parser.add_argument("--train_jsonl", default=None)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selection_threshold", type=float, default=60.0)
    parser.add_argument("--baseline_predictions", default="artifacts/main_baseline/baseline_predictions.csv")
    parser.add_argument("--skill_gold", default="data/annotations/skills_gold.jsonl")
    parser.add_argument("--disable_supervised", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = _load_jsonl(args.train_jsonl) if args.train_jsonl else None
    test_df = _load_jsonl(args.test_jsonl)
    predictions_df, analyses, evaluation_summary = run_multi_agent_evaluation(
        test_df,
        selection_threshold=args.selection_threshold,
        baseline_predictions_path=args.baseline_predictions,
        train_df=train_df,
        skill_gold_path=args.skill_gold,
        enable_supervised=not args.disable_supervised,
    )

    skill_lists = [[match.canonical_skill for match in analysis.normalized_skills] for analysis in analyses]
    clustered, cluster_summaries = cluster_candidates(skill_lists)
    rules_frame = mine_association_rules(skill_lists)
    rule_summaries = top_rules(rules_frame)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_dir / "pipeline_predictions.csv", index=False)
    if not clustered.empty:
        clustered.to_csv(output_dir / "cluster_assignments.csv", index=False)
    if not rules_frame.empty:
        rules_frame.to_csv(output_dir / "association_rules.csv", index=False)

    report_path, report_json_path = write_report_bundle(
        analyses=analyses,
        clustering_summaries=[cluster_summary_to_dict(summary) for summary in cluster_summaries],
        rule_summaries=[summary.__dict__ for summary in rule_summaries],
        fairness_summary=evaluation_summary.get("fairness", {}),
        output_dir=output_dir,
    )

    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "evaluation": evaluation_summary,
                "report_path": str(report_path),
                "report_json_path": str(report_json_path),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    if evaluation_summary.get("ranking") is not None:
        (output_dir / "ranking_metrics.json").write_text(
            json.dumps(evaluation_summary["ranking"], indent=2), encoding="utf-8"
        )
    if evaluation_summary.get("skill_extraction") is not None:
        (output_dir / "skill_extraction_metrics.json").write_text(
            json.dumps(evaluation_summary["skill_extraction"], indent=2), encoding="utf-8"
        )

    print(f"Saved pipeline summary to: {summary_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved report JSON to: {report_json_path}")


if __name__ == "__main__":
    main()
