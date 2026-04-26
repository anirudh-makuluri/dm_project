from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..fairness.fairness_metrics import fairness_result_to_dict
from ..mining.association_rules import rule_summary_to_dict
from ..mining.clustering import cluster_summary_to_dict
from ..shared import CandidateAnalysis


def format_candidate_report(analysis: CandidateAnalysis) -> str:
    evaluation = analysis.evaluation
    lines = [
        f"# Candidate Report: Resume {analysis.resume_id}",
        "",
        f"Label: {analysis.label or 'unknown'}",
        f"Fit score: {evaluation.fit_score if evaluation else 0.0}",
        f"Recommendation: {evaluation.recommendation if evaluation else 'n/a'}",
        "",
        "## Extracted Skills",
        ", ".join(match.canonical_skill for match in analysis.normalized_skills) or "None detected",
        "",
        "## Strengths",
        ", ".join(evaluation.strengths) if evaluation and evaluation.strengths else "None",
        "",
        "## Gaps",
        ", ".join(evaluation.gaps) if evaluation and evaluation.gaps else "None",
        "",
        "## Rationale",
        evaluation.rationale if evaluation else "n/a",
        "",
        "## Experience Signals",
        f"Years: {analysis.experience.total_years}",
        f"Leadership indicators: {analysis.experience.leadership_indicators}",
        f"Impact statements: {analysis.experience.impact_statements}",
    ]
    return "\n".join(lines)


def write_report_bundle(
    analyses: list[CandidateAnalysis],
    clustering_summaries: list[dict[str, object]] | None,
    rule_summaries: list[dict[str, object]] | None,
    fairness_summary: dict[str, object] | None,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "explainability_report.md"
    json_path = output_path / "explainability_report.json"

    report_lines = ["# Multi-Agent Resume Analysis Report", ""]
    report_lines.append(f"Candidates analyzed: {len(analyses)}")
    if fairness_summary:
        report_lines.extend([
            "",
            "## Fairness Audit",
            json.dumps(fairness_summary, indent=2),
        ])
    if clustering_summaries:
        report_lines.extend([
            "",
            "## Clusters",
            json.dumps(clustering_summaries, indent=2),
        ])
    if rule_summaries:
        report_lines.extend([
            "",
            "## Association Rules",
            json.dumps(rule_summaries, indent=2),
        ])
    for analysis in analyses[:20]:
        report_lines.extend(["", format_candidate_report(analysis)])

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "analyses": [asdict(analysis) for analysis in analyses],
                "clustering": clustering_summaries or [],
                "association_rules": rule_summaries or [],
                "fairness": fairness_summary or {},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return report_path, json_path


def serialize_cluster_summaries(summaries):
    return [cluster_summary_to_dict(summary) for summary in summaries]


def serialize_rule_summaries(summaries):
    return [rule_summary_to_dict(summary) for summary in summaries]


def serialize_fairness_summary(summary):
    return fairness_result_to_dict(summary) if hasattr(summary, "__dataclass_fields__") else summary
