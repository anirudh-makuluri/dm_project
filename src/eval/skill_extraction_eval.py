"""Skill extraction evaluator: P/R/F1 against an annotated gold set.

Given a gold JSONL produced by :mod:`src.data.build_skill_annotations` and
a mapping ``resume_id -> set[canonical_skill]`` from the agent pipeline,
the evaluator reports:

* **Canonical-level metrics** (primary): per-resume P/R/F1 averaged across
  resumes (macro), and a micro-aggregated P/R/F1 over all resume-skill pairs.
* **Span-level metrics** (secondary, only if the agent provides spans):
  exact-span match P/R/F1.

Per-skill confusion (true-positive / false-positive / false-negative) is
also exported so the report can show error taxonomies.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, Mapping


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    support: int = 0


@dataclass(frozen=True)
class GoldRecord:
    resume_id: int
    category: str
    skills: frozenset[str]


@dataclass
class SkillExtractionReport:
    canonical_macro: PRF
    canonical_micro: PRF
    per_resume: Dict[int, Dict[str, float]] = field(default_factory=dict)
    per_skill: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_resumes: int = 0
    n_gold_skills: int = 0
    n_pred_skills: int = 0


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> PRF:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return PRF(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        support=tp + fn,
    )


def load_gold(path: str | Path) -> list[GoldRecord]:
    """Load the gold annotations JSONL into :class:`GoldRecord` objects."""

    records: list[GoldRecord] = []
    p = Path(path)
    if not p.exists():
        return records
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            skills = payload.get("skills") or [span.get("canonical") for span in payload.get("spans", [])]
            records.append(
                GoldRecord(
                    resume_id=int(payload["resume_id"]),
                    category=str(payload.get("category", "unknown")),
                    skills=frozenset(str(s).strip().lower() for s in skills if s),
                )
            )
    return records


def evaluate_skill_extraction(
    gold: Iterable[GoldRecord],
    predictions: Mapping[int, Iterable[str]],
) -> SkillExtractionReport:
    """Compute canonical-level skill-extraction P/R/F1.

    Parameters
    ----------
    gold:
        Iterable of :class:`GoldRecord` produced by :func:`load_gold`.
    predictions:
        Mapping of ``resume_id -> iterable of canonical skill names``.
    """

    gold_list = list(gold)
    if not gold_list:
        empty = PRF(0.0, 0.0, 0.0, 0)
        return SkillExtractionReport(canonical_macro=empty, canonical_micro=empty)

    per_resume_prf: dict[int, dict[str, float]] = {}
    per_skill_counts: dict[str, Counter[str]] = defaultdict(Counter)
    per_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    macro_precisions: list[float] = []
    macro_recalls: list[float] = []
    macro_f1s: list[float] = []
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    n_gold_skills = 0
    n_pred_skills = 0

    for record in gold_list:
        gold_set = {skill.lower() for skill in record.skills}
        prediction_iter = predictions.get(record.resume_id, [])
        pred_set = {str(skill).strip().lower() for skill in prediction_iter if skill}
        n_gold_skills += len(gold_set)
        n_pred_skills += len(pred_set)

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        prf = _prf(tp, fp, fn)
        per_resume_prf[record.resume_id] = {
            "precision": prf.precision,
            "recall": prf.recall,
            "f1": prf.f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        macro_precisions.append(prf.precision)
        macro_recalls.append(prf.recall)
        macro_f1s.append(prf.f1)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        for skill in gold_set & pred_set:
            per_skill_counts[skill]["tp"] += 1
        for skill in pred_set - gold_set:
            per_skill_counts[skill]["fp"] += 1
        for skill in gold_set - pred_set:
            per_skill_counts[skill]["fn"] += 1

        cat_counter = per_category_counts[record.category]
        cat_counter["tp"] += tp
        cat_counter["fp"] += fp
        cat_counter["fn"] += fn

    canonical_macro = PRF(
        precision=round(sum(macro_precisions) / len(macro_precisions), 4),
        recall=round(sum(macro_recalls) / len(macro_recalls), 4),
        f1=round(sum(macro_f1s) / len(macro_f1s), 4),
        support=len(gold_list),
    )
    canonical_micro = _prf(micro_tp, micro_fp, micro_fn)

    per_skill_summary: dict[str, dict[str, float]] = {}
    for skill, counts in per_skill_counts.items():
        prf = _prf(counts.get("tp", 0), counts.get("fp", 0), counts.get("fn", 0))
        per_skill_summary[skill] = {
            "precision": prf.precision,
            "recall": prf.recall,
            "f1": prf.f1,
            "tp": counts.get("tp", 0),
            "fp": counts.get("fp", 0),
            "fn": counts.get("fn", 0),
        }

    per_category_summary: dict[str, dict[str, float]] = {}
    for category, counts in per_category_counts.items():
        prf = _prf(counts.get("tp", 0), counts.get("fp", 0), counts.get("fn", 0))
        per_category_summary[category] = {
            "precision": prf.precision,
            "recall": prf.recall,
            "f1": prf.f1,
            "tp": counts.get("tp", 0),
            "fp": counts.get("fp", 0),
            "fn": counts.get("fn", 0),
        }

    return SkillExtractionReport(
        canonical_macro=canonical_macro,
        canonical_micro=canonical_micro,
        per_resume=per_resume_prf,
        per_skill=per_skill_summary,
        per_category=per_category_summary,
        n_resumes=len(gold_list),
        n_gold_skills=n_gold_skills,
        n_pred_skills=n_pred_skills,
    )


def report_to_dict(report: SkillExtractionReport) -> dict[str, object]:
    return {
        "canonical_macro": asdict(report.canonical_macro),
        "canonical_micro": asdict(report.canonical_micro),
        "per_resume": report.per_resume,
        "per_skill": report.per_skill,
        "per_category": report.per_category,
        "n_resumes": report.n_resumes,
        "n_gold_skills": report.n_gold_skills,
        "n_pred_skills": report.n_pred_skills,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score skill-extraction predictions against the gold file.")
    parser.add_argument("--gold", default="data/annotations/skills_gold.jsonl")
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSONL file with one record per resume containing resume_id and a 'normalized_skills' list of canonical skill names (or a 'skills' list).",
    )
    parser.add_argument("--output", default="artifacts/skill_extraction_metrics.json")
    return parser.parse_args()


def _load_predictions_jsonl(path: str | Path) -> dict[int, list[str]]:
    predictions: dict[int, list[str]] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Predictions JSONL not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            resume_id = int(record["resume_id"])
            skills_field = (
                record.get("skills")
                or record.get("predicted_skills")
                or [
                    item.get("canonical_skill") or item.get("canonical")
                    for item in record.get("normalized_skills", [])
                ]
            )
            predictions[resume_id] = [str(s) for s in skills_field if s]
    return predictions


def main() -> None:
    args = parse_args()
    gold = load_gold(args.gold)
    predictions = _load_predictions_jsonl(args.predictions)
    report = evaluate_skill_extraction(gold, predictions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
    print(json.dumps({
        "canonical_macro": asdict(report.canonical_macro),
        "canonical_micro": asdict(report.canonical_micro),
        "n_resumes": report.n_resumes,
    }, indent=2))
    print(f"Saved skill-extraction report to: {output_path}")


if __name__ == "__main__":
    main()
