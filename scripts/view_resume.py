"""Per-resume explainability viewer.

Prints (or writes) the multi-agent record for one resume in a format that
is easy to drop into the final report: extracted fields, normalized
canonical skills, experience features, the chosen-job evaluation with
matched / missing skills, the LLM rationale, and the rationale-quality
audit for that single record.

Usage examples
--------------

  python scripts/view_resume.py --id 112
  python scripts/view_resume.py --id 112 --format markdown
  python scripts/view_resume.py --id 112 --format json --out report_card.json
  python scripts/view_resume.py --records artifacts/main_multi_agent_lmstudio/multi_agent_records.jsonl --id 1311
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running the script directly without `python -m`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rationale_quality import audit_record  # noqa: E402


DEFAULT_RECORDS = "artifacts/main_multi_agent_lmstudio/multi_agent_records.jsonl"


def _find_record(records_path: Path, resume_id: int) -> dict[str, Any]:
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if int(rec.get("resume_id", -1)) == resume_id:
                return rec
    raise SystemExit(f"resume_id={resume_id} not found in {records_path}")


def _format_skills(skills: list[dict[str, Any]], top: int = 12) -> list[str]:
    sorted_skills = sorted(skills, key=lambda s: s.get("confidence", 0.0), reverse=True)[:top]
    return [
        f"{s.get('canonical_skill', '?')} ({s.get('confidence', 0.0):.2f}) "
        f"<- {s.get('phrase') or s.get('source_phrase', '?')!r}"
        for s in sorted_skills
    ]


def render_text(rec: dict[str, Any]) -> str:
    extracted = rec.get("extracted") or {}
    normalized = rec.get("normalized_skills") or []
    experience = rec.get("experience") or {}
    evaluation = rec.get("evaluation") or {}
    extra = rec.get("extra") or {}
    audit = audit_record(rec)

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"RESUME {rec.get('resume_id')}  true_label={rec.get('label')}")
    lines.append("=" * 78)

    lines.append("")
    lines.append("# Decision")
    lines.append(
        f"  agents_only      : {extra.get('agents_only_label')}  "
        f"(top fit={extra.get('agents_only_top', 0):.2f}, margin={extra.get('agents_only_margin', 0):.2f})"
    )
    lines.append(
        f"  ensemble         : {extra.get('ensemble_label')}  "
        f"(decision_source={extra.get('decision_source')})"
    )
    lines.append(f"  baseline_label   : {extra.get('baseline_label')}")
    lines.append(f"  supervised_label : {extra.get('supervised_label')}")

    lines.append("")
    lines.append("# Extractor")
    lines.append(f"  education        : {(extracted.get('education') or [])[:4]}")
    lines.append(f"  experience[:4]   : {(extracted.get('experience') or [])[:4]}")
    lines.append(f"  skills_raw count : {len(extracted.get('skills_raw') or [])}")
    lines.append(f"  skills_raw[:10]  : {(extracted.get('skills_raw') or [])[:10]}")
    lines.append(f"  certifications   : {(extracted.get('certifications') or [])[:4]}")

    lines.append("")
    lines.append(f"# Skill miner ({len(normalized)} canonical matches, top 12)")
    for entry in _format_skills(normalized):
        lines.append(f"  {entry}")

    lines.append("")
    lines.append("# Experience")
    lines.append(
        f"  years={experience.get('total_years', 0)}  "
        f"roles={experience.get('role_count', 0)}  "
        f"leadership={experience.get('leadership_indicators', 0)}  "
        f"impact={experience.get('impact_statements', 0)}  "
        f"quantified={experience.get('quantified_impact', 0)}"
    )
    if experience.get("evidence"):
        lines.append("  evidence:")
        for ev in (experience.get("evidence") or [])[:3]:
            lines.append(f"    - {ev[:160]}")

    lines.append("")
    lines.append(
        f"# Evaluator (against ensemble label '{extra.get('ensemble_label')}')"
    )
    lines.append(f"  fit_score        : {evaluation.get('fit_score', 0):.2f}")
    lines.append(f"  matched_skills   : {list(evaluation.get('matched_skills') or [])}")
    lines.append(f"  missing_skills   : {list(evaluation.get('missing_skills') or [])[:8]}")
    cs = evaluation.get("component_scores") or {}
    lines.append(
        "  components       : "
        + ", ".join(f"{k}={v:.2f}" for k, v in cs.items())
    )
    lines.append("  strengths        : " + str(list(evaluation.get("strengths") or [])))
    lines.append("  gaps             : " + str(list(evaluation.get("gaps") or [])))
    lines.append("  rationale        : " + (evaluation.get("rationale") or "").strip())
    lines.append("  recommendation   : " + (evaluation.get("recommendation") or "").strip())

    lines.append("")
    lines.append("# Rationale-quality audit")
    lines.append(
        f"  fit_band={audit.fit_band}  recommendation_tone={audit.recommendation_tone}  "
        f"calibrated={audit.calibrated}"
    )
    lines.append(
        f"  grounded_in_matched={audit.grounded_in_matched} "
        f"(hits={audit.matched_skill_hits})"
    )
    lines.append(
        f"  grounded_in_missing={audit.grounded_in_missing} "
        f"(hits={audit.missing_skill_hits})"
    )
    lines.append(
        f"  word_count={audit.word_count}  is_specific={audit.is_specific}  "
        f"identity_clean={audit.identity_clean}"
    )
    if audit.identity_terms:
        lines.append(f"  identity_terms_flagged={audit.identity_terms}")
    lines.append(
        f"  strengths_grounded_ratio={audit.strengths_grounded_ratio:.2f}  "
        f"gaps_grounded_ratio={audit.gaps_grounded_ratio:.2f}"
    )
    if audit.strengths_invented:
        lines.append(f"  strengths_invented={audit.strengths_invented}")
    if audit.gaps_invented:
        lines.append(f"  gaps_invented={audit.gaps_invented}")

    return "\n".join(lines)


def render_markdown(rec: dict[str, Any]) -> str:
    extracted = rec.get("extracted") or {}
    normalized = rec.get("normalized_skills") or []
    experience = rec.get("experience") or {}
    evaluation = rec.get("evaluation") or {}
    extra = rec.get("extra") or {}
    audit = audit_record(rec)

    md: list[str] = []
    md.append(f"# Resume {rec.get('resume_id')} — true label `{rec.get('label')}`")
    md.append("")
    md.append("## Decision")
    md.append("")
    md.append("| Path | Label | Notes |")
    md.append("|---|---|---|")
    md.append(
        f"| agents_only | `{extra.get('agents_only_label')}` | "
        f"top fit {extra.get('agents_only_top', 0):.2f}, margin {extra.get('agents_only_margin', 0):.2f} |"
    )
    md.append(
        f"| ensemble | `{extra.get('ensemble_label')}` | source: {extra.get('decision_source')} |"
    )
    md.append(f"| baseline | `{extra.get('baseline_label')}` |  |")
    md.append(f"| supervised | `{extra.get('supervised_label')}` |  |")
    md.append("")

    md.append("## Skill miner — canonical matches")
    md.append("")
    md.append("| Canonical | Confidence | Source phrase |")
    md.append("|---|---:|---|")
    for s in sorted(normalized, key=lambda x: x.get("confidence", 0), reverse=True)[:12]:
        phrase = s.get("phrase") or s.get("source_phrase", "?")
        md.append(
            f"| `{s.get('canonical_skill', '?')}` | {s.get('confidence', 0.0):.2f} | `{phrase}` |"
        )
    md.append("")

    md.append("## Experience")
    md.append("")
    md.append(
        f"- Years: **{experience.get('total_years', 0)}**, roles: {experience.get('role_count', 0)}\n"
        f"- Leadership indicators: {experience.get('leadership_indicators', 0)}, "
        f"impact statements: {experience.get('impact_statements', 0)} "
        f"(quantified: {experience.get('quantified_impact', 0)})"
    )
    if experience.get("evidence"):
        md.append("")
        md.append("Supporting evidence:")
        for ev in (experience.get("evidence") or [])[:3]:
            md.append(f"- {ev[:200]}")
    md.append("")

    md.append(f"## Evaluator (against `{extra.get('ensemble_label')}`)")
    md.append("")
    md.append(f"- `fit_score` = **{evaluation.get('fit_score', 0):.2f}**")
    md.append(f"- Matched skills: {list(evaluation.get('matched_skills') or [])}")
    md.append(f"- Missing skills: {list(evaluation.get('missing_skills') or [])[:8]}")
    md.append("")
    md.append(f"> {(evaluation.get('rationale') or '').strip()}")
    md.append("")
    md.append(f"_Recommendation_: {(evaluation.get('recommendation') or '').strip()}")
    md.append("")

    md.append("## Rationale-quality audit")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| fit_band | `{audit.fit_band}` |")
    md.append(f"| recommendation_tone | `{audit.recommendation_tone}` |")
    md.append(f"| calibrated | `{audit.calibrated}` |")
    md.append(f"| grounded_in_matched | `{audit.grounded_in_matched}` ({audit.matched_skill_hits}) |")
    md.append(f"| grounded_in_missing | `{audit.grounded_in_missing}` ({audit.missing_skill_hits}) |")
    md.append(f"| word_count | {audit.word_count} (specific={audit.is_specific}) |")
    md.append(f"| identity_clean | `{audit.identity_clean}` |")
    md.append(
        f"| strengths_grounded_ratio | {audit.strengths_grounded_ratio:.2f} |"
    )
    md.append(f"| gaps_grounded_ratio | {audit.gaps_grounded_ratio:.2f} |")
    if audit.identity_terms:
        md.append(f"| identity_terms_flagged | {audit.identity_terms} |")
    if audit.strengths_invented:
        md.append(f"| strengths_invented | {audit.strengths_invented} |")
    if audit.gaps_invented:
        md.append(f"| gaps_invented | {audit.gaps_invented} |")
    md.append("")

    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, required=True, help="resume_id to render")
    parser.add_argument(
        "--records",
        default=DEFAULT_RECORDS,
        help=f"Path to multi_agent_records.jsonl (default: {DEFAULT_RECORDS})",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="If set, write to this file instead of stdout.",
    )
    args = parser.parse_args()

    records_path = Path(args.records)
    if not records_path.exists():
        raise SystemExit(f"records file not found: {records_path}")
    rec = _find_record(records_path, args.id)

    if args.format == "json":
        payload = dict(rec)
        payload["rationale_quality_audit"] = {
            "calibrated": audit_record(rec).calibrated,
            "grounded_in_matched": audit_record(rec).grounded_in_matched,
            "grounded_in_missing": audit_record(rec).grounded_in_missing,
            "identity_clean": audit_record(rec).identity_clean,
        }
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    elif args.format == "markdown":
        rendered = render_markdown(rec)
    else:
        rendered = render_text(rec)

    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.format} report to {args.out}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
