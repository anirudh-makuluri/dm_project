"""End-to-end multi-agent evaluation runner.

Runs the four-agent pipeline on a test JSONL, then reports two prediction
modes side-by-side so the multi-agent contribution can be isolated:

* **agents_only**: argmax over per-job deterministic fit scores produced
  by the agent stack. This is the "honest" multi-agent prediction.
* **ensemble**: combines agents_only with the TF-IDF baseline and a
  supervised LinearSVC classifier (an extra strong baseline retained for
  downstream comparisons). Toggle with ``--disable_supervised``.

Outputs (under ``--output_dir``):

* ``multi_agent_metrics.json``                -- ensemble metrics + comparisons
* ``multi_agent_metrics_agents_only.json``    -- agents-only metrics + comparisons
* ``multi_agent_predictions.csv``             -- ensemble predictions
* ``multi_agent_predictions_agents_only.csv`` -- agents-only predictions
* ``multi_agent_records.jsonl``               -- per-resume agent analyses
* ``skill_extraction_metrics.json``           -- if a gold annotations file is provided
* ``ranking_metrics.json``                    -- nDCG@k / MAP@k over per-job fit scores
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..agents.evaluator_agent import EvaluatorAgent
from ..agents.experience_agent import ExperienceAgent
from ..agents.extractor_agent import ExtractorAgent
from ..agents.skill_miner_agent import SkillMinerAgent
from ..fairness.fairness_metrics import audit_fairness, fairness_result_to_dict
from ..fairness.perturbation import build_perturbations
from ..llm.provider import LLMProvider, build_provider
from ..models.agent_features import AgentFeatureExtractor
from ..models.profile_builder import load_profiles as load_learned_profiles
from ..models.scoring import (
    DEFAULT_JOB_PROFILES,
    build_evaluation_result,
    get_job_profile,
    has_learned_profiles,
    register_learned_profiles,
)
from ..models.supervised_classifier import SupervisedResumeClassifier
from ..rag.retriever import SkillRetriever
from ..rag.vector_store import SkillVectorStore
from ..shared import CandidateAnalysis, JobProfile
from .metrics import compute_classification_metrics, metrics_to_dict
from .ranking_metrics import evaluate_ranking, report_to_dict as ranking_report_to_dict
from .rationale_quality import (
    evaluate_rationale_quality,
    report_to_dict as rationale_report_to_dict,
)
from .skill_extraction_eval import (
    evaluate_skill_extraction,
    load_gold,
    report_to_dict as skill_report_to_dict,
)

logger = logging.getLogger(__name__)


def _load_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(Path(path), lines=True)


def _load_predictions(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return pd.read_csv(candidate)


def _baseline_prediction_map(baseline_df: pd.DataFrame | None) -> dict[int, str]:
    if baseline_df is None or not {"resume_id", "y_pred"}.issubset(baseline_df.columns):
        return {}
    out: dict[int, str] = {}
    for _, row in baseline_df.iterrows():
        try:
            out[int(row["resume_id"])] = str(row["y_pred"])
        except Exception:
            continue
    return out


def _profile_label_space(baseline_df: pd.DataFrame | None) -> list[str]:
    labels = set(DEFAULT_JOB_PROFILES)
    if baseline_df is not None and "y_pred" in baseline_df.columns:
        labels.update(str(value) for value in baseline_df["y_pred"].dropna().astype(str).tolist())
    return sorted(labels)


def _supervised_prediction_map(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None,
    train_extra: "np.ndarray | None" = None,
    test_extra: "np.ndarray | None" = None,
) -> tuple[dict[int, str], dict[int, float]]:
    if train_df is None:
        return {}, {}
    required_columns = {"resume_text", "label"}
    if not required_columns.issubset(train_df.columns):
        return {}, {}

    classifier = SupervisedResumeClassifier()
    classifier.fit(
        train_df,
        text_col="resume_text",
        label_col="label",
        extra_features=train_extra,
    )
    predictions, margins = classifier.predict_with_margin(
        test_df["resume_text"].astype(str),
        extra_features=test_extra,
    )

    pred_map: dict[int, str] = {}
    margin_map: dict[int, float] = {}
    for row, label, margin in zip(test_df.itertuples(index=False), predictions, margins):
        resume_id = int(getattr(row, "resume_id", len(pred_map)))
        pred_map[resume_id] = str(label)
        margin_map[resume_id] = float(margin)
    return pred_map, margin_map


def _agents_only_choice(scores: dict[str, float]) -> tuple[str, float, float, float]:
    """Pick the highest-scoring job; return (label, top, second, margin)."""

    if not scores:
        return "unknown", 0.0, 0.0, 0.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else best_score
    return best_label, best_score, second_score, best_score - second_score


#: Threshold on the LinearSVC top-minus-runner-up decision-function margin
#: below which we consider the supervised classifier "uncertain" and allow
#: the multi-agent + baseline voters to override it.  Calibrated on the
#: main validation slice: see ``scripts/ensemble_ablation.py`` for the
#: ablation.  Lower values are more conservative (fewer overrides).
SUPERVISED_UNCERTAIN_MARGIN = 0.5


def _ensemble_choice(
    scores: dict[str, float],
    baseline_label: str | None,
    supervised_label: str | None,
    *,
    supervised_margin: float | None = None,
) -> tuple[str, str]:
    """Combine agents + baseline + supervised. Returns (label, decision_source).

    When the supervised classifier is enabled, the default rule lets it
    win in all disagreements.  However, when the supervised margin is
    below ``SUPERVISED_UNCERTAIN_MARGIN`` *and* the multi-agent vote
    independently agrees with the TF-IDF baseline (two weak signals
    aligning against a weak SVC), we promote that joint vote to the
    final decision.  The ablation in ``scripts/ensemble_ablation.py``
    shows this rule lifts ensemble accuracy by ~0.2 points on the main
    eval slice without regressing baseline or supervised behaviour.
    """

    if not scores:
        if supervised_label is not None:
            return supervised_label, "supervised-agent"
        if baseline_label is not None:
            return baseline_label, "baseline-prior"
        return "unknown", "unknown"

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else best_score
    margin = best_score - second_score

    if supervised_label is not None:
        if baseline_label is not None and supervised_label == baseline_label:
            return supervised_label, "supervised+baseline-agree"
        # New: weak-SVC + agent/baseline agreement override.
        if (
            baseline_label is not None
            and best_label == baseline_label
            and best_label != supervised_label
            and supervised_margin is not None
            and supervised_margin < SUPERVISED_UNCERTAIN_MARGIN
        ):
            return best_label, "agent+baseline-override"
        return supervised_label, "supervised-agent"

    if baseline_label is None:
        return best_label, "multi-agent"
    if baseline_label == best_label:
        return best_label, "ensemble-agree"
    if best_score < 45.0 or margin < 8.0:
        return baseline_label, "baseline-prior"
    return best_label, "multi-agent"


def _candidate_to_record(analysis: CandidateAnalysis) -> dict[str, object]:
    return {
        "resume_id": analysis.resume_id,
        "label": analysis.label,
        "experience": asdict(analysis.experience),
        "normalized_skills": [asdict(skill) for skill in analysis.normalized_skills],
        "extracted": asdict(analysis.extracted),
        "evaluation": asdict(analysis.evaluation) if analysis.evaluation else None,
        "extra": analysis.extra,
    }


def _decision_source_breakdown(prediction_rows: list[dict[str, object]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in prediction_rows:
        counter[str(row.get("decision_source", "unknown"))] += 1
    return dict(counter)


def _comparison(metrics_a, metrics_b) -> dict[str, float]:
    """Return ``a - b`` deltas across the standard classification metrics."""

    return {
        "accuracy_delta": round(metrics_a.accuracy - metrics_b.accuracy, 4),
        "precision_macro_delta": round(metrics_a.precision_macro - metrics_b.precision_macro, 4),
        "recall_macro_delta": round(metrics_a.recall_macro - metrics_b.recall_macro, 4),
        "f1_macro_delta": round(metrics_a.f1_macro - metrics_b.f1_macro, 4),
        "f1_weighted_delta": round(metrics_a.f1_weighted - metrics_b.f1_weighted, 4),
    }


def run_multi_agent_evaluation(
    test_df: pd.DataFrame,
    selection_threshold: float = 60.0,
    baseline_predictions_path: str | Path | None = None,
    train_df: pd.DataFrame | None = None,
    skill_gold_path: str | Path | None = "data/annotations/skills_gold.jsonl",
    enable_supervised: bool = True,
    use_agent_features: bool = False,
    llm_provider: Optional[LLMProvider] = None,
) -> tuple[pd.DataFrame, list[CandidateAnalysis], dict[str, object]]:
    """Run the multi-agent evaluation and return (ensemble_predictions, analyses, summary).

    The returned predictions DataFrame is the **ensemble** path (which is the
    headline number reported in artifacts). The summary dictionary contains
    both ensemble and agents-only metrics, plus skill-extraction and ranking
    evaluations when their inputs are available.
    """

    provider = llm_provider if llm_provider is not None else build_provider()
    logger.info("Multi-agent runner using LLM backend: %s", provider.name)

    extractor = ExtractorAgent(provider=provider)
    retriever = SkillRetriever(SkillVectorStore())
    skill_miner = SkillMinerAgent(retriever, provider=provider)
    experience_agent = ExperienceAgent(provider=provider)
    evaluator_with_rationale = EvaluatorAgent(provider=provider)

    baseline_df = _load_predictions(baseline_predictions_path)
    baseline_map = _baseline_prediction_map(baseline_df)
    label_space = set(_profile_label_space(baseline_df))

    train_extra: np.ndarray | None = None
    test_extra: np.ndarray | None = None
    if enable_supervised and use_agent_features and train_df is not None:
        logger.info("Computing agent feature matrices (rule-based, no LLM calls)...")
        feature_extractor = AgentFeatureExtractor()
        train_extra = feature_extractor.transform(train_df)
        test_extra = feature_extractor.transform(test_df)
        logger.info(
            "Agent features ready: train=%s test=%s",
            train_extra.shape,
            test_extra.shape,
        )

    if enable_supervised:
        supervised_map, supervised_margin_map = _supervised_prediction_map(
            test_df,
            train_df,
            train_extra=train_extra,
            test_extra=test_extra,
        )
        label_space.update(supervised_map.values())
    else:
        supervised_map, supervised_margin_map = {}, {}
    profiles: dict[str, JobProfile] = {label: get_job_profile(label) for label in sorted(label_space)}

    analyses: list[CandidateAnalysis] = []
    ensemble_rows: list[dict[str, object]] = []
    agents_only_rows: list[dict[str, object]] = []
    original_scores: list[float] = []
    perturbed_scores: list[float] = []

    for _, row in test_df.iterrows():
        resume_id = int(row.get("resume_id", len(analyses)))
        label = str(row.get("label", "unknown"))
        text = str(row.get("resume_text", ""))

        extracted = extractor.extract(text, resume_id=resume_id)
        normalized_skills = tuple(skill_miner.normalize(extracted))
        experience = experience_agent.analyze(extracted)

        baseline_label = baseline_map.get(resume_id)
        supervised_label = supervised_map.get(resume_id) if enable_supervised else None
        supervised_margin = supervised_margin_map.get(resume_id) if enable_supervised else None
        for new_label in (baseline_label, supervised_label):
            if new_label and new_label not in profiles:
                profiles[new_label] = get_job_profile(new_label)

        scores: dict[str, float] = {}
        for job_label, profile in profiles.items():
            deterministic = build_evaluation_result(normalized_skills, experience, profile, text)
            scores[job_label] = deterministic.fit_score

        agents_label, agents_top, agents_second, agents_margin = _agents_only_choice(scores)
        ensemble_label, decision_source = _ensemble_choice(
            scores,
            baseline_label,
            supervised_label,
            supervised_margin=supervised_margin,
        )

        chosen_eval = evaluator_with_rationale.evaluate(
            normalized_skills, experience, profiles[ensemble_label], text
        )
        analysis = CandidateAnalysis(
            resume_id=resume_id,
            label=label,
            extracted=extracted,
            normalized_skills=normalized_skills,
            experience=experience,
            evaluation=chosen_eval,
            extra={
                "scores": scores,
                "agents_only_label": agents_label,
                "agents_only_top": round(agents_top, 4),
                "agents_only_second": round(agents_second, 4),
                "agents_only_margin": round(agents_margin, 4),
                "ensemble_label": ensemble_label,
                "decision_source": decision_source,
                "baseline_label": baseline_label,
                "supervised_label": supervised_label,
                "supervised_margin": round(supervised_margin, 4) if supervised_margin is not None else None,
                "llm_backend": provider.name,
            },
        )
        analyses.append(analysis)
        ensemble_rows.append(
            {
                "resume_id": resume_id,
                "y_true": label,
                "y_pred": ensemble_label,
                "fit_score": chosen_eval.fit_score,
                "decision_source": decision_source,
                "scores": json.dumps(scores, sort_keys=True),
            }
        )
        agents_only_rows.append(
            {
                "resume_id": resume_id,
                "y_true": label,
                "y_pred": agents_label,
                "fit_score": scores.get(agents_label, 0.0),
                "decision_source": "agents-only",
                "scores": json.dumps(scores, sort_keys=True),
            }
        )

        perturbed_variants = build_perturbations(text, seed=resume_id)
        if perturbed_variants:
            original_scores.append(chosen_eval.fit_score)
            perturbed_text = perturbed_variants[-1].text
            perturbed_extracted = extractor.extract(perturbed_text, resume_id=resume_id)
            perturbed_skills = tuple(skill_miner.normalize(perturbed_extracted))
            perturbed_experience = experience_agent.analyze(perturbed_extracted)
            perturbed_eval = evaluator_with_rationale.evaluate(
                perturbed_skills, perturbed_experience, profiles[ensemble_label], perturbed_text
            )
            perturbed_scores.append(perturbed_eval.fit_score)

    ensemble_df = pd.DataFrame(ensemble_rows)
    agents_only_df = pd.DataFrame(agents_only_rows)

    ensemble_metrics = compute_classification_metrics(
        ensemble_df["y_true"].tolist(), ensemble_df["y_pred"].tolist()
    )
    agents_metrics = compute_classification_metrics(
        agents_only_df["y_true"].tolist(), agents_only_df["y_pred"].tolist()
    )

    baseline_summary = None
    ensemble_vs_baseline = None
    agents_vs_baseline = None
    ensemble_vs_agents = _comparison(ensemble_metrics, agents_metrics)
    if baseline_df is not None and {"y_true", "y_pred"}.issubset(baseline_df.columns):
        baseline_metrics = compute_classification_metrics(
            baseline_df["y_true"].astype(str).tolist(),
            baseline_df["y_pred"].astype(str).tolist(),
        )
        baseline_summary = metrics_to_dict(baseline_metrics)
        ensemble_vs_baseline = _comparison(ensemble_metrics, baseline_metrics)
        agents_vs_baseline = _comparison(agents_metrics, baseline_metrics)

    fairness = audit_fairness(original_scores, perturbed_scores, selection_threshold=selection_threshold)

    skill_extraction_summary: dict[str, object] | None = None
    if skill_gold_path:
        gold_records = load_gold(skill_gold_path)
        if gold_records:
            predictions_for_gold = {
                analysis.resume_id: [match.canonical_skill for match in analysis.normalized_skills]
                for analysis in analyses
                if analysis.resume_id is not None
            }
            report = evaluate_skill_extraction(gold_records, predictions_for_gold)
            skill_extraction_summary = skill_report_to_dict(report)

    ranking_summary: dict[str, object] | None = None
    try:
        ranking_summary = ranking_report_to_dict(evaluate_ranking(ensemble_df))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Ranking evaluation failed: %s", exc)
        ranking_summary = None

    # Rationale-quality audit over the chosen-ensemble-label evaluations.
    rationale_summary: dict[str, object] | None = None
    try:
        records_for_rationale = [_candidate_to_record(a) for a in analyses]
        rationale_summary = rationale_report_to_dict(
            evaluate_rationale_quality(records_for_rationale),
            include_audits=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Rationale-quality evaluation failed: %s", exc)
        rationale_summary = None

    summary = {
        "metrics": metrics_to_dict(ensemble_metrics),
        "agents_only_metrics": metrics_to_dict(agents_metrics),
        "baseline_metrics": baseline_summary,
        "comparison": ensemble_vs_baseline,
        "agents_vs_baseline": agents_vs_baseline,
        "ensemble_vs_agents": ensemble_vs_agents,
        "decision_source_breakdown": _decision_source_breakdown(ensemble_rows),
        "fairness": fairness_result_to_dict(fairness),
        "ranking": ranking_summary,
        "skill_extraction": skill_extraction_summary,
        "rationale_quality": rationale_summary,
        "providers": {
            "llm_backend": provider.name,
            "llm_model": provider.model,
            "embedding_backend": retriever.store.backend,
            "profile_source": "learned" if has_learned_profiles() else "hand-curated",
            "uses_agent_features": bool(use_agent_features),
        },
        "analyses": [_candidate_to_record(analysis) for analysis in analyses],
    }
    return ensemble_df, analyses, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-agent evaluation and fairness audit.")
    parser.add_argument("--train_jsonl", default=None)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selection_threshold", type=float, default=60.0)
    parser.add_argument("--baseline_predictions", default="artifacts/main_baseline/baseline_predictions.csv")
    parser.add_argument(
        "--skill_gold",
        default="data/annotations/skills_gold.jsonl",
        help="Optional gold annotations file for skill-extraction P/R/F1 evaluation.",
    )
    parser.add_argument(
        "--disable_supervised",
        action="store_true",
        help="Disable the LinearSVC ensemble path (agent-only ablation).",
    )
    parser.add_argument(
        "--use_agent_features",
        action="store_true",
        help=(
            "Augment the LinearSVC's TF-IDF features with rule-based "
            "agent features (per-profile fit scores, experience, skill "
            "category histogram). No additional LLM calls."
        ),
    )
    parser.add_argument(
        "--learned_profiles",
        default=None,
        help=(
            "Path to a learned-profiles JSON (produced by "
            "src.models.profile_builder). When provided, these profiles "
            "override the hand-curated DEFAULT_JOB_PROFILES inside the "
            "agent fit-score scorer."
        ),
    )
    return parser.parse_args()


def _save_run(
    output_dir: Path,
    predictions_df: pd.DataFrame,
    analyses: list[CandidateAnalysis],
    summary: dict[str, object],
    *,
    metrics_filename: str,
    predictions_filename: str,
    records_filename: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / predictions_filename
    metrics_path = output_dir / metrics_filename
    predictions_df.to_csv(predictions_path, index=False, quoting=csv.QUOTE_MINIMAL)
    metrics_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if records_filename is not None:
        records_path = output_dir / records_filename
        with records_path.open("w", encoding="utf-8") as handle:
            for analysis in analyses:
                handle.write(json.dumps(_candidate_to_record(analysis), ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    train_df = _load_jsonl(args.train_jsonl) if args.train_jsonl else None
    test_df = _load_jsonl(args.test_jsonl)
    output_dir = Path(args.output_dir)

    if args.learned_profiles:
        learned = load_learned_profiles(args.learned_profiles)
        register_learned_profiles(learned)
        logger.info(
            "Registered %d learned job profiles from %s",
            len(learned),
            args.learned_profiles,
        )
        print(f"Registered {len(learned)} learned profiles from {args.learned_profiles}", flush=True)

    if args.disable_supervised:
        predictions_df, analyses, summary = run_multi_agent_evaluation(
            test_df,
            selection_threshold=args.selection_threshold,
            baseline_predictions_path=args.baseline_predictions,
            train_df=train_df,
            skill_gold_path=args.skill_gold,
            enable_supervised=False,
            use_agent_features=args.use_agent_features,
        )
        _save_run(
            output_dir,
            predictions_df,
            analyses,
            summary,
            metrics_filename="multi_agent_metrics_agents_only.json",
            predictions_filename="multi_agent_predictions_agents_only.csv",
            records_filename="multi_agent_records.jsonl",
        )
        print(json.dumps(summary["metrics"], indent=2))
        print(f"Saved agents-only artifacts to: {output_dir}")
        return

    # Default: run the full ensemble AND save the agents-only ablation.
    ensemble_df, analyses, summary = run_multi_agent_evaluation(
        test_df,
        selection_threshold=args.selection_threshold,
        baseline_predictions_path=args.baseline_predictions,
        train_df=train_df,
        skill_gold_path=args.skill_gold,
        enable_supervised=True,
        use_agent_features=args.use_agent_features,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensemble outputs
    ensemble_df.to_csv(output_dir / "multi_agent_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    (output_dir / "multi_agent_metrics.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Agents-only outputs (built from the same analyses; no second pass needed)
    agents_rows = []
    for analysis in analyses:
        scores = analysis.extra.get("scores", {})
        label = analysis.extra.get("agents_only_label", "unknown")
        agents_rows.append(
            {
                "resume_id": analysis.resume_id,
                "y_true": analysis.label,
                "y_pred": label,
                "fit_score": scores.get(label, 0.0) if isinstance(scores, dict) else 0.0,
                "decision_source": "agents-only",
                "scores": json.dumps(scores, sort_keys=True) if isinstance(scores, dict) else "{}",
            }
        )
    agents_df = pd.DataFrame(agents_rows)
    agents_df.to_csv(
        output_dir / "multi_agent_predictions_agents_only.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    agents_only_summary = {
        "metrics": summary.get("agents_only_metrics"),
        "baseline_metrics": summary.get("baseline_metrics"),
        "comparison": summary.get("agents_vs_baseline"),
        "fairness": summary.get("fairness"),
        "providers": summary.get("providers"),
    }
    (output_dir / "multi_agent_metrics_agents_only.json").write_text(
        json.dumps(agents_only_summary, indent=2, default=str), encoding="utf-8"
    )

    # Per-resume records
    with (output_dir / "multi_agent_records.jsonl").open("w", encoding="utf-8") as handle:
        for analysis in analyses:
            handle.write(json.dumps(_candidate_to_record(analysis), ensure_ascii=False) + "\n")

    # Standalone metric files for direct downstream consumption
    if summary.get("ranking") is not None:
        (output_dir / "ranking_metrics.json").write_text(
            json.dumps(summary["ranking"], indent=2), encoding="utf-8"
        )
    if summary.get("skill_extraction") is not None:
        (output_dir / "skill_extraction_metrics.json").write_text(
            json.dumps(summary["skill_extraction"], indent=2), encoding="utf-8"
        )

    # Rationale-quality artifacts: aggregate metrics + per-record audits so the
    # report can quote individual examples.
    if summary.get("rationale_quality") is not None:
        (output_dir / "rationale_quality.json").write_text(
            json.dumps(summary["rationale_quality"], indent=2), encoding="utf-8"
        )
        try:
            records_for_audit = [_candidate_to_record(a) for a in analyses]
            full_report = evaluate_rationale_quality(records_for_audit)
            full_payload = rationale_report_to_dict(full_report, include_audits=True)
            (output_dir / "rationale_quality_audits.jsonl").write_text(
                "\n".join(json.dumps(audit, ensure_ascii=False) for audit in full_payload.get("audits", [])),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to write rationale_quality_audits.jsonl: %s", exc)

    headline = {
        "ensemble": summary["metrics"],
        "agents_only": summary["agents_only_metrics"],
        "baseline": summary["baseline_metrics"],
        "ensemble_vs_baseline": summary["comparison"],
        "agents_vs_baseline": summary["agents_vs_baseline"],
        "ensemble_vs_agents": summary["ensemble_vs_agents"],
        "decision_source_breakdown": summary["decision_source_breakdown"],
        "rationale_quality": summary.get("rationale_quality"),
        "providers": summary["providers"],
    }
    print(json.dumps(headline, indent=2))
    print(f"Saved artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
