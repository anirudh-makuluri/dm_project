from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..agents.evaluator_agent import EvaluatorAgent
from ..agents.experience_agent import ExperienceAgent
from ..agents.extractor_agent import ExtractorAgent
from ..agents.skill_miner_agent import SkillMinerAgent
from ..fairness.fairness_metrics import audit_fairness, fairness_result_to_dict
from ..fairness.perturbation import build_perturbations
from ..models.scoring import DEFAULT_JOB_PROFILES, get_job_profile
from ..models.supervised_classifier import SupervisedResumeClassifier
from ..rag.retriever import SkillRetriever
from ..rag.vector_store import SkillVectorStore
from ..shared import CandidateAnalysis, JobProfile
from .metrics import compute_classification_metrics, metrics_to_dict


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
    if baseline_df is None:
        return {}
    if {"resume_id", "y_pred"}.issubset(baseline_df.columns):
        out: dict[int, str] = {}
        for _, row in baseline_df.iterrows():
            try:
                out[int(row["resume_id"])] = str(row["y_pred"])
            except Exception:
                continue
        return out
    return {}


def _profile_label_space(baseline_df: pd.DataFrame | None) -> list[str]:
    labels = set(DEFAULT_JOB_PROFILES)
    if baseline_df is not None and "y_pred" in baseline_df.columns:
        labels.update(str(value) for value in baseline_df["y_pred"].dropna().astype(str).tolist())
    return sorted(labels)


def _supervised_prediction_map(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None,
) -> tuple[dict[int, str], dict[int, float]]:
    if train_df is None:
        return {}, {}
    required_columns = {"resume_text", "label"}
    if not required_columns.issubset(train_df.columns):
        return {}, {}

    classifier = SupervisedResumeClassifier()
    classifier.fit(train_df, text_col="resume_text", label_col="label")
    predictions, margins = classifier.predict_with_margin(test_df["resume_text"].astype(str))

    pred_map: dict[int, str] = {}
    margin_map: dict[int, float] = {}
    for row, label, margin in zip(test_df.itertuples(index=False), predictions, margins):
        resume_id = int(getattr(row, "resume_id", len(pred_map)))
        pred_map[resume_id] = str(label)
        margin_map[resume_id] = float(margin)
    return pred_map, margin_map


def _choose_label_with_ensemble(
    scores: dict[str, float],
    baseline_label: str | None,
    supervised_label: str | None,
    supervised_margin: float | None,
) -> tuple[str, str, float, float, float]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else best_score
    margin = best_score - second_score

    if supervised_label is not None:
        if baseline_label is not None and supervised_label == baseline_label:
            return supervised_label, "supervised+baseline-agree", best_score, second_score, margin
        return supervised_label, "supervised-agent", best_score, second_score, margin

    if baseline_label is None:
        return best_label, "multi-agent", best_score, second_score, margin
    if baseline_label == best_label:
        return best_label, "ensemble-agree", best_score, second_score, margin
    if best_score < 45.0 or margin < 8.0:
        return baseline_label, "baseline-prior", best_score, second_score, margin
    return best_label, "multi-agent", best_score, second_score, margin


def _candidate_to_record(analysis: CandidateAnalysis) -> dict[str, object]:
    record = {
        "resume_id": analysis.resume_id,
        "label": analysis.label,
        "experience": asdict(analysis.experience),
        "normalized_skills": [asdict(skill) for skill in analysis.normalized_skills],
        "extracted": asdict(analysis.extracted),
        "evaluation": asdict(analysis.evaluation) if analysis.evaluation else None,
        "extra": analysis.extra,
    }
    return record


def run_multi_agent_evaluation(
    test_df: pd.DataFrame,
    selection_threshold: float = 60.0,
    baseline_predictions_path: str | Path | None = None,
    train_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[CandidateAnalysis], dict[str, object]]:
    extractor = ExtractorAgent()
    retriever = SkillRetriever(SkillVectorStore())
    skill_miner = SkillMinerAgent(retriever)
    experience_agent = ExperienceAgent()
    evaluator = EvaluatorAgent()

    analyses: list[CandidateAnalysis] = []
    prediction_rows: list[dict[str, object]] = []
    original_scores: list[float] = []
    perturbed_scores: list[float] = []
    baseline_df = _load_predictions(baseline_predictions_path)
    baseline_map = _baseline_prediction_map(baseline_df)
    label_space = set(_profile_label_space(baseline_df))
    supervised_map, supervised_margin_map = _supervised_prediction_map(test_df, train_df)
    label_space.update(supervised_map.values())
    profiles = {job_label: get_job_profile(job_label) for job_label in sorted(label_space)}

    for _, row in test_df.iterrows():
        resume_id = int(row.get("resume_id", len(analyses)))
        label = str(row.get("label", "unknown"))
        text = str(row.get("resume_text", ""))
        extracted = extractor.extract(text, resume_id=resume_id)
        normalized_skills = tuple(skill_miner.normalize(extracted))
        experience = experience_agent.analyze(extracted)
        baseline_label = baseline_map.get(resume_id)
        supervised_label = supervised_map.get(resume_id)
        supervised_margin = supervised_margin_map.get(resume_id)
        if baseline_label and baseline_label not in profiles:
            profiles[baseline_label] = get_job_profile(baseline_label)
        if supervised_label and supervised_label not in profiles:
            profiles[supervised_label] = get_job_profile(supervised_label)
        scores = {
            job_label: evaluator.evaluate(normalized_skills, experience, profile, text).fit_score
            for job_label, profile in profiles.items()
        }
        chosen_label, decision_source, top_score, second_score, margin = _choose_label_with_ensemble(
            scores,
            baseline_label,
            supervised_label,
            supervised_margin,
        )
        chosen_eval = evaluator.evaluate(normalized_skills, experience, profiles[chosen_label], text)
        analysis = CandidateAnalysis(
            resume_id=resume_id,
            label=label,
            extracted=extracted,
            normalized_skills=normalized_skills,
            experience=experience,
            evaluation=chosen_eval,
            extra={
                "scores": scores,
                "decision_source": decision_source,
                "baseline_label": baseline_label,
                "supervised_label": supervised_label,
                "supervised_margin": round(supervised_margin, 4) if supervised_margin is not None else None,
                "top_score": round(top_score, 4),
                "second_score": round(second_score, 4),
                "score_margin": round(margin, 4),
            },
        )
        analyses.append(analysis)
        prediction_rows.append(
            {
                "resume_id": resume_id,
                "y_true": label,
                "y_pred": chosen_label,
                "fit_score": chosen_eval.fit_score,
                "decision_source": decision_source,
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
            perturbed_eval = evaluator.evaluate(perturbed_skills, perturbed_experience, profiles[chosen_label], perturbed_text)
            perturbed_scores.append(perturbed_eval.fit_score)

    predictions_df = pd.DataFrame(prediction_rows)
    metrics = compute_classification_metrics(predictions_df["y_true"].tolist(), predictions_df["y_pred"].tolist())
    fairness = audit_fairness(original_scores, perturbed_scores, selection_threshold=selection_threshold)
    baseline_summary = None
    comparison = None
    if baseline_df is not None and {"y_true", "y_pred"}.issubset(baseline_df.columns):
        baseline_metrics = compute_classification_metrics(baseline_df["y_true"].astype(str).tolist(), baseline_df["y_pred"].astype(str).tolist())
        baseline_summary = metrics_to_dict(baseline_metrics)
        comparison = {
            "accuracy_delta": round(metrics.accuracy - baseline_metrics.accuracy, 4),
            "precision_macro_delta": round(metrics.precision_macro - baseline_metrics.precision_macro, 4),
            "recall_macro_delta": round(metrics.recall_macro - baseline_metrics.recall_macro, 4),
            "f1_macro_delta": round(metrics.f1_macro - baseline_metrics.f1_macro, 4),
            "f1_weighted_delta": round(metrics.f1_weighted - baseline_metrics.f1_weighted, 4),
        }
    summary = {
        "metrics": metrics_to_dict(metrics),
        "baseline_metrics": baseline_summary,
        "comparison": comparison,
        "fairness": fairness_result_to_dict(fairness),
        "analyses": [
            _candidate_to_record(analysis)
            for analysis in analyses
        ],
    }
    return predictions_df, analyses, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-agent evaluation and fairness audit.")
    parser.add_argument("--train_jsonl", default=None)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selection_threshold", type=float, default=60.0)
    parser.add_argument("--baseline_predictions", default="artifacts/baseline_predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = _load_jsonl(args.train_jsonl) if args.train_jsonl else None
    test_df = _load_jsonl(args.test_jsonl)
    predictions_df, analyses, summary = run_multi_agent_evaluation(
        test_df,
        selection_threshold=args.selection_threshold,
        baseline_predictions_path=args.baseline_predictions,
        train_df=train_df,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "multi_agent_predictions.csv"
    summary_path = output_dir / "multi_agent_metrics.json"
    analyses_path = output_dir / "multi_agent_records.jsonl"

    predictions_df.to_csv(predictions_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with analyses_path.open("w", encoding="utf-8") as handle:
        for analysis in analyses:
            handle.write(json.dumps(_candidate_to_record(analysis), ensure_ascii=False) + "\n")

    print(json.dumps(summary["metrics"], indent=2))
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved records to: {analyses_path}")


if __name__ == "__main__":
    main()
