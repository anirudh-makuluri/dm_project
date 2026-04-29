"""Post-hoc re-scoring: apply the new soft-match scorer to cached
agent outputs and compute agents-only accuracy + ensemble with V4 rule.
No LLM calls, no agent rerun. Expected runtime ~30 sec.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.evaluation_runner import _ensemble_choice
from src.models.profile_builder import load_profiles as load_learned_profiles
from src.models.scoring import build_evaluation_result, get_job_profile
from src.models.supervised_classifier import SupervisedResumeClassifier
from src.shared import ExperienceFeatures, SkillMatch


def _load_records(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def _record_to_skills(rec: dict) -> list[SkillMatch]:
    out = []
    for s in rec.get("normalized_skills", []):
        out.append(
            SkillMatch(
                phrase=s.get("phrase", ""),
                canonical_skill=s.get("canonical_skill", ""),
                confidence=float(s.get("confidence", 0.0)),
                source=s.get("source", ""),
            )
        )
    return out


def _record_to_experience(rec: dict) -> ExperienceFeatures:
    e = rec.get("experience", {}) or {}
    return ExperienceFeatures(
        total_years=float(e.get("total_years", 0.0)),
        role_count=int(e.get("role_count", 0)),
        leadership_indicators=int(e.get("leadership_indicators", 0)),
        impact_statements=int(e.get("impact_statements", 0)),
        quantified_impact=int(e.get("quantified_impact", 0)),
        evidence=tuple(e.get("evidence") or ()),
    )


print("[1/5] loading cached records and profiles ...", flush=True)
t0 = time.time()
records = _load_records(ROOT / "artifacts/main_learned_profiles/multi_agent_records.jsonl")
profiles = load_learned_profiles(ROOT / "data/profiles_learned.json")
# Ensure every label seen has a profile (baseline/supervised may introduce more)
for label in list(profiles.keys()):
    pass
print(f"       {len(records)} records, {len(profiles)} profiles ({time.time()-t0:.1f}s)", flush=True)

print("[2/5] loading baseline predictions ...", flush=True)
baseline = pd.read_csv(ROOT / "artifacts/main_baseline/baseline_predictions.csv")
baseline_map = dict(zip(baseline["resume_id"].astype(int), baseline["y_pred"].astype(str)))

print("[3/5] refitting SVC ...", flush=True)
t0 = time.time()
train = pd.read_json(ROOT / "data/processed_main/train.jsonl", lines=True)
test = pd.read_json(ROOT / "data/processed_main/test.jsonl", lines=True)
clf = SupervisedResumeClassifier()
clf.fit(train)
svc_pred, svc_margin = clf.predict_with_margin(test["resume_text"].astype(str))
print(f"       done in {time.time()-t0:.1f}s", flush=True)

# Map by resume_id for safe alignment with records.jsonl
resume_id_to_svc = {}
for i, rid in enumerate(test["resume_id"].tolist()):
    resume_id_to_svc[int(rid)] = (str(svc_pred[i]), float(svc_margin[i]))

# Fill in hand-curated profiles for any baseline/SVC labels not in learned set.
for label in set(baseline_map.values()) | set(svc_pred):
    if label and label not in profiles:
        profiles[label] = get_job_profile(label)

print("[4/5] re-scoring agents with new soft-match logic ...", flush=True)
t0 = time.time()
agent_correct = 0
ensemble_correct = 0
sources: dict[str, int] = {}
results = []
for rec in records:
    rid = int(rec["resume_id"])
    y_true = str(rec["label"])
    skills = _record_to_skills(rec)
    exp = _record_to_experience(rec)
    text = rec.get("extra", {}).get("resume_text") or ""
    if not text:
        # Fall back to something - for text similarity only
        text = " ".join(s.canonical_skill for s in skills)

    # Re-score against every profile
    scores: dict[str, float] = {}
    for label, profile in profiles.items():
        ev = build_evaluation_result(skills, exp, profile, text)
        scores[label] = ev.fit_score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    agent_label = ranked[0][0] if ranked else "unknown"

    svc_label, margin = resume_id_to_svc.get(rid, (None, None))
    baseline_label = baseline_map.get(rid)

    ens_label, src = _ensemble_choice(
        scores,
        baseline_label,
        svc_label,
        supervised_margin=margin,
    )

    if agent_label == y_true:
        agent_correct += 1
    if ens_label == y_true:
        ensemble_correct += 1
    sources[src] = sources.get(src, 0) + 1
    results.append(
        {
            "resume_id": rid,
            "y_true": y_true,
            "agent_label": agent_label,
            "ens_label": ens_label,
            "decision_source": src,
        }
    )

print(f"       re-scored {len(records)} resumes in {time.time()-t0:.1f}s", flush=True)

print("[5/5] summary ...", flush=True)
n = len(records)
print(f"       Agents-only accuracy: {agent_correct/n:.4f}  ({agent_correct}/{n})", flush=True)
print(f"       Ensemble accuracy:    {ensemble_correct/n:.4f}  ({ensemble_correct}/{n})", flush=True)
print(f"       Decision sources: {sources}", flush=True)
print("DONE.", flush=True)
