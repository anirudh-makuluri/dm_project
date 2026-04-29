"""Sweep SOFT_MATCH_THRESHOLD and measure agents-only + ensemble accuracy."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.evaluation_runner import _ensemble_choice
from src.models.profile_builder import load_profiles as load_learned_profiles
from src.models import scoring
from src.models.scoring import build_evaluation_result, get_job_profile
from src.models.supervised_classifier import SupervisedResumeClassifier
from src.shared import ExperienceFeatures, SkillMatch


def _rec_skills(rec):
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


def _rec_exp(rec):
    e = rec.get("experience", {}) or {}
    return ExperienceFeatures(
        total_years=float(e.get("total_years", 0.0)),
        role_count=int(e.get("role_count", 0)),
        leadership_indicators=int(e.get("leadership_indicators", 0)),
        impact_statements=int(e.get("impact_statements", 0)),
        quantified_impact=int(e.get("quantified_impact", 0)),
        evidence=tuple(e.get("evidence") or ()),
    )


records = []
with open(ROOT / "artifacts/main_learned_profiles/multi_agent_records.jsonl") as f:
    for line in f:
        records.append(json.loads(line))
profiles = load_learned_profiles(ROOT / "data/profiles_learned.json")

baseline = pd.read_csv(ROOT / "artifacts/main_baseline/baseline_predictions.csv")
baseline_map = dict(zip(baseline["resume_id"].astype(int), baseline["y_pred"].astype(str)))

train = pd.read_json(ROOT / "data/processed_main/train.jsonl", lines=True)
test = pd.read_json(ROOT / "data/processed_main/test.jsonl", lines=True)
clf = SupervisedResumeClassifier()
clf.fit(train)
svc_pred, svc_margin = clf.predict_with_margin(test["resume_text"].astype(str))
svc_map = {int(rid): (str(svc_pred[i]), float(svc_margin[i])) for i, rid in enumerate(test["resume_id"])}

for label in set(baseline_map.values()) | set(svc_pred):
    if label and label not in profiles:
        profiles[label] = get_job_profile(label)

# Warm up per-record data.  The records.jsonl does NOT preserve raw
# resume text, so we look it up from test.jsonl via resume_id (otherwise
# the text_similarity component collapses and every profile scores
# similarly).
rid_to_text = dict(zip(test["resume_id"].astype(int), test["resume_text"].astype(str)))
skill_cache = [_rec_skills(r) for r in records]
exp_cache = [_rec_exp(r) for r in records]
ytrue_cache = [str(r["label"]) for r in records]
rid_cache = [int(r["resume_id"]) for r in records]
text_cache = [rid_to_text.get(rid, "") for rid in rid_cache]


def eval_at_threshold(threshold: float) -> tuple[float, float, dict[str, int]]:
    """Patch SOFT_MATCH_THRESHOLD, clear cache, evaluate."""
    scoring.SOFT_MATCH_THRESHOLD = threshold
    # Special case: threshold > 1.0 disables soft match entirely (only exact matches hit).
    agent_correct = 0
    ens_correct = 0
    srcs: dict[str, int] = {}
    for i, rec in enumerate(records):
        skills = skill_cache[i]
        exp = exp_cache[i]
        text = text_cache[i]
        scores = {}
        for label, profile in profiles.items():
            ev = build_evaluation_result(skills, exp, profile, text)
            scores[label] = ev.fit_score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        agent_label = ranked[0][0] if ranked else "unknown"
        svc_label, margin = svc_map.get(rid_cache[i], (None, None))
        ens_label, src = _ensemble_choice(
            scores, baseline_map.get(rid_cache[i]), svc_label, supervised_margin=margin,
        )
        if agent_label == ytrue_cache[i]:
            agent_correct += 1
        if ens_label == ytrue_cache[i]:
            ens_correct += 1
        srcs[src] = srcs.get(src, 0) + 1
    n = len(records)
    return agent_correct / n, ens_correct / n, srcs


print(f"{'threshold':>10}  {'agents':>8}  {'ensemble':>10}  decision_sources", flush=True)
for thr in [2.0, 0.70, 0.60, 0.55, 0.50, 0.45, 0.40]:
    t0 = time.time()
    agents, ens, srcs = eval_at_threshold(thr)
    marker = " (exact only)" if thr > 1.0 else ""
    print(f"  {thr:>8.2f}  {agents:>8.4f}  {ens:>10.4f}  {srcs}{marker}  ({time.time()-t0:.1f}s)", flush=True)

print("DONE.", flush=True)
