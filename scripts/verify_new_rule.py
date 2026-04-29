"""Verify the new _ensemble_choice rule produces the expected V4 result."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("loading...", flush=True)
t0 = time.time()
agent_csv = pd.read_csv(ROOT / "artifacts/main_learned_profiles/multi_agent_predictions.csv")
baseline = pd.read_csv(ROOT / "artifacts/main_baseline/baseline_predictions.csv")
train = pd.read_json(ROOT / "data/processed_main/train.jsonl", lines=True)
test = pd.read_json(ROOT / "data/processed_main/test.jsonl", lines=True)
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

from src.eval.evaluation_runner import _ensemble_choice
from src.models.supervised_classifier import SupervisedResumeClassifier

print("refitting SVC...", flush=True)
t0 = time.time()
clf = SupervisedResumeClassifier()
clf.fit(train)
svc_pred, svc_margin = clf.predict_with_margin(test["resume_text"].astype(str))
print(f"  done in {time.time()-t0:.1f}s", flush=True)

baseline_map = dict(zip(baseline["resume_id"].astype(int), baseline["y_pred"].astype(str)))
agent_score_map = dict(zip(agent_csv["resume_id"].astype(int), agent_csv["scores"].astype(str)))

correct = 0
total = 0
sources: dict[str, int] = {}
for i, row in test.reset_index(drop=True).iterrows():
    rid = int(row["resume_id"])
    y_true = str(row["label"])
    raw = agent_score_map.get(rid, "{}")
    scores = json.loads(raw) if isinstance(raw, str) and raw else {}
    label, src = _ensemble_choice(
        scores,
        baseline_map.get(rid),
        str(svc_pred[i]),
        supervised_margin=float(svc_margin[i]),
    )
    if label == y_true:
        correct += 1
    total += 1
    sources[src] = sources.get(src, 0) + 1

print(f"Ensemble accuracy with NEW rule: {correct/total:.4f}  ({correct}/{total})", flush=True)
print(f"Decision sources: {sources}", flush=True)
print("DONE.", flush=True)
