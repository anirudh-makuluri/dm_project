"""Quick post-hoc ablation: test different ensemble override rules
without rerunning any LLM calls.  Uses the saved multi-agent predictions
from the last full eval + refits the SVC (~10 sec).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("[1/5] loading artifacts ...", flush=True)
t0 = time.time()
agent_csv = pd.read_csv(ROOT / "artifacts/main_learned_profiles/multi_agent_predictions.csv")
baseline = pd.read_csv(ROOT / "artifacts/main_baseline/baseline_predictions.csv")
train = pd.read_json(ROOT / "data/processed_main/train.jsonl", lines=True)
test = pd.read_json(ROOT / "data/processed_main/test.jsonl", lines=True)
print(f"       done in {time.time()-t0:.1f}s (train={len(train)}, test={len(test)})", flush=True)

print("[2/5] refitting SVC ...", flush=True)
t0 = time.time()
from src.models.supervised_classifier import SupervisedResumeClassifier
clf = SupervisedResumeClassifier()
clf.fit(train)
print(f"       fit done in {time.time()-t0:.1f}s", flush=True)

print("[3/5] predicting with margins ...", flush=True)
t0 = time.time()
svc_pred, svc_margin = clf.predict_with_margin(test["resume_text"].astype(str))
print(f"       predict done in {time.time()-t0:.1f}s", flush=True)
print(
    f"       SVC margins: min={svc_margin.min():.3f} "
    f"q25={np.quantile(svc_margin,0.25):.3f} median={np.median(svc_margin):.3f} "
    f"q75={np.quantile(svc_margin,0.75):.3f} max={svc_margin.max():.3f}",
    flush=True,
)

print("[4/5] assembling dataframe ...", flush=True)
df = test[["resume_id", "label"]].copy().rename(columns={"label": "y_true"})
df["svc_pred"] = svc_pred
df["svc_margin"] = svc_margin
baseline_map = dict(zip(baseline["resume_id"].astype(int), baseline["y_pred"].astype(str)))
df["baseline_pred"] = df["resume_id"].map(baseline_map)
agent_score_map = dict(zip(agent_csv["resume_id"].astype(int), agent_csv["scores"].astype(str)))
parsed = df["resume_id"].map(
    lambda rid: json.loads(agent_score_map.get(int(rid), "{}")) if agent_score_map.get(int(rid)) else {}
)
df["agent_top"] = parsed.apply(lambda d: max(d.values()) if d else 0.0)
df["agent_label"] = parsed.apply(lambda d: max(d, key=d.get) if d else "unknown")
df["agent_margin"] = parsed.apply(
    lambda d: (sorted(d.values(), reverse=True)[0] - sorted(d.values(), reverse=True)[1])
    if len(d) >= 2
    else 0.0
)

print("[5/5] evaluating ensemble rules ...", flush=True)


def acc(col: str) -> float:
    return accuracy_score(df["y_true"], df[col])


df["current"] = df["svc_pred"]  # current rule == SVC wins
print(f"  CURRENT (SVC only)          acc={acc('current'):.4f}", flush=True)
print(f"  baseline-only               acc={accuracy_score(df['y_true'], df['baseline_pred']):.4f}", flush=True)
print(f"  agent-only                  acc={accuracy_score(df['y_true'], df['agent_label']):.4f}", flush=True)


def rule_v3(r):
    if r["agent_label"] == r["baseline_pred"] and r["agent_label"] != r["svc_pred"]:
        return r["agent_label"]
    return r["svc_pred"]


df["v3"] = df.apply(rule_v3, axis=1)
print(
    f"  V3 (agent==baseline)        acc={acc('v3'):.4f}  agent_won={(df['v3']!=df['svc_pred']).sum()}",
    flush=True,
)


def rule_v4(r):
    if (
        r["agent_label"] == r["baseline_pred"]
        and r["agent_label"] != r["svc_pred"]
        and r["svc_margin"] < 0.5
    ):
        return r["agent_label"]
    return r["svc_pred"]


df["v4"] = df.apply(rule_v4, axis=1)
print(
    f"  V4 (v3 + svc_margin<0.5)    acc={acc('v4'):.4f}  agent_won={(df['v4']!=df['svc_pred']).sum()}",
    flush=True,
)


def rule_v5(r):
    if r["agent_top"] >= 55 and r["agent_margin"] >= 8 and r["svc_margin"] < 0.3:
        return r["agent_label"]
    return r["svc_pred"]


df["v5"] = df.apply(rule_v5, axis=1)
print(
    f"  V5 (conf agent + weak svc)  acc={acc('v5'):.4f}  agent_won={(df['v5']!=df['svc_pred']).sum()}",
    flush=True,
)


def rule_v6(r):
    if r["agent_label"] == r["baseline_pred"] and r["agent_label"] != r["svc_pred"]:
        return r["agent_label"]
    if r["agent_top"] >= 60 and r["agent_margin"] >= 12 and r["svc_margin"] < 0.3:
        return r["agent_label"]
    return r["svc_pred"]


df["v6"] = df.apply(rule_v6, axis=1)
print(
    f"  V6 (v3 OR v5)               acc={acc('v6'):.4f}  agent_won={(df['v6']!=df['svc_pred']).sum()}",
    flush=True,
)

print("DONE.", flush=True)
