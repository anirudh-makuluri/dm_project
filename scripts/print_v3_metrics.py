"""Print headline metrics from artifacts/main_v3_soft_match/."""
import json
from pathlib import Path

ART = Path("artifacts/main_v3_soft_match")
m = json.loads((ART / "multi_agent_metrics.json").read_text())

print("=== ENSEMBLE ===")
e = m.get("metrics", {})
print(f"  acc={e.get('accuracy', 0):.4f}  f1_macro={e.get('f1_macro', 0):.4f}  f1_weighted={e.get('f1_weighted', 0):.4f}")

print("=== AGENTS-ONLY ===")
a = m.get("agents_only_metrics", {})
print(f"  acc={a.get('accuracy', 0):.4f}  f1_macro={a.get('f1_macro', 0):.4f}")

print("=== BASELINE ===")
b = m.get("baseline_metrics", {})
print(f"  acc={b.get('accuracy', 0):.4f}  f1_macro={b.get('f1_macro', 0):.4f}")

print("=== DELTAS ===")
print(f"  ensemble vs baseline: {m.get('comparison', {})}")
print(f"  agents vs baseline:   {m.get('agents_vs_baseline', {})}")
print(f"  ensemble vs agents:   {m.get('ensemble_vs_agents', {})}")

print("=== DECISION SOURCES ===")
print(f"  {m.get('decision_source_breakdown', {})}")

print("=== RATIONALE QUALITY ===")
r = m.get("rationale_quality", {})
for k in [
    "n",
    "grounded_in_matched_pct",
    "grounded_in_missing_pct",
    "grounded_in_both_pct",
    "calibrated_pct",
    "identity_clean_pct",
    "strengths_grounded_pct",
    "gaps_grounded_pct",
    "avg_word_count",
]:
    v = r.get(k)
    if isinstance(v, float):
        print(f"  {k}={v:.4f}")
    else:
        print(f"  {k}={v}")

print("=== FAIRNESS ===")
f = m.get("fairness", {})
for k in [
    "n_pairs",
    "original_mean_score",
    "perturbed_mean_score",
    "mean_score_difference",
    "score_difference_ci_lower",
    "score_difference_ci_upper",
    "selection_rate_difference",
]:
    v = f.get(k)
    if isinstance(v, float):
        print(f"  {k}={v:.4f}")
    else:
        print(f"  {k}={v}")

print("=== RANKING (macro over 24 jobs) ===")
rk = m.get("ranking", {})
for k in ["precision_at_1", "precision_at_3", "precision_at_5", "precision_at_10",
          "recall_at_5", "recall_at_10", "map_at_5", "ndcg_at_5"]:
    v = rk.get(k)
    if isinstance(v, float):
        print(f"  {k}={v:.4f}")

print("=== SKILL EXTRACTION (gold subset) ===")
sk = m.get("skill_extraction", {})
for k, v in sk.items():
    if isinstance(v, dict):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}={v}")

print("=== PROVIDERS ===")
print(f"  {m.get('providers', {})}")

