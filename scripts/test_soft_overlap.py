"""Smoke test for soft_skill_overlap."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("loading scoring module...", flush=True)
t0 = time.time()
from src.models.scoring import soft_skill_overlap, _ontology_embedding_table
print(f"  ready in {time.time()-t0:.1f}s", flush=True)

print("warming ontology embeddings...", flush=True)
t0 = time.time()
table = _ontology_embedding_table()
if table is None:
    print("  ERROR: embedding table is None", flush=True)
    sys.exit(1)
name_to_row, matrix = table
print(f"  ready in {time.time()-t0:.1f}s, n_skills={len(name_to_row)}, dim={matrix.shape[1]}", flush=True)

cases = [
    # (label, candidate_skills, target_skills, expected_behavior)
    ("exact match", {"python", "sql"}, {"python", "sql"}, "score=2.0"),
    ("partial exact", {"python"}, {"python", "sql"}, "score=1.0"),
    ("no match", {"python"}, {"adobe photoshop"}, "score=0"),
    ("near synonyms", {"financial planning"}, {"budgeting"}, "soft match expected"),
    ("recruiting synonyms", {"talent acquisition"}, {"recruiting"}, "soft match expected"),
    ("ML semantic", {"machine learning"}, {"deep learning"}, "soft match expected"),
    ("unrelated tech vs hospitality", {"python"}, {"food safety"}, "score=0"),
    ("hr semantic", {"onboarding"}, {"employee relations"}, "maybe soft"),
    ("design semantic", {"adobe illustrator"}, {"adobe photoshop"}, "soft match expected"),
    ("legal semantic", {"litigation"}, {"contract drafting"}, "maybe soft"),
]

print()
print(f"{'case':<35} {'score':>6} {'matched':<40} {'mapping'}")
for label, cand, target, expected in cases:
    score, matched, mapping = soft_skill_overlap(cand, target)
    pretty_map = ", ".join(f"{k}<-{v[0]}({v[1]:.2f})" for k, v in mapping.items())
    print(f"  {label:<33} {score:>6.3f}  {str(matched):<38}  {pretty_map}", flush=True)

print()
print("DONE.", flush=True)
