"""Inspect ontology-pair cosine similarities to calibrate SOFT_MATCH_THRESHOLD."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.scoring import _ontology_embedding_table


def main() -> None:
    table = _ontology_embedding_table()
    if table is None:
        print("ERROR: embedding table not available", flush=True)
        return
    name_to_row, matrix = table

    def sim(a: str, b: str) -> float | None:
        if a not in name_to_row or b not in name_to_row:
            return None
        return float(matrix[name_to_row[a]] @ matrix[name_to_row[b]])

    pairs = [
        # near-synonyms we expect to match
        ("machine learning", "deep learning"),
        ("machine learning", "nlp"),
        ("adobe illustrator", "adobe photoshop"),
        ("adobe illustrator", "graphic design"),
        ("sql", "mysql"),
        ("sql", "postgresql"),
        ("aws", "azure"),
        ("aws", "gcp"),
        ("docker", "kubernetes"),
        ("docker", "ci/cd"),
        ("budgeting", "financial analysis"),
        ("budgeting", "accounting"),
        ("recruiting", "onboarding"),
        ("recruiting", "employee relations"),
        ("teaching", "curriculum design"),
        ("teaching", "mentoring"),
        ("react", "angular"),
        ("react", "vue"),
        ("react", "javascript"),
        ("communication", "leadership"),
        ("communication", "mentoring"),
        ("leadership", "mentoring"),
        ("leadership", "project management"),
        ("project management", "agile"),
        ("testing", "quality control"),
        # cross-domain checks (should NOT soft-match)
        ("python", "food safety"),
        ("sql", "food safety"),
        ("adobe photoshop", "sql"),
        ("recruiting", "python"),
        ("budgeting", "docker"),
        ("teaching", "kubernetes"),
    ]

    print(f"{'pair':<55} cosine", flush=True)
    for a, b in pairs:
        s = sim(a, b)
        if s is None:
            print(f"  {a:<25} <-> {b:<25}  N/A", flush=True)
            continue
        marker = "  <-- SOFT@0.65" if s >= 0.65 else ("  (soft@0.55)" if s >= 0.55 else "")
        print(f"  {a:<25} <-> {b:<25}  {s:.3f}{marker}", flush=True)


if __name__ == "__main__":
    main()
