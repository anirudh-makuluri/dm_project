"""End-to-end smoke test for score_candidate with soft matching."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.scoring import build_evaluation_result, get_job_profile
from src.shared import ExperienceFeatures, SkillMatch


def make_skill(name: str) -> SkillMatch:
    return SkillMatch(phrase=name, canonical_skill=name, confidence=0.9)


def run(label: str, skill_names: list[str], years: float = 3.0, leadership: int = 1):
    profile = get_job_profile(label)
    skills = [make_skill(s) for s in skill_names]
    exp = ExperienceFeatures(
        total_years=years,
        role_count=1,
        leadership_indicators=leadership,
        impact_statements=0,
        quantified_impact=0,
        evidence=(),
    )
    text = "Senior professional with " + ", ".join(skill_names)
    result = build_evaluation_result(skills, exp, profile, text)
    print(f"  {label}: fit={result.fit_score:.2f}  matched={list(result.matched_skills)}", flush=True)
    print(f"    missing={list(result.missing_skills)[:6]}", flush=True)
    print(
        f"    components: req={result.component_scores['required_skill_match']:.3f}  "
        f"pref={result.component_scores['preferred_skill_match']:.3f}  "
        f"cov={result.component_scores['coverage']:.3f}  "
        f"text={result.component_scores['text_similarity']:.2f}",
        flush=True,
    )


print("=== Test: IT candidate with near-synonyms (docker+k8s+python) ===", flush=True)
run(
    "INFORMATION-TECHNOLOGY",
    ["docker", "kubernetes", "python", "aws", "sql"],
    years=5,
    leadership=2,
)

print("\n=== Test: data science candidate (should match well) ===", flush=True)
run(
    "INFORMATION-TECHNOLOGY",
    ["machine learning", "pytorch", "python", "feature engineering", "deep learning"],
    years=6,
    leadership=1,
)

print("\n=== Test: design candidate (illustrator, not photoshop) ===", flush=True)
run(
    "DESIGNER",
    ["adobe illustrator", "figma", "ui/ux"],
    years=3,
    leadership=0,
)

print("\n=== Test: cross-domain (chef applying to IT) ===", flush=True)
run(
    "INFORMATION-TECHNOLOGY",
    ["food safety", "inventory management", "customer service"],
    years=5,
    leadership=1,
)

print("\nDONE.", flush=True)
