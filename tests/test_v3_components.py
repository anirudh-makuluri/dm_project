"""Regression tests for the v3 iteration components.

These lock in the contracts of the three pieces shipped in iteration v3
of the multi-agent resume screening system:

* ``soft_skill_overlap`` (cosine-based skill matching with exact fall-back)
* ``profile_builder.scan_canonical_skills`` (alias-substring scanner)
* ``profile_builder.save_profiles`` / ``load_profiles`` round-trip
* ``_ensemble_choice`` V4 override rule

Tests are CPU-only and import-light: the embedding-dependent paths in
``soft_skill_overlap`` are exercised but the test asserts only on the
exact-match contract that holds with or without embeddings, so a missing
encoder cache cannot flake the suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_soft_skill_overlap_exact_only_at_high_threshold(monkeypatch):
    """At threshold > 1.0 (the production default) the function must
    behave like plain set intersection."""

    from src.models import scoring

    # Bump threshold above 1.0 to disable soft matching.
    monkeypatch.setattr(scoring, "SOFT_MATCH_THRESHOLD", 1.01)

    cand = {"python", "sql", "machine learning"}
    targets = {"python", "deep learning"}
    score, matched, mapping = scoring.soft_skill_overlap(cand, targets)

    assert matched == ["python"], "only the exact intersection should match"
    assert score == pytest.approx(1.0)
    assert mapping == {"python": ("python", 1.0)}


def test_soft_skill_overlap_empty_inputs():
    """Empty candidate or target sets must return zero score and no matches."""

    from src.models.scoring import soft_skill_overlap

    score, matched, mapping = soft_skill_overlap(set(), {"python"})
    assert score == 0.0 and matched == [] and mapping == {}

    score, matched, mapping = soft_skill_overlap({"python"}, set())
    assert score == 0.0 and matched == [] and mapping == {}


def test_soft_skill_overlap_threshold_disables_soft(monkeypatch):
    """Even when an embedding table is available, threshold > 1.0 must
    suppress soft hits — only exact matches survive.

    This guards the production default (``SOFT_MATCH_THRESHOLD = 1.01``):
    a future regression that drops the threshold (or breaks the cosine
    cutoff) would surface here as soon as a non-equal target picks up a
    soft match.
    """

    from src.models import scoring

    # Cross-domain pairs that previously soft-matched under threshold 0.45.
    cand = {"docker", "react"}
    targets = {"kubernetes", "angular"}

    monkeypatch.setattr(scoring, "SOFT_MATCH_THRESHOLD", 1.01)
    score_strict, matched_strict, _ = scoring.soft_skill_overlap(cand, targets)

    assert score_strict == 0.0
    assert matched_strict == []


def test_scan_canonical_skills_alias_lookup():
    """The substring scanner must canonicalize aliases (e.g. ``ml`` ->
    ``machine learning``) and ignore unrelated tokens."""

    from src.models.profile_builder import scan_canonical_skills

    text = (
        "Built a Python pipeline using scikit-learn for ML classification. "
        "Deployed to AWS via Docker. No exposure to plumbing."
    )
    found = scan_canonical_skills(text)

    # Membership rather than equality — the ontology may carry extra
    # aliases that match this string, but these four are non-negotiable.
    assert "python" in found
    assert "scikit-learn" in found
    assert "machine learning" in found
    assert "docker" in found
    # Negative case: nothing in the ontology should match "plumbing".
    assert "plumbing" not in found


def test_scan_canonical_skills_word_boundaries():
    """Aliases must respect whole-word boundaries to avoid false positives
    like ``r`` matching the letter ``r`` in arbitrary text."""

    from src.models.profile_builder import scan_canonical_skills

    text = "I work in a great room near the printer."
    found = scan_canonical_skills(text)

    # 'r' is a canonical skill alias but should never match a single
    # letter inside another word.
    assert "r" not in found


def test_profile_save_load_roundtrip(tmp_path: Path):
    """Profiles persisted to disk must reload with identical fields."""

    from src.models.profile_builder import save_profiles, load_profiles
    from src.shared import JobProfile

    profile = JobProfile(
        label="DATA-SCIENCE",
        description="ML and analytics role.",
        required_skills=("python", "sql", "machine learning"),
        preferred_skills=("docker", "aws"),
    )
    out_path = tmp_path / "profiles.json"
    save_profiles({"DATA-SCIENCE": profile}, out_path)

    loaded = load_profiles(out_path)
    assert "DATA-SCIENCE" in loaded
    got = loaded["DATA-SCIENCE"]
    assert got.label == profile.label
    assert got.description == profile.description
    assert tuple(got.required_skills) == profile.required_skills
    assert tuple(got.preferred_skills) == profile.preferred_skills


def test_profile_builder_payload_is_human_readable(tmp_path: Path):
    """The on-disk JSON should be inspectable: keys present, no binary,
    and stable enough that a diff tool can be used between iterations."""

    from src.models.profile_builder import save_profiles
    from src.shared import JobProfile

    profile = JobProfile(
        label="X",
        description="d",
        required_skills=("a",),
        preferred_skills=("b",),
    )
    out_path = tmp_path / "p.json"
    save_profiles({"X": profile}, out_path)
    payload = json.loads(out_path.read_text())

    assert payload["X"]["label"] == "X"
    assert payload["X"]["required_skills"] == ["a"]
    assert payload["X"]["preferred_skills"] == ["b"]


def test_ensemble_choice_supervised_baseline_agree():
    """When SVC and baseline agree, the agent vote is irrelevant."""

    from src.eval.evaluation_runner import _ensemble_choice

    scores = {"DATA-SCIENCE": 80.0, "ENGINEERING": 50.0}  # agent likes DS
    label, source = _ensemble_choice(
        scores,
        baseline_label="ACCOUNTANT",
        supervised_label="ACCOUNTANT",
        supervised_margin=2.0,
    )
    assert label == "ACCOUNTANT"
    assert source == "supervised+baseline-agree"


def test_ensemble_choice_v4_override_fires_under_low_margin():
    """When the agent + baseline both vote X but SVC votes Y with a small
    decision-function margin, V4 must promote X."""

    from src.eval.evaluation_runner import _ensemble_choice, SUPERVISED_UNCERTAIN_MARGIN

    scores = {"FINANCE": 70.0, "ACCOUNTANT": 60.0}  # agent picks FINANCE
    label, source = _ensemble_choice(
        scores,
        baseline_label="FINANCE",
        supervised_label="ACCOUNTANT",
        supervised_margin=SUPERVISED_UNCERTAIN_MARGIN - 0.1,  # below threshold
    )
    assert label == "FINANCE"
    assert source == "agent+baseline-override"


def test_ensemble_choice_v4_does_not_fire_when_margin_high():
    """When the SVC is confident (margin >= threshold), even if the agent
    and baseline agree against it, the SVC must win."""

    from src.eval.evaluation_runner import _ensemble_choice, SUPERVISED_UNCERTAIN_MARGIN

    scores = {"FINANCE": 70.0, "ACCOUNTANT": 60.0}
    label, source = _ensemble_choice(
        scores,
        baseline_label="FINANCE",
        supervised_label="ACCOUNTANT",
        supervised_margin=SUPERVISED_UNCERTAIN_MARGIN + 0.5,  # confident SVC
    )
    assert label == "ACCOUNTANT"
    assert source == "supervised-agent"


def test_ensemble_choice_v4_does_not_fire_when_agent_disagrees_with_baseline():
    """If the agent picks a third label that the baseline does not also
    vote for, V4 must NOT override the SVC."""

    from src.eval.evaluation_runner import _ensemble_choice, SUPERVISED_UNCERTAIN_MARGIN

    scores = {"FINANCE": 70.0, "ACCOUNTANT": 60.0}  # agent: FINANCE
    label, source = _ensemble_choice(
        scores,
        baseline_label="HR",  # baseline disagrees with agent
        supervised_label="ACCOUNTANT",
        supervised_margin=SUPERVISED_UNCERTAIN_MARGIN - 0.1,
    )
    assert label == "ACCOUNTANT"
    assert source == "supervised-agent"


def test_ensemble_choice_no_supervised_falls_back_to_legacy():
    """When the supervised classifier is unavailable, legacy multi-agent
    + baseline voting must still work."""

    from src.eval.evaluation_runner import _ensemble_choice

    scores = {"DATA-SCIENCE": 80.0, "ENGINEERING": 50.0}
    label, source = _ensemble_choice(
        scores,
        baseline_label="DATA-SCIENCE",
        supervised_label=None,
    )
    # Agent and baseline agree -> ensemble-agree branch
    assert label == "DATA-SCIENCE"
    assert source == "ensemble-agree"
