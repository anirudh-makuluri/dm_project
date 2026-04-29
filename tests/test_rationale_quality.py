"""Tests for the rationale-quality evaluation module.

The module is meant to be deterministic and rule-based, so each metric is
verifiable on small hand-written records. We test:

- the four headline metrics on a single record (grounded, specific,
  calibrated, identity-clean),
- aggregation across multiple records,
- the strengths / gaps invented-skill detection,
- edge cases (empty rationale, no matched/missing skills, JSONL loader).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.rationale_quality import (
    audit_record,
    evaluate_rationale_quality,
    evaluate_rationale_quality_jsonl,
    report_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers to build records
# ---------------------------------------------------------------------------


def _record(
    *,
    resume_id: int = 1,
    fit_score: float = 50.0,
    matched=("python", "communication"),
    missing=("docker", "kubernetes"),
    strengths=("python",),
    gaps=("docker",),
    rationale="The candidate has python and communication skills but lacks docker.",
    recommendation="Consider with reservations given the docker gap.",
) -> dict:
    return {
        "resume_id": resume_id,
        "label": "ENGINEERING",
        "evaluation": {
            "fit_score": fit_score,
            "matched_skills": list(matched),
            "missing_skills": list(missing),
            "strengths": list(strengths),
            "gaps": list(gaps),
            "rationale": rationale,
            "recommendation": recommendation,
        },
    }


# ---------------------------------------------------------------------------
# Per-record metrics
# ---------------------------------------------------------------------------


def test_grounded_when_rationale_cites_matched_and_missing():
    audit = audit_record(_record())
    assert audit.grounded_in_matched is True
    assert audit.grounded_in_missing is True
    assert "python" in audit.matched_skill_hits
    assert "docker" in audit.missing_skill_hits


def test_not_grounded_when_rationale_is_generic():
    audit = audit_record(
        _record(rationale="Strong candidate overall.", recommendation="Move forward.")
    )
    assert audit.grounded_in_matched is False
    assert audit.grounded_in_missing is False


def test_specific_threshold():
    long = _record(rationale=" ".join(["good"] * 16))
    assert audit_record(long).is_specific is True
    short = _record(rationale="great fit")
    assert audit_record(short).is_specific is False


@pytest.mark.parametrize(
    "fit_score,recommendation,expected_calibrated,expected_band",
    [
        (85.0, "Strong fit; advance to interview.", True, "advance"),
        (85.0, "Do not advance, lacks key skills.", False, "advance"),
        (50.0, "Consider with reservations.", True, "consider"),
        (50.0, "Strong recommendation to hire.", False, "consider"),
        (20.0, "Do not advance; insufficient experience.", True, "reject"),
        (20.0, "Strong fit, recommend.", False, "reject"),
    ],
)
def test_calibration_alignment(fit_score, recommendation, expected_calibrated, expected_band):
    audit = audit_record(_record(fit_score=fit_score, recommendation=recommendation))
    assert audit.fit_band == expected_band
    assert audit.calibrated is expected_calibrated


def test_identity_terms_flagged_in_rationale():
    audit = audit_record(
        _record(
            rationale="She is a strong candidate with python skills but lacks docker.",
            recommendation="Consider her for the role.",
        )
    )
    assert audit.identity_clean is False
    # We expect the female pronoun to be flagged.
    flagged = {term.lower() for term in audit.identity_terms}
    assert "she" in flagged or "her" in flagged


def test_identity_terms_flagged_when_demographic_attribute_used():
    audit = audit_record(
        _record(rationale="The young candidate has python but lacks docker.")
    )
    assert audit.identity_clean is False
    assert any(term.lower() == "young" for term in audit.identity_terms)


def test_clean_rationale_is_identity_clean():
    audit = audit_record(_record())
    assert audit.identity_clean is True
    assert audit.identity_terms == []


# ---------------------------------------------------------------------------
# Strengths / gaps invented-skill detection
# ---------------------------------------------------------------------------


def test_strengths_grounded_ratio_full_match():
    audit = audit_record(_record(strengths=("python", "communication")))
    assert audit.strengths_grounded_ratio == 1.0
    assert audit.strengths_invented == []


def test_strengths_invented_skills_detected():
    audit = audit_record(
        _record(
            matched=("python",),
            strengths=("python", "haskell", "ocaml"),
        )
    )
    assert audit.strengths_grounded_ratio == pytest.approx(1 / 3)
    assert "haskell" in audit.strengths_invented
    assert "ocaml" in audit.strengths_invented


def test_gaps_invented_skills_detected():
    audit = audit_record(
        _record(
            missing=("docker",),
            gaps=("docker", "rust", "fortran"),
        )
    )
    assert audit.gaps_grounded_ratio == pytest.approx(1 / 3)
    assert sorted(audit.gaps_invented) == ["fortran", "rust"]


def test_empty_strengths_and_gaps_treated_as_grounded():
    audit = audit_record(_record(strengths=(), gaps=()))
    assert audit.strengths_grounded_ratio == 1.0
    assert audit.gaps_grounded_ratio == 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_report_percentages():
    records = [
        _record(resume_id=1, fit_score=80.0, recommendation="Strong fit; advance."),
        _record(
            resume_id=2,
            fit_score=20.0,
            recommendation="Do not advance, lacks docker.",
        ),
        _record(
            resume_id=3,
            fit_score=20.0,
            rationale="She is great",
            recommendation="Strong fit, hire.",
        ),
    ]
    report = evaluate_rationale_quality(records)
    assert report.n == 3
    # Records 1 and 2 cite matched/missing; record 3 has a generic 4-word rationale.
    assert 0.5 <= report.grounded_in_matched_pct <= 1.0
    # Calibration: rec 1 advance/advance ✓, rec 2 reject/reject ✓, rec 3 reject/advance ✗.
    assert report.calibrated_pct == pytest.approx(2 / 3, abs=1e-3)
    # Identity-clean: only rec 3 has a pronoun.
    assert report.identity_clean_pct == pytest.approx(2 / 3, abs=1e-3)
    assert report.fit_band_distribution == {"advance": 1, "consider": 0, "reject": 2}


def test_report_to_dict_round_trips():
    report = evaluate_rationale_quality([_record()])
    payload = report_to_dict(report, include_audits=True)
    assert payload["n"] == 1
    assert "audits" in payload and len(payload["audits"]) == 1
    audit_payload = payload["audits"][0]
    assert audit_payload["resume_id"] == 1
    assert audit_payload["grounded_in_matched"] is True


def test_evaluate_jsonl_loader(tmp_path: Path):
    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps(rec)
            for rec in (
                _record(resume_id=10),
                _record(resume_id=11, rationale="generic", recommendation="advance"),
            )
        ),
        encoding="utf-8",
    )
    report = evaluate_rationale_quality_jsonl(jsonl)
    assert report.n == 2
    assert any(a.resume_id == 10 for a in report.audits)


def test_word_boundary_prevents_substring_false_positives():
    # "career" should NOT trigger a match for the canonical skill "r"
    audit = audit_record(
        _record(
            matched=("r",),
            rationale="The candidate has a long career history.",
        )
    )
    assert audit.grounded_in_matched is False


def test_empty_records_yields_zero_report():
    report = evaluate_rationale_quality([])
    assert report.n == 0
    assert report.grounded_in_matched_pct == 0.0
    assert report.calibrated_pct == 0.0
