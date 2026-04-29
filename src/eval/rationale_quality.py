"""Rationale-quality evaluation for the multi-agent pipeline.

The deterministic scorer alone cannot tell whether the LLM-generated
``rationale`` text on each evaluation is actually useful. We score every
rationale on four orthogonal axes that together justify why the multi-agent
system is worth the latency cost of an LLM call:

1. **Grounded**: the rationale explicitly names canonical skills the
   candidate has (``matched_skills``) AND skills they lack
   (``missing_skills``). Without this, the rationale is generic boilerplate.
2. **Specific**: word count above a small threshold; not a one-liner.
3. **Calibrated**: the ``recommendation`` tone (advance / consider / reject)
   matches the numeric ``fit_score`` band.
4. **Identity-clean**: the rationale and recommendation do not use
   demographic pronouns or attributes (gender, marital status, age, etc.),
   which is a basic fairness expectation.

We also evaluate the parallel structured fields ``strengths`` and ``gaps``
for consistency with the matched / missing skill sets — i.e. the LLM did
not invent skills.

The metrics here are deliberately rule-based and deterministic so the
quality numbers are reproducible without a second LLM call.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configurable lexicons
# ---------------------------------------------------------------------------

#: Tone bucket label for a fit-score band.
ToneLabel = str  # "advance" | "consider" | "reject"

#: Word lists that classify the recommendation tone.
TONE_KEYWORDS: dict[ToneLabel, tuple[str, ...]] = {
    "advance": (
        "advance",
        "recommend",
        "strong fit",
        "strong match",
        "great fit",
        "great match",
        "excellent fit",
        "excellent match",
        "proceed",
        "move forward",
        "hire",
        "good fit",
        "strong candidate",
        "well suited",
        "well-suited",
    ),
    "consider": (
        "consider",
        "reservation",
        "borderline",
        "marginal",
        "may be",
        "could be",
        "possibly",
        "moderate fit",
        "moderate match",
        "potentially",
        "worth a closer look",
        "additional review",
    ),
    "reject": (
        "do not advance",
        "do not proceed",
        "not advance",
        "not suitable",
        "not a fit",
        "not a match",
        "not recommended",
        "lacks",
        "insufficient",
        "weak match",
        "weak fit",
        "decline",
        "reject",
        "pass on",
        "not the right fit",
    ),
}

#: Demographic-attribute markers we never want to see in the rationale.
#: Kept narrow to avoid false positives on technical text.
IDENTITY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"\b{kw}\b", flags=re.IGNORECASE)
    for kw in (
        # Gender pronouns / honorifics referring to the candidate.
        "she",
        "her",
        "hers",
        "herself",
        "he",
        "him",
        "his",
        "himself",
        "mr",
        "mrs",
        "ms",
        "miss",
        "sir",
        "madam",
        "gentleman",
        "lady",
        # Direct gender descriptors.
        "woman",
        "women",
        "man",
        "men",
        "female",
        "male",
        # Age / life stage descriptors.
        "young",
        "old",
        "elderly",
        "youthful",
        "ageing",
        "aging",
        # Marital / family status.
        "married",
        "unmarried",
        "single",
        "divorced",
        "widowed",
        "spouse",
        "husband",
        "wife",
        # Religion.
        "christian",
        "muslim",
        "jewish",
        "hindu",
        "buddhist",
        "sikh",
        "atheist",
        # Race / ethnicity (a small high-confidence subset).
        "caucasian",
        "asian",
        "african-american",
        "african american",
        "hispanic",
        "latino",
        "latina",
        # Disability.
        "disabled",
        "handicapped",
        "blind",
        "deaf",
    )
)


def _fit_band(fit_score: float) -> ToneLabel:
    if fit_score >= 70.0:
        return "advance"
    if fit_score >= 40.0:
        return "consider"
    return "reject"


def _normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def _detect_tone(text: str) -> ToneLabel | None:
    """Classify a recommendation string into one of the tone buckets.

    We test the buckets in priority order ``reject`` then ``advance`` then
    ``consider`` because the consider bucket is the most permissive and we
    want a more specific match to win when keywords overlap (e.g. "may not
    be a strong fit" should count as ``reject``, not ``advance``).
    """

    norm = _normalize(text)
    if not norm:
        return None
    for tone in ("reject", "advance", "consider"):
        for kw in TONE_KEYWORDS[tone]:
            if kw in norm:
                return tone
    return None


def _mentions_any(text: str, phrases: Iterable[str]) -> tuple[bool, list[str]]:
    """Whether the (lowercased) ``text`` mentions any of the given phrases.

    Returns the boolean plus the list of phrases that hit, deduped, in order.
    Matching is substring-based after lowercasing, with a word-boundary
    requirement on the leading and trailing edges of the phrase to reduce
    spurious hits like ``r`` matching inside ``career``.
    """

    norm = _normalize(text)
    hits: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        candidate = (phrase or "").strip().lower()
        if not candidate or candidate in seen:
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(candidate)}(?![A-Za-z0-9])")
        if pattern.search(norm):
            hits.append(candidate)
            seen.add(candidate)
    return bool(hits), hits


def _identity_violations(text: str) -> list[str]:
    norm = text or ""
    hits: list[str] = []
    seen: set[str] = set()
    for pattern in IDENTITY_PATTERNS:
        m = pattern.search(norm)
        if m and m.group(0).lower() not in seen:
            hits.append(m.group(0))
            seen.add(m.group(0).lower())
    return hits


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RationaleAudit:
    """Per-rationale audit emitted for one record."""

    resume_id: int | None
    fit_score: float
    fit_band: ToneLabel
    recommendation_tone: ToneLabel | None

    grounded_in_matched: bool
    grounded_in_missing: bool
    matched_skill_hits: list[str]
    missing_skill_hits: list[str]

    word_count: int
    is_specific: bool
    calibrated: bool
    identity_clean: bool
    identity_terms: list[str]

    strengths_grounded_ratio: float
    gaps_grounded_ratio: float
    strengths_invented: list[str]
    gaps_invented: list[str]


@dataclass
class RationaleQualityReport:
    """Aggregate rationale-quality summary across a run."""

    n: int
    grounded_in_matched_pct: float
    grounded_in_missing_pct: float
    grounded_in_both_pct: float
    specific_pct: float
    calibrated_pct: float
    identity_clean_pct: float
    avg_word_count: float
    strengths_grounded_pct: float
    gaps_grounded_pct: float
    tone_distribution: dict[ToneLabel | str, int] = field(default_factory=dict)
    fit_band_distribution: dict[ToneLabel, int] = field(default_factory=dict)
    audits: list[RationaleAudit] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Audit one record
# ---------------------------------------------------------------------------


def audit_record(record: Mapping[str, Any]) -> RationaleAudit:
    """Score a single multi-agent record's rationale quality.

    ``record`` is one entry from ``multi_agent_records.jsonl`` or any mapping
    with the same shape (``evaluation``, ``resume_id``).
    """

    evaluation = record.get("evaluation") or {}
    rationale = evaluation.get("rationale") or ""
    recommendation = evaluation.get("recommendation") or ""
    matched = list(evaluation.get("matched_skills") or [])
    missing = list(evaluation.get("missing_skills") or [])
    strengths = list(evaluation.get("strengths") or [])
    gaps = list(evaluation.get("gaps") or [])
    fit_score = float(evaluation.get("fit_score") or 0.0)

    combined = f"{rationale}\n{recommendation}".strip()

    grounded_matched, matched_hits = _mentions_any(combined, matched)
    grounded_missing, missing_hits = _mentions_any(combined, missing)

    word_count = len(rationale.split())
    is_specific = word_count >= 15

    fit_band = _fit_band(fit_score)
    tone = _detect_tone(recommendation)
    calibrated = tone == fit_band if tone is not None else False

    identity_terms = _identity_violations(combined)
    identity_clean = not identity_terms

    # Strengths/gaps grounding: each entry should match a canonical skill
    # in matched_skills / missing_skills (case-insensitive equality OR a
    # substring match in either direction to allow harmless paraphrases like
    # "Adobe Photoshop" vs "adobe photoshop").
    def _grounding_ratio(items: list[str], reference: list[str]) -> tuple[float, list[str]]:
        if not items:
            return 1.0, []
        ref_lower = [r.lower() for r in reference]
        invented: list[str] = []
        grounded = 0
        for item in items:
            cand = (item or "").strip().lower()
            if not cand:
                continue
            if any(cand == r or cand in r or r in cand for r in ref_lower):
                grounded += 1
            else:
                invented.append(item)
        ratio = grounded / max(1, len(items))
        return ratio, invented

    strengths_ratio, strengths_invented = _grounding_ratio(strengths, matched)
    gaps_ratio, gaps_invented = _grounding_ratio(gaps, missing)

    return RationaleAudit(
        resume_id=record.get("resume_id"),
        fit_score=fit_score,
        fit_band=fit_band,
        recommendation_tone=tone,
        grounded_in_matched=grounded_matched,
        grounded_in_missing=grounded_missing,
        matched_skill_hits=matched_hits,
        missing_skill_hits=missing_hits,
        word_count=word_count,
        is_specific=is_specific,
        calibrated=calibrated,
        identity_clean=identity_clean,
        identity_terms=identity_terms,
        strengths_grounded_ratio=strengths_ratio,
        gaps_grounded_ratio=gaps_ratio,
        strengths_invented=strengths_invented,
        gaps_invented=gaps_invented,
    )


# ---------------------------------------------------------------------------
# Aggregate over a run
# ---------------------------------------------------------------------------


def evaluate_rationale_quality(
    records: Iterable[Mapping[str, Any]],
) -> RationaleQualityReport:
    """Audit every rationale and return the aggregate report."""

    audits: list[RationaleAudit] = [audit_record(r) for r in records]
    n = len(audits)
    if n == 0:
        return RationaleQualityReport(
            n=0,
            grounded_in_matched_pct=0.0,
            grounded_in_missing_pct=0.0,
            grounded_in_both_pct=0.0,
            specific_pct=0.0,
            calibrated_pct=0.0,
            identity_clean_pct=0.0,
            avg_word_count=0.0,
            strengths_grounded_pct=0.0,
            gaps_grounded_pct=0.0,
        )

    def _pct(predicate) -> float:
        return round(sum(1 for a in audits if predicate(a)) / n, 4)

    tone_dist: dict[str, int] = {"advance": 0, "consider": 0, "reject": 0, "unclassified": 0}
    band_dist: dict[ToneLabel, int] = {"advance": 0, "consider": 0, "reject": 0}
    for a in audits:
        tone_key = a.recommendation_tone or "unclassified"
        tone_dist[tone_key] = tone_dist.get(tone_key, 0) + 1
        band_dist[a.fit_band] = band_dist.get(a.fit_band, 0) + 1

    return RationaleQualityReport(
        n=n,
        grounded_in_matched_pct=_pct(lambda a: a.grounded_in_matched),
        grounded_in_missing_pct=_pct(lambda a: a.grounded_in_missing),
        grounded_in_both_pct=_pct(lambda a: a.grounded_in_matched and a.grounded_in_missing),
        specific_pct=_pct(lambda a: a.is_specific),
        calibrated_pct=_pct(lambda a: a.calibrated),
        identity_clean_pct=_pct(lambda a: a.identity_clean),
        avg_word_count=round(sum(a.word_count for a in audits) / n, 2),
        strengths_grounded_pct=round(sum(a.strengths_grounded_ratio for a in audits) / n, 4),
        gaps_grounded_pct=round(sum(a.gaps_grounded_ratio for a in audits) / n, 4),
        tone_distribution=tone_dist,
        fit_band_distribution=band_dist,
        audits=audits,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def report_to_dict(report: RationaleQualityReport, include_audits: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "n": report.n,
        "grounded_in_matched_pct": report.grounded_in_matched_pct,
        "grounded_in_missing_pct": report.grounded_in_missing_pct,
        "grounded_in_both_pct": report.grounded_in_both_pct,
        "specific_pct": report.specific_pct,
        "calibrated_pct": report.calibrated_pct,
        "identity_clean_pct": report.identity_clean_pct,
        "avg_word_count": report.avg_word_count,
        "strengths_grounded_pct": report.strengths_grounded_pct,
        "gaps_grounded_pct": report.gaps_grounded_pct,
        "tone_distribution": report.tone_distribution,
        "fit_band_distribution": report.fit_band_distribution,
    }
    if include_audits:
        payload["audits"] = [
            {
                "resume_id": a.resume_id,
                "fit_score": a.fit_score,
                "fit_band": a.fit_band,
                "recommendation_tone": a.recommendation_tone,
                "grounded_in_matched": a.grounded_in_matched,
                "grounded_in_missing": a.grounded_in_missing,
                "matched_skill_hits": a.matched_skill_hits,
                "missing_skill_hits": a.missing_skill_hits,
                "word_count": a.word_count,
                "is_specific": a.is_specific,
                "calibrated": a.calibrated,
                "identity_clean": a.identity_clean,
                "identity_terms": a.identity_terms,
                "strengths_grounded_ratio": round(a.strengths_grounded_ratio, 4),
                "gaps_grounded_ratio": round(a.gaps_grounded_ratio, 4),
                "strengths_invented": a.strengths_invented,
                "gaps_invented": a.gaps_invented,
            }
            for a in report.audits
        ]
    return payload


def evaluate_rationale_quality_jsonl(records_path: str | Path) -> RationaleQualityReport:
    """Convenience wrapper: load a multi_agent_records.jsonl and audit it."""

    path = Path(records_path)
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return evaluate_rationale_quality(records)


__all__ = [
    "RationaleAudit",
    "RationaleQualityReport",
    "audit_record",
    "evaluate_rationale_quality",
    "evaluate_rationale_quality_jsonl",
    "report_to_dict",
]
