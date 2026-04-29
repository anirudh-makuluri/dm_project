from __future__ import annotations

import logging
from dataclasses import asdict
from functools import lru_cache
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..shared import EvaluationResult, ExperienceFeatures, JobProfile, SkillMatch

logger = logging.getLogger(__name__)


def _normalize_label_key(label: str) -> str:
    return label.strip().upper().replace("_", "-")


PROFILE_SEEDS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "ACCOUNTANT": (
        "Accounting and audit workflows with spreadsheets, financial controls, reporting, and compliance.",
        ("excel", "sql", "compliance", "communication"),
        ("statistics", "dashboarding", "project management", "stakeholder management"),
    ),
    "ADVOCATE": (
        "Legal advocacy and advisory services with client communication and compliance-intensive case management.",
        ("communication", "compliance", "stakeholder management", "project management"),
        ("leadership", "documentation", "analytics", "negotiation"),
    ),
    "AGRICULTURE": (
        "Agriculture domain operations with planning, monitoring, reporting, and quality/compliance controls.",
        ("project management", "communication", "compliance", "excel"),
        ("statistics", "dashboarding", "stakeholder management", "leadership"),
    ),
    "APPAREL": (
        "Apparel operations across design coordination, production planning, vendor alignment, and quality controls.",
        ("project management", "communication", "stakeholder management", "compliance"),
        ("excel", "leadership", "dashboarding", "sales"),
    ),
    "ARTS": (
        "Creative arts delivery with strong communication, execution planning, and stakeholder collaboration.",
        ("communication", "project management", "stakeholder management", "leadership"),
        ("design", "presentation", "documentation", "marketing"),
    ),
    "AUTOMOBILE": (
        "Automobile domain engineering/operations with testing, quality, production execution, and process controls.",
        ("testing", "project management", "communication", "compliance"),
        ("data modeling", "observability", "leadership", "stakeholder management"),
    ),
    "AVIATION": (
        "Aviation operations requiring safety-first compliance, coordination, and high-reliability execution.",
        ("compliance", "communication", "project management", "leadership"),
        ("stakeholder management", "observability", "documentation", "analytics"),
    ),
    "BANKING": (
        "Banking processes with regulatory compliance, financial reporting, reconciliation, and customer operations.",
        ("sql", "excel", "compliance", "communication"),
        ("statistics", "dashboarding", "stakeholder management", "project management"),
    ),
    "BPO": (
        "Business process outsourcing operations with service quality, communication, and process efficiency.",
        ("communication", "stakeholder management", "project management", "excel"),
        ("leadership", "dashboarding", "documentation", "compliance"),
    ),
    "BUSINESS-DEVELOPMENT": (
        "Business development focused on pipeline growth, stakeholder engagement, and conversion operations.",
        ("communication", "stakeholder management", "project management", "leadership"),
        ("sales", "dashboarding", "excel", "statistics"),
    ),
    "CHEF": (
        "Culinary operations with menu execution, hygiene compliance, planning, and team coordination.",
        ("project management", "communication", "compliance", "leadership"),
        ("stakeholder management", "operations", "quality", "documentation"),
    ),
    "CONSTRUCTION": (
        "Construction project lifecycle management with planning, site execution, quality, and compliance.",
        ("project management", "compliance", "communication", "leadership"),
        ("stakeholder management", "documentation", "dashboarding", "operations"),
    ),
    "CONSULTANT": (
        "Consulting delivery with diagnostics, recommendations, stakeholder alignment, and execution tracking.",
        ("communication", "stakeholder management", "project management", "leadership"),
        ("statistics", "dashboarding", "sql", "excel"),
    ),
    "DESIGNER": (
        "Design role focused on visual communication, creative iteration, and collaborative project execution.",
        ("graphic design", "communication", "adobe photoshop", "presentation"),
        ("adobe creative suite", "adobe illustrator", "ui/ux", "figma", "video editing", "marketing"),
    ),
    "DIGITAL-MEDIA": (
        "Digital media role with content strategy, campaign execution, analytics, and channel operations.",
        ("marketing", "communication", "social media", "seo"),
        ("graphic design", "video editing", "analytics", "dashboarding", "brand management", "presentation"),
    ),
    "ENGINEERING": (
        "Software engineering, APIs, cloud deployment, containers, testing, and scalable backend systems.",
        ("python", "apis", "testing", "docker", "git"),
        ("java", "javascript", "kubernetes", "microservices", "ci/cd", "cloud"),
    ),
    "FINANCE": (
        "Finance role focused on analysis, controls, reporting, and compliance-aware planning.",
        ("excel", "sql", "statistics", "compliance"),
        ("dashboarding", "project management", "communication", "stakeholder management"),
    ),
    "FITNESS": (
        "Fitness and wellness services requiring client communication, planning, and quality outcomes.",
        ("communication", "leadership", "project management", "stakeholder management"),
        ("compliance", "documentation", "operations", "analytics"),
    ),
    "HEALTHCARE": (
        "Healthcare operations and service delivery with strict compliance and multidisciplinary coordination.",
        ("compliance", "communication", "project management", "stakeholder management"),
        ("leadership", "documentation", "analytics", "operations"),
    ),
    "HR": (
        "Recruiting, onboarding, employee relations, compliance, benefits, payroll, and applicant tracking systems.",
        ("recruiting", "onboarding", "compliance", "communication"),
        ("employee relations", "benefits administration", "payroll", "applicant tracking systems", "hr analytics"),
    ),
    "INFORMATION-TECHNOLOGY": (
        "Information technology operations across software, infrastructure, support, and delivery automation.",
        ("python", "sql", "apis", "git"),
        ("cloud", "docker", "testing", "ci/cd", "kubernetes", "microservices"),
    ),
    "PUBLIC-RELATIONS": (
        "Public relations with campaign communication, brand messaging, stakeholder outreach, and event coordination.",
        ("communication", "stakeholder management", "project management", "leadership"),
        ("dashboarding", "content", "analytics", "documentation"),
    ),
    "SALES": (
        "Sales operations with pipeline management, customer communication, and performance-driven execution.",
        ("communication", "stakeholder management", "project management", "leadership"),
        ("dashboarding", "excel", "statistics", "documentation"),
    ),
    "TEACHER": (
        "Teaching and learning delivery with curriculum execution, communication, and classroom leadership.",
        ("communication", "leadership", "project management", "stakeholder management"),
        ("documentation", "analytics", "dashboarding", "compliance"),
    ),
    "DATA SCIENCE": (
        "Machine learning, Python, SQL, experimentation, statistics, analytics, and model evaluation.",
        ("python", "sql", "machine learning", "statistics", "model evaluation"),
        ("pandas", "scikit-learn", "feature engineering", "dashboarding", "experiment design"),
    ),
}


DEFAULT_JOB_PROFILES: dict[str, JobProfile] = {
    label: JobProfile(label=label, description=description, required_skills=required, preferred_skills=preferred)
    for label, (description, required, preferred) in PROFILE_SEEDS.items()
}

_NORMALIZED_PROFILE_INDEX = {_normalize_label_key(label): profile for label, profile in DEFAULT_JOB_PROFILES.items()}


# ---------------------------------------------------------------------------
# Optional learned-profile registry.
#
# When ``register_learned_profiles`` is called, ``get_job_profile`` will
# prefer the learned profile for a given label and fall back to the
# hand-curated ``DEFAULT_JOB_PROFILES`` only when no learned profile is
# available.  Callers (e.g. the evaluation runner) pass
# ``--learned_profiles_path`` to populate this registry before running.
# ---------------------------------------------------------------------------
_LEARNED_PROFILE_INDEX: dict[str, JobProfile] = {}


def register_learned_profiles(profiles: dict[str, JobProfile] | None) -> None:
    """Replace the in-memory learned-profile registry."""

    _LEARNED_PROFILE_INDEX.clear()
    if not profiles:
        return
    for label, profile in profiles.items():
        _LEARNED_PROFILE_INDEX[_normalize_label_key(label)] = profile


def has_learned_profiles() -> bool:
    return bool(_LEARNED_PROFILE_INDEX)


def get_job_profile(label: str) -> JobProfile:
    normalized = _normalize_label_key(label)
    if normalized in _LEARNED_PROFILE_INDEX:
        learned = _LEARNED_PROFILE_INDEX[normalized]
        if learned.label == label:
            return learned
        return JobProfile(
            label=label,
            description=learned.description,
            required_skills=learned.required_skills,
            preferred_skills=learned.preferred_skills,
        )
    if normalized in _NORMALIZED_PROFILE_INDEX:
        profile = _NORMALIZED_PROFILE_INDEX[normalized]
        if profile.label == label:
            return profile
        return JobProfile(
            label=label,
            description=profile.description,
            required_skills=profile.required_skills,
            preferred_skills=profile.preferred_skills,
        )
    fallback_tokens = label.lower().replace("_", " ").replace("-", " ")
    return JobProfile(label=label, description=f"Role focused on {fallback_tokens}.", required_skills=(fallback_tokens,))


def _skill_names(skills: Iterable[SkillMatch]) -> list[str]:
    return [skill.canonical_skill for skill in skills]


def _jaccard(candidate: set[str], target: set[str]) -> float:
    if not candidate and not target:
        return 0.0
    intersection = len(candidate & target)
    union = len(candidate | target)
    return intersection / union if union else 0.0


def _text_similarity(candidate_text: str, job_description: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([candidate_text, job_description])
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


# ---------------------------------------------------------------------------
# Semantic skill matching.
#
# Hand-curated and learned profiles often list a slightly different
# canonical skill than the candidate has, e.g. profile asks for
# ``budgeting`` but the candidate has ``financial planning``.  Both are
# semantically close, share an ontology category, and should count as a
# (partial) match.  ``soft_skill_overlap`` complements the exact-set
# intersection used by ``score_candidate`` with a cosine-similarity
# fallback against the embedded ontology, gated by a threshold to keep
# noise low.  When the threshold is unmet OR the embeddings backend is
# unavailable, the fallback degrades gracefully to exact match.
# ---------------------------------------------------------------------------


#: Cosine-similarity threshold above which a non-exact skill is counted
#: as a "soft" match.  Calibrated against the MiniLM-L6 ontology
#: embeddings in ``scripts/inspect_cosine_pairs.py``:
#:
#: * Same-category pairs cluster in [0.45, 0.65]
#:   (docker<->k8s=0.64, ML<->DL=0.63, aws<->azure=0.56,
#:    react<->angular=0.45, adobe-illustrator<->photoshop=0.45).
#: * Cross-domain pairs cluster in [0.00, 0.16]
#:   (python<->food safety=0.08, recruiting<->python=0.02).
#:
#: A threshold of >1.0 effectively disables soft matching, falling back
#: to exact set intersection (the ``score`` returned by
#: ``soft_skill_overlap`` is identical in that regime).  The default is
#: set high because the threshold sweep in
#: ``scripts/threshold_sweep.py`` showed soft matching monotonically
#: hurt agents-only accuracy (0.294 -> 0.235 from threshold 2.0 -> 0.45)
#: while leaving ensemble accuracy essentially flat (0.682-0.686): the
#: cosine fallback dilutes argmax discrimination across profiles
#: without offsetting the loss elsewhere.  Lower this (e.g. to 0.50)
#: only after a fresh ablation that confirms a robust gain on the
#: current corpus.
SOFT_MATCH_THRESHOLD = 1.01

#: Soft matches contribute less than exact ones so the scorer still
#: rewards crisp ontology hits.  An exact match is worth 1.0; a soft
#: match scales linearly between this floor (at threshold) and 1.0
#: (at cosine = 1.0).
SOFT_MATCH_FLOOR_WEIGHT = 0.5


@lru_cache(maxsize=1)
def _ontology_embedding_table() -> tuple[dict[str, int], np.ndarray] | None:
    """Return ``({canonical_name: row_index}, embedding_matrix)`` for the
    process-wide ontology, or None if embeddings cannot be loaded."""

    try:
        from ..rag.embeddings import get_default_skill_index
    except Exception as exc:  # pragma: no cover - defensive import guard
        logger.info("Soft skill matching unavailable (import error): %s", exc)
        return None

    try:
        index = get_default_skill_index()
    except Exception as exc:  # pragma: no cover - encoder may fail on cold start
        logger.info("Soft skill matching unavailable (encoder error): %s", exc)
        return None

    matrix = np.asarray(index.matrix, dtype=np.float32)
    if matrix.size == 0:
        return None

    # Pre-normalize rows so dot product == cosine similarity.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms

    name_to_row: dict[str, int] = {
        entry.canonical: i for i, entry in enumerate(index.entries)
    }
    return name_to_row, matrix


def _soft_match_weight(cosine: float) -> float:
    """Map a cosine similarity to a contribution weight in [floor, 1]."""

    if cosine >= 1.0:
        return 1.0
    if cosine < SOFT_MATCH_THRESHOLD:
        return 0.0
    span = max(1e-6, 1.0 - SOFT_MATCH_THRESHOLD)
    scaled = (cosine - SOFT_MATCH_THRESHOLD) / span
    return SOFT_MATCH_FLOOR_WEIGHT + (1.0 - SOFT_MATCH_FLOOR_WEIGHT) * scaled


def soft_skill_overlap(
    candidate_skills: set[str],
    target_skills: set[str],
    *,
    threshold: float = SOFT_MATCH_THRESHOLD,
) -> tuple[float, list[str], dict[str, tuple[str, float]]]:
    """Compute a fractional overlap between candidate and target skills.

    Returns ``(score, matched_target_skills, mapping)`` where:

    * ``score`` is the soft overlap count (sum of per-target weights), so
      it is bounded by ``len(target_skills)`` from above.
    * ``matched_target_skills`` is the sorted subset of targets that were
      matched (exactly OR soft).  Used downstream for ``matched`` /
      ``missing`` reporting.
    * ``mapping`` maps each matched target -> (matched_candidate_skill,
      cosine_similarity).  Exact matches map to themselves with
      similarity 1.0.

    The function falls back to plain set intersection when the embedding
    matrix is unavailable, in which case only exact matches are counted.
    """

    if not candidate_skills or not target_skills:
        return 0.0, [], {}

    table = _ontology_embedding_table()
    matched: list[str] = []
    mapping: dict[str, tuple[str, float]] = {}
    score = 0.0

    if table is None:
        # No embeddings available: degrade to exact match.
        for target in target_skills:
            if target in candidate_skills:
                matched.append(target)
                mapping[target] = (target, 1.0)
                score += 1.0
        return score, sorted(matched), mapping

    name_to_row, matrix = table
    candidate_list = list(candidate_skills)
    candidate_indices = [name_to_row[c] for c in candidate_list if c in name_to_row]
    candidate_known = [c for c in candidate_list if c in name_to_row]

    candidate_matrix = matrix[candidate_indices] if candidate_indices else None

    for target in target_skills:
        if target in candidate_skills:
            matched.append(target)
            mapping[target] = (target, 1.0)
            score += 1.0
            continue
        target_idx = name_to_row.get(target)
        if target_idx is None or candidate_matrix is None:
            continue
        target_vec = matrix[target_idx]
        sims = candidate_matrix @ target_vec  # shape (n_known_candidates,)
        if sims.size == 0:
            continue
        best_local = int(np.argmax(sims))
        best_sim = float(sims[best_local])
        if best_sim < threshold:
            continue
        weight = _soft_match_weight(best_sim)
        if weight == 0.0:
            continue
        matched.append(target)
        mapping[target] = (candidate_known[best_local], best_sim)
        score += weight

    return score, sorted(matched), mapping


def score_candidate(
    normalized_skills: Iterable[SkillMatch],
    experience: ExperienceFeatures,
    job_profile: JobProfile,
    candidate_text: str,
) -> tuple[float, dict[str, float], tuple[str, ...], tuple[str, ...]]:
    candidate_skill_names = set(_skill_names(normalized_skills))
    required = set(job_profile.required_skills)
    preferred = set(job_profile.preferred_skills)

    # Soft (cosine-similarity) overlap.  Falls back to exact-set
    # intersection when ontology embeddings are unavailable.
    required_soft, matched_required, _ = soft_skill_overlap(candidate_skill_names, required)
    preferred_soft, matched_preferred, _ = soft_skill_overlap(candidate_skill_names, preferred)
    union_soft, matched_union, _ = soft_skill_overlap(
        candidate_skill_names, required | preferred
    )

    required_score = required_soft / max(1, len(required))
    preferred_score = preferred_soft / max(1, len(preferred))
    # Coverage can exceed 1.0 under soft matching because a single
    # candidate skill may soft-hit several targets (e.g. "adobe
    # illustrator" -> {illustrator, photoshop, graphic design}).  Cap to
    # preserve the original [0, 1] contract that the fit-score formula
    # assumes.
    coverage_score = min(1.0, union_soft / max(1, len(candidate_skill_names)))
    years_score = min(experience.total_years / 8.0, 1.0)
    leadership_score = min(experience.leadership_indicators / 2.0, 1.0)
    text_score = _text_similarity(candidate_text, job_profile.description)
    # ``general_overlap`` keeps the original Jaccard formulation (counts
    # only exact set intersection) for backwards-comparable component
    # reporting; the headline match-quality signals above already use the
    # softer cosine matching.
    general_overlap = _jaccard(candidate_skill_names, required | preferred)

    component_scores = {
        "required_skill_match": round(required_score, 4),
        "preferred_skill_match": round(preferred_score, 4),
        "coverage": round(coverage_score, 4),
        "experience_years": round(years_score, 4),
        "leadership": round(leadership_score, 4),
        "text_similarity": round(text_score, 4),
        "general_overlap": round(general_overlap, 4),
    }

    fit_score = 100.0 * (
        0.28 * required_score
        + 0.16 * preferred_score
        + 0.14 * coverage_score
        + 0.14 * years_score
        + 0.08 * leadership_score
        + 0.12 * text_score
        + 0.08 * general_overlap
    )
    fit_score = round(max(0.0, min(100.0, fit_score)), 2)

    matched = tuple(sorted(set(matched_union)))
    missing = tuple(sorted((required | preferred) - set(matched_union)))
    return fit_score, component_scores, matched, missing


def build_evaluation_result(
    normalized_skills: Iterable[SkillMatch],
    experience: ExperienceFeatures,
    job_profile: JobProfile,
    candidate_text: str,
) -> EvaluationResult:
    fit_score, component_scores, matched, missing = score_candidate(
        normalized_skills=normalized_skills,
        experience=experience,
        job_profile=job_profile,
        candidate_text=candidate_text,
    )
    strengths = matched[:5] if matched else tuple()
    gaps = missing[:5] if missing else tuple()
    if fit_score >= 80:
        recommendation = "Strong match: move forward to interview."
    elif fit_score >= 60:
        recommendation = "Moderate match: review carefully and probe gaps."
    else:
        recommendation = "Weak match: consider for a different role or upskilling path."

    rationale_parts = [
        f"Matched {len(matched)} target skills",
        f"experience signal {experience.total_years:.1f} years",
        f"leadership indicators {experience.leadership_indicators}",
        f"text alignment {component_scores['text_similarity']:.2f}",
    ]
    rationale = "; ".join(rationale_parts)
    return EvaluationResult(
        label=job_profile.label,
        fit_score=fit_score,
        rationale=rationale,
        strengths=strengths,
        gaps=gaps,
        recommendation=recommendation,
        component_scores=component_scores,
        matched_skills=matched,
        missing_skills=missing,
    )


def evaluation_to_dict(result: EvaluationResult) -> dict[str, object]:
    return asdict(result)
