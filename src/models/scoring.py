from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..shared import EvaluationResult, ExperienceFeatures, JobProfile, SkillMatch


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
        ("communication", "project management", "stakeholder management", "leadership"),
        ("react", "javascript", "design", "presentation"),
    ),
    "DIGITAL-MEDIA": (
        "Digital media role with content strategy, campaign execution, analytics, and channel operations.",
        ("communication", "project management", "stakeholder management", "dashboarding"),
        ("javascript", "react", "sql", "analytics"),
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


def get_job_profile(label: str) -> JobProfile:
    normalized = _normalize_label_key(label)
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


def score_candidate(
    normalized_skills: Iterable[SkillMatch],
    experience: ExperienceFeatures,
    job_profile: JobProfile,
    candidate_text: str,
) -> tuple[float, dict[str, float], tuple[str, ...], tuple[str, ...]]:
    candidate_skill_names = set(_skill_names(normalized_skills))
    required = set(job_profile.required_skills)
    preferred = set(job_profile.preferred_skills)

    required_overlap = len(candidate_skill_names & required)
    preferred_overlap = len(candidate_skill_names & preferred)
    required_score = required_overlap / max(1, len(required))
    preferred_score = preferred_overlap / max(1, len(preferred))
    coverage_score = len(candidate_skill_names & (required | preferred)) / max(1, len(candidate_skill_names))
    years_score = min(experience.total_years / 8.0, 1.0)
    leadership_score = min(experience.leadership_indicators / 2.0, 1.0)
    text_score = _text_similarity(candidate_text, job_profile.description)
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

    matched = tuple(sorted(candidate_skill_names & (required | preferred)))
    missing = tuple(sorted((required | preferred) - candidate_skill_names))
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
