from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillEntry:
    canonical: str
    aliases: tuple[str, ...] = ()
    category: str = "general"
    description: str = ""

    def embedding_text(self) -> str:
        return " ".join(
            part
            for part in [self.canonical, self.category, self.description, " ".join(self.aliases)]
            if part
        )


@dataclass(frozen=True)
class SkillMatch:
    phrase: str
    canonical_skill: str
    confidence: float
    source: str = "ontology"


@dataclass
class ExtractedResume:
    resume_id: int | None
    raw_text: str
    education: list[str] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    skills_raw: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    contact: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExperienceFeatures:
    total_years: float = 0.0
    role_count: int = 0
    leadership_indicators: int = 0
    impact_statements: int = 0
    quantified_impact: int = 0
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class JobProfile:
    label: str
    description: str
    required_skills: tuple[str, ...] = ()
    preferred_skills: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    label: str
    fit_score: float
    rationale: str
    strengths: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    recommendation: str = ""
    component_scores: dict[str, float] = field(default_factory=dict)
    matched_skills: tuple[str, ...] = ()
    missing_skills: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateAnalysis:
    resume_id: int | None
    label: str | None
    extracted: ExtractedResume
    normalized_skills: tuple[SkillMatch, ...]
    experience: ExperienceFeatures
    evaluation: EvaluationResult | None = None
    extra: dict[str, Any] = field(default_factory=dict)
