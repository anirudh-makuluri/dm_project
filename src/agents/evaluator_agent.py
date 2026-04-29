"""Evaluator agent.

The fit_score and component_scores remain deterministic (so the comparison
against the baseline is reproducible and auditable). When an LLM provider
is available, it writes a calibrated rationale and recommendation that is
constrained by the deterministic signals.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from pydantic import ValidationError

from ..llm.provider import LLMProvider, LLMUnavailable, parse_json_block
from ..llm.prompts import EVALUATOR_PROMPT
from ..llm.schemas import EvaluatorRationale
from ..models.scoring import build_evaluation_result
from ..shared import EvaluationResult, ExperienceFeatures, JobProfile, SkillMatch

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """Deterministic scorer + optional LLM-authored rationale."""

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self.provider = provider

    @property
    def backend(self) -> str:
        return self.provider.name if self.provider is not None else "rule_based"

    def evaluate(
        self,
        normalized_skills: Iterable[SkillMatch],
        experience: ExperienceFeatures,
        job_profile: JobProfile,
        candidate_text: str,
    ) -> EvaluationResult:
        skills_tuple = tuple(normalized_skills)
        deterministic = build_evaluation_result(skills_tuple, experience, job_profile, candidate_text)
        if self.provider is None:
            return deterministic
        try:
            rationale = self._call_llm(deterministic, experience, job_profile)
        except LLMUnavailable:
            return deterministic
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Evaluator LLM call failed: %s", exc)
            return deterministic

        return EvaluationResult(
            label=deterministic.label,
            fit_score=deterministic.fit_score,
            rationale=(rationale.rationale or deterministic.rationale).strip(),
            strengths=tuple(rationale.strengths) or deterministic.strengths,
            gaps=tuple(rationale.gaps) or deterministic.gaps,
            recommendation=(rationale.recommendation or deterministic.recommendation).strip(),
            component_scores=deterministic.component_scores,
            matched_skills=deterministic.matched_skills,
            missing_skills=deterministic.missing_skills,
        )

    def _call_llm(
        self,
        deterministic: EvaluationResult,
        experience: ExperienceFeatures,
        job_profile: JobProfile,
    ) -> EvaluatorRationale:
        prompt = EVALUATOR_PROMPT.format(
            job_description=job_profile.description,
            required_skills=", ".join(job_profile.required_skills) or "n/a",
            preferred_skills=", ".join(job_profile.preferred_skills) or "n/a",
            matched_skills=", ".join(deterministic.matched_skills) or "n/a",
            missing_skills=", ".join(deterministic.missing_skills) or "n/a",
            total_years=f"{experience.total_years:.1f}",
            leadership=experience.leadership_indicators,
            text_similarity=deterministic.component_scores.get("text_similarity", 0.0),
            fit_score=deterministic.fit_score,
        )
        response = self.provider.complete(prompt, schema_name="evaluator_rationale")  # type: ignore[union-attr]
        try:
            payload = parse_json_block(response.text)
        except ValueError as exc:
            raise LLMUnavailable(f"Evaluator JSON parse failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise LLMUnavailable("Evaluator expected a JSON object")
        try:
            return EvaluatorRationale.model_validate(payload)
        except ValidationError as exc:
            raise LLMUnavailable(f"Evaluator schema validation failed: {exc}") from exc
