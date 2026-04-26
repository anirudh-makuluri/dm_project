from __future__ import annotations

from typing import Iterable

from ..models.scoring import build_evaluation_result
from ..shared import EvaluationResult, ExperienceFeatures, JobProfile, SkillMatch


class EvaluatorAgent:
    def evaluate(
        self,
        normalized_skills: Iterable[SkillMatch],
        experience: ExperienceFeatures,
        job_profile: JobProfile,
        candidate_text: str,
    ) -> EvaluationResult:
        return build_evaluation_result(normalized_skills, experience, job_profile, candidate_text)
