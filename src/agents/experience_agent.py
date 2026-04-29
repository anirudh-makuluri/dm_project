"""Experience agent.

LLM-based experience extraction with deterministic regex fallback. The
LLM result is sanity-checked against the regex baseline so a wildly
hallucinated value cannot dominate the final ``ExperienceFeatures``.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import ValidationError

from ..llm.provider import LLMProvider, LLMUnavailable, parse_json_block
from ..llm.prompts import EXPERIENCE_PROMPT
from ..llm.schemas import ExperienceExtraction
from ..shared import ExperienceFeatures, ExtractedResume

logger = logging.getLogger(__name__)


YEAR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE)
LEADERSHIP_RE = re.compile(
    r"\b(led|managed|mentored|supervised|headed|owned|coordinated|directed)\b",
    re.IGNORECASE,
)
IMPACT_RE = re.compile(
    r"\b(improved|increased|decreased|reduced|delivered|built|created|launched|optimized|saved)\b",
    re.IGNORECASE,
)
QUANT_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _rule_based_features(extracted: ExtractedResume) -> ExperienceFeatures:
    text = " ".join(extracted.experience or [extracted.raw_text])
    years = [float(match.group(1)) for match in YEAR_RE.finditer(text)]
    total_years = max(years) if years else 0.0

    sentences = _split_sentences(text)
    leadership = sum(1 for sentence in sentences if LEADERSHIP_RE.search(sentence))
    impact = [sentence for sentence in sentences if IMPACT_RE.search(sentence)]
    quantified = sum(1 for sentence in impact if QUANT_RE.search(sentence))

    role_markers = sum(
        1
        for sentence in sentences
        if re.search(
            r"\b(engineer|developer|analyst|specialist|manager|lead|coordinator|scientist|recruiter|consultant|administrator)\b",
            sentence,
            re.IGNORECASE,
        )
    )
    evidence = tuple(impact[:3])
    return ExperienceFeatures(
        total_years=total_years,
        role_count=role_markers,
        leadership_indicators=leadership,
        impact_statements=len(impact),
        quantified_impact=quantified,
        evidence=evidence,
    )


class ExperienceAgent:
    """LLM-backed experience analyzer with deterministic fallback."""

    def __init__(self, provider: Optional[LLMProvider] = None, max_input_chars: int = 12000) -> None:
        self.provider = provider
        self.max_input_chars = max_input_chars

    @property
    def backend(self) -> str:
        return self.provider.name if self.provider is not None else "rule_based"

    def analyze(self, extracted: ExtractedResume) -> ExperienceFeatures:
        rule_features = _rule_based_features(extracted)
        if self.provider is None:
            return rule_features
        try:
            llm = self._call_llm(extracted)
        except LLMUnavailable:
            return rule_features
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Experience LLM call failed: %s", exc)
            return rule_features
        return self._merge(rule_features, llm)

    def _call_llm(self, extracted: ExtractedResume) -> ExperienceExtraction:
        text = (extracted.raw_text or "")[: self.max_input_chars]
        if not text:
            raise LLMUnavailable("Empty resume text")
        prompt = EXPERIENCE_PROMPT.format(resume_text=text)
        response = self.provider.complete(prompt, schema_name="experience_extraction")  # type: ignore[union-attr]
        try:
            payload = parse_json_block(response.text)
        except ValueError as exc:
            raise LLMUnavailable(f"Experience JSON parse failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise LLMUnavailable("Experience expected a JSON object")
        try:
            return ExperienceExtraction.model_validate(payload)
        except ValidationError as exc:
            raise LLMUnavailable(f"Experience schema validation failed: {exc}") from exc

    @staticmethod
    def _merge(rule_features: ExperienceFeatures, llm: ExperienceExtraction) -> ExperienceFeatures:
        # Take the larger of the two for count-based features (the LLM tends to find
        # more nuanced phrasing) but cap years at 60 as a sanity bound.
        total_years = max(0.0, min(60.0, max(rule_features.total_years, float(llm.total_years))))
        role_count = max(rule_features.role_count, int(llm.role_count))
        leadership = max(rule_features.leadership_indicators, int(llm.leadership_indicators))
        impact = max(rule_features.impact_statements, int(llm.impact_statements))
        quantified = max(rule_features.quantified_impact, int(llm.quantified_impact))
        evidence = tuple(item.strip() for item in llm.evidence if item.strip()) or rule_features.evidence
        return ExperienceFeatures(
            total_years=total_years,
            role_count=role_count,
            leadership_indicators=leadership,
            impact_statements=impact,
            quantified_impact=quantified,
            evidence=evidence[:3],
        )
