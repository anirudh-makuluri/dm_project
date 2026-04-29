"""Skill mining agent.

Combines LLM-based skill phrase extraction with retrieval-augmented
normalization. The LLM identifies candidate skill phrases from the resume
text; the retriever maps each phrase to a canonical entry in the ontology
using dense embeddings (sentence-transformers by default).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from typing import Iterable, Optional

from pydantic import ValidationError

from ..llm.provider import LLMProvider, LLMUnavailable, parse_json_block
from ..llm.prompts import SKILL_MINER_PROMPT
from ..llm.schemas import SkillMiningResult
from ..rag.retriever import SkillRetriever
from ..shared import ExtractedResume, SkillMatch

logger = logging.getLogger(__name__)


def _candidate_phrases(items: Iterable[str]) -> list[str]:
    phrases: list[str] = []
    for item in items:
        for chunk in re.split(r"[,;/|\n]", item):
            cleaned = chunk.strip(" -:\t").lower()
            if cleaned and cleaned not in phrases:
                phrases.append(cleaned)
    return phrases


class SkillMinerAgent:
    """LLM + RAG skill normalization with deterministic fallback."""

    def __init__(
        self,
        retriever: SkillRetriever,
        provider: Optional[LLMProvider] = None,
        min_confidence: float = 0.4,
        max_input_chars: int = 12000,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.min_confidence = min_confidence
        self.max_input_chars = max_input_chars

    @property
    def backend(self) -> str:
        return self.provider.name if self.provider is not None else "rule_based"

    def normalize(self, extracted: ExtractedResume) -> list[SkillMatch]:
        phrases = self._gather_phrases(extracted)
        normalized: list[SkillMatch] = []
        seen: set[str] = set()
        for phrase in phrases:
            matches = self.retriever.retrieve(phrase, top_k=3)
            if not matches:
                continue
            best = matches[0]
            if best.confidence < self.min_confidence:
                continue
            if best.canonical_skill in seen:
                continue
            normalized.append(best)
            seen.add(best.canonical_skill)
        return normalized

    def _gather_phrases(self, extracted: ExtractedResume) -> list[str]:
        # Try LLM first.
        llm_phrases: list[str] = []
        if self.provider is not None:
            try:
                llm_phrases = self._llm_phrases(extracted.raw_text)
            except LLMUnavailable:
                llm_phrases = []
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Skill miner LLM call failed: %s", exc)
                llm_phrases = []

        # Always include rule-based phrases as a safety net.
        rule_based_phrases = _candidate_phrases(extracted.skills_raw)
        if not rule_based_phrases:
            rule_based_phrases = _candidate_phrases(
                extracted.experience + extracted.projects + extracted.certifications
            )

        merged: list[str] = []
        for phrase in llm_phrases + rule_based_phrases:
            cleaned = phrase.strip().lower()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged

    def _llm_phrases(self, text: str) -> list[str]:
        truncated = (text or "")[: self.max_input_chars]
        if not truncated:
            return []
        prompt = SKILL_MINER_PROMPT.format(resume_text=truncated)
        response = self.provider.complete(prompt, schema_name="skill_mining")  # type: ignore[union-attr]
        try:
            payload = parse_json_block(response.text)
        except ValueError as exc:
            raise LLMUnavailable(f"Skill miner JSON parse failed: {exc}") from exc
        if isinstance(payload, list):
            payload = {"phrases": payload}
        if not isinstance(payload, dict):
            raise LLMUnavailable("Skill miner expected a JSON object or array")
        try:
            parsed = SkillMiningResult.model_validate(payload)
        except ValidationError as exc:
            raise LLMUnavailable(f"Skill miner schema validation failed: {exc}") from exc
        return [phrase for phrase in (p.strip() for p in parsed.phrases) if phrase]

    def to_dicts(self, matches: Iterable[SkillMatch]) -> list[dict[str, object]]:
        return [asdict(match) for match in matches]
