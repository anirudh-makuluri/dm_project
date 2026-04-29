"""Resume extractor agent.

Tries the configured LLM provider first; on any failure (provider
unavailable, parse / validation error) it falls back to the deterministic
regex-based extractor so the rest of the pipeline keeps working.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from typing import Iterable, Optional

from pydantic import ValidationError

from ..llm.provider import LLMProvider, LLMUnavailable, parse_json_block
from ..llm.prompts import EXTRACTOR_PROMPT
from ..llm.schemas import ResumeExtraction
from ..shared import ExtractedResume

logger = logging.getLogger(__name__)


SECTION_HEADERS = {
    "education": re.compile(r"^education\b[:\s-]*", re.IGNORECASE),
    "experience": re.compile(r"^(experience|work experience|employment)\b[:\s-]*", re.IGNORECASE),
    "skills": re.compile(r"^skills\b[:\s-]*", re.IGNORECASE),
    "projects": re.compile(r"^projects?\b[:\s-]*", re.IGNORECASE),
    "certifications": re.compile(r"^(certifications?|licenses?)\b[:\s-]*", re.IGNORECASE),
}

CONTACT_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b|\b\S+@\S+\b|https?://\S+|www\.\S+",
    re.IGNORECASE,
)
SKILL_HINT_RE = re.compile(
    r"\b(python|sql|pytorch|tensorflow|scikit-learn|sklearn|tableau|docker|kubernetes|react|javascript|"
    r"java|flask|nlp|machine learning|data science|hr|recruiting|onboarding|communication|leadership|"
    r"excel|aws|azure|gcp|powerbi|power bi|tableau|figma|photoshop|autocad)\b",
    re.IGNORECASE,
)


def _split_sections(text: str) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {name: [] for name in SECTION_HEADERS}
    current = "experience"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = next((name for name, pattern in SECTION_HEADERS.items() if pattern.match(line)), None)
        if matched:
            current = matched
            line = SECTION_HEADERS[matched].sub("", line).strip(" :-")
            if line:
                buckets[current].append(line)
            continue
        buckets.setdefault(current, []).append(line)
    return buckets


def _collect_skill_hints(lines: Iterable[str], text: str) -> list[str]:
    hints: list[str] = []
    for line in lines:
        if SKILL_HINT_RE.search(line):
            for chunk in re.split(r"[,;/|]", line):
                cleaned = chunk.strip(" -:\t")
                if cleaned and cleaned not in hints:
                    hints.append(cleaned)
    if not hints:
        for match in SKILL_HINT_RE.findall(text):
            if match and match not in hints:
                hints.append(match)
    return hints


def _detect_contact_lines(text: str) -> list[str]:
    contact_lines: list[str] = []
    for match in CONTACT_RE.finditer(text):
        contact_lines.append(match.group(0))
    return contact_lines


def _rule_based_extract(text: str, resume_id: int | None) -> ExtractedResume:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    sections = _split_sections(normalized)
    experience = sections.get("experience") or [normalized]
    skills_raw = _collect_skill_hints(sections.get("skills", []), normalized)
    if not skills_raw:
        flattened: list[str] = []
        for chunk in re.split(r"[\n\u2022;|]", normalized):
            cleaned = chunk.strip()
            if cleaned:
                flattened.append(cleaned)
        skills_raw = _collect_skill_hints(flattened, normalized)
    return ExtractedResume(
        resume_id=resume_id,
        raw_text=normalized,
        education=list(sections.get("education", [])),
        experience=list(experience),
        skills_raw=skills_raw,
        projects=list(sections.get("projects", [])),
        certifications=list(sections.get("certifications", [])),
        contact=_detect_contact_lines(normalized),
    )


class ExtractorAgent:
    """LLM-backed resume parser with deterministic fallback."""

    def __init__(self, provider: Optional[LLMProvider] = None, max_input_chars: int = 12000) -> None:
        self.provider = provider
        self.max_input_chars = max_input_chars

    @property
    def backend(self) -> str:
        return self.provider.name if self.provider is not None else "rule_based"

    def extract(self, text: str, resume_id: int | None = None) -> ExtractedResume:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        rule_based = _rule_based_extract(normalized, resume_id)
        if self.provider is None:
            return rule_based
        try:
            llm_result = self._call_llm(normalized)
        except LLMUnavailable:
            return rule_based
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Extractor LLM call failed: %s", exc)
            return rule_based
        return self._merge(rule_based, llm_result, resume_id, normalized)

    def _call_llm(self, normalized: str) -> ResumeExtraction:
        truncated = normalized[: self.max_input_chars]
        prompt = EXTRACTOR_PROMPT.format(resume_text=truncated)
        response = self.provider.complete(prompt, schema_name="resume_extraction")  # type: ignore[union-attr]
        try:
            payload = parse_json_block(response.text)
        except ValueError as exc:
            raise LLMUnavailable(f"Extractor JSON parse failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise LLMUnavailable("Extractor expected a JSON object")
        try:
            return ResumeExtraction.model_validate(payload)
        except ValidationError as exc:
            raise LLMUnavailable(f"Extractor schema validation failed: {exc}") from exc

    @staticmethod
    def _merge(
        rule_based: ExtractedResume,
        llm_result: ResumeExtraction,
        resume_id: int | None,
        normalized: str,
    ) -> ExtractedResume:
        def pick(llm_field: list[str], rule_field: list[str]) -> list[str]:
            cleaned = [item.strip() for item in llm_field if isinstance(item, str) and item.strip()]
            if cleaned:
                return cleaned
            return list(rule_field)

        return ExtractedResume(
            resume_id=resume_id,
            raw_text=normalized,
            education=pick(llm_result.education, rule_based.education),
            experience=pick(llm_result.experience, rule_based.experience),
            skills_raw=pick(llm_result.skills_raw, rule_based.skills_raw),
            projects=pick(llm_result.projects, rule_based.projects),
            certifications=pick(llm_result.certifications, rule_based.certifications),
            contact=list(rule_based.contact),
        )

    def to_dict(self, extracted: ExtractedResume) -> dict[str, object]:
        return asdict(extracted)
