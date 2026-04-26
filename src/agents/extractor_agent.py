from __future__ import annotations

import re
from dataclasses import asdict
from typing import Iterable

from ..shared import ExtractedResume

SECTION_HEADERS = {
    "education": re.compile(r"^education\b[:\s-]*", re.IGNORECASE),
    "experience": re.compile(r"^(experience|work experience|employment)\b[:\s-]*", re.IGNORECASE),
    "skills": re.compile(r"^skills\b[:\s-]*", re.IGNORECASE),
    "projects": re.compile(r"^projects?\b[:\s-]*", re.IGNORECASE),
    "certifications": re.compile(r"^(certifications?|licenses?)\b[:\s-]*", re.IGNORECASE),
}

CONTACT_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b|\b\S+@\S+\b|https?://\S+|www\.\S+", re.IGNORECASE)
SKILL_HINT_RE = re.compile(
    r"\b(python|sql|pytorch|tensorflow|scikit-learn|sklearn|tableau|docker|kubernetes|react|javascript|java|flask|nlp|machine learning|data science|hr|recruiting|onboarding|communication|leadership)\b",
    re.IGNORECASE,
)


def _split_sections(text: str) -> dict[str, list[str]]:
    buckets = {name: [] for name in SECTION_HEADERS}
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


class ExtractorAgent:
    def extract(self, text: str, resume_id: int | None = None) -> ExtractedResume:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        sections = _split_sections(normalized)
        experience = sections.get("experience") or [normalized]
        skills_raw = _collect_skill_hints(sections.get("skills", []), normalized)
        if not skills_raw:
            flattened = []
            for chunk in re.split(r"[\n•;|]", normalized):
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

    def to_dict(self, extracted: ExtractedResume) -> dict[str, object]:
        return asdict(extracted)
