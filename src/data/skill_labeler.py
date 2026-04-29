"""High-precision rule-based skill labeler used for gold annotations.

The labeler walks every alias surface form in the curated ontology and
matches it against the resume text using a word-boundary regex. Each match
yields a (start, end, surface_text, canonical_skill) record. The output is
intentionally **conservative** (high precision, lower recall) so the
resulting annotations are trustworthy as a lower-bound benchmark even
without manual review.

The output is human-editable JSONL. Reviewers are expected to:

* delete spurious matches,
* add canonical skills the labeler missed,
* keep ``resume_id`` stable so evaluation can join against predictions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..rag.ontology import SKILL_ONTOLOGY
from ..shared import SkillEntry


@dataclass(frozen=True)
class SkillSpan:
    canonical: str
    surface: str
    start: int
    end: int


def _surface_forms(entry: SkillEntry) -> list[str]:
    forms: list[str] = [entry.canonical]
    forms.extend(alias for alias in entry.aliases if alias)
    # Deduplicate but keep order so longer forms come first when canonical is long.
    seen: set[str] = set()
    unique: list[str] = []
    for form in forms:
        cleaned = form.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    # Sort longer forms first so "machine learning" is preferred over "ml".
    unique.sort(key=len, reverse=True)
    return unique


def _build_pattern(forms: Iterable[str]) -> re.Pattern[str]:
    escaped = sorted({re.escape(form) for form in forms}, key=len, reverse=True)
    if not escaped:
        # Match nothing.
        return re.compile(r"(?!x)x")
    body = "|".join(escaped)
    return re.compile(rf"(?<![A-Za-z0-9_+#\.]){body}(?![A-Za-z0-9_+#])", re.IGNORECASE)


class SkillLabeler:
    """Match curated ontology surface forms against resume text."""

    def __init__(self, ontology: tuple[SkillEntry, ...] = SKILL_ONTOLOGY) -> None:
        self.ontology = ontology
        self._form_to_canonical: dict[str, str] = {}
        all_forms: list[str] = []
        for entry in ontology:
            for form in _surface_forms(entry):
                # Keep first canonical wins; longer forms come first naturally.
                self._form_to_canonical.setdefault(form.lower(), entry.canonical)
                all_forms.append(form)
        self._pattern = _build_pattern(all_forms)

    def label(self, text: str) -> list[SkillSpan]:
        if not text:
            return []
        spans: list[SkillSpan] = []
        used_canonicals: set[str] = set()
        for match in self._pattern.finditer(text):
            surface = match.group(0)
            canonical = self._form_to_canonical.get(surface.lower())
            if canonical is None:
                continue
            # Keep at most the FIRST span per canonical to avoid double-counting
            # the same skill mentioned dozens of times in a resume.
            if canonical in used_canonicals:
                continue
            used_canonicals.add(canonical)
            spans.append(
                SkillSpan(
                    canonical=canonical,
                    surface=surface,
                    start=match.start(),
                    end=match.end(),
                )
            )
        spans.sort(key=lambda span: span.start)
        return spans

    def label_canonicals(self, text: str) -> list[str]:
        return [span.canonical for span in self.label(text)]
