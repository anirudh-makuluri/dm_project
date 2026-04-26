from __future__ import annotations

import re
from dataclasses import asdict
from typing import Iterable

from ..rag.retriever import SkillRetriever
from ..shared import ExtractedResume, SkillMatch


def _candidate_phrases(items: Iterable[str]) -> list[str]:
    phrases: list[str] = []
    for item in items:
        for chunk in re.split(r"[,;/|\n]", item):
            cleaned = chunk.strip(" -:\t").lower()
            if cleaned and cleaned not in phrases:
                phrases.append(cleaned)
    return phrases


class SkillMinerAgent:
    def __init__(self, retriever: SkillRetriever, min_confidence: float = 0.2):
        self.retriever = retriever
        self.min_confidence = min_confidence

    def normalize(self, extracted: ExtractedResume) -> list[SkillMatch]:
        raw_phrases = _candidate_phrases(extracted.skills_raw)
        if not raw_phrases:
            raw_phrases = _candidate_phrases(extracted.experience + extracted.projects + extracted.certifications)

        normalized: list[SkillMatch] = []
        seen: set[str] = set()
        for phrase in raw_phrases:
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

    def to_dicts(self, matches: Iterable[SkillMatch]) -> list[dict[str, object]]:
        return [asdict(match) for match in matches]
