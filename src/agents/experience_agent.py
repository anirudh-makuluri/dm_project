from __future__ import annotations

import re
from typing import Iterable

from ..shared import ExperienceFeatures, ExtractedResume

YEAR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE)
LEADERSHIP_RE = re.compile(r"\b(led|managed|mentored|supervised|headed|owned|coordinated|directed)\b", re.IGNORECASE)
IMPACT_RE = re.compile(r"\b(improved|increased|decreased|reduced|delivered|built|created|launched|optimized|saved)\b", re.IGNORECASE)
QUANT_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


class ExperienceAgent:
    def analyze(self, extracted: ExtractedResume) -> ExperienceFeatures:
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
            if re.search(r"\b(engineer|developer|analyst|specialist|manager|lead|coordinator|scientist|recruiter|consultant|administrator)\b", sentence, re.IGNORECASE)
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
