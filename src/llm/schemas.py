"""Pydantic schemas used for structured LLM outputs.

These schemas are the contract that every LLM backend must honour. When an
LLM backend is unavailable or returns malformed JSON, callers should fall
back to the rule-based implementations while preserving these exact shapes.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ResumeExtraction(BaseModel):
    """Structured resume fields produced by the extractor agent."""

    education: List[str] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)
    skills_raw: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)


class SkillMiningResult(BaseModel):
    """Skill phrases identified by the skill-miner agent prior to RAG normalization."""

    phrases: List[str] = Field(default_factory=list)


class ExperienceExtraction(BaseModel):
    """Structured experience features produced by the experience agent."""

    total_years: float = 0.0
    role_count: int = 0
    leadership_indicators: int = 0
    impact_statements: int = 0
    quantified_impact: int = 0
    evidence: List[str] = Field(default_factory=list)


class EvaluatorRationale(BaseModel):
    """LLM-authored rationale attached to a deterministic fit score."""

    rationale: str = ""
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    recommendation: str = ""
