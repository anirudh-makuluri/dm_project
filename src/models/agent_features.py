"""Per-resume feature vector built from the rule-based agent stack.

These are the features we feed into the supervised classifier on top of
TF-IDF.  All features are computed **without any LLM calls** by using
each agent's deterministic fallback path, which makes the extraction
fast enough to run on both train (~2000 resumes) and test (~500
resumes) in a few seconds.

Feature layout (83 dims total):

1. ``fit_scores`` (24 dims): deterministic ``fit_score`` against each of
   the ``DEFAULT_JOB_PROFILES`` job profiles.
2. ``fit_argmax_onehot`` (24 dims): one-hot encoding of the argmax
   profile.  Captures the agent-stack's top-1 vote directly.
3. ``experience`` (7 dims): years, years_capped, role_count,
   leadership_indicators, impact_statements, quantified_impact,
   has_significant_experience (years >= 5).
4. ``skill_stats`` (4 dims): n_raw_skills, n_normalized_skills,
   avg_skill_confidence, max_skill_confidence.
5. ``skill_category_hist`` (19 dims): count of normalized skills in each
   ontology category, normalized to a fraction.
6. ``resume_structure`` (5 dims): n_education_lines,
   n_experience_lines, n_certifications, n_projects,
   log_resume_length_chars.

Usage
-----
::

    from src.models.agent_features import AgentFeatureExtractor

    extractor = AgentFeatureExtractor()
    train_features = extractor.transform(train_df)
    test_features = extractor.transform(test_df)
    # train_features.shape == (len(train_df), 83)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..agents.experience_agent import ExperienceAgent
from ..agents.extractor_agent import ExtractorAgent
from ..agents.skill_miner_agent import SkillMinerAgent
from ..models.scoring import DEFAULT_JOB_PROFILES, score_candidate
from ..rag.ontology import SKILL_ONTOLOGY
from ..rag.retriever import SkillRetriever
from ..rag.vector_store import SkillVectorStore
from ..shared import ExperienceFeatures, SkillMatch


# Stable, alphabetical ordering so features are reproducible across runs
# and across different ``DEFAULT_JOB_PROFILES`` insertion orders.
_FIT_PROFILES: tuple[str, ...] = tuple(sorted(DEFAULT_JOB_PROFILES.keys()))
_SKILL_CATEGORIES: tuple[str, ...] = tuple(
    sorted({entry.category for entry in SKILL_ONTOLOGY})
)


def _build_feature_names() -> list[str]:
    names: list[str] = []
    names.extend(f"fit::{label}" for label in _FIT_PROFILES)
    names.extend(f"top_profile::{label}" for label in _FIT_PROFILES)
    names.extend(
        [
            "exp::years",
            "exp::years_capped_8",
            "exp::role_count",
            "exp::leadership",
            "exp::impact",
            "exp::quantified",
            "exp::has_senior_experience",
        ]
    )
    names.extend(
        [
            "skills::n_raw",
            "skills::n_normalized",
            "skills::avg_confidence",
            "skills::max_confidence",
        ]
    )
    names.extend(f"skill_cat::{cat}" for cat in _SKILL_CATEGORIES)
    names.extend(
        [
            "struct::n_education",
            "struct::n_experience",
            "struct::n_certifications",
            "struct::n_projects",
            "struct::log_length",
        ]
    )
    return names


FEATURE_NAMES: tuple[str, ...] = tuple(_build_feature_names())
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class AgentFeatureExtractor:
    """Run the rule-based agent stack and return a dense feature matrix."""

    min_confidence: float = 0.4

    def __post_init__(self) -> None:
        # ``provider=None`` triggers the rule-based-only code path inside
        # each agent (see ``ExtractorAgent.extract`` etc.) so we do not
        # require LM Studio / Gemini to be running for feature extraction.
        self._extractor = ExtractorAgent(provider=None)
        self._experience_agent = ExperienceAgent(provider=None)
        self._retriever = SkillRetriever(SkillVectorStore())
        self._skill_miner = SkillMinerAgent(
            retriever=self._retriever,
            provider=None,
            min_confidence=self.min_confidence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def transform(
        self,
        df: pd.DataFrame,
        text_col: str = "resume_text",
        id_col: str = "resume_id",
    ) -> np.ndarray:
        """Compute a ``(n_rows, N_FEATURES)`` float32 feature matrix."""

        n = len(df)
        out = np.zeros((n, N_FEATURES), dtype=np.float32)
        for i, row in enumerate(df.itertuples(index=False)):
            resume_id = getattr(row, id_col, i) if hasattr(row, id_col) else i
            text = str(getattr(row, text_col, ""))
            out[i, :] = self._features_for_resume(text, int(resume_id))
        return out

    def transform_one(self, text: str, resume_id: int = 0) -> np.ndarray:
        return self._features_for_resume(text, resume_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _features_for_resume(self, text: str, resume_id: int) -> np.ndarray:
        extracted = self._extractor.extract(text, resume_id=resume_id)
        normalized = self._skill_miner.normalize(extracted)
        experience = self._experience_agent.analyze(extracted)

        fit_scores = _fit_score_vector(normalized, experience, text)
        top_profile = _one_hot_top_profile(fit_scores)
        exp_vec = _experience_vector(experience)
        skill_stats = _skill_stats_vector(extracted.skills_raw, normalized)
        skill_hist = _skill_category_histogram(normalized)
        structure = _resume_structure_vector(extracted, text)

        return np.concatenate(
            [fit_scores, top_profile, exp_vec, skill_stats, skill_hist, structure]
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Pure functions — easy to unit-test in isolation.
# ---------------------------------------------------------------------------


def _fit_score_vector(
    normalized: Sequence[SkillMatch],
    experience: ExperienceFeatures,
    text: str,
) -> np.ndarray:
    scores = np.zeros(len(_FIT_PROFILES), dtype=np.float32)
    for i, label in enumerate(_FIT_PROFILES):
        profile = DEFAULT_JOB_PROFILES[label]
        fit, _components, _matched, _missing = score_candidate(
            normalized, experience, profile, text
        )
        scores[i] = fit
    # Fit scores are in [0, 100]; scale to [0, 1] so they are on the same
    # order of magnitude as the other features (LinearSVC is not scale
    # invariant).
    return scores / 100.0


def _one_hot_top_profile(fit_scores: np.ndarray) -> np.ndarray:
    out = np.zeros(len(_FIT_PROFILES), dtype=np.float32)
    if fit_scores.size == 0:
        return out
    top = int(np.argmax(fit_scores))
    out[top] = 1.0
    return out


def _experience_vector(experience: ExperienceFeatures) -> np.ndarray:
    return np.array(
        [
            float(experience.total_years),
            min(float(experience.total_years), 8.0) / 8.0,
            float(experience.role_count),
            float(experience.leadership_indicators),
            float(experience.impact_statements),
            float(experience.quantified_impact),
            1.0 if experience.total_years >= 5 else 0.0,
        ],
        dtype=np.float32,
    )


def _skill_stats_vector(
    raw_skills: Sequence[str], normalized: Sequence[SkillMatch]
) -> np.ndarray:
    n_raw = float(len(raw_skills))
    n_norm = float(len(normalized))
    if normalized:
        confidences = [float(s.confidence) for s in normalized]
        avg_conf = sum(confidences) / len(confidences)
        max_conf = max(confidences)
    else:
        avg_conf = 0.0
        max_conf = 0.0
    return np.array([n_raw, n_norm, avg_conf, max_conf], dtype=np.float32)


def _skill_category_histogram(normalized: Sequence[SkillMatch]) -> np.ndarray:
    out = np.zeros(len(_SKILL_CATEGORIES), dtype=np.float32)
    if not normalized:
        return out
    # Map canonical skill -> category via the ontology.
    cat_index = {cat: idx for idx, cat in enumerate(_SKILL_CATEGORIES)}
    skill_to_cat = {entry.canonical: entry.category for entry in SKILL_ONTOLOGY}
    for skill in normalized:
        cat = skill_to_cat.get(skill.canonical_skill)
        if cat is None:
            continue
        out[cat_index[cat]] += 1.0
    total = out.sum()
    if total > 0:
        out /= total
    return out


def _resume_structure_vector(extracted, text: str) -> np.ndarray:
    return np.array(
        [
            float(len(extracted.education)),
            float(len(extracted.experience)),
            float(len(extracted.certifications)),
            float(len(extracted.projects)),
            math.log1p(float(len(text))),
        ],
        dtype=np.float32,
    )


def build_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "resume_text",
    id_col: str = "resume_id",
    min_confidence: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Convenience wrapper that returns ``(train, test, feature_names)``."""

    extractor = AgentFeatureExtractor(min_confidence=min_confidence)
    train = extractor.transform(train_df, text_col=text_col, id_col=id_col)
    test = extractor.transform(test_df, text_col=text_col, id_col=id_col)
    return train, test, extractor.feature_names
