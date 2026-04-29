"""Learn job profiles from a labelled training corpus.

The hand-curated ``DEFAULT_JOB_PROFILES`` in :mod:`src.models.scoring`
have shallow coverage outside of tech (CHEF, ADVOCATE, BANKING,
TEACHER, ...). This module replaces them with profiles derived from the
training resumes themselves:

1. For each category C, take the in-class train resumes.
2. Run the rule-based skill miner over each (no LLM calls). This pulls
   out canonical skill names from the existing 154-entry ontology.
3. Aggregate canonical-skill coverage = (#in-class resumes containing
   that canonical skill) / (#in-class resumes).
4. Promote skills with coverage >= 30% to ``required_skills`` (capped
   at 6) and skills with 10% <= coverage < 30% to ``preferred_skills``
   (capped at 8). For under-represented categories (n < 35) fall back
   to a top-K rule.
5. Auto-generate a description paragraph from the top characteristic
   word/bigram terms (TF-IDF mean-in - mean-out scoring) so the
   ``text_similarity`` component of ``score_candidate`` still works
   even when the canonical-skill match is weak.

The hand-curated description from ``PROFILE_SEEDS`` (when available) is
used as a textual fallback when the auto description is too sparse,
because the existing seed descriptions are already domain-correct.

Output schema is identical to ``JobProfile`` so the rest of the
pipeline does not change.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..agents.extractor_agent import ExtractorAgent
from ..agents.skill_miner_agent import SkillMinerAgent
from ..rag.ontology import SKILL_ONTOLOGY
from ..rag.retriever import SkillRetriever
from ..rag.vector_store import SkillVectorStore
from ..shared import JobProfile, SkillEntry

logger = logging.getLogger(__name__)


# Defaults below were chosen by inspecting the per-category train
# distribution: median train_count is ~92 resumes, min 18 (BPO).  A 30%
# coverage threshold + cap of 6 produces compact profiles (<= 6 required,
# <= 8 preferred) that mirror the structure of the hand-curated ones.
REQUIRED_THRESHOLD = 0.30
PREFERRED_THRESHOLD = 0.10
MAX_REQUIRED = 6
MAX_PREFERRED = 8
SPARSE_CLASS_FLOOR = 35
TOP_NGRAM_TERMS = 8

_NAME_TOKEN_RE = re.compile(r"[a-z][a-z\-+#]+(?: [a-z][a-z\-+#]+){0,2}", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Direct ontology-substring scanner.
#
# Profile mining via the rule-based ``SkillMinerAgent`` was found to miss
# most tech skills (python, apis, docker, ...) because its regex hint
# list is intentionally narrow.  The LLM-backed miner has the right
# recall but costs ~90 minutes over the ~2000 train resumes.  The third
# option below is the one we use: for each canonical skill in the
# ontology, look for the canonical name **or** any of its aliases as a
# whole-word substring in the lowercased resume text.  This recovers
# ~2x more canonical skills per resume than the rule-based miner at a
# fraction of the cost (the whole train set scans in ~10 seconds).
# ---------------------------------------------------------------------------


def _build_alias_patterns(
    ontology: tuple[SkillEntry, ...] = SKILL_ONTOLOGY,
) -> list[tuple[str, re.Pattern[str]]]:
    """Compile ``(canonical_name, regex)`` pairs for whole-word matching."""

    pairs: list[tuple[str, re.Pattern[str]]] = []
    for entry in ontology:
        surface_forms: set[str] = {entry.canonical}
        surface_forms.update(entry.aliases)
        # Build one alternation regex per skill.  ``\b`` whole-word match
        # is too strict for terms with punctuation (``c++``, ``c#``,
        # ``ci/cd``, ``a/b``); for those we fall back to lookarounds.
        escaped: list[str] = []
        for term in surface_forms:
            cleaned = term.strip().lower()
            if not cleaned:
                continue
            esc = re.escape(cleaned)
            # Only enforce word boundaries on alphanumeric chars.
            left = r"(?<![a-z0-9])" if cleaned[0].isalnum() else ""
            right = r"(?![a-z0-9])" if cleaned[-1].isalnum() else ""
            escaped.append(f"{left}{esc}{right}")
        if not escaped:
            continue
        pattern = re.compile("|".join(escaped), re.IGNORECASE)
        pairs.append((entry.canonical, pattern))
    return pairs


_ALIAS_PATTERNS_CACHE: list[tuple[str, re.Pattern[str]]] | None = None


def _alias_patterns() -> list[tuple[str, re.Pattern[str]]]:
    global _ALIAS_PATTERNS_CACHE
    if _ALIAS_PATTERNS_CACHE is None:
        _ALIAS_PATTERNS_CACHE = _build_alias_patterns()
    return _ALIAS_PATTERNS_CACHE


def scan_canonical_skills(text: str) -> set[str]:
    """Return the set of canonical skill names that appear (verbatim or
    via any alias) in ``text``.  Substring-match, lowercase, no LLM."""

    if not text:
        return set()
    lowered = text.lower()
    found: set[str] = set()
    for canonical, pattern in _alias_patterns():
        if pattern.search(lowered):
            found.add(canonical)
    return found


def _build_extractor_and_miner(min_confidence: float = 0.4) -> tuple[ExtractorAgent, SkillMinerAgent]:
    """Build the rule-based agent stack used to mine canonical skills."""

    extractor = ExtractorAgent(provider=None)
    retriever = SkillRetriever(SkillVectorStore())
    miner = SkillMinerAgent(retriever=retriever, provider=None, min_confidence=min_confidence)
    return extractor, miner


def _mine_canonical_skills(
    text: str,
    resume_id: int,
    extractor: ExtractorAgent,
    miner: SkillMinerAgent,
) -> set[str]:
    extracted = extractor.extract(text, resume_id=resume_id)
    normalized = miner.normalize(extracted)
    return {skill.canonical_skill for skill in normalized}


def _aggregate_skill_coverage(
    in_class_texts: Iterable[tuple[int, str]],
    extractor: ExtractorAgent | None = None,
    miner: SkillMinerAgent | None = None,
    *,
    use_alias_scan: bool = True,
) -> tuple[Counter[str], int]:
    """Return (skill -> count_of_resumes_containing_it, n_resumes).

    When ``use_alias_scan`` is True (the default), this scans each
    resume directly against the ontology aliases.  Pass False to use
    the rule-based ``SkillMinerAgent`` instead (legacy behaviour).
    """

    counter: Counter[str] = Counter()
    n = 0
    for resume_id, text in in_class_texts:
        if use_alias_scan:
            skills = scan_canonical_skills(text)
        else:
            assert extractor is not None and miner is not None
            skills = _mine_canonical_skills(text, resume_id, extractor, miner)
        counter.update(skills)
        n += 1
    return counter, n


def _split_required_preferred(
    coverage_counter: Counter[str],
    n_in_class: int,
    *,
    required_threshold: float,
    preferred_threshold: float,
    max_required: int,
    max_preferred: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if n_in_class == 0:
        return (), ()

    coverage_pairs = sorted(
        ((skill, count / n_in_class) for skill, count in coverage_counter.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )

    required: list[str] = []
    preferred: list[str] = []

    for skill, coverage in coverage_pairs:
        if coverage >= required_threshold and len(required) < max_required:
            required.append(skill)
        elif coverage >= preferred_threshold and len(preferred) < max_preferred:
            preferred.append(skill)

    # Sparse class fallback: ensure we have *something* in required even
    # if coverage is low.  Pick the top 3 most-frequent canonical skills
    # in the class as a soft floor.  Avoids empty ``required_skills``
    # which would force ``required_score = 0`` for everyone in
    # ``score_candidate``.
    if not required and coverage_pairs:
        required = [skill for skill, _ in coverage_pairs[:3]]
        # Anything not in required falls into preferred
        preferred = [
            skill
            for skill, _ in coverage_pairs[3 : 3 + max_preferred]
            if skill not in required
        ]

    return tuple(required), tuple(preferred)


def _characteristic_terms(
    train_df: pd.DataFrame,
    label: str,
    *,
    text_col: str = "resume_text",
    label_col: str = "label",
    top_k: int = TOP_NGRAM_TERMS,
) -> list[str]:
    """Top-K terms by (mean_TFIDF_in - mean_TFIDF_out)."""

    in_mask = (train_df[label_col] == label).values
    if int(in_mask.sum()) < 2:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        max_features=10_000,
        stop_words="english",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(train_df[text_col].astype(str).tolist())
    feature_names = vectorizer.get_feature_names_out()

    mean_in = np.asarray(matrix[in_mask].mean(axis=0)).ravel()
    mean_out = np.asarray(matrix[~in_mask].mean(axis=0)).ravel()
    score = mean_in - mean_out

    top_indices = np.argsort(-score)[: top_k * 3]
    selected: list[str] = []
    for idx in top_indices:
        term = feature_names[idx]
        # Drop tokens that look like proper nouns or pure numbers.
        if not _NAME_TOKEN_RE.fullmatch(term):
            continue
        if term in selected:
            continue
        selected.append(term)
        if len(selected) >= top_k:
            break
    return selected


def _auto_description(label: str, top_terms: Iterable[str], fallback: str | None) -> str:
    pretty = label.replace("_", " ").replace("-", " ").lower().title()
    if not top_terms:
        return fallback or f"Role focused on {pretty}."
    head = ", ".join(list(top_terms)[:5])
    return (
        f"{pretty} role typically involves {head}." +
        (f" {fallback}" if fallback else "")
    )


def _hand_curated_description(label: str) -> str | None:
    # Lazy import to avoid a circular dependency at module load time.
    from .scoring import DEFAULT_JOB_PROFILES, _normalize_label_key

    profile = DEFAULT_JOB_PROFILES.get(label) or DEFAULT_JOB_PROFILES.get(_normalize_label_key(label))
    return profile.description if profile is not None else None


def build_profile_for_category(
    label: str,
    train_df: pd.DataFrame,
    *,
    text_col: str = "resume_text",
    label_col: str = "label",
    id_col: str = "resume_id",
    extractor: ExtractorAgent | None = None,
    miner: SkillMinerAgent | None = None,
    use_alias_scan: bool = True,
    required_threshold: float = REQUIRED_THRESHOLD,
    preferred_threshold: float = PREFERRED_THRESHOLD,
    max_required: int = MAX_REQUIRED,
    max_preferred: int = MAX_PREFERRED,
    sparse_class_floor: int = SPARSE_CLASS_FLOOR,
) -> JobProfile:
    """Learn one ``JobProfile`` from the in-class slice of ``train_df``."""

    if not use_alias_scan and (extractor is None or miner is None):
        extractor, miner = _build_extractor_and_miner()

    in_class = train_df[train_df[label_col] == label]
    n = len(in_class)
    if n == 0:
        # Fall back to the hand-curated profile if it exists.
        from .scoring import get_job_profile

        return get_job_profile(label)

    # Adjust thresholds for sparse classes.
    if n < sparse_class_floor:
        required_threshold = max(0.20, required_threshold - 0.10)
        preferred_threshold = max(0.05, preferred_threshold - 0.05)

    records = in_class.to_dict(orient="records")
    in_class_texts = (
        (int(row.get(id_col, 0) or 0), str(row.get(text_col, "")))
        for row in records
    )
    coverage_counter, n_in_class = _aggregate_skill_coverage(
        in_class_texts, extractor, miner, use_alias_scan=use_alias_scan
    )
    required, preferred = _split_required_preferred(
        coverage_counter,
        n_in_class,
        required_threshold=required_threshold,
        preferred_threshold=preferred_threshold,
        max_required=max_required,
        max_preferred=max_preferred,
    )

    top_terms = _characteristic_terms(train_df, label, text_col=text_col, label_col=label_col)
    description = _auto_description(label, top_terms, _hand_curated_description(label))

    return JobProfile(
        label=label,
        description=description,
        required_skills=required,
        preferred_skills=preferred,
    )


def build_all_profiles(
    train_df: pd.DataFrame,
    *,
    text_col: str = "resume_text",
    label_col: str = "label",
    id_col: str = "resume_id",
    use_alias_scan: bool = True,
    log_progress: bool = True,
) -> dict[str, JobProfile]:
    """Learn one profile per unique label found in ``train_df``."""

    extractor: ExtractorAgent | None = None
    miner: SkillMinerAgent | None = None
    if not use_alias_scan:
        extractor, miner = _build_extractor_and_miner()

    labels = sorted(train_df[label_col].dropna().astype(str).unique())
    out: dict[str, JobProfile] = {}
    for i, label in enumerate(labels, start=1):
        if log_progress:
            logger.info("Building profile %d/%d for %s ...", i, len(labels), label)
            print(f"  [{i:>2}/{len(labels)}] {label} ...", flush=True)
        profile = build_profile_for_category(
            label,
            train_df,
            text_col=text_col,
            label_col=label_col,
            id_col=id_col,
            extractor=extractor,
            miner=miner,
            use_alias_scan=use_alias_scan,
        )
        out[label] = profile
        if log_progress:
            print(
                f"        required={profile.required_skills}\n"
                f"        preferred={profile.preferred_skills}",
                flush=True,
            )
    return out


def save_profiles(profiles: Mapping[str, JobProfile], path: str | Path) -> None:
    payload = {
        label: {
            "label": profile.label,
            "description": profile.description,
            "required_skills": list(profile.required_skills),
            "preferred_skills": list(profile.preferred_skills),
        }
        for label, profile in profiles.items()
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_profiles(path: str | Path) -> dict[str, JobProfile]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, JobProfile] = {}
    for label, entry in raw.items():
        out[label] = JobProfile(
            label=str(entry.get("label", label)),
            description=str(entry.get("description", "")),
            required_skills=tuple(entry.get("required_skills", [])),
            preferred_skills=tuple(entry.get("preferred_skills", [])),
        )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data-driven job profiles from a train JSONL.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output", default="data/profiles_learned.json")
    parser.add_argument("--required_threshold", type=float, default=REQUIRED_THRESHOLD)
    parser.add_argument("--preferred_threshold", type=float, default=PREFERRED_THRESHOLD)
    parser.add_argument(
        "--use_rule_based_miner",
        action="store_true",
        help="Use the rule-based SkillMinerAgent for skill discovery (slower, lower recall).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    train_df = pd.read_json(args.train_jsonl, lines=True)
    print(f"Loaded {len(train_df)} train resumes across {train_df['label'].nunique()} labels.", flush=True)
    profiles = build_all_profiles(train_df, use_alias_scan=not args.use_rule_based_miner)
    save_profiles(profiles, args.output)
    print(f"Saved {len(profiles)} learned profiles to {args.output}", flush=True)


if __name__ == "__main__":
    main()
