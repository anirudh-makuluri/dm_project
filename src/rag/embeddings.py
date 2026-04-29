"""Embedding backends and skill index used by the RAG retriever.

Two encoders are supported:

* :class:`SentenceTransformerEncoder` — dense semantic embeddings via the
  ``sentence-transformers`` package. Default for skill normalization.
* :class:`TfidfEncoder` — classic TF-IDF over n-grams. Used as a graceful
  fallback when ``sentence-transformers`` cannot be imported (e.g. offline
  environment without the model cached).

Both expose the same ``encode(list[str]) -> np.ndarray`` interface. The
:class:`SkillEmbeddingIndex` dispatches to whichever encoder is configured.

Backend selection (highest priority first):

1. ``encoder`` argument passed to :func:`SkillEmbeddingIndex.build`.
2. Environment variable ``EMBEDDING_BACKEND`` set to ``semantic`` or
   ``tfidf``. ``EMBEDDING_BACKEND=tfidf`` forces the legacy path even if
   sentence-transformers is installed.
3. Auto-detection: try semantic, fall back to TF-IDF if the import or the
   model load fails.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..shared import SkillEntry
from .ontology import SKILL_ONTOLOGY, get_default_ontology

logger = logging.getLogger(__name__)


# Re-export the curated ontology so existing imports (``from .embeddings import
# DEFAULT_SKILL_ONTOLOGY``) keep working after the ontology was moved.
DEFAULT_SKILL_ONTOLOGY: tuple[SkillEntry, ...] = SKILL_ONTOLOGY


class SkillEncoder(Protocol):
    """Protocol implemented by every encoder backend."""

    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# TF-IDF encoder (deterministic, no network)
# ---------------------------------------------------------------------------


class TfidfEncoder:
    """Sparse TF-IDF encoder. Cheap and deterministic; used as fallback."""

    name = "tfidf"

    def __init__(self, corpus: Iterable[str] | None = None, ngram_max: int = 3) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, ngram_max), lowercase=True)
        seed_corpus = list(corpus) if corpus is not None else [entry.embedding_text() for entry in SKILL_ONTOLOGY]
        if not seed_corpus:
            seed_corpus = ["placeholder"]
        self._matrix = self._vectorizer.fit_transform(seed_corpus)
        self.dim = int(self._matrix.shape[1])

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = self._vectorizer.transform(texts)
        return np.asarray(matrix.todense(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Semantic encoder (sentence-transformers, dense)
# ---------------------------------------------------------------------------


class SentenceTransformerEncoder:
    """Dense semantic encoder backed by sentence-transformers."""

    name = "sentence_transformer"

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(f"sentence-transformers not available: {exc}") from exc
        resolved = model_name or os.environ.get(
            "SENTENCE_TRANSFORMER_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self._model = SentenceTransformer(resolved)
        self.model_name = resolved
        # Probe dim once.
        probe = self._model.encode(["__probe__"], normalize_embeddings=True)
        self.dim = int(np.asarray(probe).shape[-1])

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Skill embedding index
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("/", " ").split())


@dataclass
class SkillEmbeddingIndex:
    """Indexed ontology embeddings with cosine similarity search."""

    entries: tuple[SkillEntry, ...]
    matrix: np.ndarray
    encoder: SkillEncoder

    @classmethod
    def build(
        cls,
        entries: Iterable[SkillEntry] | None = None,
        encoder: SkillEncoder | None = None,
    ) -> "SkillEmbeddingIndex":
        ontology = tuple(entries or get_default_ontology())
        encoder = encoder or get_default_encoder()
        documents = [entry.embedding_text() for entry in ontology]
        matrix = encoder.encode(documents)
        return cls(entries=ontology, matrix=matrix, encoder=encoder)

    @property
    def backend(self) -> str:
        return self.encoder.name

    def embed(self, text: str) -> np.ndarray:
        return self.encoder.encode([text])

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        query_vector = self.embed(query)
        if query_vector.size == 0 or self.matrix.size == 0:
            return []
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        order = np.argsort(-scores)[:top_k]
        results: list[dict[str, object]] = []
        for idx in order:
            entry = self.entries[int(idx)]
            results.append(
                {
                    "canonical": entry.canonical,
                    "aliases": entry.aliases,
                    "category": entry.category,
                    "description": entry.description,
                    "score": float(max(0.0, min(1.0, scores[int(idx)]))),
                }
            )
        return results


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def _resolve_backend_choice() -> str:
    raw = os.environ.get("EMBEDDING_BACKEND", "auto").lower().strip()
    if raw in {"semantic", "sentence_transformer", "sentence-transformers"}:
        return "semantic"
    if raw in {"tfidf", "sparse"}:
        return "tfidf"
    return "auto"


@lru_cache(maxsize=1)
def get_default_encoder() -> SkillEncoder:
    """Return a process-wide default encoder according to env config."""

    choice = _resolve_backend_choice()
    if choice == "tfidf":
        return TfidfEncoder()
    if choice == "semantic":
        return SentenceTransformerEncoder()
    # auto: prefer semantic, fall back silently to tfidf.
    try:
        return SentenceTransformerEncoder()
    except Exception as exc:
        logger.info("Sentence-transformer unavailable, using TF-IDF fallback: %s", exc)
        return TfidfEncoder()


@lru_cache(maxsize=1)
def get_default_skill_index() -> SkillEmbeddingIndex:
    return SkillEmbeddingIndex.build(encoder=get_default_encoder())


def normalize_skill_phrase(phrase: str) -> str:
    return _normalize_text(phrase)
