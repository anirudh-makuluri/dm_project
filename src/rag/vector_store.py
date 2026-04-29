"""Skill vector store backed by an in-memory index, optionally persisted to ChromaDB.

The store wraps a :class:`SkillEmbeddingIndex` and uses dense embeddings from
the configured encoder (sentence-transformers by default, TF-IDF fallback).

ChromaDB is used purely for persistence + metadata querying; the underlying
embeddings come from our encoder so the dense semantic space is preserved.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .embeddings import (
    SkillEmbeddingIndex,
    get_default_skill_index,
    normalize_skill_phrase,
)
from ..shared import SkillEntry

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency fallback
    chromadb = None  # type: ignore


class SkillVectorStore:
    """Wraps the in-memory skill index and (optionally) mirrors it into Chroma."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = "skill_ontology",
    ) -> None:
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.collection_name = collection_name
        self.index = get_default_skill_index()
        self.client = None
        self.collection = None
        if chromadb is not None:
            try:
                if self.persist_directory is not None:
                    self.persist_directory.mkdir(parents=True, exist_ok=True)
                    self.client = chromadb.PersistentClient(path=str(self.persist_directory))
                else:
                    self.client = chromadb.Client()
                # Cosine distance matches the normalized sentence-transformer
                # embeddings; the default (L2) makes ``1 - distance`` meaningless
                # for our score, so we pin it explicitly.
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._populate()
            except Exception:
                # Chroma initialisation is best-effort; the in-memory index is enough.
                self.client = None
                self.collection = None

    @property
    def backend(self) -> str:
        return self.index.backend

    def _embeddings_as_lists(self) -> list[list[float]]:
        matrix = np.asarray(self.index.matrix)
        if matrix.size == 0:
            return [[0.0]] * len(self.index.entries)
        return matrix.astype(float).tolist()

    def _populate(self) -> None:
        if self.collection is None:
            return
        try:
            documents = [entry.embedding_text() for entry in self.index.entries]
            embeddings = self._embeddings_as_lists()
            ids = [normalize_skill_phrase(entry.canonical) for entry in self.index.entries]
            metadatas = [
                {
                    "canonical": entry.canonical,
                    "aliases": " | ".join(entry.aliases),
                    "category": entry.category,
                    "description": entry.description,
                }
                for entry in self.index.entries
            ]
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception:
            self.collection = None

    def query(self, text: str, top_k: int = 5) -> list[dict[str, object]]:
        # The in-memory index uses sklearn's cosine_similarity over normalized
        # sentence-transformer embeddings, so scores are well-calibrated cosine
        # similarities in [0, 1]. We prefer it for queries and treat Chroma as a
        # write-through persistence layer (its default HNSW distance metric and
        # collection-level config can desync from the encoder, producing
        # zero-valued scores for valid matches).
        return self.index.similarity_search(text, top_k=top_k)


def build_skill_vector_store(
    entries: Iterable[SkillEntry] | None = None,
    persist_directory: str | Path | None = None,
) -> SkillVectorStore:
    if entries is None:
        return SkillVectorStore(persist_directory=persist_directory)
    store = SkillVectorStore(persist_directory=persist_directory)
    store.index = SkillEmbeddingIndex.build(entries, encoder=store.index.encoder)
    store._populate()
    return store
