from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .embeddings import SkillEmbeddingIndex, get_default_skill_index, normalize_skill_phrase
from ..shared import SkillEntry

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency fallback
    chromadb = None


class SkillVectorStore:
    def __init__(self, persist_directory: str | Path | None = None, collection_name: str = "skill_ontology"):
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.collection_name = collection_name
        self.index = get_default_skill_index()
        self.client = None
        self.collection = None
        if chromadb is not None:
            if self.persist_directory is not None:
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            else:
                self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            self._populate()

    def _populate(self) -> None:
        if self.collection is None:
            return
        documents = [entry.embedding_text() for entry in self.index.entries]
        embeddings = self.index.matrix.toarray().tolist()
        ids = [normalize_skill_phrase(entry.canonical) for entry in self.index.entries]
        metadatas = [asdict(entry) for entry in self.index.entries]
        try:
            self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        except Exception:
            # Chroma is optional; a fallback in-memory index is still fully usable.
            self.collection = None

    def query(self, text: str, top_k: int = 5) -> list[dict[str, object]]:
        query_embedding = self.index.embed(text).toarray().tolist()[0]
        if self.collection is not None:
            try:
                result = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                rows: list[dict[str, object]] = []
                for metadata, distance in zip(result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
                    rows.append(
                        {
                            "canonical": metadata.get("canonical", ""),
                            "aliases": tuple(metadata.get("aliases", ())),
                            "category": metadata.get("category", "general"),
                            "description": metadata.get("description", ""),
                            "score": float(max(0.0, 1.0 - float(distance))),
                        }
                    )
                if rows:
                    return rows
            except Exception:
                pass
        return self.index.similarity_search(text, top_k=top_k)


def build_skill_vector_store(entries: Iterable[SkillEntry] | None = None, persist_directory: str | Path | None = None) -> SkillVectorStore:
    if entries is None and persist_directory is None:
        return SkillVectorStore()
    if entries is None:
        return SkillVectorStore(persist_directory=persist_directory)
    store = SkillVectorStore(persist_directory=persist_directory)
    store.index = SkillEmbeddingIndex.build(entries)
    store._populate()
    return store
