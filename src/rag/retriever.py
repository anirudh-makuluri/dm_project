from __future__ import annotations

from dataclasses import dataclass

from .vector_store import SkillVectorStore
from ..shared import SkillMatch


@dataclass
class SkillRetriever:
    store: SkillVectorStore
    min_confidence: float = 0.15

    def retrieve(self, phrase: str, top_k: int = 5) -> list[SkillMatch]:
        results = self.store.query(phrase, top_k=top_k)
        matches: list[SkillMatch] = []
        for row in results:
            score = float(row.get("score", 0.0))
            if score < self.min_confidence:
                continue
            matches.append(
                SkillMatch(
                    phrase=phrase,
                    canonical_skill=str(row.get("canonical", "")).strip(),
                    confidence=score,
                    source=str(row.get("category", "ontology")),
                )
            )
        return matches
