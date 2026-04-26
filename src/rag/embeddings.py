from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..shared import SkillEntry


DEFAULT_SKILL_ONTOLOGY: tuple[SkillEntry, ...] = (
    SkillEntry("python", ("python programming",), "programming", "General-purpose programming language."),
    SkillEntry("sql", ("structured query language",), "data", "Database querying and relational data work."),
    SkillEntry("pandas", ("dataframes",), "data", "Tabular data manipulation library."),
    SkillEntry("numpy", ("numerical python",), "data", "Scientific computing library."),
    SkillEntry("scikit-learn", ("sklearn", "machine learning"), "ml", "Classical machine learning toolkit."),
    SkillEntry("tensorflow", ("tf",), "ml", "Deep learning framework."),
    SkillEntry("pytorch", ("torch",), "ml", "Deep learning framework."),
    SkillEntry("nlp", ("natural language processing",), "ml", "Language modeling and text analytics."),
    SkillEntry("transformers", ("transformer models",), "ml", "Transformer architectures and LLM tooling."),
    SkillEntry("feature engineering", ("features",), "ml", "Creating predictive features."),
    SkillEntry("statistics", ("statistical analysis",), "analysis", "Inferential and descriptive statistics."),
    SkillEntry("regression", ("linear regression", "logistic regression"), "analysis", "Regression methods."),
    SkillEntry("clustering", ("unsupervised learning",), "analysis", "Unsupervised segmentation methods."),
    SkillEntry("model evaluation", ("metrics", "validation"), "ml", "Model performance analysis."),
    SkillEntry("experiment design", ("a/b testing", "ab testing", "experimentation"), "analysis", "Controlled experiments and testing."),
    SkillEntry("dashboarding", ("dashboards", "visualization"), "analytics", "Reporting and decision support."),
    SkillEntry("tableau", (), "analytics", "Analytics dashboarding platform."),
    SkillEntry("power bi", ("powerbi",), "analytics", "Business intelligence and reporting."),
    SkillEntry("excel", ("spreadsheets",), "general", "Spreadsheet analysis."),
    SkillEntry("docker", (), "engineering", "Containerization platform."),
    SkillEntry("kubernetes", ("k8s",), "engineering", "Container orchestration platform."),
    SkillEntry("cloud", ("aws", "azure", "gcp", "cloud tooling"), "engineering", "Cloud infrastructure and deployment."),
    SkillEntry("apis", ("rest api", "rest apis", "web services"), "engineering", "Interface design and integration."),
    SkillEntry("microservices", (), "engineering", "Distributed service architecture."),
    SkillEntry("git", ("version control",), "engineering", "Source control systems."),
    SkillEntry("ci/cd", ("continuous integration", "continuous deployment"), "engineering", "Delivery automation."),
    SkillEntry("testing", ("unit testing", "automated tests", "qa"), "engineering", "Quality assurance and validation."),
    SkillEntry("java", (), "engineering", "Java programming language."),
    SkillEntry("javascript", ("js",), "engineering", "JavaScript programming language."),
    SkillEntry("react", (), "engineering", "Frontend UI library."),
    SkillEntry("node.js", ("node",), "engineering", "JavaScript runtime for server-side apps."),
    SkillEntry("flask", (), "engineering", "Lightweight Python web framework."),
    SkillEntry("spring boot", (), "engineering", "Java application framework."),
    SkillEntry("postgresql", ("postgres",), "data", "Relational database platform."),
    SkillEntry("mysql", (), "data", "Relational database platform."),
    SkillEntry("data pipelines", ("etl", "pipeline"), "data", "Operational data movement and transformation."),
    SkillEntry("data modeling", (), "data", "Modeling structured data systems."),
    SkillEntry("machine learning", ("ml",), "ml", "Predictive modeling and learning systems."),
    SkillEntry("leadership", ("team lead", "leading", "managed"), "soft", "People and project leadership."),
    SkillEntry("communication", (), "soft", "Written and verbal communication."),
    SkillEntry("stakeholder management", (), "soft", "Cross-functional alignment and coordination."),
    SkillEntry("recruiting", ("talent acquisition", "hiring"), "hr", "Talent acquisition and hiring operations."),
    SkillEntry("onboarding", (), "hr", "New hire onboarding and enablement."),
    SkillEntry("employee relations", (), "hr", "Employee support and relations."),
    SkillEntry("benefits administration", (), "hr", "Benefits operations."),
    SkillEntry("payroll", (), "hr", "Payroll coordination and processing."),
    SkillEntry("compliance", (), "hr", "Policy and regulatory compliance."),
    SkillEntry("hr analytics", (), "hr", "Analytic support for HR operations."),
    SkillEntry("applicant tracking systems", ("ats",), "hr", "Applicant tracking platform workflows."),
    SkillEntry("model deployment", (), "ml", "Deploying models to production."),
    SkillEntry("observability", (), "engineering", "Monitoring and tracing systems."),
    SkillEntry("distributed systems", (), "engineering", "Large-scale distributed architecture."),
    SkillEntry("database optimization", (), "engineering", "Database performance tuning."),
    SkillEntry("project management", (), "soft", "Planning and execution across stakeholders."),
)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("/", " ").split())


@dataclass
class SkillEmbeddingIndex:
    entries: tuple[SkillEntry, ...]
    vectorizer: TfidfVectorizer
    matrix: np.ndarray

    @classmethod
    def build(cls, entries: Iterable[SkillEntry] | None = None) -> "SkillEmbeddingIndex":
        ontology = tuple(entries or DEFAULT_SKILL_ONTOLOGY)
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)
        documents = [entry.embedding_text() for entry in ontology]
        matrix = vectorizer.fit_transform(documents)
        return cls(entries=ontology, vectorizer=vectorizer, matrix=matrix)

    def embed(self, text: str):
        return self.vectorizer.transform([text])

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vector = self.embed(query)
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
                    "score": float(scores[int(idx)]),
                }
            )
        return results


@lru_cache(maxsize=1)
def get_default_skill_index() -> SkillEmbeddingIndex:
    return SkillEmbeddingIndex.build()


def normalize_skill_phrase(phrase: str) -> str:
    return _normalize_text(phrase)
