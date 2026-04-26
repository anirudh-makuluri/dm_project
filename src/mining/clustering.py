from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: int
    size: int
    top_skills: tuple[str, ...]


def build_skill_matrix(skill_lists: list[list[str]]) -> tuple[pd.DataFrame, MultiLabelBinarizer]:
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(skill_lists)
    frame = pd.DataFrame(matrix, columns=mlb.classes_)
    return frame, mlb


def cluster_candidates(skill_lists: list[list[str]], n_clusters: int | None = None, random_state: int = 42) -> tuple[pd.DataFrame, list[ClusterSummary]]:
    if not skill_lists:
        return pd.DataFrame(), []
    skill_frame, _ = build_skill_matrix(skill_lists)
    if len(skill_frame) == 1:
        clustered = skill_frame.copy()
        clustered["cluster"] = 0
        summary = [ClusterSummary(cluster_id=0, size=1, top_skills=tuple(skill_frame.columns[:5]))]
        return clustered, summary

    cluster_count = n_clusters or max(2, min(4, int(np.sqrt(len(skill_frame)))))
    cluster_count = min(cluster_count, len(skill_frame))
    model = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10)
    clustered = skill_frame.copy()
    clustered["cluster"] = model.fit_predict(skill_frame.values)

    summaries: list[ClusterSummary] = []
    for cluster_id in sorted(clustered["cluster"].unique()):
        cluster_rows = clustered[clustered["cluster"] == cluster_id].drop(columns=["cluster"])
        mean_values = cluster_rows.mean(axis=0).sort_values(ascending=False)
        top_skills = tuple(mean_values.head(5).index.tolist())
        summaries.append(ClusterSummary(cluster_id=int(cluster_id), size=int(len(cluster_rows)), top_skills=top_skills))
    return clustered, summaries


def cluster_summary_to_dict(summary: ClusterSummary) -> dict[str, object]:
    return asdict(summary)
