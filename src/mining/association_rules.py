from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except Exception:  # pragma: no cover - optional dependency fallback
    apriori = None
    association_rules = None
    TransactionEncoder = None


@dataclass(frozen=True)
class AssociationRuleSummary:
    antecedents: tuple[str, ...]
    consequents: tuple[str, ...]
    support: float
    confidence: float
    lift: float


def mine_association_rules(
    skill_lists: list[list[str]],
    min_support: float = 0.2,
    min_confidence: float = 0.6,
) -> pd.DataFrame:
    if not skill_lists or apriori is None or association_rules is None or TransactionEncoder is None:
        return pd.DataFrame()

    encoder = TransactionEncoder()
    encoded = encoder.fit(skill_lists).transform(skill_lists)
    frame = pd.DataFrame(encoded, columns=encoder.columns_)
    frequent = apriori(frame, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return frequent
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return rules
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return rules


def top_rules(rules: pd.DataFrame, limit: int = 10) -> list[AssociationRuleSummary]:
    if rules.empty:
        return []
    summaries: list[AssociationRuleSummary] = []
    for _, row in rules.head(limit).iterrows():
        summaries.append(
            AssociationRuleSummary(
                antecedents=tuple(sorted(map(str, row["antecedents"]))),
                consequents=tuple(sorted(map(str, row["consequents"]))),
                support=float(row["support"]),
                confidence=float(row["confidence"]),
                lift=float(row["lift"]),
            )
        )
    return summaries


def rule_summary_to_dict(summary: AssociationRuleSummary) -> dict[str, object]:
    return asdict(summary)
