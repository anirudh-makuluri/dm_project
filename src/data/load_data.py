from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame


def _guess_text_column(columns: list[str]) -> str:
    candidates = [
        "resume_text",
        "text",
        "resume",
        "content",
        "description",
    ]
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError(f"Unable to infer text column from: {columns}")


def _guess_label_column(columns: list[str]) -> Optional[str]:
    candidates = [
        "label",
        "category",
        "class",
        "target",
        "occupation",
        "job_role",
        "genre",
    ]
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def load_resume_csv(
    input_csv: str | Path,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    columns = list(df.columns)

    resolved_text_col = text_col or _guess_text_column(columns)
    if resolved_text_col not in df.columns:
        raise ValueError(f"text_col '{resolved_text_col}' not in CSV columns: {columns}")

    resolved_label_col = label_col or _guess_label_column(columns)
    if resolved_label_col is not None and resolved_label_col not in df.columns:
        raise ValueError(f"label_col '{resolved_label_col}' not in CSV columns: {columns}")

    out = pd.DataFrame()
    out["resume_id"] = range(len(df))
    out["resume_text"] = df[resolved_text_col].fillna("").astype(str)
    out["label"] = df[resolved_label_col].astype(str) if resolved_label_col else "unknown"

    # Remove rows with empty text after casting/cleanup.
    out = out[out["resume_text"].str.strip() != ""].reset_index(drop=True)
    return out


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    stratify = df["label"] if df["label"].nunique() > 1 else None
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        # Fallback for datasets with ultra-rare classes where stratified split is impossible.
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
    return DatasetBundle(
        train=train_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def save_jsonl(df: pd.DataFrame, output_path: str | Path) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
