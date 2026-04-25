from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .load_data import load_resume_csv, save_jsonl, split_dataset
from .pii_masking import mask_pii

WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WS_RE.sub(" ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["resume_text"] = out["resume_text"].astype(str).map(mask_pii).map(normalize_text)
    out = out[out["resume_text"].str.len() > 0].reset_index(drop=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess resume CSV into JSONL train/test splits.")
    parser.add_argument("--input_csv", required=True, help="Path to source CSV.")
    parser.add_argument("--output_dir", required=True, help="Output directory for train/test JSONL.")
    parser.add_argument("--text_col", default=None, help="Optional text column name.")
    parser.add_argument("--label_col", default=None, help="Optional label column name.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_resume_csv(args.input_csv, text_col=args.text_col, label_col=args.label_col)
    clean_df = preprocess_dataframe(df)
    bundle = split_dataset(clean_df, test_size=args.test_size, random_state=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    save_jsonl(bundle.train, train_path)
    save_jsonl(bundle.test, test_path)

    print(f"Saved train split: {train_path} ({len(bundle.train)} rows)")
    print(f"Saved test split:  {test_path} ({len(bundle.test)} rows)")
    print(f"Unique labels: {clean_df['label'].nunique()}")


if __name__ == "__main__":
    main()
