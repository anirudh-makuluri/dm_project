"""Build the skill-extraction gold annotations file.

Strategy
========
1. Load the processed test split (JSONL).
2. Stratify a small sample (default ~100 rows) across occupational categories
   so every class is represented.
3. Apply the high-precision :class:`SkillLabeler` to each sampled resume to
   produce an initial set of ``(canonical, surface, start, end)`` spans.
4. Write the result to ``data/annotations/skills_gold.jsonl`` as one record
   per resume:

   .. code-block:: json

       {"resume_id": 17, "category": "ENGINEERING", "skills": ["python", "sql"],
        "spans": [{"canonical": "python", "surface": "Python", "start": 42, "end": 48}],
        "review_status": "auto"}

The file is intended to be human-reviewable: the ``review_status`` field is
``"auto"`` for first-pass labels and should be flipped to ``"reviewed"`` after
a manual pass. Reviewers can edit ``skills`` / ``spans`` freely; the evaluator
treats whatever is in the file as ground truth.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .skill_labeler import SkillLabeler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the skill-extraction gold annotations file.")
    parser.add_argument(
        "--source_jsonl",
        default="data/processed_main/test.jsonl",
        help="Source resume JSONL (must contain resume_id, resume_text, label).",
    )
    parser.add_argument(
        "--output",
        default="data/annotations/skills_gold.jsonl",
        help="Output path for the gold annotations JSONL.",
    )
    parser.add_argument("--per_class", type=int, default=4, help="Resumes sampled per occupational class.")
    parser.add_argument("--max_total", type=int, default=120, help="Hard cap on the number of sampled resumes.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling RNG seed.")
    return parser.parse_args()


def stratified_sample(df: pd.DataFrame, per_class: int, max_total: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, row in df.iterrows():
        by_class[str(row.get("label", "unknown"))].append(int(index))
    chosen: list[int] = []
    for class_label, indices in by_class.items():
        rng.shuffle(indices)
        chosen.extend(indices[:per_class])
    rng.shuffle(chosen)
    chosen = chosen[:max_total]
    return df.iloc[sorted(chosen)].reset_index(drop=True)


def build(args: argparse.Namespace) -> None:
    src = Path(args.source_jsonl)
    if not src.exists():
        raise FileNotFoundError(f"Source JSONL not found: {src}")
    df = pd.read_json(src, lines=True)
    required = {"resume_id", "resume_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source JSONL is missing required columns: {missing}")
    if "label" not in df.columns:
        df["label"] = "unknown"

    sampled = stratified_sample(df, per_class=args.per_class, max_total=args.max_total, seed=args.seed)
    labeler = SkillLabeler()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for _, row in sampled.iterrows():
            text = str(row.get("resume_text", ""))
            spans = labeler.label(text)
            record = {
                "resume_id": int(row["resume_id"]),
                "category": str(row.get("label", "unknown")),
                "skills": sorted({span.canonical for span in spans}),
                "spans": [asdict(span) for span in spans],
                "review_status": "auto",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} gold annotations to: {out_path}")
    print(
        "Note: spans were produced by the high-precision SkillLabeler; review and "
        "edit before treating the file as final ground truth."
    )


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
