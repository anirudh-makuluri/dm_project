"""Tabulate per-resume true vs predicted classes for the smoke run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        default="artifacts/main_multi_agent_lmstudio_smoke/multi_agent_records.jsonl",
    )
    args = parser.parse_args()

    print(f"{'true':<22} {'agents_only':<22} {'ensemble':<22} {'top_fit':>7} {'margin':>7}  matched")
    print("-" * 110)
    correct = 0
    total = 0
    by_true: Counter[str] = Counter()
    correct_by_true: Counter[str] = Counter()
    confused_to: Counter[tuple[str, str]] = Counter()

    with Path(args.records).open() as fh:
        for line in fh:
            rec = json.loads(line)
            true = rec["label"]
            extra = rec["extra"]
            ev = rec["evaluation"]
            agents = extra.get("agents_only_label", "?")
            ensemble = extra.get("ensemble_label", "?")
            top = float(extra.get("agents_only_top", 0))
            margin = float(extra.get("agents_only_margin", 0))
            matched = ", ".join(list(ev.get("matched_skills", []))[:4])
            print(f"{true:<22} {agents:<22} {ensemble:<22} {top:7.2f} {margin:7.2f}  {matched}")
            total += 1
            by_true[true] += 1
            if agents == true:
                correct += 1
                correct_by_true[true] += 1
            else:
                confused_to[(true, agents)] += 1

    print()
    print(f"agents_only correct: {correct}/{total} = {correct/total:.2%}")
    print()
    print("per-true accuracy (agents_only):")
    for t in sorted(by_true):
        print(f"  {t:<22} {correct_by_true[t]}/{by_true[t]}")
    print()
    print("most common confusions (true -> predicted):")
    for (t, p), n in confused_to.most_common(8):
        print(f"  {t:<22} -> {p:<22}  ({n})")


if __name__ == "__main__":
    main()
