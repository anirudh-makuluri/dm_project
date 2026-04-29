"""Quick inspection of a multi_agent_records.jsonl produced by the runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="artifacts/main_multi_agent_lmstudio_smoke/multi_agent_records.jsonl")
    parser.add_argument("--n", type=int, default=1, help="How many records to print.")
    args = parser.parse_args()

    path = Path(args.records)
    with path.open() as fh:
        for idx, line in enumerate(fh):
            if idx >= args.n:
                break
            rec = json.loads(line)
            ex = rec["extracted"]
            extra = rec["extra"]
            ev = rec["evaluation"]

            print("=" * 78)
            print(f"RECORD #{idx}  resume_id={rec['resume_id']}  true_label={rec['label']}")
            print("=" * 78)
            print(f"extractor.skills_raw count : {len(ex.get('skills_raw', []))}")
            print(f"extractor.skills_raw[:10]  : {ex.get('skills_raw', [])[:10]}")
            print()

            sk = rec.get("normalized_skills", [])
            print(f"skill_miner normalized ({len(sk)} total, sorted by confidence):")
            for s in sorted(sk, key=lambda x: x.get("confidence", 0), reverse=True):
                phrase = s.get("phrase") or s.get("source_phrase", "?")
                print(f"  {s['canonical_skill']:<24} conf={s.get('confidence', 0):.3f}  <- {phrase!r}")
            print()

            e = rec["experience"]
            print(
                f"experience: years={e.get('total_years', 0)}  roles={e.get('role_count', 0)}  "
                f"leadership={e.get('leadership_indicators', 0)}  impact={e.get('impact_statements', 0)}  "
                f"quantified={e.get('quantified_impact', 0)}"
            )
            print()

            print(
                f"agents_only={extra.get('agents_only_label')}  "
                f"top={extra.get('agents_only_top', 0):.2f}  margin={extra.get('agents_only_margin', 0):.2f}"
            )
            print(
                f"ensemble={extra.get('ensemble_label')}  baseline={extra.get('baseline_label')}  "
                f"supervised={extra.get('supervised_label')}  source={extra.get('decision_source')}"
            )
            print()

            print(f"fit_score        : {ev.get('fit_score', 0):.2f}")
            print(f"matched_skills   : {list(ev.get('matched_skills', []))[:8]}")
            print(f"missing_skills   : {list(ev.get('missing_skills', []))[:8]}")
            print(f"component_scores : {ev.get('component_scores', {})}")
            print(f"rationale        : {(ev.get('rationale') or '')[:220]}")
            print(f"recommendation   : {(ev.get('recommendation') or '')[:200]}")
            print()


if __name__ == "__main__":
    main()
