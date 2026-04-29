# Multi-Agent Resume Screening and Skill Mining System

## 1. Goal
Production-style prototype that:

- Ingests unstructured resumes and a job-profile catalogue.
- Extracts standardized skills and experience signals via four cooperating
  agents (Extractor / Skill Miner / Experience / Evaluator).
- Normalises raw skill phrases against a curated ontology using **dense
  semantic embeddings** (sentence-transformers) backed by ChromaDB.
- Scores candidate-job fit with explainable component scores plus an
  LLM-authored rationale.
- Audits demographic bias via name / pronoun perturbations.
- Supports downstream mining (clustering + association rules).

## 2. Architecture overview
```text
            +--------------+   +----------------+   +-----------------+
resume txt  |  Extractor   |   |  Skill miner   |   |  Experience     |
----------->|   agent      +-->+  agent + RAG   +-->+  agent          |
            +------+-------+   +--------+-------+   +--------+--------+
                   |                    |                    |
                   v                    v                    v
                                  +-----------+
                                  | Evaluator |  deterministic fit_score
                                  |  agent    |  +  LLM rationale
                                  +-----+-----+
                                        |
                                        v
                              +---------+----------+
                              | Multi-agent runner |
                              +---+------------+---+
                                  |            |
                  agents-only path|            | ensemble path
                                  v            v
                       agents_only.csv   ensemble.csv
                              + ranking_metrics + skill_extraction_metrics
                              + fairness_audit
```

Each agent calls the LLM provider for its primary path and falls back to
the deterministic rule-based logic when the provider is unavailable, the
response fails to parse, or the response fails Pydantic validation. This
makes the pipeline runnable offline while still benefiting from LLM quality
when keys / a daemon are available.

## 3. Repository layout
```text
src/
  data/
    load_data.py                # CSV -> normalized DataFrame
    preprocess.py               # PII masking + JSONL splits
    pii_masking.py
    skill_labeler.py            # high-precision rule labeler over the ontology
    build_skill_annotations.py  # builds data/annotations/skills_gold.jsonl
  llm/
    provider.py                 # Gemini / Ollama / RuleBased backends + sqlite cache
    prompts.py                  # Extractor / SkillMiner / Experience / Evaluator templates
    schemas.py                  # Pydantic schemas for structured outputs
  rag/
    ontology.py                 # ~140-entry curated skill ontology
    embeddings.py               # SentenceTransformer + TfidfEncoder backends
    vector_store.py             # in-memory + ChromaDB persistence
    retriever.py                # confidence-thresholded retrieval
  agents/
    extractor_agent.py          # LLM-first, regex fallback
    skill_miner_agent.py        # LLM phrases -> RAG normalisation
    experience_agent.py         # LLM features merged with regex bounds
    evaluator_agent.py          # deterministic score + LLM rationale
  models/
    baseline_tfidf.py           # ATS-style baseline
    scoring.py                  # job profiles + deterministic fit scorer
    supervised_classifier.py    # LinearSVC over TF-IDF (ensemble member)
  fairness/
    perturbation.py             # name / pronoun / identity-cue swaps
    fairness_metrics.py         # mean delta, selection rate, bootstrap CI
  mining/
    clustering.py               # KMeans over canonical skill vectors
    association_rules.py        # mlxtend Apriori
  eval/
    metrics.py                  # classification metrics
    skill_extraction_eval.py    # E2: P/R/F1 vs gold annotations
    ranking_metrics.py          # E3: nDCG@k, MAP@k, P@k, R@k
    evaluation_runner.py        # full pipeline: agents-only + ensemble
  app/
    pipeline.py                 # milestone-4 driver (eval + mining + report)
    report_generator.py         # explainability bundle
data/
  Main DataSet Resume.csv       # Kaggle resume corpus (~2,500 resumes)
  processed_main/               # full-corpus train/test JSONL
  annotations/                  # skills_gold.jsonl (built on demand)
artifacts/
  main_baseline/                # TF-IDF baseline metrics + predictions
  main_multi_agent/             # ensemble + agents-only metrics, ranking,
                                # skills, records, rationale_quality.json,
                                # rationale_quality_audits.jsonl
scripts/
  view_resume.py                # per-record explainability viewer
  inspect_smoke.py              # tabular inspection of records.jsonl
  diagnose_smoke.py             # per-category accuracy breakdown
tests/
  test_rationale_quality.py     # rationale-audit unit tests (21 cases)
```

## 4. Tech stack
- Python 3.10+ (3.11 / 3.12 / 3.13 tested).
- LangChain (`langchain`, `langchain-google-genai`) for the Gemini path.
- google-generativeai client (transitive).
- Ollama HTTP API for local LLM backend (no extra Python dep needed).
- sentence-transformers + transformers for dense embeddings.
- ChromaDB for vector persistence.
- scikit-learn for baseline + LinearSVC + clustering + cosine similarity.
- mlxtend for Apriori association rules.
- pandas / numpy for data handling.
- pydantic for structured-output validation.
- pytest for tests.

## 5. LLM provider adapter (`src/llm/provider.py`)
Backends, in auto-selection order:

1. **Gemini** via `langchain-google-genai`. Active when `GEMINI_API_KEY`
   (or `GOOGLE_API_KEY`) is set. Configurable model via `GEMINI_MODEL`
   (default `gemini-1.5-flash`).
2. **OpenAI-compatible** chat-completions server (LM Studio, vLLM,
   llama.cpp server, oobabooga, ...). Active when `OPENAI_BASE_URL`
   responds at `GET /models`. Configurable via `OPENAI_BASE_URL`
   (default `http://localhost:1234/v1` — LM Studio's default),
   `OPENAI_MODEL` (empty -> auto-discover from `/models`), and
   `OPENAI_API_KEY` (any string; LM Studio ignores the value but the
   header must be present). The aliases `lmstudio`, `lm-studio`,
   `openai`, and `vllm` for `LLM_BACKEND` all resolve to this backend.
3. **Ollama** local server. Active when an Ollama daemon is reachable at
   `OLLAMA_HOST` (default `http://127.0.0.1:11434`). Configurable model
   via `OLLAMA_MODEL` (default `llama3.2:3b`).
4. **RuleBasedProvider**. Always available; raises `LLMUnavailable` so
   each agent uses its deterministic regex / TF-IDF fallback.

Common knobs (env vars): `LLM_BACKEND`, `LLM_TEMPERATURE`,
`LLM_MAX_OUTPUT_TOKENS`, `LLM_TIMEOUT_SECONDS`, `LLM_ENABLE_CACHE`,
`LLM_CACHE_PATH`.

Quick LM Studio runbook:
```bash
# LM Studio: load a model (e.g. llama-3.2-3b-instruct) and click "Start Server".
export LLM_BACKEND=lmstudio
export OPENAI_BASE_URL=http://localhost:1234/v1
# OPENAI_MODEL is auto-discovered; set it explicitly only if multiple models are loaded.
python -m src.eval.evaluation_runner ...
```

Every completion is keyed by `sha256(backend, model, prompt, schema)` and
cached in `data/llm_cache.sqlite`. Reruns are cheap and reproducible; delete
the file to force a fresh run.

## 6. Embedding backends (`src/rag/embeddings.py`)
Two encoder backends share a common interface:

- `SentenceTransformerEncoder` — default. Uses
  `sentence-transformers/all-MiniLM-L6-v2` (configurable via
  `SENTENCE_TRANSFORMER_MODEL`). The first run downloads ~80MB into the
  HuggingFace cache and reruns are offline.
- `TfidfEncoder` — fallback. Used automatically when
  `sentence-transformers` cannot be imported, or forced via
  `EMBEDDING_BACKEND=tfidf`.

Both embed the curated ontology (`src/rag/ontology.py`, ~140 entries).
The `SkillVectorStore` mirrors the dense matrix into ChromaDB for
persistence; queries reuse the same encoder so similarity remains in the
intended semantic space.

## 7. Agents and their schemas
Every agent is structured as **(prompt -> JSON -> Pydantic -> dataclass)**
with a deterministic fallback if any step fails:

| Agent       | Prompt                  | Schema                    | Fallback                       |
|-------------|-------------------------|---------------------------|--------------------------------|
| Extractor   | `EXTRACTOR_PROMPT`      | `ResumeExtraction`        | regex section splitter         |
| Skill Miner | `SKILL_MINER_PROMPT`    | `SkillMiningResult`       | rule-based phrase tokenizer    |
| Experience  | `EXPERIENCE_PROMPT`     | `ExperienceExtraction`    | regex year / verb counters     |
| Evaluator   | `EVALUATOR_PROMPT`      | `EvaluatorRationale`      | deterministic recommendation   |

The Evaluator's **fit score is always deterministic** (computed by
`build_evaluation_result` in `src/models/scoring.py`); the LLM only
authors the rationale, strengths, gaps and recommendation around it. This
keeps comparative numbers reproducible while still giving an
LLM-quality narrative.

## 8. Evaluation pipeline (`src/eval/evaluation_runner.py`)
Each test resume produces deterministic fit scores against every job
profile. We then materialise **both** prediction modes:

- **agents-only**: `argmax_j fit_score(resume, job_j)`. This is the
  honest multi-agent classification.
- **ensemble**: agents-only blended with the TF-IDF baseline and a
  supervised LinearSVC. Toggle off with `--disable_supervised`. The
  decision-source for each row is recorded in the predictions CSV and
  summarised in `decision_source_breakdown`.

Both modes are saved side-by-side. The summary JSON exposes:

- `metrics`                       — ensemble classification metrics
- `agents_only_metrics`           — agents-only classification metrics
- `baseline_metrics`              — TF-IDF baseline metrics
- `comparison`                    — ensemble - baseline deltas
- `agents_vs_baseline`            — agents-only - baseline deltas
- `ensemble_vs_agents`            — ensemble - agents-only deltas
- `decision_source_breakdown`     — counts of `agents-only`,
  `multi-agent`, `ensemble-agree`, `supervised-agent`, `baseline-prior`,
  `supervised+baseline-agree`
- `fairness`                      — perturbation audit
- `ranking`                       — nDCG@k / MAP@k / P@k / R@k
- `skill_extraction`              — P/R/F1 vs the gold annotations
- `rationale_quality`             — grounding, specificity, calibration,
  identity-clean rate over the LLM-authored rationales
- `providers`                     — backend names actually used

## 9. Skill-extraction evaluation (E2)
1. **Build the gold file** with the high-precision rule labeler:
   ```bash
   python -m src.data.build_skill_annotations \
       --source_jsonl data/processed_main/test.jsonl \
       --output data/annotations/skills_gold.jsonl \
       --per_class 4
   ```
   Produces ~100 stratified records with auto-tagged spans. Reviewers can
   edit the JSONL freely (delete spurious entries, add missing ones,
   flip `review_status` to `"reviewed"`).
2. **Score predictions** in two ways:
   - Inline as part of `evaluation_runner` (when the gold file exists).
   - Standalone, against any predictions JSONL:
     ```bash
     python -m src.eval.skill_extraction_eval \
         --gold data/annotations/skills_gold.jsonl \
         --predictions artifacts/main_multi_agent/multi_agent_records.jsonl \
         --output artifacts/main_multi_agent/skill_extraction_metrics.json
     ```
   The evaluator reports macro & micro canonical-level P/R/F1, per-resume
   detail, per-skill confusion (TP / FP / FN), and per-category breakdown.

## 10. Ranking evaluation (E3)
For each occupational category C (treated as a "job"), we rank all test
resumes by `fit_score(resume, job=C)` and use binary relevance
`1[true_label == C]`. Implementation in `src/eval/ranking_metrics.py`
reports macro-averaged `Precision@k`, `Recall@k`, `MAP@k` and `nDCG@k`
for `k ∈ {1, 3, 5, 10}` plus per-job breakdowns. Standalone CLI:
```bash
python -m src.eval.ranking_metrics \
    --predictions_csv artifacts/main_multi_agent/multi_agent_predictions.csv \
    --output artifacts/main_multi_agent/ranking_metrics.json
```

## 11. Bias audit
`src/fairness/perturbation.py` produces name / pronoun / identity-cue
variants. The runner currently feeds the "combined" perturbation through
the full agent pipeline and reports mean score delta, selection-rate
delta and bootstrap CIs in `fairness`. Two known limitations are
documented in the next-gaps doc: deterministic scoring is partially
identity-blind by construction (a feature, not a bug, for fairness) but
this means the audit primarily probes the LLM rationale and any
identity-leaking text similarity in the deterministic path.

## 12. Rationale-quality evaluation (`src/eval/rationale_quality.py`)
The deterministic scorer alone cannot tell whether the LLM-authored
``rationale`` text is actually useful, so each run also audits every
rationale on four orthogonal axes. The metrics are deliberately
**rule-based** so the numbers are reproducible without a second LLM call:

| Metric | Definition |
|---|---|
| `grounded_in_matched_pct` | rationale text mentions ≥ 1 canonical skill that is in the candidate's `matched_skills` |
| `grounded_in_missing_pct` | rationale text mentions ≥ 1 canonical skill that is in `missing_skills` |
| `specific_pct` | rationale word-count ≥ 15 (not a one-liner) |
| `calibrated_pct` | `recommendation` tone (advance / consider / reject) matches the `fit_score` band (≥70 / 40-69 / <40) |
| `identity_clean_pct` | rationale + recommendation contain no demographic pronouns or attributes |
| `strengths_grounded_pct` | average fraction of `strengths` array entries that map back to a canonical skill in `matched_skills` (catches invented skills) |
| `gaps_grounded_pct` | same idea for `gaps` vs `missing_skills` |

Standalone CLI for any existing run's records:
```bash
python -c "
from src.eval.rationale_quality import evaluate_rationale_quality_jsonl, report_to_dict
print(report_to_dict(evaluate_rationale_quality_jsonl(
    'artifacts/main_multi_agent/multi_agent_records.jsonl'
)))
"
```

The evaluation runner writes two artifacts:

- `rationale_quality.json` — aggregate metrics + tone / fit-band distributions.
- `rationale_quality_audits.jsonl` — per-record audit so the report can
  quote individual examples (matched-skill hits, identity flags, invented
  strengths/gaps).

A complementary explainability viewer renders any single resume's full
agent record (extraction → normalized skills → experience → evaluation →
audit) as text, markdown, or JSON:
```bash
python scripts/view_resume.py --id 112 --format markdown \
    --records artifacts/main_multi_agent/multi_agent_records.jsonl
```

## 13. Reproducible runbook
```bash
# 0. Create an isolated environment.
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 1. (Optional) Configure providers.
cp .env.example .env  # then edit; export GEMINI_API_KEY=... if you have a key

# 2. Preprocess the Kaggle CSV.
python -m src.data.preprocess \
    --input_csv "data/Main DataSet Resume.csv" \
    --output_dir data/processed_main \
    --text_col Resume_str --label_col Category

# 3. Run the TF-IDF baseline.
python -m src.models.baseline_tfidf \
    --train_jsonl data/processed_main/train.jsonl \
    --test_jsonl  data/processed_main/test.jsonl \
    --output_dir  artifacts/main_baseline

# 4. Build skill-extraction gold annotations.
python -m src.data.build_skill_annotations \
    --source_jsonl data/processed_main/test.jsonl \
    --output data/annotations/skills_gold.jsonl

# 5. Run the multi-agent evaluation (ensemble + agents-only).
python -m src.eval.evaluation_runner \
    --train_jsonl data/processed_main/train.jsonl \
    --test_jsonl  data/processed_main/test.jsonl \
    --output_dir  artifacts/main_multi_agent \
    --baseline_predictions artifacts/main_baseline/baseline_predictions.csv \
    --skill_gold data/annotations/skills_gold.jsonl

# 6. (Optional) Force the agents-only ablation as the only mode.
python -m src.eval.evaluation_runner --disable_supervised \
    --test_jsonl  data/processed_main/test.jsonl \
    --output_dir  artifacts/main_multi_agent_agents_only \
    --baseline_predictions artifacts/main_baseline/baseline_predictions.csv \
    --skill_gold data/annotations/skills_gold.jsonl

# 7. Run the milestone-4 pipeline (mining + explainability report).
python -m src.app.pipeline \
    --train_jsonl data/processed_main/train.jsonl \
    --test_jsonl  data/processed_main/test.jsonl \
    --output_dir  artifacts/main_pipeline \
    --baseline_predictions artifacts/main_baseline/baseline_predictions.csv \
    --skill_gold data/annotations/skills_gold.jsonl
```

## 14. Honest reporting expectations
- Headline classification numbers must report **agents_only**,
  **ensemble**, and **baseline** side by side; the SVM contribution is
  visible as the gap between agents_only and ensemble (also as
  `decision_source_breakdown`).
- LLM provider in use is recorded in `summary.providers.llm_backend`. A
  `rule_based` value indicates the deterministic fallback was used and
  must be disclosed when the run is described.
- The skill-extraction gold file is initially auto-labeled; reviewers
  must mark records as `"reviewed"` before treating per-resume P/R/F1 as
  human-validated ground truth.
