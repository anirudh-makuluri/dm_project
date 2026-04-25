# Multi-Agent Resume Screening and Skill Mining System

## 1. Goal
Build a production-style prototype that:
- Ingests unstructured resumes and job descriptions.
- Extracts standardized skills and experience signals.
- Scores candidate-job fit with explainability.
- Measures and reduces demographic bias.
- Supports downstream mining (clustering + association rules).

## 2. System Scope (MVP -> Final)
### MVP
- Resume parser + cleaner.
- Skill extraction agent.
- Candidate evaluator agent.
- Baseline ATS model (TF-IDF + cosine).
- Core metrics: Precision, Recall, F1.

### Final
- Full multi-agent orchestration.
- RAG-based skill normalization with vector DB.
- Bias audit pipeline with perturbation tests.
- Explainability report generation.
- Cluster and rule mining from structured skill vectors.

## 3. Proposed Repository Layout
```text
project-implementation/
  IMPLEMENTATION.md
  src/
    data/
      load_data.py
      preprocess.py
      pii_masking.py
    agents/
      extractor_agent.py
      skill_miner_agent.py
      experience_agent.py
      evaluator_agent.py
    rag/
      embeddings.py
      vector_store.py
      retriever.py
    models/
      baseline_tfidf.py
      scoring.py
    fairness/
      perturbation.py
      fairness_metrics.py
    mining/
      clustering.py
      association_rules.py
    eval/
      evaluation_runner.py
      metrics.py
    app/
      pipeline.py
      report_generator.py
  data/
    raw/
    processed/
    annotations/
  notebooks/
    experiments.ipynb
  tests/
  requirements.txt
  README.md
```

## 4. Tech Stack
- Python 3.10+ (3.11 recommended)
- LangChain for agent orchestration
- LLM API: Gemini (configurable provider adapter)
- ChromaDB for vector storage
- scikit-learn for baseline + clustering
- pandas / numpy for data handling
- mlxtend for association rule mining
- pytest for testing

## 5. Data Pipeline
1. Load dataset from Kaggle resume data.
2. Split into train/test (80/20) with fixed random seed.
3. Apply PII masking:
   - names, emails, phone numbers, links, addresses.
4. Normalize text:
   - whitespace cleanup, unicode normalization, lowercasing policy.
5. Chunk long resumes for context-window-safe processing.
6. Store processed records as JSONL for reproducibility.

## 6. Agent Design
### Agent A: Data Extractor
- Input: cleaned resume text.
- Output: structured JSON with sections:
  - education, experience, skills_raw, projects, certifications.

### Agent B: Skill Miner
- Input: extracted JSON + skill ontology context from retriever.
- Output: normalized skills list with confidence scores.

### Agent C: Experience Agent
- Input: extracted experience text.
- Output: features:
  - total years, role count, leadership indicators, impact statements.

### Agent D: Evaluator
- Input: normalized skills + experience features + job description.
- Output:
  - fit score, rationale, strengths, gaps, recommendation.

## 7. RAG and Ontology Strategy
- Build skill ontology from:
  - public tech skill lists + manually curated aliases.
- Embed ontology entries and persist in ChromaDB.
- Retrieve top-k skills for each extracted candidate skill phrase.
- Map candidate phrase -> canonical skill by semantic similarity + threshold.

## 8. Baseline and Comparative Evaluation
### Baseline
- TF-IDF vectors of resume and job description.
- Cosine similarity fit score.

### Multi-agent
- Score from evaluator agent plus calibrated sub-scores.

### Metrics
- Skill extraction: Precision, Recall, F1 vs annotated subset.
- Ranking quality: nDCG@k, MAP@k (if relevance labels available).
- Runtime: avg latency per resume.
- Cost: token and API cost per 100 resumes.

## 9. Bias and Fairness Evaluation
1. Create demographic perturbation variants per resume:
   - name swaps, pronoun swaps, culturally varied identity cues.
2. Keep qualifications constant; alter only demographic signals.
3. Measure score deltas and subgroup disparities.
4. Report:
   - mean score difference,
   - selection-rate difference,
   - confidence interval via bootstrap.
5. Add mitigation checks:
   - PII suppression before evaluation,
   - evaluator prompt constraints,
   - post-hoc calibration.

## 10. Explainability Output
Generate per-candidate report:
- Extracted canonical skills.
- Top matched job requirements.
- Missing skills and recommended upskilling.
- Why score assigned (concise reasoning trace).
- Fairness audit flags (if triggered).

## 11. Implementation Milestones
### Milestone 1 (Week 1): Foundation
- Set up repo structure.
- Implement data ingestion + preprocessing.
- Build baseline TF-IDF model.

### Milestone 2 (Week 2): Core Agents
- Implement extractor + skill miner.
- Build ontology + ChromaDB retrieval.
- Save structured outputs.

### Milestone 3 (Week 3): Evaluation and Fairness
- Implement evaluator and metrics runner.
- Add perturbation-based fairness audit.
- Compare against baseline.

### Milestone 4 (Week 4): Mining + Reporting
- Add clustering and association rules.
- Build explainability report pipeline.
- Final ablation and documentation.

## 12. Definition of Done
- End-to-end pipeline runs on test split without manual intervention.
- Metrics table generated for baseline vs multi-agent model.
- Fairness audit report generated with perturbation analysis.
- Reproducible scripts and environment setup documented.
- Presentation-ready plots and summary tables exported.

## 13. Immediate Next Tasks (Actionable)
1. Create Python package skeleton under src.
2. Add requirements.txt with pinned versions.
3. Implement preprocess.py and pii_masking.py first.
4. Build baseline_tfidf.py and verify benchmark metrics.
5. Integrate first LLM call in extractor_agent.py with fixed prompt template.
