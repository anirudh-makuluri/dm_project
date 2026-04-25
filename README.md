# Project Implementation

This folder contains the implementation scaffold for the Multi-Agent Resume Screening and Skill Mining System.

## Milestone 1 Included
- Data loading and schema normalization
- PII masking and text normalization
- Train/test split + JSONL export
- Baseline TF-IDF cosine-label benchmark

## Setup
```powershell
cd c:\Users\aniru\OneDrive\Desktop\DM\project-implementation
c:\Users\aniru\OneDrive\Desktop\DM\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 1) Preprocess Dataset
```powershell
c:\Users\aniru\OneDrive\Desktop\DM\.venv\Scripts\python.exe -m src.data.preprocess --input_csv .\data\raw\resumes.csv --output_dir .\data\processed --text_col text --label_col label
```

If your CSV uses different column names, pass them with `--text_col` and `--label_col`.

## 2) Run Baseline
```powershell
c:\Users\aniru\OneDrive\Desktop\DM\.venv\Scripts\python.exe -m src.models.baseline_tfidf --train_jsonl .\data\processed\train.jsonl --test_jsonl .\data\processed\test.jsonl --output_dir .\artifacts
```

## Output Artifacts
- `data/processed/train.jsonl`
- `data/processed/test.jsonl`
- `artifacts/baseline_metrics.json`
- `artifacts/baseline_predictions.csv`
