from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity


def load_jsonl(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")
    return pd.read_json(p, lines=True)


def build_class_centroids(train_vectors, labels: pd.Series):
    centroids = {}
    for label in sorted(labels.unique()):
        idx = labels[labels == label].index
        centroids[label] = train_vectors[idx].mean(axis=0)
    return centroids


def predict_with_centroids(test_vectors, centroids: dict[str, object]) -> list[str]:
    labels = list(centroids.keys())
    centroid_matrix = [centroids[label] for label in labels]
    centroid_matrix = pd.DataFrame([c.A1 if hasattr(c, "A1") else c for c in centroid_matrix])

    sims = cosine_similarity(test_vectors, centroid_matrix.values)
    best_idx = sims.argmax(axis=1)
    return [labels[i] for i in best_idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF cosine baseline on preprocessed JSONL splits.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_features", type=int, default=30000)
    parser.add_argument("--ngram_max", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_jsonl(args.train_jsonl)
    test_df = load_jsonl(args.test_jsonl)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=2,
    )

    x_train = vectorizer.fit_transform(train_df["resume_text"].astype(str))
    x_test = vectorizer.transform(test_df["resume_text"].astype(str))

    centroids = build_class_centroids(x_train, train_df["label"].astype(str))
    y_pred = predict_with_centroids(x_test, centroids)
    y_true = test_df["label"].astype(str).tolist()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "num_classes": int(train_df["label"].nunique()),
    }

    pred_df = test_df[["resume_id", "label"]].copy()
    pred_df.rename(columns={"label": "y_true"}, inplace=True)
    pred_df["y_pred"] = y_pred

    metrics_path = out_dir / "baseline_metrics.json"
    preds_path = out_dir / "baseline_predictions.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pred_df.to_csv(preds_path, index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {preds_path}")


if __name__ == "__main__":
    main()
