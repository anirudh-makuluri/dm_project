from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


@dataclass
class SupervisedResumeClassifier:
    """A strong text classifier used as an expert inside the multi-agent stack."""

    word_features: int = 5000
    char_features: int = 5000
    c_value: float = 1.0
    random_state: int = 42

    def __post_init__(self) -> None:
        self.vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 1),
                        max_features=self.word_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 3),
                        max_features=self.char_features,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        self.classifier = LinearSVC(C=self.c_value, random_state=self.random_state, max_iter=1000)
        self._is_fitted = False

    def fit(self, train_df: pd.DataFrame, text_col: str = "resume_text", label_col: str = "label") -> None:
        text_series = train_df[text_col].astype(str)
        label_series = train_df[label_col].astype(str)
        features = self.vectorizer.fit_transform(text_series)
        self.classifier.fit(features, label_series)
        self._is_fitted = True

    def predict(self, text_series: pd.Series) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("SupervisedResumeClassifier must be fitted before prediction.")
        features = self.vectorizer.transform(text_series.astype(str))
        return self.classifier.predict(features)

    def predict_with_margin(self, text_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("SupervisedResumeClassifier must be fitted before prediction.")
        features = self.vectorizer.transform(text_series.astype(str))
        predictions = self.classifier.predict(features)
        decision = self.classifier.decision_function(features)
        if decision.ndim == 1:
            margins = np.abs(decision)
        else:
            sorted_scores = np.sort(decision, axis=1)
            margins = sorted_scores[:, -1] - sorted_scores[:, -2]
        return predictions, margins
