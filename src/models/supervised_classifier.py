from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass
class SupervisedResumeClassifier:
    """A strong text classifier used as an expert inside the multi-agent stack.

    Optionally augments the TF-IDF bag-of-words representation with a dense
    block of agent-derived features (fit-scores against every job profile,
    experience signals, skill-category histogram, etc).  The extra features
    are z-scored so they are on the same scale as the L2-normalized TF-IDF
    block before ``LinearSVC`` sees them.
    """

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
        self._extra_scaler: StandardScaler | None = None
        self._uses_extras: bool = False

    def _transform_features(
        self,
        text_series: pd.Series,
        extra: np.ndarray | None,
        *,
        fit: bool,
    ) -> sparse.csr_matrix:
        """Build the combined (TF-IDF, optional extras) feature matrix."""

        if fit:
            tfidf = self.vectorizer.fit_transform(text_series)
        else:
            tfidf = self.vectorizer.transform(text_series)

        if extra is None:
            return tfidf

        if extra.shape[0] != tfidf.shape[0]:
            raise ValueError(
                f"Extra features row count ({extra.shape[0]}) does not match "
                f"text row count ({tfidf.shape[0]})."
            )

        if fit:
            self._extra_scaler = StandardScaler(with_mean=True)
            scaled = self._extra_scaler.fit_transform(extra)
        else:
            if self._extra_scaler is None:
                raise RuntimeError("Extras were not seen at fit time.")
            scaled = self._extra_scaler.transform(extra)

        return sparse.hstack(
            [tfidf, sparse.csr_matrix(scaled.astype(np.float32))],
            format="csr",
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        text_col: str = "resume_text",
        label_col: str = "label",
        extra_features: np.ndarray | None = None,
    ) -> None:
        text_series = train_df[text_col].astype(str)
        label_series = train_df[label_col].astype(str)
        features = self._transform_features(text_series, extra_features, fit=True)
        self.classifier.fit(features, label_series)
        self._is_fitted = True
        self._uses_extras = extra_features is not None

    def predict(
        self,
        text_series: pd.Series,
        extra_features: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("SupervisedResumeClassifier must be fitted before prediction.")
        if self._uses_extras and extra_features is None:
            raise ValueError("Classifier was fitted with agent features; must provide them at predict time.")
        features = self._transform_features(
            text_series.astype(str),
            extra_features if self._uses_extras else None,
            fit=False,
        )
        return self.classifier.predict(features)

    def predict_with_margin(
        self,
        text_series: pd.Series,
        extra_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("SupervisedResumeClassifier must be fitted before prediction.")
        if self._uses_extras and extra_features is None:
            raise ValueError("Classifier was fitted with agent features; must provide them at predict time.")
        features = self._transform_features(
            text_series.astype(str),
            extra_features if self._uses_extras else None,
            fit=False,
        )
        predictions = self.classifier.predict(features)
        decision = self.classifier.decision_function(features)
        if decision.ndim == 1:
            margins = np.abs(decision)
        else:
            sorted_scores = np.sort(decision, axis=1)
            margins = sorted_scores[:, -1] - sorted_scores[:, -2]
        return predictions, margins
