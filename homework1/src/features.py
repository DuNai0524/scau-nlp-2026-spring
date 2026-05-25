"""Text feature extraction utilities for linear text classifiers."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureExtractor:
    """Wrapper for text vectorization with count and TF-IDF features."""

    def __init__(
        self,
        method: str = "bow",
        max_features: int = 5000,
        ngram_range: tuple = (1, 1),
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = False,
    ):
        """
        Args:
            method: 'bow', 'ngram', 'tfidf', 'char_tfidf', or 'char_wb_tfidf'
            max_features: maximum vocabulary size
            ngram_range: tuple (min_n, max_n) for n-gram
            min_df: ignore terms that appear in fewer than ``min_df`` documents
            max_df: ignore overly common terms above this ratio
            sublinear_tf: whether to apply sublinear tf scaling for TF-IDF
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self._vectorizer = None

    def _build_vectorizer(self):
        common_kwargs = {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
        }

        if self.method == "bow":
            return CountVectorizer(
                **common_kwargs,
                token_pattern=r"(?u)\b\w+\b",
            )
        if self.method == "ngram":
            return CountVectorizer(
                **common_kwargs,
                token_pattern=r"(?u)\b\w+\b",
            )
        if self.method == "tfidf":
            return TfidfVectorizer(
                **common_kwargs,
                token_pattern=r"(?u)\b\w+\b",
                sublinear_tf=self.sublinear_tf,
            )
        if self.method == "char_tfidf":
            return TfidfVectorizer(
                **common_kwargs,
                analyzer="char",
                sublinear_tf=self.sublinear_tf,
            )
        if self.method == "char_wb_tfidf":
            return TfidfVectorizer(
                **common_kwargs,
                analyzer="char_wb",
                sublinear_tf=self.sublinear_tf,
            )
        raise ValueError(f"Unknown method: {self.method}")

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit on training texts and return feature matrix."""
        self._vectorizer = self._build_vectorizer()
        X = self._vectorizer.fit_transform(texts)
        return X.toarray().astype(np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer."""
        if self._vectorizer is None:
            raise RuntimeError("Must call fit_transform first")
        X = self._vectorizer.transform(texts)
        return X.toarray().astype(np.float32)

    @property
    def vocab_size(self) -> int:
        if self._vectorizer is None:
            return 0
        return len(self._vectorizer.vocabulary_)
