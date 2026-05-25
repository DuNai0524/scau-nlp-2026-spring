"""Text feature extraction: Bag-of-Words and N-gram."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureExtractor:
    """Wrapper for text vectorization with BoW, N-gram, or TF-IDF."""

    def __init__(self, method: str = "bow", max_features: int = 5000, ngram_range: tuple = (1, 1)):
        """
        Args:
            method: 'bow', 'ngram', or 'tfidf'
            max_features: maximum vocabulary size
            ngram_range: tuple (min_n, max_n) for n-gram
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit on training texts and return feature matrix."""
        if self.method == "bow":
            self._vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 1),
            )
        elif self.method == "ngram":
            self._vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            )
        elif self.method == "tfidf":
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

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
