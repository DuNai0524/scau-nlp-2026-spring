"""Data loading and text preprocessing for SLU intent detection."""

import re

import jieba
import numpy as np
import pandas as pd


STOP_WORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 那 她 他 它 们 什么 吗 呢 吧 啊 呀 哦 嗯 哈 嘛 哎 唉 啦 嘞 哇 哪 这个 那个 "
    "请 您 帮 一下 还 那个 这个 嗯 对".split()
)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file with Id, Sentence, Category columns."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def clean_text(text: str) -> str:
    """Clean text: remove special characters, normalize whitespace."""
    text = str(text)
    text = re.sub(r"[^一-龥a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> str:
    """Tokenize Chinese text with jieba, optionally remove stop words."""
    text = clean_text(text)
    words = jieba.lcut(text)
    if remove_stopwords:
        words = [w for w in words if w.strip() and w not in STOP_WORDS]
    else:
        words = [w for w in words if w.strip()]
    return " ".join(words)


class LabelEncoder:
    """Map category strings to integer indices."""

    def __init__(self):
        self.label_to_idx: dict[str, int] = {}
        self.idx_to_label: dict[int, str] = {}

    def fit(self, labels: list[str]) -> "LabelEncoder":
        unique = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        return self

    def transform(self, labels: list[str]) -> np.ndarray:
        return np.array([self.label_to_idx[l] for l in labels])

    def inverse_transform(self, indices: np.ndarray) -> list[str]:
        return [self.idx_to_label[int(i)] for i in indices]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)
