"""Data loading and preprocessing helpers for Homework 2 (DADGNN)."""

import json
import os
from typing import Any

import jieba
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and normalize column names."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def save_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON with UTF-8 output."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict[str, Any]:
    """Read UTF-8 JSON content."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_label_metadata(train_df: pd.DataFrame, dev_df: pd.DataFrame) -> dict[str, Any]:
    """Build a stable label order and numerical mapping."""
    train_pairs = (
        train_df[["label_raw", "c_numerical"]]
        .drop_duplicates()
        .sort_values("c_numerical")
    )
    dev_labels = set(dev_df["label_raw"].tolist())
    train_labels = set(train_pairs["label_raw"].tolist())
    missing_from_train = sorted(dev_labels - train_labels)
    if missing_from_train:
        raise ValueError(
            "Dev split contains labels missing from train: "
            + ", ".join(missing_from_train)
        )

    combined_pairs = (
        pd.concat(
            [
                train_df[["label_raw", "c_numerical"]],
                dev_df[["label_raw", "c_numerical"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values("c_numerical")
    )

    labels_in_order = train_pairs["label_raw"].tolist()
    label_to_c_numerical = dict(
        zip(combined_pairs["label_raw"], combined_pairs["c_numerical"], strict=False)
    )
    return {
        "labels_in_order": labels_in_order,
        "label_to_c_numerical": label_to_c_numerical,
    }


def tokenize_chinese(text: str) -> list[str]:
    """Tokenize Chinese text using jieba."""
    return list(jieba.cut(str(text)))


def build_vocab(documents: list[str], min_freq: int = 1) -> tuple[dict[str, int], dict[int, str]]:
    """Build vocabulary from tokenized documents. Returns word2id and id2word."""
    freq: dict[str, int] = {}
    for doc in documents:
        for token in tokenize_chinese(doc):
            freq[token] = freq.get(token, 0) + 1

    word2id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    idx = len(word2id)
    for word, count in sorted(freq.items()):
        if count >= min_freq:
            word2id[word] = idx
            idx += 1

    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word


def encode_document(tokens: list[str], word2id: dict[str, int], max_length: int) -> list[int]:
    """Convert tokens to id sequence, with UNK fallback and truncation."""
    ids = [word2id.get(t, UNK_ID) for t in tokens]
    return ids[:max_length]


def load_chinese_embeddings(path: str, word2id: dict[str, int], embed_dim: int = 300) -> np.ndarray:
    """Load Chinese Word2Vec embeddings via gensim. Uncovered words are randomly initialized."""
    from gensim.models import KeyedVectors

    vocab_size = len(word2id)
    embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.1
    embeddings[PAD_ID] = np.zeros(embed_dim, dtype=np.float32)

    if not os.path.exists(path):
        print(f"  Embedding file not found: {path}. Using random initialization.")
        return embeddings

    print(f"  Loading embeddings from: {path}")
    kv = KeyedVectors.load(path, mmap="r") if path.endswith(".kv") else KeyedVectors.load_word2vec_format(path, binary=False)
    found = 0
    for word, idx in word2id.items():
        if word in kv:
            embeddings[idx] = kv[word]
            found += 1
    print(f"  Covered {found}/{vocab_size} words from pre-trained embeddings.")
    return embeddings


class GraphTextDataset(Dataset):
    """Dataset returning token id sequences and labels for graph construction."""

    def __init__(self, doc_ids_list: list[list[int]], labels: list[int] | None = None):
        self.doc_ids_list = doc_ids_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.doc_ids_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {"doc_ids": torch.tensor(self.doc_ids_list[idx], dtype=torch.long)}
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
