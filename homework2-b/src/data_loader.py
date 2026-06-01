"""Data loading and preprocessing for Homework 2-b (Tiny DeBERTa)."""

import json
import os
import re
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

STOP_WORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 那 她 他 它 们 什么 吗 呢 吧 啊 呀 哦 嗯 哈 嘛 哎 唉 啦 嘞 哇 哪 这个 那个 "
    "请 您 帮 一下 还 那个 这个 嗯 对".split()
)


def clean_text(text: str) -> str:
    """Clean text: remove special characters, normalize whitespace."""
    text = str(text)
    text = re.sub(r"\[sep\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^一-龥a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_label_metadata(train_df: pd.DataFrame, dev_df: pd.DataFrame) -> dict[str, Any]:
    train_pairs = (
        train_df[["label_raw", "c_numerical"]]
        .drop_duplicates()
        .sort_values("c_numerical")
    )
    combined_pairs = (
        pd.concat(
            [train_df[["label_raw", "c_numerical"]], dev_df[["label_raw", "c_numerical"]]],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values("c_numerical")
    )
    labels_in_order = combined_pairs["label_raw"].tolist()
    label_to_c_numerical = dict(
        zip(combined_pairs["label_raw"], combined_pairs["c_numerical"], strict=False)
    )
    return {"labels_in_order": labels_in_order, "label_to_c_numerical": label_to_c_numerical}


def tokenize_chinese(text: str, remove_stopwords: bool = True) -> list[str]:
    text = clean_text(text)
    words = jieba.lcut(text)
    if remove_stopwords:
        words = [w for w in words if w.strip() and w not in STOP_WORDS]
    else:
        words = [w for w in words if w.strip()]
    return words


class TextDataset(Dataset):
    """Dataset returning padded token id tensors and labels."""

    def __init__(self, doc_ids_list: list[list[int]], labels: list[int] | None = None, pad_id: int = 0):
        self.doc_ids_list = doc_ids_list
        self.labels = labels
        self.pad_id = pad_id
        self.max_len = max(len(ids) for ids in doc_ids_list) if doc_ids_list else 0

    def __len__(self) -> int:
        return len(self.doc_ids_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ids = self.doc_ids_list[idx]
        padded = ids + [self.pad_id] * (self.max_len - len(ids))
        item = {"input_ids": torch.tensor(padded, dtype=torch.long)}
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
