"""Data loading and preprocessing for Homework 2-b (Tiny DeBERTa)."""

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


def tokenize_chinese(text: str) -> list[str]:
    return list(jieba.cut(str(text)))


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
