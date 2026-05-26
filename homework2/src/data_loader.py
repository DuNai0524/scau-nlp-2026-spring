"""Data loading and preprocessing helpers for Homework 2."""

import json
import re
from collections.abc import Sequence
from typing import Any

import pandas as pd
from torch.utils.data import Dataset


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and normalize column names."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def normalize_dialogue(text: str, sep_token: str) -> str:
    """Preserve dialogue turns by replacing literal [SEP] markers with the tokenizer SEP token."""
    text = str(text)
    text = re.sub(r"\[\s*sep\s*\]", f" {sep_token} ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def save_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON with UTF-8 output."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict[str, Any]:
    """Read UTF-8 JSON content."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class EncodedTextDataset(Dataset):
    """Dataset wrapper for tokenized text classification inputs."""

    def __init__(self, encodings: dict[str, Sequence[Any]], labels: Sequence[int] | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item
