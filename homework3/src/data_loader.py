"""Data loading and label mapping for homework3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "nlp-text-classification-experiments"


def load_train() -> pd.DataFrame:
    """Load training set (5-shot)."""
    return pd.read_csv(DATA_DIR / "train_new_5shot.csv")


def load_dev() -> pd.DataFrame:
    """Load validation set."""
    return pd.read_csv(DATA_DIR / "dev_new.csv")


def load_test() -> pd.DataFrame:
    """Load test set for prediction."""
    return pd.read_csv(DATA_DIR / "kaggle_test.csv")


def load_sample_submission() -> pd.DataFrame:
    """Load sample submission format."""
    return pd.read_csv(DATA_DIR / "sample_submission.csv")


def get_label_mapping(df: pd.DataFrame) -> dict[int, str]:
    """Extract {c_numerical: label_raw} mapping from a labelled DataFrame."""
    return (
        df[["c_numerical", "label_raw"]]
        .drop_duplicates()
        .set_index("c_numerical")["label_raw"]
        .sort_index()
        .to_dict()
    )


def select_fewshot_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Select one shortest example per category for few-shot prompting.

    Returns a DataFrame with columns [c_numerical, label_raw, sentence_sep],
    sorted by c_numerical.
    """
    df = df.copy()
    df["_text_len"] = df["sentence_sep"].str.len()
    examples = (
        df.sort_values("_text_len")
        .drop_duplicates(subset="c_numerical", keep="first")
        .sort_values("c_numerical")[["c_numerical", "label_raw", "sentence_sep"]]
        .reset_index(drop=True)
    )
    return examples
