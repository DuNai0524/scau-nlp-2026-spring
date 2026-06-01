"""Evaluate prompt strategies on the dev set."""

from __future__ import annotations

import time

import pandas as pd
from tqdm import tqdm

from .data_loader import load_dev
from .evaluator import parse_prediction, print_eval_report
from .llm_client import LLMClient
from .prompt_builder import build_prompt, build_system_prompt


def run_eval(
    *,
    mode: str,
    label_mapping: dict[int, str],
    examples: pd.DataFrame | None = None,
    client: LLMClient | None = None,
    max_samples: int | None = None,
) -> list[int | None]:
    """Evaluate a prompt strategy on the dev set.

    Args:
        mode: 'zero-shot' or 'few-shot'.
        label_mapping: {c_numerical: label_raw} mapping.
        examples: Few-shot examples (required if mode='few-shot').
        client: LLMClient instance (created with defaults if None).
        max_samples: Limit number of samples to evaluate (None = all).

    Returns:
        List of predictions (None for parse failures).
    """
    if client is None:
        client = LLMClient()

    dev_df = load_dev()
    if max_samples is not None:
        dev_df = dev_df.head(max_samples)

    system_prompt = build_system_prompt()
    predictions: list[int | None] = []
    ground_truth = dev_df["c_numerical"].tolist()

    for _, row in tqdm(dev_df.iterrows(), total=len(dev_df), desc=f"Evaluating ({mode})"):
        user_prompt = build_prompt(
            text=row["sentence_sep"],
            label_mapping=label_mapping,
            mode=mode,
            examples=examples,
        )

        pred = None
        for attempt in range(3):
            try:
                raw = client.chat(
                    message=user_prompt,
                    system=system_prompt,
                    temperature=0.0,
                    max_tokens=16,
                )
                pred = parse_prediction(raw, max_label=max(label_mapping))
                if pred is not None:
                    break
            except Exception as e:
                print(f"\n[WARN] attempt {attempt+1} failed: {e}")
                time.sleep(1)

        predictions.append(pred)

    print_eval_report(predictions, ground_truth, label_mapping)
    return predictions
