"""Generate predictions for the test set using LLM."""

from __future__ import annotations

import time

import pandas as pd
from tqdm import tqdm

from .data_loader import load_test
from .evaluator import parse_prediction
from .llm_client import LLMClient
from .prompt_builder import build_prompt, build_system_prompt


def run_predict(
    *,
    mode: str,
    label_mapping: dict[int, str],
    examples: pd.DataFrame | None = None,
    client: LLMClient | None = None,
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """Run prediction on the test set and save submission CSV.

    Args:
        mode: 'zero-shot' or 'few-shot'.
        label_mapping: {c_numerical: label_raw} mapping.
        examples: Few-shot examples (required if mode='few-shot').
        client: LLMClient instance (created with defaults if None).
        output_path: Where to save the submission CSV.

    Returns:
        The submission DataFrame.
    """
    if client is None:
        client = LLMClient()

    test_df = load_test()
    system_prompt = build_system_prompt()

    ids = []
    predictions = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Predicting ({mode})"):
        user_prompt = build_prompt(
            text=row["sentence_sep"],
            label_mapping=label_mapping,
            mode=mode,
            examples=examples,
        )

        # Retry up to 3 times on failure
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
                print(f"\n[WARN] {row['ID']} attempt {attempt+1} failed: {e}")
                time.sleep(1)

        ids.append(row["ID"])
        predictions.append(pred if pred is not None else 0)  # fallback to 0

    submission = pd.DataFrame({"ID": ids, "c_numerical": predictions})
    submission.to_csv(output_path, index=False)
    print(f"\nSaved {len(submission)} predictions to {output_path}")
    return submission
