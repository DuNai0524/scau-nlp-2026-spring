"""Prediction helpers for Homework 2."""

import os

import numpy as np
import pandas as pd

# Keep download behavior aligned with training so saved artifacts can be re-fetched reliably.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .data_loader import EncodedTextDataset, load_json, normalize_dialogue


def require_artifact(path: str, message: str) -> None:
    """Ensure a required artifact exists before prediction starts."""
    if not os.path.exists(path):
        raise FileNotFoundError(message)


def predict_and_save(
    test_df: pd.DataFrame,
    results_dir: str,
    output_path: str = "submission.csv",
) -> str:
    """Load the saved BERT artifacts and generate a Kaggle-style submission file."""
    model_dir = os.path.join(results_dir, "model")
    config_path = os.path.join(results_dir, "training_config.json")
    label_path = os.path.join(results_dir, "label_metadata.json")
    require_artifact(
        model_dir,
        "Missing trained model artifacts. Please run `python main.py train` in homework2 first.",
    )
    require_artifact(
        config_path,
        "Missing training configuration. Please run `python main.py train` in homework2 first.",
    )
    require_artifact(
        label_path,
        "Missing label metadata. Please run `python main.py train` in homework2 first.",
    )

    config = load_json(config_path)
    label_metadata = load_json(label_path)
    labels_in_order = label_metadata["labels_in_order"]
    label_to_c_numerical = label_metadata["label_to_c_numerical"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    texts = [
        normalize_dialogue(text, tokenizer.sep_token)
        for text in test_df["sentence_sep"].tolist()
    ]
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=int(config["max_length"]),
    )
    dataset = EncodedTextDataset(encodings=encodings)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(results_dir, "predict_tmp"),
            per_device_eval_batch_size=int(config["eval_batch_size"]),
            report_to="none",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        ),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    predictions = trainer.predict(dataset).predictions
    predicted_ids = np.argmax(predictions, axis=-1)
    predicted_labels = [labels_in_order[int(idx)] for idx in predicted_ids]
    c_numerical = [label_to_c_numerical[label] for label in predicted_labels]

    submission = pd.DataFrame(
        {
            "ID": test_df["ID"],
            "c_numerical": c_numerical,
        }
    )
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} rows)")
    return output_path
