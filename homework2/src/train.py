"""Training helpers for Homework 2 BERT fine-tuning."""

import os
from typing import Any

import numpy as np
import pandas as pd

# Prefer the standard Hub download path because Xet downloads can stall on some local setups.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data_loader import EncodedTextDataset, build_label_metadata, normalize_dialogue, save_json


DEFAULT_CONFIG = {
    "model_name": "bert-base-chinese",
    "max_length": 512,
    "learning_rate": 2e-5,
    "batch_size": 8,
    "eval_batch_size": 8,
    "num_train_epochs": 5,
    "weight_decay": 0.01,
    "seed": 42,
}


def normalize_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge custom config with defaults and normalize numeric types."""
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)

    merged["max_length"] = int(merged["max_length"])
    merged["learning_rate"] = float(merged["learning_rate"])
    merged["batch_size"] = int(merged["batch_size"])
    merged["eval_batch_size"] = int(merged["eval_batch_size"])
    merged["num_train_epochs"] = int(merged["num_train_epochs"])
    merged["weight_decay"] = float(merged["weight_decay"])
    merged["seed"] = int(merged["seed"])
    return merged


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute plain accuracy without adding extra dependencies."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float(np.mean(predictions == labels))
    return {"accuracy": accuracy}


def prepare_split(tokenizer, texts: list[str], labels: list[int] | None, max_length: int) -> EncodedTextDataset:
    """Tokenize a split once and wrap it into a lightweight dataset."""
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
    )
    return EncodedTextDataset(encodings=encodings, labels=labels)


def train_and_save(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    results_dir: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fine-tune BERT and save reusable artifacts for prediction."""
    config = normalize_config(config)
    os.makedirs(results_dir, exist_ok=True)
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    model_dir = os.path.join(results_dir, "model")

    label_metadata = build_label_metadata(train_df, dev_df)
    labels_in_order = label_metadata["labels_in_order"]
    label_to_id = {label: idx for idx, label in enumerate(labels_in_order)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_texts = [
        normalize_dialogue(text, tokenizer.sep_token)
        for text in train_df["sentence_sep"].tolist()
    ]
    dev_texts = [
        normalize_dialogue(text, tokenizer.sep_token)
        for text in dev_df["sentence_sep"].tolist()
    ]
    train_labels = [label_to_id[label] for label in train_df["label_raw"].tolist()]
    dev_labels = [label_to_id[label] for label in dev_df["label_raw"].tolist()]

    set_seed(config["seed"])
    train_dataset = prepare_split(tokenizer, train_texts, train_labels, config["max_length"])
    dev_dataset = prepare_split(tokenizer, dev_texts, dev_labels, config["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=len(labels_in_order),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        optim="adamw_torch",
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        seed=config["seed"],
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    metrics = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "train_samples": len(train_df),
        "eval_samples": len(dev_df),
    }
    save_json(os.path.join(results_dir, "training_config.json"), config)
    save_json(os.path.join(results_dir, "label_metadata.json"), label_metadata)
    save_json(os.path.join(results_dir, "metrics.json"), metrics)

    return {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "model_dir": model_dir,
        "eval_metrics": eval_metrics,
    }
