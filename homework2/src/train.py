"""Training helpers for Homework 2 DADGNN."""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .data_loader import (
    GraphTextDataset,
    build_label_metadata,
    build_vocab,
    encode_document,
    load_chinese_embeddings,
    save_json,
    tokenize_chinese,
)
from .model import DADGNNModel

DEFAULT_CONFIG = {
    "embed_dim": 300,
    "num_hidden": 64,
    "num_layers": 5,
    "num_heads": 2,
    "k": 5,
    "alpha": 0.5,
    "ngram": 4,
    "max_length": 350,
    "dropout": 0.5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "batch_size": 64,
    "num_train_epochs": 100,
    "early_stop_patience": 10,
    "seed": 42,
}


def normalize_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)
    for key in ["embed_dim", "num_hidden", "num_layers", "num_heads", "k", "max_length", "batch_size", "num_train_epochs", "early_stop_patience", "seed"]:
        merged[key] = int(merged[key])
    for key in ["alpha", "ngram", "dropout", "learning_rate", "weight_decay"]:
        merged[key] = float(merged[key])
    return merged


def collate_fn(batch):
    """Custom collate: return lists of doc_ids tensors and stacked labels."""
    doc_ids_list = [item["doc_ids"] for item in batch]
    labels = None
    if "label" in batch[0]:
        labels = torch.stack([item["label"] for item in batch])
    return doc_ids_list, labels


def evaluate(model, dataset, device, batch_size):
    """Evaluate model accuracy on a dataset."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for doc_ids_list, labels in loader:
            logits = model(doc_ids_list, device)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels.to(device)).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


def train_and_save(
    train_df,
    dev_df,
    results_dir: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = normalize_config(config)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Label metadata
    label_metadata = build_label_metadata(train_df, dev_df)
    labels_in_order = label_metadata["labels_in_order"]
    label_to_id = {label: idx for idx, label in enumerate(labels_in_order)}
    num_classes = len(labels_in_order)

    # Tokenize and build vocab
    print("Tokenizing and building vocabulary...")
    train_texts = train_df["sentence_sep"].tolist()
    dev_texts = dev_df["sentence_sep"].tolist()
    word2id, id2word = build_vocab(train_texts + dev_texts)
    print(f"  Vocabulary size: {len(word2id)}")

    # Encode documents
    print("Encoding documents...")
    train_doc_ids = [encode_document(tokenize_chinese(t), word2id, config["max_length"]) for t in train_texts]
    dev_doc_ids = [encode_document(tokenize_chinese(t), word2id, config["max_length"]) for t in dev_texts]
    train_labels = [label_to_id[l] for l in train_df["label_raw"].tolist()]
    dev_labels = [label_to_id[l] for l in dev_df["label_raw"].tolist()]

    train_dataset = GraphTextDataset(train_doc_ids, train_labels)
    dev_dataset = GraphTextDataset(dev_doc_ids, dev_labels)

    # Load embeddings
    embed_path = os.path.join(os.path.dirname(results_dir), "data", "sgns.merge.word")
    embeddings = load_chinese_embeddings(embed_path, word2id, config["embed_dim"])
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Build model
    print("Building DADGNN model...")
    model = DADGNNModel(
        vocab_size=len(word2id),
        embed_dim=config["embed_dim"],
        num_hidden=config["num_hidden"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        k=config["k"],
        alpha=config["alpha"],
        ngram=int(config["ngram"]),
        max_length=config["max_length"],
        dropout=config["dropout"],
        pretrained_embeddings=embeddings_tensor,
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"Training on {device}...")
    best_dev_acc = 0.0
    patience_counter = 0
    model_path = os.path.join(results_dir, "model", "dadgnn_model.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(config["num_train_epochs"]):
        model.train()
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn,
        )
        total_loss = 0.0
        for doc_ids_list, labels in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()
            logits = model(doc_ids_list, device)
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        avg_loss = total_loss / len(train_dataset)
        dev_acc = evaluate(model, dev_dataset, device, config["batch_size"])
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, dev_acc={dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  -> Best model saved (dev_acc={dev_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stop_patience"]:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Save artifacts
    save_json(os.path.join(results_dir, "training_config.json"), config)
    save_json(os.path.join(results_dir, "label_metadata.json"), label_metadata)
    save_json(os.path.join(results_dir, "vocab.json"), {"word2id": word2id, "id2word": {str(k): v for k, v in id2word.items()}})

    metrics = {
        "best_dev_accuracy": best_dev_acc,
        "train_samples": len(train_df),
        "eval_samples": len(dev_df),
    }
    save_json(os.path.join(results_dir, "metrics.json"), metrics)

    print(f"\nTraining finished. Best dev accuracy: {best_dev_acc:.4f}")
    return {
        "model_path": model_path,
        "eval_metrics": {"accuracy": best_dev_acc},
    }
