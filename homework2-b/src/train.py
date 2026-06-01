"""Training helpers for Homework 2-b (Tiny DeBERTa)."""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_loader import TextDataset, build_label_metadata, save_json, tokenize_chinese
from .model import TinyDeBERTa
from .vocab import build_vocab, encode_document, load_chinese_embeddings

DEFAULT_CONFIG = {
    "embed_dim": 200,
    "hidden_dim": 128,
    "num_layers": 2,
    "num_heads": 2,
    "ffn_dim": 512,
    "max_seq_len": 512,
    "max_length": 350,
    "dropout": 0.5,
    "learning_rate": 2e-4,
    "weight_decay": 1e-3,
    "label_smoothing": 0.1,
    "batch_size": 16,
    "num_train_epochs": 1000,
    "early_stop_patience": 0,
    "seed": 42,
}


def normalize_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)
    for key in [
        "embed_dim", "hidden_dim", "num_layers", "num_heads", "ffn_dim",
        "max_seq_len", "max_length", "batch_size", "num_train_epochs",
        "early_stop_patience", "seed",
    ]:
        merged[key] = int(merged[key])
    for key in ["dropout", "learning_rate", "weight_decay", "label_smoothing"]:
        merged[key] = float(merged[key])
    return merged


def collate_fn(batch):
    """Pad sequences to the max length in the batch and stack labels."""
    input_ids = [item["input_ids"] for item in batch]
    max_len = max(ids.size(0) for ids in input_ids)
    padded = []
    for ids in input_ids:
        pad_size = max_len - ids.size(0)
        if pad_size > 0:
            ids = torch.cat([ids, torch.zeros(pad_size, dtype=torch.long)])
        padded.append(ids)
    input_ids_tensor = torch.stack(padded)

    labels = None
    if "label" in batch[0]:
        labels = torch.stack([item["label"] for item in batch])
    return input_ids_tensor, labels


def evaluate(model: nn.Module, dataset: TextDataset, device: torch.device, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            logits = model(input_ids.to(device))
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

    train_dataset = TextDataset(train_doc_ids, train_labels)
    dev_dataset = TextDataset(dev_doc_ids, dev_labels)

    # Load embeddings — prefer Tencent, fallback to sgns
    data_dir = os.path.join(os.path.dirname(results_dir), "data")
    embed_path = os.path.join(data_dir, "Tencent_AILab_ChineseEmbedding.bin")
    if not os.path.exists(embed_path):
        embed_path = os.path.join(data_dir, "sgns.merge.word")
    # Also check homework2/data/ as fallback
    if not os.path.exists(embed_path):
        embed_path = os.path.join(os.path.dirname(results_dir), "..", "homework2", "data", "Tencent_AILab_ChineseEmbedding.bin")
    if not os.path.exists(embed_path):
        embed_path = os.path.join(os.path.dirname(results_dir), "..", "homework2", "data", "sgns.merge.word")
    embeddings = load_chinese_embeddings(embed_path, word2id, config["embed_dim"])
    actual_embed_dim = embeddings.shape[1]
    if actual_embed_dim != config["embed_dim"]:
        print(f"  Adjusting embed_dim from {config['embed_dim']} to {actual_embed_dim}")
        config["embed_dim"] = actual_embed_dim
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Build model
    print("Building Tiny DeBERTa model...")
    model = TinyDeBERTa(
        vocab_size=len(word2id),
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        max_seq_len=config["max_seq_len"],
        max_length=config["max_length"],
        dropout=config["dropout"],
        pretrained_embeddings=embeddings_tensor,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    # Training loop
    print(f"Training on {device}...")
    best_dev_acc = 0.0
    patience_counter = 0
    model_path = os.path.join(results_dir, "model", "tiny_deberta.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(config["num_train_epochs"]):
        model.train()
        loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
        total_loss = 0.0
        for input_ids, labels in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()
            logits = model(input_ids.to(device))
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
            if config["early_stop_patience"] > 0 and patience_counter >= config["early_stop_patience"]:
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
        "total_params": total_params,
    }
    save_json(os.path.join(results_dir, "metrics.json"), metrics)

    print(f"\nTraining finished. Best dev accuracy: {best_dev_acc:.4f}")
    return {"model_path": model_path, "eval_metrics": {"accuracy": best_dev_acc}}
