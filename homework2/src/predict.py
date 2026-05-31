"""Prediction helpers for Homework 2 DADGNN."""

import os

import numpy as np
import pandas as pd
import torch

from .data_loader import (
    GraphTextDataset,
    encode_document,
    load_json,
    load_chinese_embeddings,
    tokenize_chinese,
)
from .model import DADGNNModel
from .train import collate_fn


def predict_and_save(
    test_df: pd.DataFrame,
    results_dir: str,
    output_path: str = "submission.csv",
) -> str:
    """Load saved DADGNN model and generate submission CSV."""
    model_path = os.path.join(results_dir, "model", "dadgnn_model.pt")
    config_path = os.path.join(results_dir, "training_config.json")
    label_path = os.path.join(results_dir, "label_metadata.json")
    vocab_path = os.path.join(results_dir, "vocab.json")

    for p, msg in [
        (model_path, "Missing trained model. Run `python main.py train` first."),
        (config_path, "Missing training config. Run `python main.py train` first."),
        (label_path, "Missing label metadata. Run `python main.py train` first."),
        (vocab_path, "Missing vocab. Run `python main.py train` first."),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(msg)

    config = load_json(config_path)
    label_metadata = load_json(label_path)
    vocab_data = load_json(vocab_path)
    word2id = vocab_data["word2id"]
    labels_in_order = label_metadata["labels_in_order"]
    label_to_c_numerical = label_metadata["label_to_c_numerical"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load embeddings
    embed_path = os.path.join(os.path.dirname(results_dir), "data", "sgns.merge.word")
    embeddings = load_chinese_embeddings(embed_path, word2id, config["embed_dim"])
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Build model
    model = DADGNNModel(
        vocab_size=len(word2id),
        embed_dim=int(config["embed_dim"]),
        num_hidden=int(config["num_hidden"]),
        num_classes=len(labels_in_order),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        k=int(config["k"]),
        alpha=float(config["alpha"]),
        ngram=int(float(config["ngram"])),
        max_length=int(config["max_length"]),
        dropout=float(config["dropout"]),
        pretrained_embeddings=embeddings_tensor,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Encode test documents
    test_texts = test_df["sentence_sep"].tolist()
    test_doc_ids = [encode_document(tokenize_chinese(t), word2id, int(config["max_length"])) for t in test_texts]
    test_dataset = GraphTextDataset(test_doc_ids)

    # Batch inference
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(config["batch_size"]), shuffle=False, collate_fn=collate_fn,
    )
    all_preds = []
    with torch.no_grad():
        for doc_ids_list, _ in loader:
            logits = model(doc_ids_list, device)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)

    # Map predictions to c_numerical
    predicted_labels = [labels_in_order[int(idx)] for idx in all_preds]
    c_numerical = [label_to_c_numerical[label] for label in predicted_labels]

    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "c_numerical": c_numerical,
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} rows)")
    return output_path
