"""Prediction helpers for Homework 2-b (Tiny DeBERTa)."""

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data_loader import TextDataset, load_json, tokenize_chinese
from .model import TinyDeBERTa
from .train import collate_fn
from .vocab import encode_document, load_chinese_embeddings


def predict_and_save(
    test_df: pd.DataFrame,
    results_dir: str,
    output_path: str = "submission.csv",
) -> str:
    model_path = os.path.join(results_dir, "model", "tiny_deberta.pt")
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
    data_dir = os.path.join(os.path.dirname(results_dir), "data")
    embed_path = os.path.join(data_dir, "Tencent_AILab_ChineseEmbedding.bin")
    if not os.path.exists(embed_path):
        embed_path = os.path.join(data_dir, "sgns.merge.word")
    if not os.path.exists(embed_path):
        embed_path = os.path.join(os.path.dirname(results_dir), "..", "homework2", "data", "Tencent_AILab_ChineseEmbedding.bin")
    if not os.path.exists(embed_path):
        embed_path = os.path.join(os.path.dirname(results_dir), "..", "homework2", "data", "sgns.merge.word")
    embeddings = load_chinese_embeddings(embed_path, word2id, int(config["embed_dim"]))
    actual_embed_dim = embeddings.shape[1]
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Build model
    model = TinyDeBERTa(
        vocab_size=len(word2id),
        embed_dim=actual_embed_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_classes=len(labels_in_order),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        ffn_dim=int(config["ffn_dim"]),
        max_seq_len=int(config["max_seq_len"]),
        max_length=int(config["max_length"]),
        dropout=float(config["dropout"]),
        pretrained_embeddings=embeddings_tensor,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Encode test documents
    test_texts = test_df["sentence_sep"].tolist()
    test_doc_ids = [encode_document(tokenize_chinese(t), word2id, int(config["max_length"])) for t in test_texts]
    test_dataset = TextDataset(test_doc_ids)

    # Batch inference
    loader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False, collate_fn=collate_fn)
    all_preds = []
    with torch.no_grad():
        for input_ids, _ in loader:
            logits = model(input_ids.to(device))
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)

    # Map predictions to c_numerical
    predicted_labels = [labels_in_order[int(idx)] for idx in all_preds]
    c_numerical = [label_to_c_numerical[label] for label in predicted_labels]

    submission = pd.DataFrame({"ID": test_df["ID"], "c_numerical": c_numerical})
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} rows)")
    return output_path
