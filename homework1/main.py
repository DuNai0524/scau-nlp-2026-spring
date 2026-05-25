"""SLU Intent Detection - Homework 1 Entry Point.

Usage:
    python main.py search
    python main.py train
    python main.py predict
    python main.py compare_features
"""

import argparse
import json
import os
from typing import Any

from src.data_loader import LabelEncoder, load_csv, tokenize
from src.predict import predict_and_save
from src.train import build_feature_matrices, grid_search, train_model

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "nlp-text-classification-experiments")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

DEFAULT_CONFIG = {
    "feature_method": "char_wb_tfidf",
    "ngram_range": (2, 5),
    "max_features": 12000,
    "sublinear_tf": True,
    "optimizer": "adam",
    "lr": 0.03,
    "l2": 1e-4,
    "batch_size": 16,
    "best_epoch": 80,
}


def normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge config with defaults and keep backward compatibility with old files."""
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)

    merged["ngram_range"] = tuple(merged["ngram_range"])
    merged["max_features"] = int(merged["max_features"])
    merged["lr"] = float(merged["lr"])
    merged["l2"] = float(merged.get("l2", 0.0))
    merged["batch_size"] = int(merged.get("batch_size", 32))
    merged["best_epoch"] = int(merged.get("best_epoch", 100))
    merged["sublinear_tf"] = bool(merged.get("sublinear_tf", False))
    return merged


def load_tokenized_splits():
    """Load CSV files and tokenize all text splits once."""
    print("Loading data...")
    train_df = load_csv(os.path.join(DATA_DIR, "train_new_5shot.csv"))
    dev_df = load_csv(os.path.join(DATA_DIR, "dev_new.csv"))
    test_df = load_csv(os.path.join(DATA_DIR, "kaggle_test.csv"))

    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    print("Tokenizing...")

    train_texts = [tokenize(text) for text in train_df["sentence_sep"]]
    dev_texts = [tokenize(text) for text in dev_df["sentence_sep"]]
    test_texts = [tokenize(text) for text in test_df["sentence_sep"]]

    return train_df, dev_df, test_df, train_texts, dev_texts, test_texts


def build_label_to_c_numerical(train_df, dev_df) -> dict[str, int]:
    """Build a stable mapping from label text to the required numeric code."""
    label_df = (
        train_df[["label_raw", "c_numerical"]]
        .drop_duplicates()
        .sort_values("c_numerical")
    )
    dev_label_df = dev_df[["label_raw", "c_numerical"]].drop_duplicates()
    combined = (
        label_df.merge(dev_label_df, on=["label_raw", "c_numerical"], how="outer")
        .sort_values("c_numerical")
    )
    return dict(zip(combined["label_raw"], combined["c_numerical"], strict=False))


def prepare_data(config: dict[str, Any]):
    """Build feature matrices from the selected feature configuration."""
    config = normalize_config(config)
    train_df, dev_df, test_df, train_texts, dev_texts, test_texts = load_tokenized_splits()
    label_to_c_numerical = build_label_to_c_numerical(train_df, dev_df)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label_raw"].tolist())

    y_train = label_encoder.transform(train_df["label_raw"].tolist())
    y_dev = label_encoder.transform(dev_df["label_raw"].tolist())

    X_train, X_dev, X_test, extractor = build_feature_matrices(
        train_texts,
        dev_texts,
        test_texts,
        config,
    )

    print(f"Extracting features ({config['feature_method']})...")
    print(f"  Vocabulary size: {extractor.vocab_size}")
    print(
        f"  Train shape: {X_train.shape} | Dev shape: {X_dev.shape} | Test shape: {X_test.shape}"
    )
    return (
        X_train,
        y_train,
        X_dev,
        y_dev,
        X_test,
        test_df,
        label_encoder,
        extractor,
        label_to_c_numerical,
    )


def load_best_config() -> dict[str, Any]:
    """Load the saved config, or fall back to defaults."""
    config_path = os.path.join(RESULTS_DIR, "best_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
        config = normalize_config(saved)
        print(f"Loaded best config: {config}")
        return config

    config = normalize_config(None)
    print(f"Using default config: {config}")
    return config


def run_search():
    """Grid search for the best feature and optimization configuration."""
    train_df, dev_df, _, train_texts, dev_texts, _ = load_tokenized_splits()

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label_raw"].tolist())
    y_train = label_encoder.transform(train_df["label_raw"].tolist())
    y_dev = label_encoder.transform(dev_df["label_raw"].tolist())

    print(f"\nClasses ({label_encoder.num_classes}): {list(label_encoder.label_to_idx.keys())}")

    results = grid_search(
        train_texts=train_texts,
        y_train=y_train,
        dev_texts=dev_texts,
        y_val=y_dev,
        num_classes=label_encoder.num_classes,
        output_dir=RESULTS_DIR,
    )

    best_config = normalize_config(results["best_config"])
    with open(os.path.join(RESULTS_DIR, "best_config.json"), "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    print(f"\nBest config saved to {RESULTS_DIR}/best_config.json")


def run_train(config: dict[str, Any] | None = None):
    """Train the classifier and save model parameters."""
    config = normalize_config(config or load_best_config())
    (
        X_train,
        y_train,
        X_dev,
        y_dev,
        X_test,
        test_df,
        label_encoder,
        _,
        label_to_c_numerical,
    ) = prepare_data(config)

    print(f"\nTraining with: {config}")
    epochs = 2000 # max(config["best_epoch"], 60)
    patience = 100000

    model, info = train_model(
        X_train,
        y_train,
        X_dev,
        y_dev,
        input_dim=X_train.shape[1],
        num_classes=label_encoder.num_classes,
        optimizer_name=config["optimizer"],
        lr=config["lr"],
        l2=config["l2"],
        epochs=epochs,
        batch_size=config["batch_size"],
        patience=patience,
        verbose=True,
    )

    print(f"\nFinal validation accuracy: {info['best_val_acc']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    params = {k: v.tolist() for k, v in model.get_params().items()}
    params["_meta"] = {
        "input_dim": X_train.shape[1],
        "num_classes": label_encoder.num_classes,
        "config": config,
    }
    with open(os.path.join(RESULTS_DIR, "model_params.json"), "w") as f:
        json.dump(params, f)

    return model, label_encoder, label_to_c_numerical, X_test, test_df


def run_predict():
    """Generate submission CSV."""
    config = load_best_config()
    model, label_encoder, label_to_c_numerical, X_test, test_df = run_train(config)

    output_path = os.path.join(os.path.dirname(__file__), "submission.csv")
    predict_and_save(
        model,
        X_test,
        test_df,
        label_encoder,
        label_to_c_numerical,
        output_path,
    )


def run_compare_features():
    """Compare several high-value feature configurations."""
    print("=" * 60)
    print("Feature Comparison")
    print("=" * 60)

    feature_configs = [
        {"feature_method": "bow", "ngram_range": (1, 1), "max_features": 5000},
        {
            "feature_method": "tfidf",
            "ngram_range": (1, 2),
            "max_features": 10000,
            "sublinear_tf": True,
        },
        {
            "feature_method": "char_wb_tfidf",
            "ngram_range": (2, 5),
            "max_features": 12000,
            "sublinear_tf": True,
        },
    ]
    base_config = load_best_config()

    for feature_config in feature_configs:
        config = normalize_config({**base_config, **feature_config})
        print(f"\n{'─' * 40}")
        print(f"Feature config: {feature_config}")
        print(f"{'─' * 40}")

        X_train, y_train, X_dev, y_dev, _, _, label_encoder, _, _ = prepare_data(config)
        _, info = train_model(
            X_train,
            y_train,
            X_dev,
            y_dev,
            input_dim=X_train.shape[1],
            num_classes=label_encoder.num_classes,
            optimizer_name=config["optimizer"],
            lr=config["lr"],
            l2=config["l2"],
            epochs=120,
            batch_size=config["batch_size"],
            patience=12,
            verbose=False,
        )
        print(f"  Val accuracy: {info['best_val_acc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="SLU Intent Detection - Homework 1")
    parser.add_argument(
        "mode",
        choices=["search", "train", "predict", "compare_features"],
        help="Operation mode",
    )
    args = parser.parse_args()

    if args.mode == "search":
        run_search()
    elif args.mode == "train":
        run_train()
    elif args.mode == "predict":
        run_predict()
    elif args.mode == "compare_features":
        run_compare_features()


if __name__ == "__main__":
    main()
