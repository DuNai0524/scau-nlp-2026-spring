"""SLU Intent Detection - Homework 1 Entry Point.

Usage:
    python main.py search           # Grid search for best hyperparameters
    python main.py train            # Train with best/default config
    python main.py predict          # Generate submission.csv
    python main.py compare_features # Compare BoW vs N-gram vs TF-IDF
"""

import argparse
import json
import os
import sys

import numpy as np

from src.data_loader import LabelEncoder, load_csv, tokenize
from src.features import FeatureExtractor
from src.model import SoftmaxRegression
from src.predict import predict_and_save
from src.train import grid_search, train_model

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "nlp-text-classification-experiments")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def prepare_data(feature_method: str = "bow", max_features: int = 5000, ngram_range: tuple = (1, 2)):
    """Load data, preprocess, and extract features. Returns everything needed."""
    print("Loading data...")
    train_df = load_csv(os.path.join(DATA_DIR, "train_new_5shot.csv"))
    dev_df = load_csv(os.path.join(DATA_DIR, "dev_new.csv"))
    test_df = load_csv(os.path.join(DATA_DIR, "kaggle_test.csv"))

    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    # Label encoding
    le = LabelEncoder()
    le.fit(train_df["label_raw"].tolist())

    y_train = le.transform(train_df["label_raw"].tolist())
    y_dev = le.transform(dev_df["label_raw"].tolist())

    # Tokenize
    print("Tokenizing...")
    train_texts = [tokenize(s) for s in train_df["sentence_sep"]]
    dev_texts = [tokenize(s) for s in dev_df["sentence_sep"]]
    test_texts = [tokenize(s) for s in test_df["sentence_sep"]]

    # Feature extraction
    print(f"Extracting features ({feature_method})...")
    extractor = FeatureExtractor(method=feature_method, max_features=max_features, ngram_range=ngram_range)
    X_train = extractor.fit_transform(train_texts)
    X_dev = extractor.transform(dev_texts)
    X_test = extractor.transform(test_texts)

    print(f"  Vocabulary size: {extractor.vocab_size}")
    print(f"  Feature shape: {X_train.shape}")

    return X_train, y_train, X_dev, y_dev, X_test, test_df, le, extractor


def run_search():
    """Grid search for best hyperparameters."""
    X_train, y_train, X_dev, y_dev, X_test, test_df, le, extractor = prepare_data()

    print(f"\nClasses ({le.num_classes}): {list(le.label_to_idx.keys())}")

    results = grid_search(
        X_train, y_train, X_dev, y_dev,
        input_dim=X_train.shape[1],
        num_classes=le.num_classes,
        output_dir=RESULTS_DIR,
    )

    # Save best config
    with open(os.path.join(RESULTS_DIR, "best_config.json"), "w") as f:
        json.dump(results["best_config"], f, indent=2, ensure_ascii=False)
    print(f"\nBest config saved to {RESULTS_DIR}/best_config.json")


def run_train(config: dict | None = None):
    """Train with given or best config."""
    if config is None:
        config_path = os.path.join(RESULTS_DIR, "best_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            print(f"Loaded best config: {config}")
        else:
            config = {
                "activation": "relu",
                "loss": "cross_entropy",
                "optimizer": "adam",
                "lr": 0.01,
                "dropout": 0.1,
            }
            print(f"Using default config: {config}")

    X_train, y_train, X_dev, y_dev, X_test, test_df, le, extractor = prepare_data()

    print(f"\nTraining with: {config}")
    model, info = train_model(
        X_train, y_train, X_dev, y_dev,
        input_dim=X_train.shape[1],
        num_classes=le.num_classes,
        activation=config["activation"],
        loss_name=config["loss"],
        optimizer_name=config["optimizer"],
        lr=config["lr"],
        dropout=config["dropout"],
        epochs=200,
        batch_size=32,
        patience=20,
        verbose=True,
    )

    print(f"\nFinal validation accuracy: {info['best_val_acc']:.4f}")

    # Save model params
    os.makedirs(RESULTS_DIR, exist_ok=True)
    params = {k: v.tolist() for k, v in model.get_params().items()}
    params["_meta"] = {
        "input_dim": X_train.shape[1],
        "num_classes": le.num_classes,
        "config": config,
    }
    with open(os.path.join(RESULTS_DIR, "model_params.json"), "w") as f:
        json.dump(params, f)

    return model, le, X_test, test_df


def run_predict():
    """Generate submission file."""
    # Load config
    config_path = os.path.join(RESULTS_DIR, "best_config.json")
    if not os.path.exists(config_path):
        print("No best_config.json found. Run 'python main.py search' first.")
        print("Using default config for prediction...")
        config = {
            "activation": "relu",
            "loss": "cross_entropy",
            "optimizer": "adam",
            "lr": 0.01,
            "dropout": 0.1,
        }
    else:
        with open(config_path) as f:
            config = json.load(f)

    model, le, X_test, test_df = run_train(config)

    output_path = os.path.join(os.path.dirname(__file__), "submission.csv")
    predict_and_save(model, X_test, test_df, le, output_path)


def run_compare_features():
    """Compare BoW, N-gram, and TF-IDF feature extraction methods."""
    print("=" * 60)
    print("Feature Comparison: BoW vs N-gram vs TF-IDF")
    print("=" * 60)

    config = {
        "activation": "relu",
        "loss": "cross_entropy",
        "optimizer": "adam",
        "lr": 0.01,
        "dropout": 0.1,
    }

    for method in ["bow", "ngram", "tfidf"]:
        print(f"\n{'─' * 40}")
        print(f"Feature method: {method}")
        print(f"{'─' * 40}")

        X_train, y_train, X_dev, y_dev, _, _, le, _ = prepare_data(feature_method=method)

        model, info = train_model(
            X_train, y_train, X_dev, y_dev,
            input_dim=X_train.shape[1],
            num_classes=le.num_classes,
            activation=config["activation"],
            loss_name=config["loss"],
            optimizer_name=config["optimizer"],
            lr=config["lr"],
            dropout=config["dropout"],
            epochs=100,
            batch_size=32,
            patience=10,
            verbose=False,
        )
        print(f"  Val accuracy: {info['best_val_acc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="SLU Intent Detection - Homework 1")
    parser.add_argument("mode", choices=["search", "train", "predict", "compare_features"],
                        help="Operation mode")
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
