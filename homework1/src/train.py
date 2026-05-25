"""Training utilities and grid search for softmax regression."""

import json
import os
from itertools import product
from typing import Any

import numpy as np
from tqdm import tqdm

from .features import FeatureExtractor
from .model import SoftmaxRegression, cross_entropy_loss
from .optimizer import Adam, SGD


def create_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
):
    """Yield mini-batches."""
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        indices = rng.permutation(indices)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        yield X[idx], y[idx]


def build_feature_matrices(
    train_texts: list[str],
    dev_texts: list[str],
    test_texts: list[str] | None,
    feature_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, FeatureExtractor]:
    """Fit the vectorizer on train texts and transform all splits."""
    extractor = FeatureExtractor(
        method=feature_config["feature_method"],
        max_features=feature_config["max_features"],
        ngram_range=feature_config["ngram_range"],
        min_df=feature_config.get("min_df", 1),
        max_df=feature_config.get("max_df", 1.0),
        sublinear_tf=feature_config.get("sublinear_tf", False),
    )

    X_train = extractor.fit_transform(train_texts)
    X_dev = extractor.transform(dev_texts)
    X_test = extractor.transform(test_texts) if test_texts is not None else None
    return X_train, X_dev, X_test, extractor


def create_optimizer(optimizer_name: str, lr: float):
    """Create an optimizer by name."""
    if optimizer_name == "sgd":
        return SGD(lr=lr, momentum=0.9)
    if optimizer_name == "adam":
        return Adam(lr=lr)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(
    model: SoftmaxRegression,
    optimizer,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    l2: float,
    rng: np.random.Generator,
) -> float:
    """Train for one epoch and return the average loss."""
    total_loss = 0.0
    total_examples = 0

    for X_batch, y_batch in create_batches(X, y, batch_size, rng):
        probs = model.forward(X_batch)
        loss = cross_entropy_loss(probs, y_batch, weights=model.W, l2=l2)
        grads = model.backward(X_batch, y_batch, l2=l2)
        optimizer.step(model.get_params(), grads)
        total_loss += loss * X_batch.shape[0]
        total_examples += X_batch.shape[0]

    return total_loss / total_examples


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    input_dim: int,
    num_classes: int,
    optimizer_name: str = "sgd",
    lr: float = 0.01,
    l2: float = 0.0,
    epochs: int = 1000,
    batch_size: int = 32,
    patience: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[SoftmaxRegression, dict[str, Any]]:
    """Train a model and keep the best validation checkpoint when available."""
    model = SoftmaxRegression(input_dim, num_classes, seed=seed)
    optimizer = create_optimizer(optimizer_name, lr)
    rng = np.random.default_rng(seed)

    best_params = {k: v.copy() for k, v in model.get_params().items()}
    best_score = -1.0
    best_epoch = 0
    patience_counter = 0
    use_validation = X_val is not None and y_val is not None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, optimizer, X_train, y_train, batch_size, l2, rng)
        train_acc = model.accuracy(X_train, y_train)
        score = model.accuracy(X_val, y_val) if use_validation else train_acc

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_params = {k: v.copy() for k, v in model.get_params().items()}
            patience_counter = 0
        elif use_validation:
            patience_counter += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            metrics = (
                f"  Epoch {epoch:3d} | loss={train_loss:.4f} | train_acc={train_acc:.4f}"
            )
            if use_validation:
                metrics += f" | val_acc={score:.4f}"
            print(metrics)

        if use_validation and patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    model.set_params(best_params)
    info = {
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "best_val_acc": best_score if use_validation else None,
        "final_train_acc": model.accuracy(X_train, y_train),
    }
    return model, info


def grid_search(
    train_texts: list[str],
    y_train: np.ndarray,
    dev_texts: list[str],
    y_val: np.ndarray,
    num_classes: int,
    output_dir: str = "results",
    seed: int = 42,
) -> dict[str, Any]:
    """Search feature and optimization settings for the best validation score."""
    os.makedirs(output_dir, exist_ok=True)

    feature_configs = [
        {"feature_method": "bow", "ngram_range": (1, 1), "max_features": 5000},
        {"feature_method": "ngram", "ngram_range": (1, 2), "max_features": 8000},
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
    optimizers = ["sgd", "adam"]
    learning_rates = [0.1, 0.03, 0.01]
    l2_values = [0.0, 1e-4, 1e-3]
    batch_sizes = [16, 32]

    results = []
    best_acc = -1.0
    best_config = None

    combos = list(product(feature_configs, optimizers, learning_rates, l2_values, batch_sizes))
    print(f"\nGrid search: {len(combos)} combinations")

    for feature_config, optimizer_name, lr, l2, batch_size in tqdm(combos, desc="Grid search"):
        try:
            X_train, X_val, _, extractor = build_feature_matrices(
                train_texts,
                dev_texts,
                None,
                feature_config,
            )
            _, info = train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                input_dim=X_train.shape[1],
                num_classes=num_classes,
                optimizer_name=optimizer_name,
                lr=lr,
                l2=l2,
                epochs=120,
                batch_size=batch_size,
                patience=12,
                seed=seed,
                verbose=False,
            )
            val_acc = info["best_val_acc"]
        except Exception as exc:
            val_acc = 0.0
            info = {"best_epoch": 0, "epochs_trained": 0}
            extractor = None
            print(
                "  Failed:",
                feature_config,
                f"optimizer={optimizer_name}",
                f"lr={lr}",
                f"l2={l2}",
                f"batch_size={batch_size}",
                exc,
            )

        entry = {
            **feature_config,
            "optimizer": optimizer_name,
            "lr": lr,
            "l2": l2,
            "batch_size": batch_size,
            "val_acc": val_acc,
            "best_epoch": info["best_epoch"],
            "vocab_size": extractor.vocab_size if extractor is not None else 0,
        }
        results.append(entry)

        if val_acc > best_acc:
            best_acc = val_acc
            best_config = entry

    with open(os.path.join(output_dir, "grid_search_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nBest config: {best_config}")
    print(f"Best val accuracy: {best_acc:.4f}")

    return {"best_config": best_config, "results": results}
