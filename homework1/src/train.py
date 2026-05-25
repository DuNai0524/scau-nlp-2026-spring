"""Training loop with hyperparameter grid search."""

import json
import os
from itertools import product
from typing import Any

import numpy as np
from tqdm import tqdm

from .model import SoftmaxRegression, cross_entropy_loss, mse_loss
from .optimizer import SGD, Adam


LOSSES_FN = {
    "cross_entropy": cross_entropy_loss,
    "mse": mse_loss,
}


def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """Yield (X_batch, y_batch) tuples."""
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        yield X[idx], y[idx]


def train_epoch(
    model: SoftmaxRegression,
    optimizer,
    X: np.ndarray,
    y: np.ndarray,
    loss_name: str,
    batch_size: int = 32,
) -> float:
    """Train for one epoch, return average loss."""
    total_loss = 0.0
    count = 0
    for X_batch, y_batch in create_batches(X, y, batch_size):
        probs = model.forward(X_batch, training=True)
        loss = LOSSES_FN[loss_name](probs, y_batch, model.num_classes)
        grads = model.backward(X_batch, y_batch, loss_fn=loss_name)
        optimizer.step(model.get_params(), grads)
        total_loss += loss * X_batch.shape[0]
        count += X_batch.shape[0]
    return total_loss / count


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    num_classes: int,
    activation: str = "relu",
    loss_name: str = "cross_entropy",
    optimizer_name: str = "sgd",
    lr: float = 0.01,
    dropout: float = 0.0,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    verbose: bool = True,
) -> tuple[SoftmaxRegression, dict[str, Any]]:
    """Train a model and return the best one based on validation accuracy."""
    model = SoftmaxRegression(
        input_dim, num_classes, activation=activation, dropout=dropout
    )

    if optimizer_name == "sgd":
        optimizer = SGD(lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    best_val_acc = 0.0
    best_params = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, optimizer, X_train, y_train, loss_name, batch_size
        )
        train_acc = model.accuracy(X_train, y_train)
        val_acc = model.accuracy(X_val, y_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.get_params().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 10 == 0:
            print(
                f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
                f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    if best_params is not None:
        model.set_params(best_params)

    info = {
        "best_val_acc": best_val_acc,
        "epochs_trained": epoch,
    }
    return model, info


def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    num_classes: int,
    output_dir: str = "results",
) -> dict[str, Any]:
    """Grid search over all hyperparameter combinations."""
    os.makedirs(output_dir, exist_ok=True)

    feature_methods = ["bow", "ngram"]
    activations = ["relu", "leaky_relu", "tanh"]
    losses = ["cross_entropy", "mse"]
    optimizers = ["sgd", "adam"]
    learning_rates = [0.1, 0.01, 0.001]
    dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]

    results = []
    best_acc = 0.0
    best_config = None

    combos = list(product(activations, losses, optimizers, learning_rates, dropouts))
    print(f"\nGrid search: {len(combos)} combinations")

    for act, loss, opt, lr, drop in tqdm(combos, desc="Grid search"):
        try:
            model, info = train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                input_dim=input_dim,
                num_classes=num_classes,
                activation=act,
                loss_name=loss,
                optimizer_name=opt,
                lr=lr,
                dropout=drop,
                epochs=100,
                batch_size=32,
                patience=10,
                verbose=False,
            )
            val_acc = info["best_val_acc"]
        except Exception as e:
            val_acc = 0.0
            print(f"  Failed: act={act} loss={loss} opt={opt} lr={lr} drop={drop}: {e}")

        entry = {
            "activation": act,
            "loss": loss,
            "optimizer": opt,
            "lr": lr,
            "dropout": drop,
            "val_acc": val_acc,
        }
        results.append(entry)

        if val_acc > best_acc:
            best_acc = val_acc
            best_config = entry

    # Save results
    with open(os.path.join(output_dir, "grid_search_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nBest config: {best_config}")
    print(f"Best val accuracy: {best_acc:.4f}")

    return {"best_config": best_config, "results": results}
