"""Standard multiclass softmax regression implemented with NumPy."""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def cross_entropy_loss(
    probs: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    l2: float = 0.0,
) -> float:
    """Cross-entropy loss with optional L2 regularization."""
    n = y.shape[0]
    log_probs = -np.log(probs[np.arange(n), y] + 1e-12)
    loss = float(np.mean(log_probs))
    if weights is not None and l2 > 0:
        loss += 0.5 * l2 * float(np.sum(weights ** 2))
    return loss


class SoftmaxRegression:
    """Single-layer linear classifier trained with softmax cross-entropy."""

    def __init__(self, input_dim: int, num_classes: int, seed: int = 42):
        self.num_classes = num_classes
        rng = np.random.default_rng(seed)

        # Xavier-style initialization is stable for linear classifiers.
        scale = np.sqrt(2.0 / (input_dim + num_classes))
        self.W = (rng.standard_normal((input_dim, num_classes)).astype(np.float32) * scale)
        self.b = np.zeros(num_classes, dtype=np.float32)

        self._cache: dict[str, np.ndarray] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: X -> logits -> softmax."""
        logits = X @ self.W + self.b
        probs = softmax(logits)
        self._cache["probs"] = probs
        return probs

    def backward(self, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> dict[str, np.ndarray]:
        """Compute gradients for the linear parameters."""
        probs = self._cache["probs"]
        grad_logits = probs.copy()
        grad_logits[np.arange(y.shape[0]), y] -= 1.0
        grad_logits /= y.shape[0]

        dW = X.T @ grad_logits
        if l2 > 0:
            dW += l2 * self.W
        db = np.sum(grad_logits, axis=0)

        return {"W": dW, "b": db}

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def get_params(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        self.W = params["W"]
        self.b = params["b"]
