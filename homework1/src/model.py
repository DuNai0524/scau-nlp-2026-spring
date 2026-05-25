"""Softmax Regression model with multiple activation functions and loss functions."""

import numpy as np


# ── Activation functions ──────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def leaky_relu_grad(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "leaky_relu": (leaky_relu, leaky_relu_grad),
    "tanh": (tanh, tanh_grad),
}


# ── Loss functions ────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy loss. y is integer class indices."""
    n = y.shape[0]
    log_probs = -np.log(probs[np.arange(n), y] + 1e-12)
    return float(np.mean(log_probs))


def cross_entropy_grad(probs: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    """Gradient of cross-entropy w.r.t. logits (before softmax)."""
    n = y.shape[0]
    grad = probs.copy()
    grad[np.arange(n), y] -= 1
    return grad / n


def mse_loss(probs: np.ndarray, y: np.ndarray, num_classes: int) -> float:
    """MSE loss with one-hot encoded targets."""
    n = y.shape[0]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y] = 1.0
    return float(np.mean((probs - one_hot) ** 2))


def mse_grad(probs: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    """Gradient of MSE w.r.t. logits (through softmax)."""
    n = y.shape[0]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y] = 1.0
    diff = probs - one_hot
    # Jacobian of softmax times diff
    grad = diff * probs * (1 - probs) + probs * (-diff * probs).sum(axis=1, keepdims=True)
    return grad / n


LOSSES = {
    "cross_entropy": (cross_entropy_loss, cross_entropy_grad),
    "mse": (mse_loss, mse_grad),
}


# ── Softmax Regression ───────────────────────────────────────────────────

class SoftmaxRegression:
    """Single-layer softmax regression with optional hidden activation and dropout."""

    def __init__(self, input_dim: int, num_classes: int,
                 activation: str = "relu", dropout: float = 0.0):
        self.num_classes = num_classes
        self.activation_name = activation
        self.dropout = dropout
        self._act_fn, self._act_grad = ACTIVATIONS[activation]

        # Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + num_classes))
        self.W = np.random.randn(input_dim, num_classes).astype(np.float32) * scale
        self.b = np.zeros(num_classes, dtype=np.float32)

        # Cache for backward pass
        self._cache: dict = {}

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: X -> activation -> dropout -> softmax."""
        logits = X @ self.W + self.b  # (N, C)

        # Apply activation
        activated = self._act_fn(logits)
        self._cache["logits"] = logits

        # Dropout
        if training and self.dropout > 0:
            mask = (np.random.rand(*activated.shape) > self.dropout).astype(np.float32)
            activated = activated * mask / (1 - self.dropout)
            self._cache["dropout_mask"] = mask

        self._cache["activated"] = activated
        probs = softmax(activated)
        self._cache["probs"] = probs
        return probs

    def backward(self, X: np.ndarray, y: np.ndarray,
                 loss_fn: str = "cross_entropy") -> dict[str, np.ndarray]:
        """Backward pass: compute gradients of W and b."""
        probs = self._cache["probs"]
        activated = self._cache["activated"]
        logits = self._cache["logits"]

        _, grad_fn = LOSSES[loss_fn]
        d_activated = grad_fn(probs, y, self.num_classes)

        # Dropout gradient
        if self.dropout > 0 and "dropout_mask" in self._cache:
            d_activated = d_activated * self._cache["dropout_mask"] / (1 - self.dropout)

        # Activation gradient
        d_logits = d_activated * self._act_grad(logits)

        # Weight and bias gradients
        n = X.shape[0]
        dW = X.T @ d_logits / n
        db = np.sum(d_logits, axis=0) / n

        return {"W": dW, "b": db}

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X, training=False)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def get_params(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        self.W = params["W"]
        self.b = params["b"]
