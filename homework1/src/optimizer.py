"""SGD and Adam optimizers for manual gradient descent."""


class SGD:
    """Stochastic Gradient Descent with optional momentum."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict[str, any] = {}

    def step(self, params: dict[str, any], grads: dict[str, any]) -> None:
        """Update parameters in-place."""
        for key in params:
            if key not in self._velocity:
                self._velocity[key] = 0.0

            v = self.momentum * self._velocity[key] - self.lr * grads[key]
            self._velocity[key] = v
            params[key] += v


class Adam:
    """Adam optimizer."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: dict[str, any] = {}
        self._v: dict[str, any] = {}
        self._t = 0

    def step(self, params: dict[str, any], grads: dict[str, any]) -> None:
        """Update parameters in-place."""
        self._t += 1
        for key in params:
            if key not in self._m:
                self._m[key] = 0.0
                self._v[key] = 0.0

            self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * grads[key]
            self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self._m[key] / (1 - self.beta1 ** self._t)
            v_hat = self._v[key] / (1 - self.beta2 ** self._t)

            params[key] -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
