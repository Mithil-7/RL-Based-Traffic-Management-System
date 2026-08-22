"""Pure-NumPy dueling MLP Q-network with a hand-written forward/backward pass.

No autograd, no PyTorch. This is the backend used for edge inference (or
anywhere PyTorch isn't installed / isn't worth the footprint) and is the
backend exercised by this repo's automated tests, since it has zero heavy
native dependencies. Architecture:

  obs (obs_dim) -> Dense(hidden, ReLU) -> Dense(hidden, ReLU) -> split into
     value head:      Dense(hidden -> 1)
     advantage head:  Dense(hidden -> n_actions)
  Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))

Trained with plain mini-batch gradient descent (Adam) on the Huber loss
between predicted Q(s, a) and the DQN target.
"""
from __future__ import annotations

import numpy as np

from traffic_system.agents.backends.base import QNetworkBackend


def _he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(np.float32)


class _Adam:
    """Minimal per-parameter Adam optimizer."""

    def __init__(self, shape: tuple[int, ...], lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class NumpyDuelingQNetwork(QNetworkBackend):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, lr: float = 5e-4, seed: int | None = None) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        rng = np.random.default_rng(seed)

        self.W1 = _he_init(rng, obs_dim, hidden)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = _he_init(rng, hidden, hidden)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.Wv = _he_init(rng, hidden, 1)
        self.bv = np.zeros(1, dtype=np.float32)
        self.Wa = _he_init(rng, hidden, n_actions)
        self.ba = np.zeros(n_actions, dtype=np.float32)

        self._opt = {
            name: _Adam(getattr(self, name).shape, lr)
            for name in ("W1", "b1", "W2", "b2", "Wv", "bv", "Wa", "ba")
        }

    def _forward(self, states: np.ndarray) -> dict[str, np.ndarray]:
        z1 = states @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0)
        v = h2 @ self.Wv + self.bv  # (batch, 1)
        a = h2 @ self.Wa + self.ba  # (batch, n_actions)
        q = v + (a - a.mean(axis=1, keepdims=True))
        return {"z1": z1, "h1": h1, "z2": z2, "h2": h2, "v": v, "a": a, "q": q}

    def predict(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        return self._forward(states)["q"]

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> float:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)
        batch = states.shape[0]
        weights = np.ones(batch, dtype=np.float32) if sample_weights is None else np.asarray(sample_weights, dtype=np.float32)

        cache = self._forward(states)
        q_pred_all = cache["q"]
        q_pred_taken = q_pred_all[np.arange(batch), actions]

        # Huber loss (delta=1.0) for robustness to reward-scale outliers.
        error = q_pred_taken - targets
        delta = 1.0
        abs_err = np.abs(error)
        huber_grad = np.where(abs_err <= delta, error, delta * np.sign(error))
        loss = np.mean(np.where(abs_err <= delta, 0.5 * error**2, delta * (abs_err - 0.5 * delta)) * weights)

        # dL/dQ_taken, scaled by importance weights and batch size.
        dQ_taken = (huber_grad * weights) / batch

        # Backprop through dueling head: Q = V + A - mean(A)
        dQ_all = np.zeros_like(q_pred_all)
        dQ_all[np.arange(batch), actions] = dQ_taken
        dV = dQ_all.sum(axis=1, keepdims=True)
        dA = dQ_all - dQ_all.mean(axis=1, keepdims=True)

        dWa = cache["h2"].T @ dA
        dba = dA.sum(axis=0)
        dWv = cache["h2"].T @ dV
        dbv = dV.sum(axis=0)

        dh2 = dA @ self.Wa.T + dV @ self.Wv.T
        dz2 = dh2 * (cache["z2"] > 0)
        dW2 = cache["h1"].T @ dz2
        db2 = dz2.sum(axis=0)

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (cache["z1"] > 0)
        dW1 = states.T @ dz1
        db1 = dz1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "Wv": dWv, "bv": dbv, "Wa": dWa, "ba": dba}
        for name, grad in grads.items():
            new_val = self._opt[name].step(getattr(self, name), grad)
            setattr(self, name, new_val.astype(np.float32))

        return float(loss)

    def hard_update_to(self, target: QNetworkBackend) -> None:
        assert isinstance(target, NumpyDuelingQNetwork)
        for name in ("W1", "b1", "W2", "b2", "Wv", "bv", "Wa", "ba"):
            setattr(target, name, getattr(self, name).copy())

    def save(self, path: str) -> None:
        np.savez(
            path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, Wv=self.Wv, bv=self.bv, Wa=self.Wa, ba=self.ba,
            obs_dim=self.obs_dim, n_actions=self.n_actions, hidden=self.hidden,
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        for name in ("W1", "b1", "W2", "b2", "Wv", "bv", "Wa", "ba"):
            setattr(self, name, data[name])
