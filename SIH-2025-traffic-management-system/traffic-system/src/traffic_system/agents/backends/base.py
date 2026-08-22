"""Abstract interface for a Q-network backend.

Why this exists: the DQN agent's *logic* (epsilon-greedy action selection,
replay, target-network sync, double-DQN target computation) is completely
independent of *how* Q(s, a) is approximated. Separating them means:

1. Training on a real GPU/CPU box uses the PyTorch backend for full
   autograd-based learning.
2. A resource-constrained edge device (e.g. a Raspberry Pi coordinating a
   small number of local decisions, or any environment without PyTorch
   installed) can run the exact same agent logic against a lightweight
   pure-NumPy backend for inference, and even for training at reduced
   scale.
3. Tests and CI can run the full agent/environment loop without pulling in
   a multi-hundred-MB PyTorch install.

Both backends implement a dueling architecture (separate value and
advantage streams, recombined as Q = V + (A - mean(A))), which is a
well-established improvement over vanilla DQN for exactly this kind of
low-dimensional, discrete-action control problem.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class QNetworkBackend(ABC):
    """A Q-network with training-loop hooks. All arrays are float32 numpy."""

    @abstractmethod
    def predict(self, states: np.ndarray) -> np.ndarray:
        """states: (batch, obs_dim) -> returns (batch, n_actions) Q-values."""

    @abstractmethod
    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> float:
        """One gradient step toward `targets` for the taken `actions`. Returns loss (float)."""

    @abstractmethod
    def hard_update_to(self, target: QNetworkBackend) -> None:
        """Copy this network's weights into `target` (e.g. online -> target-net sync)."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
