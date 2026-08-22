"""Factory for picking a Q-network backend at runtime.

`TRAFFIC_QNET_BACKEND=torch|numpy|auto` (see common/config.py). `auto` tries
torch first (best training performance) and silently falls back to the
NumPy implementation if torch isn't installed -- this is what lets the
exact same agent code run on a full training box and on a lightweight
edge/CI environment.
"""
from __future__ import annotations

from traffic_system.agents.backends.base import QNetworkBackend
from traffic_system.common.logging import get_logger

logger = get_logger(__name__)


def make_backend(
    backend: str, obs_dim: int, n_actions: int, hidden: int = 64, lr: float = 5e-4, seed: int | None = None
) -> QNetworkBackend:
    if backend == "numpy":
        from traffic_system.agents.backends.numpy_backend import NumpyDuelingQNetwork

        return NumpyDuelingQNetwork(obs_dim, n_actions, hidden, lr, seed)

    if backend == "torch":
        from traffic_system.agents.backends.torch_backend import TorchDuelingQNetwork

        return TorchDuelingQNetwork(obs_dim, n_actions, hidden, lr, seed)

    if backend == "auto":
        try:
            from traffic_system.agents.backends.torch_backend import TorchDuelingQNetwork

            return TorchDuelingQNetwork(obs_dim, n_actions, hidden, lr, seed)
        except ImportError:
            logger.warning("qnet_backend.torch_unavailable_falling_back_to_numpy")
            from traffic_system.agents.backends.numpy_backend import NumpyDuelingQNetwork

            return NumpyDuelingQNetwork(obs_dim, n_actions, hidden, lr, seed)

    raise ValueError(f"Unknown backend: {backend}")
