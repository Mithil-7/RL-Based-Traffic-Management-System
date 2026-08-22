"""PyTorch dueling Q-network backend -- the recommended backend for real
training runs (GPU or multi-core CPU). Mirrors `numpy_backend.py`'s
architecture and public interface exactly, so `dqn_agent.py` is agnostic to
which one is loaded.

This module imports torch lazily (inside functions/`__init__`) so the rest
of the codebase can be imported and unit-tested in environments where torch
isn't installed -- see `agents/dqn_agent.py::make_backend`.
"""
from __future__ import annotations

import numpy as np

from traffic_system.agents.backends.base import QNetworkBackend


class TorchDuelingQNetwork(QNetworkBackend):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, lr: float = 5e-4, seed: int | None = None, device: str | None = None) -> None:
        import torch
        from torch import nn

        if seed is not None:
            torch.manual_seed(seed)

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, hidden

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(obs_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                )
                self.value_head = nn.Linear(hidden, 1)
                self.advantage_head = nn.Linear(hidden, n_actions)

            def forward(self, x):  # noqa: ANN001
                h = self.trunk(x)
                v = self.value_head(h)
                a = self.advantage_head(h)
                return v + (a - a.mean(dim=1, keepdim=True))

        self.net = _Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # Huber loss

    def predict(self, states: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            q = self.net(t)
            return q.cpu().numpy()

    def train_step(self, states, actions, targets, sample_weights=None) -> float:  # noqa: ANN001
        torch = self.torch
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        y = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        w = (
            torch.ones(s.shape[0], dtype=torch.float32, device=self.device)
            if sample_weights is None
            else torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device)
        )

        q_all = self.net(s)
        q_taken = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        per_sample_loss = self.loss_fn(q_taken, y)
        loss = (per_sample_loss * w).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def hard_update_to(self, target: QNetworkBackend) -> None:
        assert isinstance(target, TorchDuelingQNetwork)
        target.net.load_state_dict(self.net.state_dict())

    def save(self, path: str) -> None:
        self.torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(self.torch.load(path, map_location=self.device))
