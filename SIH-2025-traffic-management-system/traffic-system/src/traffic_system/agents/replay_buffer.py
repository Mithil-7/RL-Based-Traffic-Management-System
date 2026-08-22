"""Proportional prioritized experience replay (Schaul et al., 2016).

Sampling transitions in proportion to their TD-error means the agent spends
more training time on surprising transitions (e.g. a rare emergency-vehicle
event or a sudden congestion spike) instead of being dominated by the very
common "empty intersection" transitions -- which is exactly the class
imbalance this problem has.
"""
from __future__ import annotations

import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, seed: int | None = None) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self._rng = np.random.default_rng(seed)

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self._size = 0
        self._pos = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._size

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        i = self._pos
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.priorities[i] = self._max_priority  # new transitions get max priority so they're seen at least once
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> dict[str, np.ndarray]:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        priorities = self.priorities[: self._size] ** self.alpha
        probs = priorities / priorities.sum()
        indices = self._rng.choice(self._size, size=min(batch_size, self._size), p=probs, replace=self._size < batch_size)

        weights = (self._size * probs[indices]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "indices": indices,
            "weights": weights,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, eps: float = 1e-3) -> None:
        new_priorities = np.abs(td_errors) + eps
        self.priorities[indices] = new_priorities
        self._max_priority = max(self._max_priority, float(new_priorities.max()))
