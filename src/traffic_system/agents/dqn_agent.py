"""Double Dueling DQN agent that controls one intersection's signal phase.

- **Dueling** architecture (see backends/*) separates "how good is this
  state" from "how much better is action A than the average action here" --
  helps a lot when many actions have similar value (e.g. queues are short
  on every approach, so which phase runs barely matters).
- **Double DQN**: the online network picks the best next action, but the
  *target* network evaluates it. This decouples action selection from
  value estimation and is a standard, well-validated fix for DQN's
  well-known overestimation bias.
- **Prioritized replay** (see replay_buffer.py) focuses learning on
  surprising transitions.

One `IntersectionDQNAgent` instance is created per intersection during
training/inference. Weights can optionally be shared across intersections
of the same "type" (see `scripts/train_dqn.py --share-weights`) to scale to
a full city without training hundreds of independent networks.
"""
from __future__ import annotations

import numpy as np

from traffic_system.agents.backends import make_backend
from traffic_system.agents.base_agent import Agent
from traffic_system.agents.replay_buffer import PrioritizedReplayBuffer
from traffic_system.common.config import get_settings
from traffic_system.common.logging import get_logger

logger = get_logger(__name__)


class IntersectionDQNAgent(Agent):
    name = "dqn_agent"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        intersection_id: str = "generic",
        backend: str | None = None,
        seed: int | None = None,
    ) -> None:
        settings = get_settings()
        self.intersection_id = intersection_id
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = settings.gamma
        self.batch_size = settings.batch_size
        self.target_sync_every_steps = settings.target_sync_every_steps

        self.epsilon = settings.epsilon_start
        self.epsilon_end = settings.epsilon_end
        self.epsilon_decay = (settings.epsilon_start - settings.epsilon_end) / max(settings.epsilon_decay_steps, 1)

        backend_name = backend or settings.qnet_backend
        self.online = make_backend(backend_name, obs_dim, n_actions, lr=settings.learning_rate, seed=seed)
        self.target = make_backend(backend_name, obs_dim, n_actions, lr=settings.learning_rate, seed=seed)
        self.online.hard_update_to(self.target)

        self.buffer = PrioritizedReplayBuffer(settings.replay_buffer_size, obs_dim, seed=seed)
        self._train_steps = 0
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        q = self.online.predict(obs.reshape(1, -1))
        return int(np.argmax(q[0]))

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards = batch["states"], batch["actions"], batch["rewards"]
        next_states, dones, weights = batch["next_states"], batch["dones"], batch["weights"]

        next_q_online = self.online.predict(next_states)
        best_next_actions = np.argmax(next_q_online, axis=1)
        next_q_target = self.target.predict(next_states)
        next_q_selected = next_q_target[np.arange(len(next_q_target)), best_next_actions]

        targets = rewards + self.gamma * next_q_selected * (~dones)

        current_q = self.online.predict(states)
        current_q_taken = current_q[np.arange(len(current_q)), actions]
        td_errors = targets - current_q_taken

        loss = self.online.train_step(states, actions, targets, sample_weights=weights)
        self.buffer.update_priorities(batch["indices"], td_errors)

        self._train_steps += 1
        if self._train_steps % self.target_sync_every_steps == 0:
            self.online.hard_update_to(self.target)

        self.decay_epsilon()
        return loss

    def save(self, path: str) -> None:
        self.online.save(path)

    def load(self, path: str) -> None:
        self.online.load(path)
        self.online.hard_update_to(self.target)
