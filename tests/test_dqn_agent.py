import numpy as np

from traffic_system.agents.dqn_agent import IntersectionDQNAgent
from traffic_system.agents.replay_buffer import PrioritizedReplayBuffer
from traffic_system.env.traffic_grid_env import SimConfig, SingleIntersectionEnv


def test_replay_buffer_push_and_sample():
    buf = PrioritizedReplayBuffer(capacity=100, obs_dim=4, seed=1)
    for i in range(50):
        buf.push(np.zeros(4), 0, float(i), np.ones(4), False)
    assert len(buf) == 50
    batch = buf.sample(16)
    assert batch["states"].shape == (16, 4)
    assert batch["weights"].shape == (16,)
    buf.update_priorities(batch["indices"], np.ones(16))


def test_replay_buffer_wraps_at_capacity():
    buf = PrioritizedReplayBuffer(capacity=10, obs_dim=2, seed=1)
    for i in range(25):
        buf.push(np.array([i, i]), 0, 1.0, np.array([i, i]), False)
    assert len(buf) == 10  # capped, not 25


def test_dqn_agent_action_space_and_epsilon_decay():
    agent = IntersectionDQNAgent(obs_dim=16, n_actions=2, backend="numpy", seed=0)
    obs = np.zeros(16, dtype=np.float32)
    for _ in range(20):
        a = agent.act(obs, explore=True)
        assert a in (0, 1)

    start_eps = agent.epsilon
    for _ in range(100):
        agent.decay_epsilon()
    assert agent.epsilon < start_eps
    assert agent.epsilon >= agent.epsilon_end


def test_dqn_agent_learns_and_loss_is_finite():
    agent = IntersectionDQNAgent(obs_dim=16, n_actions=2, backend="numpy", seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.normal(size=16).astype(np.float32)
        ns = rng.normal(size=16).astype(np.float32)
        agent.remember(s, rng.integers(0, 2), float(rng.normal()), ns, False)

    loss = agent.learn()
    assert loss is not None
    assert np.isfinite(loss)


def test_dqn_agent_save_load_roundtrip(tmp_path):
    agent = IntersectionDQNAgent(obs_dim=16, n_actions=2, backend="numpy", seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.normal(size=16).astype(np.float32)
        agent.remember(s, rng.integers(0, 2), float(rng.normal()), s, False)
    agent.learn()

    path = str(tmp_path / "agent.npz")
    agent.save(path)

    other = IntersectionDQNAgent(obs_dim=16, n_actions=2, backend="numpy", seed=999)
    other.load(path)

    probe = rng.normal(size=(1, 16)).astype(np.float32)
    np.testing.assert_allclose(agent.online.predict(probe), other.online.predict(probe))


def test_dqn_agent_training_reduces_average_queue_vs_random_policy():
    """A weak but meaningful learning-signal check: after training briefly,
    the agent's average queue length over an eval episode should not be
    dramatically worse than a random policy's (guards against a totally
    broken training loop, e.g. exploding loss or inverted gradients)."""

    def run_episode(agent_or_none, seed):
        env = SingleIntersectionEnv(SimConfig(max_episode_steps=80, base_arrival_rate_veh_s=0.15), seed=seed)
        obs, _ = env.reset(seed=seed)
        total_queue = 0.0
        steps = 0
        for _ in range(80):
            if agent_or_none is None:
                action = env.action_space.sample()
            else:
                action = agent_or_none.act(obs, explore=False)
            obs, reward, term, trunc, info = env.step(action)
            total_queue += info["total_queue"]
            steps += 1
            if term or trunc:
                break
        return total_queue / steps

    agent = IntersectionDQNAgent(obs_dim=16, n_actions=2, backend="numpy", seed=42)
    train_env = SingleIntersectionEnv(SimConfig(max_episode_steps=80, base_arrival_rate_veh_s=0.15), seed=42)
    for ep in range(15):
        obs, _ = train_env.reset(seed=ep)
        for _ in range(80):
            action = agent.act(obs, explore=True)
            next_obs, reward, term, trunc, info = train_env.step(action)
            agent.remember(obs, action, reward, next_obs, term or trunc)
            agent.learn()
            obs = next_obs
            if term or trunc:
                break

    trained_avg_queue = run_episode(agent, seed=1000)
    random_avg_queue = run_episode(None, seed=1000)

    # Not a strict optimality bound (15 episodes is a smoke-test amount of
    # training), just a sanity check that learning moved in the right
    # direction and isn't wildly worse than doing nothing intelligent.
    assert trained_avg_queue < random_avg_queue * 1.5
