
from traffic_system.env.traffic_grid_env import SimConfig, SingleIntersectionEnv, TrafficGridEnv


def test_single_intersection_env_obs_and_reward_shapes():
    env = SingleIntersectionEnv(SimConfig(max_episode_steps=20), seed=1)
    obs, info = env.reset(seed=1)
    assert obs.shape == (16,)
    assert env.action_space.n == 2

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (16,)
        assert isinstance(reward, float)
        if terminated or truncated:
            break
    assert truncated  # episode should end via the step cap, not an error


def test_single_intersection_queues_never_go_negative():
    env = SingleIntersectionEnv(SimConfig(max_episode_steps=100, base_arrival_rate_veh_s=0.3), seed=3)
    env.reset(seed=3)
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        for a in env._state.approaches.values():
            assert a.queue >= 0
        if terminated or truncated:
            break


def test_traffic_grid_env_matches_city_graph_topology(city_graph):
    env = TrafficGridEnv(city_graph, SimConfig(max_episode_steps=10), seed=1)
    obs = env.reset(seed=1)
    assert set(obs.keys()) == set(city_graph.intersection_ids)

    # I_B2 sits in the middle of the grid; every direction should resolve to a real neighbor.
    neighbors = city_graph.direction_neighbors("I_B2")
    assert neighbors == {"N": "I_C2", "S": "I_A2", "E": "I_B3", "W": "I_B1"}

    # Corner intersection I_A1 should have exactly two real neighbors and two boundary (None) directions.
    corner_neighbors = city_graph.direction_neighbors("I_A1")
    assert sum(v is not None for v in corner_neighbors.values()) == 2


def test_traffic_grid_env_routes_discharged_vehicles_downstream(city_graph):
    """A vehicle discharged from one intersection's approach must eventually
    show up on the correct downstream intersection's matching approach."""
    env = TrafficGridEnv(city_graph, SimConfig(max_episode_steps=50, base_arrival_rate_veh_s=0.0), seed=5)
    env.reset(seed=5)

    # Manually inject a queue at I_B2's S approach and force NS_THROUGH so it discharges.
    env.states["I_B2"].approaches["S"].queue = 20.0
    actions = {iid: 0 for iid in env.agents}  # 0 = NS_THROUGH for everyone
    # Hold I_C2 on EW_THROUGH (red for N/S) so arriving vehicles accumulate in its
    # queue instead of immediately discharging onward in the same step -- otherwise
    # we'd just be observing a (correct!) green-wave pass-through with nothing to see.
    actions["I_C2"] = 1

    max_seen = 0.0
    for _ in range(30):
        env.step(actions)
        max_seen = max(max_seen, env.states["I_C2"].approaches["S"].queue)

    # I_C2 is I_B2's N-neighbor; discharged S-approach traffic from I_B2 should
    # arrive at I_C2's S approach after the free-flow travel delay and sit there
    # (since I_C2 is deliberately held red for that direction).
    assert max_seen > 0, "vehicles discharged from I_B2's S approach never reached I_C2's S approach"
    assert max_seen == 20.0  # both discharge batches (10 + 10) should have arrived intact


def test_safety_layer_enforces_min_green(city_graph):
    """Requesting a phase change immediately after a switch should be ignored
    until min_green_s has elapsed."""
    cfg = SimConfig(dt_s=5.0, min_green_s=20.0, max_episode_steps=10)
    env = SingleIntersectionEnv(cfg, seed=1)
    env.reset(seed=1)
    starting_phase = env._state.current_phase

    # Immediately request the opposite phase every step; with dt=5s and
    # min_green=20s it should take at least 4 steps before a switch begins.
    switched_at = None
    for step_i in range(10):
        opposite_action = 1  # request EW_THROUGH regardless of current phase
        env.step(opposite_action)
        if env._state.transitioning or env._state.current_phase != starting_phase:
            switched_at = step_i
            break
    assert switched_at is None or switched_at >= 3  # 4th step (index 3) is the earliest min_green allows
