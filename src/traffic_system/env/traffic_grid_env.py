"""Traffic simulation environments used for training and for the live demo.

Two environments are provided, deliberately:

`SingleIntersectionEnv` -- a standard `gymnasium.Env` (single agent, one
intersection, exogenous Poisson arrivals on all four approaches, vehicles
exit the system on discharge). This is the fast, fully spec-compliant
Gymnasium environment used to train and unit-test one DQN agent in
isolation, the way most traffic-RL papers benchmark a single controller.

`TrafficGridEnv` -- composes N `SingleIntersectionEnv`-style intersections
over the real city graph topology (see `city_graph.py`), routing vehicles
that are discharged from one intersection into the correct downstream
approach of its neighbor after a free-flow travel delay. This is what lets
a "green wave" emerge and is what the Coordinator agent and the full brain
service actually run against. It exposes a PettingZoo-`ParallelEnv`-style
API (dict-in / dict-out) rather than subclassing `gymnasium.Env`, because
gymnasium itself is a single-agent spec.

Both share the same per-approach queueing physics (see `intersection.py`)
so a policy trained on `SingleIntersectionEnv` is a drop-in prior for
`TrafficGridEnv`.

Modeling simplifications (documented, not hidden):
- Only two phases are simulated (NS_THROUGH / EW_THROUGH). Protected left
  turns (`SignalPhase.NS_LEFT` / `EW_LEFT`) exist in the schema and safety
  layer for intersections with dedicated turn lanes, but the discharge
  physics for them is a documented extension point, not yet implemented.
- Vehicles that would turn are not separately modeled; every discharged
  vehicle continues straight through to the corresponding downstream
  approach, or exits the network at a grid boundary.
- Average wait time uses an M/M/1-style queue-length/service-rate proxy
  rather than per-vehicle timestamps -- standard practice for a training
  signal, not a microsimulation-grade estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from traffic_system.common.schemas import SignalPhase, VehicleClass
from traffic_system.env.city_graph import CityGraph
from traffic_system.env.intersection import ApproachState, IntersectionState, SafetyLayer

APPROACH_ORDER = ("N", "S", "E", "W")
ACTION_PHASES = (SignalPhase.NS_THROUGH, SignalPhase.EW_THROUGH)
OBS_DIM = 16

W_QUEUE = 0.25
W_WAIT = 0.05
W_THROUGHPUT = 1.0
W_SWITCH = 2.0
W_EMERGENCY_WAIT = 5.0

EMERGENCY_CLASSES = (VehicleClass.AMBULANCE, VehicleClass.FIRE_TRUCK, VehicleClass.POLICE)


def build_observation(state: IntersectionState, max_green_s: float) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    for i, d in enumerate(APPROACH_ORDER):
        a = state.approaches[d]
        obs[i] = min(a.queue / 50.0, 3.0)
        obs[4 + i] = min(a.avg_wait_s / 60.0, 3.0)
        obs[8 + i] = 1.0 if a.emergency_present else 0.0
    obs[12] = 1.0 if state.current_phase == SignalPhase.NS_THROUGH else 0.0
    obs[13] = 1.0 if state.current_phase == SignalPhase.EW_THROUGH else 0.0
    obs[14] = 1.0 if state.transitioning else 0.0
    obs[15] = min(state.time_in_phase_s / max_green_s, 1.0)
    return obs


def _arrival_rate(base_rate: float, t: float, cycle_period_s: float, rng: np.random.Generator) -> float:
    variation = 0.5 * base_rate * math.sin(2 * math.pi * t / cycle_period_s)
    noise = rng.normal(0, 0.1 * base_rate)
    return max(base_rate + variation + noise, 0.0)


def _step_physics(
    state: IntersectionState,
    safety: SafetyLayer,
    action_index: int,
    dt: float,
    force_emergency_phase: SignalPhase | None,
) -> tuple[IntersectionState, dict[str, float], bool]:
    """Advance one intersection's signal + queues by `dt` seconds.

    Returns (state, discharged_per_approach, did_switch_this_step).
    """
    requested_phase = ACTION_PHASES[action_index]
    was_transitioning = state.transitioning
    state = safety.resolve(state, requested_phase, dt, force_emergency_phase)
    did_switch = (not was_transitioning) and state.transitioning

    discharged: dict[str, float] = {d: 0.0 for d in APPROACH_ORDER}
    if not state.transitioning:
        active_approaches = (
            ("N", "S") if state.current_phase == SignalPhase.NS_THROUGH else ("E", "W")
        )
        for d in active_approaches:
            a = state.approaches[d]
            served = min(a.queue, a.discharge_rate_per_s * dt)
            a.queue -= served
            discharged[d] = served
            state.total_discharged += int(served)
            if a.emergency_present and served > 0:
                a.emergency_present = False
                a.emergency_class = None

    return state, discharged, did_switch


def _reward(state: IntersectionState, discharged: dict[str, float], did_switch: bool) -> float:
    total_queue = state.total_queue()
    total_wait = sum(a.avg_wait_s for a in state.approaches.values())
    throughput = sum(discharged.values())
    has_emergency = state.has_emergency()
    return (
        -W_QUEUE * total_queue
        - W_WAIT * total_wait
        + W_THROUGHPUT * throughput
        - (W_SWITCH if did_switch else 0.0)
        - (W_EMERGENCY_WAIT if has_emergency else 0.0)
    )


@dataclass
class SimConfig:
    dt_s: float = 10.0
    min_green_s: float = 10.0
    max_green_s: float = 90.0
    yellow_s: float = 3.0
    all_red_s: float = 2.0
    base_arrival_rate_veh_s: float = 0.12
    arrival_cycle_period_s: float = 1800.0
    emergency_probability_per_step: float = 0.0015
    lanes_per_approach: int = 2
    max_episode_steps: int = 360  # 1 hour of simulated time at dt=10s


class SingleIntersectionEnv(gym.Env):
    """Standard-conforming Gymnasium environment for one isolated intersection."""

    metadata = {"render_modes": []}

    def __init__(self, config: SimConfig | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.config = config or SimConfig()
        self.action_space = spaces.Discrete(len(ACTION_PHASES))
        self.observation_space = spaces.Box(low=0.0, high=3.0, shape=(OBS_DIM,), dtype=np.float32)
        self._safety = SafetyLayer(
            self.config.min_green_s, self.config.max_green_s, self.config.yellow_s, self.config.all_red_s
        )
        self._rng = np.random.default_rng(seed)
        self._state = self._new_state()
        self._t = 0.0
        self._step_count = 0

    def _new_state(self) -> IntersectionState:
        approaches = {
            d: ApproachState(lanes=self.config.lanes_per_approach) for d in APPROACH_ORDER
        }
        return IntersectionState(intersection_id="standalone", approaches=approaches)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._new_state()
        self._t = 0.0
        self._step_count = 0
        obs = build_observation(self._state, self.config.max_green_s)
        return obs, {}

    def step(self, action: int):
        for d in APPROACH_ORDER:
            a = self._state.approaches[d]
            rate = _arrival_rate(self.config.base_arrival_rate_veh_s, self._t, self.config.arrival_cycle_period_s, self._rng)
            new_vehicles = self._rng.poisson(rate * self.config.dt_s)
            a.queue += new_vehicles
            if (not a.emergency_present) and self._rng.random() < self.config.emergency_probability_per_step:
                a.emergency_present = True
                a.emergency_class = EMERGENCY_CLASSES[self._rng.integers(0, len(EMERGENCY_CLASSES))]

        self._state, discharged, did_switch = _step_physics(
            self._state, self._safety, int(action), self.config.dt_s, force_emergency_phase=None
        )
        reward = _reward(self._state, discharged, did_switch)

        self._t += self.config.dt_s
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.config.max_episode_steps

        obs = build_observation(self._state, self.config.max_green_s)
        info = {
            "total_queue": self._state.total_queue(),
            "has_emergency": self._state.has_emergency(),
            "total_discharged": self._state.total_discharged,
        }
        return obs, reward, terminated, truncated, info


class TrafficGridEnv:
    """Multi-intersection network simulation over the real city graph topology.

    Boundary approaches (no neighbor in that compass direction) are sources
    of exogenous Poisson arrivals and sinks for completed trips. Interior
    approaches only receive vehicles routed from an upstream intersection's
    discharge, after a free-flow travel delay -- this coupling is what
    makes network-level coordination (the Coordinator agent) meaningful.
    """

    def __init__(self, city_graph: CityGraph, config: SimConfig | None = None, seed: int | None = None) -> None:
        self.city_graph = city_graph
        self.config = config or SimConfig()
        self.agents = city_graph.intersection_ids
        self._safety = SafetyLayer(
            self.config.min_green_s, self.config.max_green_s, self.config.yellow_s, self.config.all_red_s
        )
        self._rng = np.random.default_rng(seed)
        self._states: dict[str, IntersectionState] = {}
        self._in_transit: list[tuple[float, str, str, float]] = []  # (arrival_t, dest_node, dest_approach, count)
        self._t = 0.0
        self._step_count = 0
        self.reset(seed=seed)

    def _new_states(self) -> dict[str, IntersectionState]:
        states = {}
        for iid in self.agents:
            approaches = {d: ApproachState(lanes=self.config.lanes_per_approach) for d in APPROACH_ORDER}
            states[iid] = IntersectionState(intersection_id=iid, approaches=approaches)
        return states

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._states = self._new_states()
        self._in_transit = []
        self._t = 0.0
        self._step_count = 0
        return {iid: build_observation(s, self.config.max_green_s) for iid, s in self._states.items()}

    def _is_boundary(self, iid: str, direction: str) -> bool:
        return self.city_graph.direction_neighbors(iid)[direction] is None

    def _route_downstream(self, iid: str, direction: str, count: float) -> None:
        if count <= 0:
            return
        downstream = self.city_graph.direction_neighbors(iid)[self.city_graph.opposite(direction)]
        if downstream is None:
            return  # exits the network at the grid boundary -- trip completed
        travel_time = self.city_graph.road(iid, downstream).free_flow_travel_time_s
        self._in_transit.append((self._t + travel_time, downstream, direction, count))

    def step(
        self, actions: dict[str, int], forced_phases: dict[str, SignalPhase] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        """`forced_phases` lets the Emergency Preemption agent override an
        intersection's action for this step regardless of what the DQN chose."""
        forced_phases = forced_phases or {}

        # 1. Exogenous arrivals at network boundary approaches.
        for iid, state in self._states.items():
            for d in APPROACH_ORDER:
                if not self._is_boundary(iid, d):
                    continue
                a = state.approaches[d]
                rate = _arrival_rate(self.config.base_arrival_rate_veh_s, self._t, self.config.arrival_cycle_period_s, self._rng)
                new_vehicles = self._rng.poisson(rate * self.config.dt_s)
                a.queue += new_vehicles
                if (not a.emergency_present) and self._rng.random() < self.config.emergency_probability_per_step:
                    a.emergency_present = True
                    a.emergency_class = EMERGENCY_CLASSES[self._rng.integers(0, len(EMERGENCY_CLASSES))]

        # 2. Deliver in-transit vehicles that have arrived by now.
        still_in_transit = []
        for arrival_t, dest_node, dest_approach, count in self._in_transit:
            if arrival_t <= self._t:
                self._states[dest_node].approaches[dest_approach].queue += count
            else:
                still_in_transit.append((arrival_t, dest_node, dest_approach, count))
        self._in_transit = still_in_transit

        # 3. Advance signal physics + discharge + route onward.
        rewards, terminations, truncations, infos = {}, {}, {}, {}
        for iid, state in self._states.items():
            action_index = actions.get(iid, 0)
            force = forced_phases.get(iid)
            state, discharged, did_switch = _step_physics(state, self._safety, action_index, self.config.dt_s, force)
            for d, qty in discharged.items():
                self._route_downstream(iid, d, qty)
            rewards[iid] = _reward(state, discharged, did_switch)
            terminations[iid] = False
            infos[iid] = {
                "total_queue": state.total_queue(),
                "has_emergency": state.has_emergency(),
                "total_discharged": state.total_discharged,
            }

        self._t += self.config.dt_s
        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps
        for iid in self.agents:
            truncations[iid] = truncated

        obs = {iid: build_observation(s, self.config.max_green_s) for iid, s in self._states.items()}
        return obs, rewards, terminations, truncations, infos

    @property
    def states(self) -> dict[str, IntersectionState]:
        return self._states
