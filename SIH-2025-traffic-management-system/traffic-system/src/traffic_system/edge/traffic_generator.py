"""Standalone synthetic traffic generator used by the simulated edge agent
when no real camera is present.

Deliberately reuses `env.intersection.ApproachState`/`IntersectionState`
(the exact same queueing physics the training environment uses) so
telemetry an edge agent publishes in "live demo" mode is dynamically
consistent with what the brain was trained against. Unlike the training
environment, this generator does not choose its own action -- it receives
real `SignalPhase` commands (published by the brain over MQTT) and evolves
the queues accordingly, exactly like a real intersection would react to a
real signal.
"""
from __future__ import annotations

import numpy as np

from traffic_system.common.schemas import SignalPhase, VehicleClass
from traffic_system.env.intersection import ApproachState, IntersectionState

APPROACH_ORDER: tuple[str, ...] = ("N", "S", "E", "W")
EMERGENCY_CLASSES = (VehicleClass.AMBULANCE, VehicleClass.FIRE_TRUCK, VehicleClass.POLICE)


class SyntheticTrafficGenerator:
    def __init__(
        self,
        intersection_id: str,
        lanes: int = 2,
        base_arrival_rate_veh_s: float = 0.12,
        arrival_cycle_period_s: float = 1800.0,
        emergency_probability_per_tick: float = 0.001,
        seed: int | None = None,
    ) -> None:
        self.intersection_id = intersection_id
        self.state = IntersectionState(
            intersection_id, approaches={d: ApproachState(lanes=lanes) for d in APPROACH_ORDER}
        )
        self.base_arrival_rate_veh_s = base_arrival_rate_veh_s
        self.arrival_cycle_period_s = arrival_cycle_period_s
        self.emergency_probability_per_tick = emergency_probability_per_tick
        self._rng = np.random.default_rng(seed)
        self._t = 0.0

    def tick(self, phase: SignalPhase, dt: float) -> None:
        """Advance the simulated queues by `dt` seconds under the given
        (already safety-validated) active phase."""
        for direction in APPROACH_ORDER:
            approach = self.state.approaches[direction]
            variation = 0.3 * self.base_arrival_rate_veh_s * np.sin(2 * np.pi * self._t / self.arrival_cycle_period_s)
            rate = max(self.base_arrival_rate_veh_s + variation, 0.0)
            approach.queue += self._rng.poisson(rate * dt)
            if not approach.emergency_present and self._rng.random() < self.emergency_probability_per_tick:
                approach.emergency_present = True
                approach.emergency_class = EMERGENCY_CLASSES[self._rng.integers(0, len(EMERGENCY_CLASSES))]

        active_approaches: tuple[str, ...] = ()
        if phase == SignalPhase.NS_THROUGH:
            active_approaches = ("N", "S")
        elif phase == SignalPhase.EW_THROUGH:
            active_approaches = ("E", "W")

        for direction in active_approaches:
            approach = self.state.approaches[direction]
            served = min(approach.queue, approach.discharge_rate_per_s * dt)
            approach.queue -= served
            if approach.emergency_present and served > 0:
                approach.emergency_present = False
                approach.emergency_class = None

        self.state.current_phase = phase
        self._t += dt

    def trigger_emergency(self, direction: str, vehicle_class: VehicleClass = VehicleClass.AMBULANCE) -> None:
        """Manually inject an emergency vehicle -- used by demo scripts to
        showcase preemption on demand rather than waiting for a random event."""
        self.state.approaches[direction].emergency_present = True
        self.state.approaches[direction].emergency_class = vehicle_class

    def trigger_congestion(self, direction: str, extra_vehicles: float) -> None:
        """Manually spike a queue -- used by demo scripts to showcase the
        route allocation and incident detection agents on demand."""
        self.state.approaches[direction].queue += extra_vehicles
