"""Single-intersection queueing model and the safety layer that sits
between the RL agent's requested phase and what the signal actually does.

Design note: the DQN agent never controls the signal directly. It requests
a phase; `SafetyLayer.resolve` enforces minimum/maximum green time and
inserts the mandatory yellow + all-red clearance interval before any phase
change takes effect. This mirrors how real traffic controllers are built
(NEMA/ITS standards mandate hard timing floors and ceilings) and means a
buggy or adversarial model literally cannot create an unsafe signal
sequence -- it can only pick *when*, within safe bounds, to switch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from traffic_system.common.schemas import SignalPhase, VehicleClass

APPROACHES: tuple[str, ...] = ("N", "S", "E", "W")

PHASE_APPROACHES: dict[SignalPhase, tuple[str, ...]] = {
    SignalPhase.NS_THROUGH: ("N", "S"),
    SignalPhase.NS_LEFT: ("N", "S"),
    SignalPhase.EW_THROUGH: ("E", "W"),
    SignalPhase.EW_LEFT: ("E", "W"),
    SignalPhase.ALL_RED: (),
}

# Fraction of a through-queue that is left-turning traffic; only discharged
# during the corresponding protected *_LEFT phase. Kept as a simple constant
# for the base simulation -- swap for measured turn ratios when real
# intersection data is available.
LEFT_TURN_FRACTION = 0.18

# Standard traffic-engineering saturation flow assumption: ~1800 veh/hr/lane.
DEFAULT_SATURATION_FLOW_PER_LANE_PER_S = 1800 / 3600


@dataclass
class ApproachState:
    lanes: int = 2
    queue: float = 0.0
    saturation_flow_per_lane_per_s: float = DEFAULT_SATURATION_FLOW_PER_LANE_PER_S
    emergency_present: bool = False
    emergency_class: VehicleClass | None = None

    @property
    def discharge_rate_per_s(self) -> float:
        return self.lanes * self.saturation_flow_per_lane_per_s

    @property
    def avg_wait_s(self) -> float:
        """M/M/1-style approximation: expected wait scales with queue length
        over service rate. Good enough as a learnable reward signal without
        needing per-vehicle arrival timestamps."""
        rate = self.discharge_rate_per_s
        return self.queue / rate if rate > 0 else 0.0


@dataclass
class IntersectionState:
    intersection_id: str
    approaches: dict[str, ApproachState] = field(default_factory=dict)
    current_phase: SignalPhase = SignalPhase.NS_THROUGH
    time_in_phase_s: float = 0.0
    transitioning: bool = False
    transition_remaining_s: float = 0.0
    pending_phase: SignalPhase | None = None
    total_discharged: int = 0
    total_switches: int = 0

    def total_queue(self) -> float:
        return sum(a.queue for a in self.approaches.values())

    def has_emergency(self) -> bool:
        return any(a.emergency_present for a in self.approaches.values())


class SafetyLayer:
    """Mediates every phase-change request against hard timing constraints."""

    def __init__(self, min_green_s: float, max_green_s: float, yellow_s: float, all_red_s: float) -> None:
        self.min_green_s = min_green_s
        self.max_green_s = max_green_s
        self.yellow_s = yellow_s
        self.all_red_s = all_red_s

    def resolve(self, state: IntersectionState, requested_phase: SignalPhase, dt: float, force_emergency_phase: SignalPhase | None = None) -> IntersectionState:
        """Advance one simulation step's worth of phase-timing logic.

        Returns the (mutated) state with `current_phase` set to what is
        actually active *this* step, after applying min/max green and the
        yellow/all-red clearance. `force_emergency_phase`, when set, bypasses
        min-green (but never bypasses the yellow/all-red clearance -- that
        would be unsafe even for an ambulance).
        """
        if state.transitioning:
            state.transition_remaining_s -= dt
            if state.transition_remaining_s <= 0:
                state.current_phase = state.pending_phase or state.current_phase
                state.pending_phase = None
                state.transitioning = False
                state.time_in_phase_s = 0.0
                state.total_switches += 1
            return state

        state.time_in_phase_s += dt

        target = force_emergency_phase or requested_phase
        must_switch_for_max_green = state.time_in_phase_s >= self.max_green_s
        may_switch = state.time_in_phase_s >= self.min_green_s or force_emergency_phase is not None

        wants_switch = target != state.current_phase and (may_switch or must_switch_for_max_green)

        if wants_switch:
            state.transitioning = True
            state.transition_remaining_s = self.yellow_s + self.all_red_s
            state.pending_phase = target if target != SignalPhase.ALL_RED else state.current_phase

        return state
