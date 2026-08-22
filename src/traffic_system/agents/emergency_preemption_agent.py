"""Rule-based emergency vehicle preemption -- highest priority in the brain.

This agent is intentionally *not* learned. When the edge CV pipeline flags
an ambulance, fire truck, or police vehicle on an approach, this agent
forces that intersection's signal to serve that approach immediately,
bypassing the DQN's suggestion entirely (subject only to the hard
yellow/all-red safety clearance -- see `SafetyLayer`, which even an
emergency override cannot skip). A rule-based approach is deliberate here:
this is a safety-critical path and its behavior needs to be simple, 100%
predictable, and auditable, not something a reward function could
theoretically get wrong.

It also projects a *soft* preemption corridor: the next `corridor_depth`
intersections the vehicle is heading toward. Those are not force-switched
(we don't have a precise ETA yet -- see the module-level TODO), but
`brain_service.py` uses the returned bonus to nudge each corridor
intersection's action selection toward the matching phase.
"""
from __future__ import annotations

from traffic_system.agents.base_agent import Agent
from traffic_system.common.schemas import EmergencyAlert, SignalPhase, VehicleClass
from traffic_system.env.city_graph import CityGraph
from traffic_system.env.intersection import IntersectionState

# TODO(future work): replace the fixed corridor_depth soft-bias below with an
# ETA-based model (distance / current approach speed) so corridor
# intersections are only nudged shortly before the vehicle is actually
# projected to arrive, instead of for the entire time it's anywhere upstream.


class EmergencyPreemptionAgent(Agent):
    name = "emergency_preemption"

    def __init__(self, city_graph: CityGraph, corridor_depth: int = 2, corridor_bias: float = 0.3) -> None:
        self.city_graph = city_graph
        self.corridor_depth = corridor_depth
        self.corridor_bias = corridor_bias

    def act(
        self, states: dict[str, IntersectionState]
    ) -> tuple[dict[str, SignalPhase], dict[str, tuple[SignalPhase, float]], list[EmergencyAlert]]:
        """Returns:
        - forced_phases: {intersection_id: SignalPhase} -- hard override, apply unconditionally
        - corridor_bias: {intersection_id: (preferred_phase, strength in [0,1])} -- soft nudge,
          applied by brain_service only when the DQN's own choice disagrees
        - alerts: one EmergencyAlert per detected emergency vehicle, for the dashboard/API
        """
        forced_phases: dict[str, SignalPhase] = {}
        corridor_bias: dict[str, tuple[SignalPhase, float]] = {}
        alerts: list[EmergencyAlert] = []

        for intersection_id, state in states.items():
            for direction, approach in state.approaches.items():
                if not approach.emergency_present:
                    continue

                phase = SignalPhase.NS_THROUGH if direction in ("N", "S") else SignalPhase.EW_THROUGH
                forced_phases[intersection_id] = phase

                corridor = self._project_corridor(intersection_id, direction)
                for node in corridor:
                    existing = corridor_bias.get(node)
                    if existing is None or self.corridor_bias > existing[1]:
                        corridor_bias[node] = (phase, self.corridor_bias)

                alerts.append(
                    EmergencyAlert(
                        intersection_id=intersection_id,
                        vehicle_class=approach.emergency_class or VehicleClass.AMBULANCE,
                        approach=direction,
                        confidence=0.95,
                        corridor=corridor,
                    )
                )

        return forced_phases, corridor_bias, alerts

    def _project_corridor(self, intersection_id: str, direction: str) -> list[str]:
        """Walk downstream in the vehicle's direction of travel for `corridor_depth` hops."""
        corridor: list[str] = []
        current = intersection_id
        for _ in range(self.corridor_depth):
            nxt = self.city_graph.direction_neighbors(current)[self.city_graph.opposite(direction)]
            if nxt is None:
                break
            corridor.append(nxt)
            current = nxt
        return corridor
