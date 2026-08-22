"""Network-level coordinator that arbitrates between independent per-
intersection DQN proposals.

Each `IntersectionDQNAgent` only sees its own queues -- it has no idea
whether the road it's about to send traffic down is already backed up. The
coordinator adds exactly one thing a purely local agent cannot see:
**spillback risk**. If serving a phase would discharge vehicles onto a
downstream link that is already saturated, sending more traffic into it
doesn't help throughput, it just moves the jam one intersection over (and
in the worst case, blocks the upstream intersection too, a classic gridlock
failure mode). This is a simplified version of the "max-pressure" family of
network traffic-signal control algorithms (Varaiya, 2013).

This is deliberately a *thin* rule sitting on top of the DQN's judgement,
not a second learned model -- it only overrides when the alternative phase
is clearly safer, and always defers to the DQN otherwise.
"""
from __future__ import annotations

from traffic_system.agents.base_agent import Agent
from traffic_system.common.logging import get_logger
from traffic_system.env.city_graph import CityGraph
from traffic_system.env.intersection import PHASE_APPROACHES, IntersectionState
from traffic_system.env.traffic_grid_env import ACTION_PHASES

logger = get_logger(__name__)


class CoordinatorAgent(Agent):
    name = "coordinator"

    def __init__(self, city_graph: CityGraph, spillback_queue_threshold: float = 40.0) -> None:
        self.city_graph = city_graph
        self.spillback_queue_threshold = spillback_queue_threshold

    def _phase_approaches(self, action_index: int) -> tuple[str, ...]:
        return PHASE_APPROACHES[ACTION_PHASES[action_index]]

    def _is_spillback_risk(self, intersection_id: str, action_index: int, states: dict[str, IntersectionState]) -> bool:
        for d in self._phase_approaches(action_index):
            downstream = self.city_graph.direction_neighbors(intersection_id)[self.city_graph.opposite(d)]
            if downstream is None or downstream not in states:
                continue
            if states[downstream].approaches[d].queue >= self.spillback_queue_threshold:
                return True
        return False

    def act(
        self, proposed_actions: dict[str, int], states: dict[str, IntersectionState]
    ) -> tuple[dict[str, int], dict[str, str]]:
        """Returns (final_actions, override_reasons). `override_reasons` only
        contains entries for intersections the coordinator actually changed."""
        final_actions: dict[str, int] = {}
        reasons: dict[str, str] = {}

        for intersection_id, action in proposed_actions.items():
            if intersection_id not in states:
                final_actions[intersection_id] = action
                continue

            alternative = 1 - action
            proposed_risky = self._is_spillback_risk(intersection_id, action, states)
            alternative_risky = self._is_spillback_risk(intersection_id, alternative, states)

            if proposed_risky and not alternative_risky:
                final_actions[intersection_id] = alternative
                reasons[intersection_id] = "spillback_prevention"
                logger.info("coordinator.override", intersection_id=intersection_id, reason="spillback_prevention")
            else:
                final_actions[intersection_id] = action

        return final_actions, reasons
