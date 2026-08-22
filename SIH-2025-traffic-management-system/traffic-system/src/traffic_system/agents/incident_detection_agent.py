"""Anomaly detection over each intersection's telemetry history.

Heuristic, not learned: flags an intersection whose queue is growing
significantly faster than it is being discharged, even though it is
receiving green time -- the signature of a blocked lane, a stalled
vehicle, or a minor collision rather than ordinary demand. A purely
reactive DQN agent would eventually "notice" via falling reward, but by
then the queue is already long; this agent exists to surface the
*diagnosis* ("something is physically wrong here, not just busy") early
enough for a human operator or the route allocation agent to act on it.
"""
from __future__ import annotations

from collections import deque

from traffic_system.agents.base_agent import Agent
from traffic_system.common.schemas import IncidentAlert
from traffic_system.env.intersection import IntersectionState


class IncidentDetectionAgent(Agent):
    name = "incident_detection"

    def __init__(self, window: int = 6, min_absolute_queue: float = 8.0, stall_discharge_ratio: float = 0.25) -> None:
        self.window = window
        self.min_absolute_queue = min_absolute_queue
        self.stall_discharge_ratio = stall_discharge_ratio
        self._queue_history: dict[str, deque[float]] = {}
        self._discharge_history: dict[str, deque[int]] = {}

    def act(self, intersection_id: str, state: IntersectionState) -> IncidentAlert | None:
        queue_hist = self._queue_history.setdefault(intersection_id, deque(maxlen=self.window))
        discharge_hist = self._discharge_history.setdefault(intersection_id, deque(maxlen=self.window))

        queue_hist.append(state.total_queue())
        discharge_hist.append(state.total_discharged)

        if len(queue_hist) < self.window:
            return None  # not enough history yet

        queue_growth = queue_hist[-1] - queue_hist[0]
        discharged_in_window = discharge_hist[-1] - discharge_hist[0]

        is_growing = queue_growth > 0 and queue_hist[-1] >= self.min_absolute_queue
        is_stalled = discharged_in_window < queue_growth * self.stall_discharge_ratio

        if is_growing and is_stalled:
            severity = min(1.0, queue_growth / (self.min_absolute_queue * 3))
            return IncidentAlert(
                intersection_id=intersection_id,
                kind="unusual_queue_growth",
                severity=round(severity, 2),
                description=(
                    f"Queue grew by {queue_growth:.1f} vehicles over the last {self.window} cycles "
                    f"but only {discharged_in_window} discharged -- possible blocked lane or incident."
                ),
            )
        return None
