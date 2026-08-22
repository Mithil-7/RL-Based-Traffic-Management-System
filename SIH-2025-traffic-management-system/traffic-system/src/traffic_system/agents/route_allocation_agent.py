"""Graph-based dynamic route allocation.

This is deliberately *not* part of the signal-control RL loop -- it answers
a different question ("which path should this specific vehicle/driver
take") using classical weighted shortest-path search (Dijkstra via
networkx) over live, congestion-adjusted edge weights. Every decision cycle
it refreshes each road's weight from the current queue lengths at both
endpoints, so `city_graph.shortest_path` naturally routes around
intersections the brain's own telemetry says are backed up -- this is what
implements "the system detects the cause of traffic and allocates a
specialised route for that vehicle" from the original spec.
"""
from __future__ import annotations

from traffic_system.agents.base_agent import Agent
from traffic_system.common.schemas import RouteRequest, RouteResponse
from traffic_system.env.city_graph import CityGraph
from traffic_system.env.intersection import IntersectionState

# Tuned so a combined endpoint queue of ~40 vehicles roughly doubles the
# perceived travel time of that road; a queue of ~120 caps the penalty at 4x
# so pathfinding never sees an effectively "infinite cost" edge.
CONGESTION_QUEUE_SCALE = 40.0
MAX_CONGESTION_FACTOR = 4.0


class RouteAllocationAgent(Agent):
    name = "route_allocation"

    def __init__(self, city_graph: CityGraph) -> None:
        self.city_graph = city_graph

    def _directional_label(self, u: str, v: str) -> str | None:
        """The approach label shared by both ends of edge (u, v) -- see the
        routing derivation in `traffic_grid_env._route_downstream`: a vehicle
        queued at approach `label` on one end is exactly the traffic that
        travels this edge and arrives at approach `label` on the other end."""
        for direction, neighbor in self.city_graph.direction_neighbors(u).items():
            if neighbor == v:
                return self.city_graph.opposite(direction)
        return None

    def update_congestion(self, states: dict[str, IntersectionState]) -> None:
        """Refresh every road's congestion factor from current queue telemetry.

        Uses only the queue *specific to this edge's direction* at each
        endpoint (not the intersection's total queue across all approaches)
        -- otherwise heavy congestion on one approach would incorrectly
        make every road touching that intersection look congested, even
        roads facing an entirely different direction. Call this once per
        decision cycle before `act`.
        """
        for u, v in self.city_graph.graph.edges():
            label = self._directional_label(u, v)
            q_u = states[u].approaches[label].queue if (u in states and label) else 0.0
            q_v = states[v].approaches[label].queue if (v in states and label) else 0.0
            avg_queue = (q_u + q_v) / 2
            factor = 1.0 + min(avg_queue / CONGESTION_QUEUE_SCALE, MAX_CONGESTION_FACTOR - 1.0)
            self.city_graph.set_congestion_factor(u, v, factor)

    def act(self, request: RouteRequest) -> RouteResponse:
        congested_path, congested_cost = self.city_graph.shortest_path(
            request.origin_intersection_id,
            request.destination_intersection_id,
            avoid=request.avoid_intersection_ids,
            use_congestion=True,
        )
        free_flow_path, _ = self.city_graph.shortest_path(
            request.origin_intersection_id,
            request.destination_intersection_id,
            avoid=request.avoid_intersection_ids,
            use_congestion=False,
        )

        return RouteResponse(
            path=congested_path,
            estimated_travel_time_s=congested_cost,
            congestion_avoided=(congested_path != free_flow_path),
        )
