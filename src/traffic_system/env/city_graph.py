"""Loads the city road graph and derives the compass-direction adjacency
that the simulation and the routing agent both need.

The map file only lists roads as (from, to) pairs with lat/lon on each
node. We derive, for every intersection, which neighbor sits to its N/S/E/W
-- that mapping is what lets the traffic simulation know "a vehicle
discharged from the N approach of I_B2 continues on to the N approach of
I_A2" and what lets the route allocation agent do turn-by-turn pathing.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass(frozen=True)
class RoadEdge:
    lanes: int
    length_m: float
    free_flow_speed_kmh: float

    @property
    def free_flow_travel_time_s(self) -> float:
        speed_ms = self.free_flow_speed_kmh / 3.6
        return self.length_m / speed_ms


class CityGraph:
    """Wraps a networkx.Graph of intersections and roads with traffic-domain helpers."""

    def __init__(self, graph: nx.Graph, city_name: str) -> None:
        self.graph = graph
        self.city_name = city_name
        self._direction_cache: dict[str, dict[str, str | None]] = {}

    @classmethod
    def load(cls, path: str | Path) -> CityGraph:
        data = json.loads(Path(path).read_text())
        g = nx.Graph()
        for node in data["intersections"]:
            g.add_node(
                node["id"],
                name=node.get("name", node["id"]),
                lat=node["lat"],
                lon=node["lon"],
                approaches=tuple(node.get("approaches", ["N", "S", "E", "W"])),
            )
        for road in data["roads"]:
            g.add_edge(
                road["from"],
                road["to"],
                data=RoadEdge(
                    lanes=road["lanes"],
                    length_m=road["length_m"],
                    free_flow_speed_kmh=road["free_flow_speed_kmh"],
                ),
                congestion_factor=1.0,
            )
        return cls(g, data.get("city_name", "unnamed"))

    @property
    def intersection_ids(self) -> list[str]:
        return list(self.graph.nodes)

    def neighbors(self, intersection_id: str) -> list[str]:
        return list(self.graph.neighbors(intersection_id))

    def road(self, u: str, v: str) -> RoadEdge:
        return self.graph[u][v]["data"]

    def direction_neighbors(self, intersection_id: str) -> dict[str, str | None]:
        """Classify each neighbor of `intersection_id` as N/S/E/W using lat/lon deltas.

        Cached per node since the graph topology is static after load.
        """
        if intersection_id in self._direction_cache:
            return self._direction_cache[intersection_id]

        self_node = self.graph.nodes[intersection_id]
        result: dict[str, str | None] = {"N": None, "S": None, "E": None, "W": None}
        for neighbor_id in self.graph.neighbors(intersection_id):
            neighbor = self.graph.nodes[neighbor_id]
            d_lat = neighbor["lat"] - self_node["lat"]
            d_lon = neighbor["lon"] - self_node["lon"]
            if abs(d_lat) >= abs(d_lon):
                direction = "N" if d_lat > 0 else "S"
            else:
                direction = "E" if d_lon > 0 else "W"
            result[direction] = neighbor_id
        self._direction_cache[intersection_id] = result
        return result

    def opposite(self, direction: str) -> str:
        return {"N": "S", "S": "N", "E": "W", "W": "E"}[direction]

    def set_congestion_factor(self, u: str, v: str, factor: float) -> None:
        """Used by the route allocation agent to make live-congested roads 'longer'
        in path-cost terms without changing physical topology."""
        self.graph[u][v]["congestion_factor"] = max(factor, 0.01)

    def export_congestion(self) -> dict[str, float]:
        """Serialize every edge's congestion factor to a flat, JSON-friendly
        dict -- used by the brain service to publish live congestion to
        Redis so other processes (e.g. API replicas) can pick it up without
        sharing this in-memory object."""
        return {f"{u}|{v}": data.get("congestion_factor", 1.0) for u, v, data in self.graph.edges(data=True)}

    def import_congestion(self, snapshot: dict[str, float]) -> None:
        for key, factor in snapshot.items():
            u, v = key.split("|")
            if self.graph.has_edge(u, v):
                self.graph[u][v]["congestion_factor"] = factor

    def travel_cost_s(self, u: str, v: str) -> float:
        edge = self.road(u, v)
        return edge.free_flow_travel_time_s * self.graph[u][v].get("congestion_factor", 1.0)

    def shortest_path(
        self, origin: str, destination: str, avoid: list[str] | None = None, use_congestion: bool = True
    ) -> tuple[list[str], float]:
        """Dijkstra shortest path weighted by travel time.

        `use_congestion=False` computes the free-flow (uncongested) path,
        which the route allocation agent uses as a baseline to tell whether
        it actually rerouted traffic around congestion or not.
        """
        avoid = set(avoid or [])
        working_graph = self.graph
        if avoid:
            working_graph = self.graph.copy()
            working_graph.remove_nodes_from(n for n in avoid if n not in (origin, destination))

        def weight(u: str, v: str, data: dict) -> float:
            edge = data["data"]
            factor = data.get("congestion_factor", 1.0) if use_congestion else 1.0
            return edge.free_flow_travel_time_s * factor

        path = nx.shortest_path(working_graph, origin, destination, weight=weight)
        cost = nx.shortest_path_length(working_graph, origin, destination, weight=weight)
        return path, cost

    def haversine_m(self, u: str, v: str) -> float:
        lat1, lon1 = self.graph.nodes[u]["lat"], self.graph.nodes[u]["lon"]
        lat2, lon2 = self.graph.nodes[v]["lat"], self.graph.nodes[v]["lon"]
        r = 6_371_000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))
