"""Route allocation endpoint -- the "give this vehicle a specialised route
around congestion" piece from the original spec.

Loads the latest congestion snapshot the brain service published to Redis
before computing the route. This is the deliberate design point that keeps
the API stateless and horizontally scalable: any API replica can serve any
request because live congestion lives in Redis, not in this process's
memory.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from traffic_system.agents.route_allocation_agent import RouteAllocationAgent
from traffic_system.api.deps import get_city_graph, get_redis_state
from traffic_system.common.schemas import RouteRequest, RouteResponse
from traffic_system.env.city_graph import CityGraph
from traffic_system.ingestion.redis_state import RedisState

router = APIRouter(prefix="/api/routes", tags=["routing"])


@router.post("/suggest", response_model=RouteResponse)
def suggest_route(
    request: RouteRequest,
    city_graph: CityGraph = Depends(get_city_graph),
    redis_state: RedisState = Depends(get_redis_state),
) -> RouteResponse:
    if request.origin_intersection_id not in city_graph.intersection_ids:
        raise HTTPException(status_code=404, detail=f"Unknown origin '{request.origin_intersection_id}'")
    if request.destination_intersection_id not in city_graph.intersection_ids:
        raise HTTPException(status_code=404, detail=f"Unknown destination '{request.destination_intersection_id}'")

    snapshot = redis_state.get_congestion_snapshot()
    if snapshot:
        city_graph.import_congestion(snapshot)

    agent = RouteAllocationAgent(city_graph)
    try:
        return agent.act(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No route found: {exc}") from exc
