"""Intersection listing, live state, and signal history endpoints."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException

from traffic_system.api.deps import get_city_graph, get_redis_state, get_session_factory
from traffic_system.env.city_graph import CityGraph
from traffic_system.ingestion.db_models import SignalCommandRecord, TelemetryRecord
from traffic_system.ingestion.redis_state import RedisState

router = APIRouter(prefix="/api/intersections", tags=["intersections"])


@router.get("")
def list_intersections(city_graph: CityGraph = Depends(get_city_graph)) -> list[dict]:
    return [
        {
            "id": node_id,
            "name": data.get("name", node_id),
            "lat": data["lat"],
            "lon": data["lon"],
            "approaches": list(data.get("approaches", [])),
        }
        for node_id, data in city_graph.graph.nodes(data=True)
    ]


@router.get("/{intersection_id}/state")
def get_intersection_state(intersection_id: str, redis_state: RedisState = Depends(get_redis_state)) -> dict:
    telemetry = redis_state.get_latest_telemetry(intersection_id)
    command = redis_state.get_latest_command(intersection_id)
    if telemetry is None and command is None:
        raise HTTPException(status_code=404, detail=f"No live state for intersection '{intersection_id}' yet")
    return {
        "intersection_id": intersection_id,
        "telemetry": telemetry.model_dump(mode="json") if telemetry else None,
        "signal": command.model_dump(mode="json") if command else None,
    }


@router.get("/{intersection_id}/history")
def get_intersection_history(
    intersection_id: str,
    minutes: int = 60,
    session_factory=Depends(get_session_factory),
) -> list[dict]:
    since = datetime.now(UTC) - timedelta(minutes=minutes)
    with session_factory() as session:
        records = (
            session.query(TelemetryRecord)
            .filter(TelemetryRecord.intersection_id == intersection_id, TelemetryRecord.timestamp >= since)
            .order_by(TelemetryRecord.timestamp.asc())
            .limit(2000)
            .all()
        )
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "total_vehicles": r.total_vehicles,
                "has_emergency": r.has_emergency,
                "frame_processing_ms": r.frame_processing_ms,
            }
            for r in records
        ]


@router.get("/{intersection_id}/signal-history")
def get_signal_history(
    intersection_id: str,
    minutes: int = 60,
    session_factory=Depends(get_session_factory),
) -> list[dict]:
    since = datetime.now(UTC) - timedelta(minutes=minutes)
    with session_factory() as session:
        records = (
            session.query(SignalCommandRecord)
            .filter(SignalCommandRecord.intersection_id == intersection_id, SignalCommandRecord.issued_at >= since)
            .order_by(SignalCommandRecord.issued_at.asc())
            .limit(2000)
            .all()
        )
        return [
            {
                "timestamp": r.issued_at.isoformat(),
                "phase": r.phase,
                "duration_s": r.duration_s,
                "issued_by": r.issued_by,
                "reason": r.reason,
                "preempted": r.preempted,
            }
            for r in records
        ]


@router.get("/state/all", tags=["intersections"])
def get_all_states(
    city_graph: CityGraph = Depends(get_city_graph), redis_state: RedisState = Depends(get_redis_state)
) -> list[dict]:
    """One-shot fetch of every intersection's live telemetry + signal state --
    what the dashboard polls instead of making one request per intersection."""
    result = []
    for node_id, data in city_graph.graph.nodes(data=True):
        telemetry = redis_state.get_latest_telemetry(node_id)
        command = redis_state.get_latest_command(node_id)
        result.append(
            {
                "id": node_id,
                "name": data.get("name", node_id),
                "lat": data["lat"],
                "lon": data["lon"],
                "telemetry": telemetry.model_dump(mode="json") if telemetry else None,
                "signal": command.model_dump(mode="json") if command else None,
            }
        )
    return result


@router.get("/alerts/recent", tags=["intersections"])
def get_recent_alerts(minutes: int = 60, session_factory=Depends(get_session_factory)) -> dict:
    from traffic_system.ingestion.db_models import EmergencyAlertRecord, IncidentAlertRecord

    since = datetime.now(UTC) - timedelta(minutes=minutes)
    with session_factory() as session:
        emergencies = (
            session.query(EmergencyAlertRecord)
            .filter(EmergencyAlertRecord.detected_at >= since)
            .order_by(EmergencyAlertRecord.detected_at.desc())
            .limit(100)
            .all()
        )
        incidents = (
            session.query(IncidentAlertRecord)
            .filter(IncidentAlertRecord.detected_at >= since)
            .order_by(IncidentAlertRecord.detected_at.desc())
            .limit(100)
            .all()
        )
        return {
            "emergencies": [
                {
                    "intersection_id": e.intersection_id,
                    "vehicle_class": e.vehicle_class,
                    "approach": e.approach,
                    "confidence": e.confidence,
                    "detected_at": e.detected_at.isoformat(),
                    "corridor": e.corridor,
                }
                for e in emergencies
            ],
            "incidents": [
                {
                    "intersection_id": i.intersection_id,
                    "kind": i.kind,
                    "severity": i.severity,
                    "description": i.description,
                    "detected_at": i.detected_at.isoformat(),
                }
                for i in incidents
            ],
        }
