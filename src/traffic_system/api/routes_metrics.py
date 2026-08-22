"""KPI aggregation (from durable history) and a Prometheus scrape endpoint
(from live Redis state) -- the two different metrics use cases need
different data sources, deliberately: KPIs answer "how did intersection X
perform over the last hour", which needs the database; Prometheus scraping
answers "what is true right now", which needs to be cheap and fast, so it
reads Redis instead of hitting Postgres on every scrape.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from sqlalchemy import func

from traffic_system.api.deps import get_city_graph, get_redis_state, get_session_factory
from traffic_system.common.schemas import IntersectionKPIs
from traffic_system.env.city_graph import CityGraph
from traffic_system.env.intersection import DEFAULT_SATURATION_FLOW_PER_LANE_PER_S
from traffic_system.ingestion.db_models import IntersectionSnapshotRecord, TelemetryRecord
from traffic_system.ingestion.redis_state import RedisState

router = APIRouter(tags=["metrics"])

# Matches the default 2-lanes-per-approach assumption used throughout the
# simulation (SimConfig.lanes_per_approach) -- used only to convert an
# average queue length into an M/M/1-style wait-time proxy consistent with
# `env/intersection.py::ApproachState.avg_wait_s`.
_ASSUMED_LANES = 2
_ASSUMED_DISCHARGE_RATE_PER_S = _ASSUMED_LANES * DEFAULT_SATURATION_FLOW_PER_LANE_PER_S


@router.get("/api/metrics/kpis", response_model=list[IntersectionKPIs])
def get_kpis(
    minutes: int = 60,
    session_factory=Depends(get_session_factory),
    city_graph: CityGraph = Depends(get_city_graph),
) -> list[IntersectionKPIs]:
    since = datetime.now(UTC) - timedelta(minutes=minutes)
    now = datetime.now(UTC)
    results: list[IntersectionKPIs] = []

    with session_factory() as session:
        for iid in city_graph.intersection_ids:
            avg_queue = (
                session.query(func.avg(TelemetryRecord.total_vehicles))
                .filter(TelemetryRecord.intersection_id == iid, TelemetryRecord.timestamp >= since)
                .scalar()
                or 0.0
            )
            avg_processing_ms = (
                session.query(func.avg(TelemetryRecord.frame_processing_ms))
                .filter(TelemetryRecord.intersection_id == iid, TelemetryRecord.timestamp >= since)
                .scalar()
                or 0.0
            )
            emergency_ticks = (
                session.query(func.count(TelemetryRecord.id))
                .filter(
                    TelemetryRecord.intersection_id == iid,
                    TelemetryRecord.timestamp >= since,
                    TelemetryRecord.has_emergency.is_(True),
                )
                .scalar()
                or 0
            )

            snapshots = (
                session.query(IntersectionSnapshotRecord)
                .filter(IntersectionSnapshotRecord.intersection_id == iid, IntersectionSnapshotRecord.timestamp >= since)
                .order_by(IntersectionSnapshotRecord.timestamp.asc())
                .all()
            )
            throughput = max(0, snapshots[-1].total_discharged - snapshots[0].total_discharged) if len(snapshots) >= 2 else 0

            avg_queue_f = float(avg_queue)
            avg_wait = avg_queue_f / _ASSUMED_DISCHARGE_RATE_PER_S if _ASSUMED_DISCHARGE_RATE_PER_S > 0 else 0.0

            results.append(
                IntersectionKPIs(
                    intersection_id=iid,
                    window_start=since,
                    window_end=now,
                    avg_queue_vehicles=round(avg_queue_f, 2),
                    avg_waiting_time_s=round(avg_wait, 2),
                    throughput_vehicles=int(throughput),
                    emergency_ticks=int(emergency_ticks),
                    avg_frame_processing_ms=round(float(avg_processing_ms), 2),
                )
            )
    return results


@router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics(
    redis_state: RedisState = Depends(get_redis_state), city_graph: CityGraph = Depends(get_city_graph)
) -> PlainTextResponse:
    registry = CollectorRegistry()
    queue_gauge = Gauge(
        "traffic_intersection_queue_vehicles", "Vehicles currently queued", ["intersection_id"], registry=registry
    )
    emergency_gauge = Gauge(
        "traffic_intersection_emergency_active", "1 if an emergency vehicle is currently present", ["intersection_id"], registry=registry
    )
    processing_gauge = Gauge(
        "traffic_frame_processing_ms", "Most recent CV frame processing time", ["intersection_id"], registry=registry
    )

    for iid in city_graph.intersection_ids:
        event = redis_state.get_latest_telemetry(iid)
        if event is None:
            continue
        queue_gauge.labels(intersection_id=iid).set(event.total_vehicles)
        emergency_gauge.labels(intersection_id=iid).set(1 if event.has_emergency else 0)
        processing_gauge.labels(intersection_id=iid).set(event.frame_processing_ms)

    return PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
