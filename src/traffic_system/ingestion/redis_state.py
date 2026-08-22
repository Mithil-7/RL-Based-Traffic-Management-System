"""Redis is the low-latency "what is true right now" store, separate from
Postgres/TimescaleDB (which is the durable history store). Every decision
cycle, the brain reads current intersection state from here rather than
querying Postgres -- sub-millisecond reads matter when a decision has to be
made every few seconds for every intersection in the city.

Also doubles as the pub/sub bridge that lets the FastAPI WebSocket handler
push live updates to connected dashboard clients without polling Postgres.
"""
from __future__ import annotations

import json
from typing import Any

import redis

from traffic_system.common.config import get_settings
from traffic_system.common.schemas import SignalCommand, TelemetryEvent

UPDATES_CHANNEL = "traffic:updates"


class RedisState:
    def __init__(self, client: redis.Redis | None = None) -> None:
        settings = get_settings()
        self._client = client or redis.from_url(settings.redis_url, decode_responses=True)
        self._ttl = settings.redis_state_ttl_seconds

    # --- telemetry ---
    def set_latest_telemetry(self, event: TelemetryEvent) -> None:
        key = f"telemetry:{event.intersection_id}"
        self._client.set(key, event.model_dump_json(), ex=self._ttl)

    def get_latest_telemetry(self, intersection_id: str) -> TelemetryEvent | None:
        raw = self._client.get(f"telemetry:{intersection_id}")
        return TelemetryEvent.model_validate_json(raw) if raw else None

    def get_all_latest_telemetry(self, intersection_ids: list[str]) -> dict[str, TelemetryEvent]:
        result = {}
        for iid in intersection_ids:
            event = self.get_latest_telemetry(iid)
            if event is not None:
                result[iid] = event
        return result

    # --- signal state ---
    def set_latest_command(self, command: SignalCommand) -> None:
        key = f"signal:{command.intersection_id}"
        self._client.set(key, command.model_dump_json(), ex=self._ttl * 4)

    def get_latest_command(self, intersection_id: str) -> SignalCommand | None:
        raw = self._client.get(f"signal:{intersection_id}")
        return SignalCommand.model_validate_json(raw) if raw else None

    # --- live congestion snapshot (shared across brain/API processes) ---
    def set_congestion_snapshot(self, snapshot: dict[str, float]) -> None:
        self._client.set("congestion:snapshot", json.dumps(snapshot), ex=self._ttl * 6)

    def get_congestion_snapshot(self) -> dict[str, float] | None:
        raw = self._client.get("congestion:snapshot")
        return json.loads(raw) if raw else None

    # --- pub/sub bridge for the dashboard/API WebSocket ---
    def publish_update(self, kind: str, payload: dict[str, Any]) -> None:
        self._client.publish(UPDATES_CHANNEL, json.dumps({"kind": kind, "payload": payload}, default=str))

    def subscribe_updates(self) -> redis.client.PubSub:
        pubsub = self._client.pubsub()
        pubsub.subscribe(UPDATES_CHANNEL)
        return pubsub
