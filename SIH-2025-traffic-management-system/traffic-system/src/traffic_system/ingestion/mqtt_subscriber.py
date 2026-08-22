"""The ingestion service: the one process that actually subscribes to
`traffic/telemetry/+` on the MQTT broker. Every edge agent in the city
publishes to this same wildcard topic; this service fans each message out
to:

  1. Redis (`RedisState`) -- overwrites the "current state" key, used by
     the brain for its next decision and by the API for `GET /state`.
  2. Postgres/TimescaleDB (`TelemetryRecord`) -- durable history, used for
     KPI queries, retraining data export, and audits.
  3. Redis pub/sub (`traffic:updates`) -- so the FastAPI WebSocket handler
     can push the update straight to connected dashboard clients.

Keeping this as its own service (rather than folding ingestion into the
brain or the API) means it can be scaled and restarted independently, and
the brain never blocks on a database write mid-decision.
"""
from __future__ import annotations

import argparse

from traffic_system.common.config import get_settings
from traffic_system.common.logging import configure_logging, get_logger
from traffic_system.common.mqtt_client import MqttClient
from traffic_system.common.schemas import TelemetryEvent
from traffic_system.ingestion.db_models import TelemetryRecord, init_db, make_engine, make_session_factory
from traffic_system.ingestion.redis_state import RedisState

logger = get_logger(__name__)


class IngestionService:
    def __init__(self, mqtt_client: MqttClient, redis_state: RedisState, session_factory) -> None:
        self.mqtt = mqtt_client
        self.redis_state = redis_state
        self.session_factory = session_factory
        self._events_ingested = 0

    def start(self) -> None:
        settings = get_settings()
        self.mqtt.connect()
        self.mqtt.subscribe_json(settings.mqtt_telemetry_topic, self._on_message)
        logger.info("ingestion.started", topic=settings.mqtt_telemetry_topic)

    def _on_message(self, topic: str, payload: dict) -> None:
        try:
            event = TelemetryEvent.model_validate(payload)
        except Exception:
            logger.exception("ingestion.bad_payload", topic=topic)
            return

        self.redis_state.set_latest_telemetry(event)
        self._persist(event)
        self.redis_state.publish_update("telemetry", event.model_dump(mode="json"))
        self._events_ingested += 1

    def _persist(self, event: TelemetryEvent) -> None:
        record = TelemetryRecord(
            intersection_id=event.intersection_id,
            timestamp=event.timestamp,
            total_vehicles=event.total_vehicles,
            has_emergency=event.has_emergency,
            frame_processing_ms=event.frame_processing_ms,
            payload=event.model_dump(mode="json"),
        )
        with self.session_factory() as session:
            session.add(record)
            session.commit()


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the MQTT -> Redis/Postgres ingestion service.")
    parser.parse_args()

    settings = get_settings()
    engine = make_engine(settings.postgres_dsn)
    init_db(engine)
    session_factory = make_session_factory(engine)

    service = IngestionService(
        mqtt_client=MqttClient(client_id_suffix="ingestion"),
        redis_state=RedisState(),
        session_factory=session_factory,
    )
    service.start()

    import time

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("ingestion.shutting_down")
        service.mqtt.disconnect()


if __name__ == "__main__":
    main()
