"""Thin wrapper around paho-mqtt used by both edge publishers and the
ingestion subscriber.

MQTT (not raw HTTP polling) is the transport between edge devices and the
cloud because: (1) it is designed for thousands of small, low-power,
possibly-intermittently-connected clients -- exactly what a city's worth of
Raspberry Pi intersection controllers look like; (2) pub/sub with QoS and
retained "last known state" messages means a dashboard or a newly-started
brain instance can recover current state without replaying history; (3) it
is the de-facto standard in IoT/ITS (Intelligent Transport Systems)
deployments, so this choice keeps us compatible with real traffic hardware
vendors later.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

import paho.mqtt.client as mqtt
from tenacity import retry, stop_after_attempt, wait_exponential

from traffic_system.common.config import get_settings
from traffic_system.common.logging import get_logger

logger = get_logger(__name__)


class MqttClient:
    """Synchronous MQTT client with JSON (de)serialization helpers.

    Wraps paho-mqtt's callback API in a small class so publishers and
    subscribers share one connection/reconnect implementation instead of
    reimplementing it in every service.
    """

    def __init__(self, client_id_suffix: str | None = None) -> None:
        settings = get_settings()
        client_id = f"{settings.mqtt_client_id_prefix}-{client_id_suffix or uuid.uuid4().hex[:8]}"
        self._settings = settings
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._subscriptions: dict[str, Callable[[str, dict[str, Any]], None]] = {}

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=15))
    def connect(self) -> None:
        logger.info("mqtt.connecting", host=self._settings.mqtt_host, port=self._settings.mqtt_port)
        self._client.connect(self._settings.mqtt_host, self._settings.mqtt_port, self._settings.mqtt_keepalive)
        self._client.loop_start()

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def publish_json(self, topic: str, payload: dict[str, Any], qos: int = 1, retain: bool = False) -> None:
        self._client.publish(topic, json.dumps(payload, default=str), qos=qos, retain=retain)

    def subscribe_json(self, topic: str, callback: Callable[[str, dict[str, Any]], None], qos: int = 1) -> None:
        """Register a callback invoked with (topic, parsed_json_payload) for every matching message."""
        self._subscriptions[topic] = callback
        self._client.subscribe(topic, qos=qos)
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:  # noqa: ANN001
        if reason_code == 0:
            logger.info("mqtt.connected")
            for topic in self._subscriptions:
                client.subscribe(topic)
        else:
            logger.warning("mqtt.connect_failed", reason_code=str(reason_code))

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None) -> None:  # noqa: ANN001
        logger.warning("mqtt.disconnected", reason_code=str(reason_code))

    def _on_message(self, client, userdata, msg) -> None:  # noqa: ANN001
        for pattern, callback in self._subscriptions.items():
            if mqtt.topic_matches_sub(pattern, msg.topic):
                try:
                    payload = json.loads(msg.payload.decode("utf-8"))
                except json.JSONDecodeError:
                    logger.error("mqtt.bad_payload", topic=msg.topic)
                    return
                try:
                    callback(msg.topic, payload)
                except Exception:
                    logger.exception("mqtt.callback_error", topic=msg.topic)
