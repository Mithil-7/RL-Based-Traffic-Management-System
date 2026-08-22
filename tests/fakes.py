"""Test doubles shared across the test suite."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class FakeMqttClient:
    """Synchronous, in-process stand-in for `MqttClient` implementing the
    same public interface. Dispatches published messages directly to any
    matching subscriber callback with no network/threads involved -- keeps
    brain/ingestion unit tests fast and independent of a running broker.
    A real-broker integration test also exists (see the manual test in
    edge_agent's module docstring / README) and runs in CI via a Mosquitto
    service container.
    """

    def __init__(self, client_id_suffix: str | None = None) -> None:
        self.client_id_suffix = client_id_suffix
        self.published: list[tuple[str, dict]] = []
        self._subscriptions: dict[str, Callable[[str, dict[str, Any]], None]] = {}
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def publish_json(self, topic: str, payload: dict, qos: int = 1, retain: bool = False) -> None:
        self.published.append((topic, payload))
        for pattern, callback in self._subscriptions.items():
            if _topic_matches(pattern, topic):
                callback(topic, payload)

    def subscribe_json(self, topic: str, callback: Callable[[str, dict[str, Any]], None], qos: int = 1) -> None:
        self._subscriptions[topic] = callback


def _topic_matches(pattern: str, topic: str) -> bool:
    pattern_parts = pattern.split("/")
    topic_parts = topic.split("/")
    for i, part in enumerate(pattern_parts):
        if part == "+":
            if i >= len(topic_parts):
                return False
            continue
        if part == "#":
            return True
        if i >= len(topic_parts) or part != topic_parts[i]:
            return False
    return len(pattern_parts) == len(topic_parts)
