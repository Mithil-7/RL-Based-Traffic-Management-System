"""Dependency-injection helpers. Everything pulls from `.app.state`, which
is populated once at startup by the lifespan handler in `main.py` --
keeps route modules free of global state and easy to unit test.

Typed against `HTTPConnection` (the common base of `Request` and
`WebSocket`) rather than `Request` specifically, since `websocket.py`'s
`/ws` route needs the same dependencies as ordinary REST routes.
"""
from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker
from starlette.requests import HTTPConnection

from traffic_system.env.city_graph import CityGraph
from traffic_system.ingestion.redis_state import RedisState


def get_city_graph(conn: HTTPConnection) -> CityGraph:
    return conn.app.state.city_graph


def get_redis_state(conn: HTTPConnection) -> RedisState:
    return conn.app.state.redis_state


def get_session_factory(conn: HTTPConnection) -> sessionmaker[Session]:
    return conn.app.state.session_factory
