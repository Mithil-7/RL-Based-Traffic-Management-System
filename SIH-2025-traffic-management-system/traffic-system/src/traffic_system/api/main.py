"""FastAPI application factory.

Startup wires up the city graph, Redis connection, and database engine
once and stores them on `app.state` (see `api/deps.py`). The lifespan
handler only initializes state that isn't already set -- this is what lets
tests pre-populate `app.state` with fakes/SQLite *before* the app starts,
without needing real Redis/Postgres running, while production simply lets
the defaults kick in.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from traffic_system.api import routes_metrics, routes_routing, routes_signals, websocket
from traffic_system.common.config import get_settings
from traffic_system.common.logging import configure_logging, get_logger
from traffic_system.env.city_graph import CityGraph
from traffic_system.ingestion.db_models import init_db, make_engine, make_session_factory
from traffic_system.ingestion.redis_state import RedisState

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()

    if not hasattr(app.state, "city_graph"):
        app.state.city_graph = CityGraph.load(settings.city_graph_path)
    if not hasattr(app.state, "redis_state"):
        app.state.redis_state = RedisState()
    if not hasattr(app.state, "session_factory"):
        engine = make_engine(settings.postgres_dsn)
        init_db(engine)
        app.state.session_factory = make_session_factory(engine)

    logger.info("api.started", intersections=len(app.state.city_graph.intersection_ids))
    yield
    logger.info("api.shutting_down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Adaptive Traffic Management System",
        description="Multi-agent RL traffic signal control, emergency preemption, and dynamic routing (SIH 2025).",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_signals.router)
    app.include_router(routes_routing.router)
    app.include_router(routes_metrics.router)
    app.include_router(websocket.router)

    @app.get("/health", tags=["health"])
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
