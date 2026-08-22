"""Durable storage models.

In production these tables live in TimescaleDB (Postgres + the timescaledb
extension) -- `infra/postgres/init.sql` converts `telemetry_records` and
`signal_commands` into hypertables, which gives automatic time-based
partitioning and much faster range queries ("last 24h for intersection X")
at city scale than plain Postgres tables would. The SQLAlchemy models below
are just plain relational tables, so they also work unmodified against
plain Postgres or SQLite (used by the test suite) -- TimescaleDB's
hypertable behavior is a transparent extension of a normal table, not a
different schema.

Each record keeps both queryable summary columns (for fast dashboard
queries) and the full original JSON payload (for reprocessing/audits/
retraining data export) -- a common "raw + derived" pattern for telemetry
pipelines.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TelemetryRecord(Base):
    __tablename__ = "telemetry_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    intersection_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=_utcnow)
    total_vehicles: Mapped[int] = mapped_column(Integer)
    has_emergency: Mapped[bool] = mapped_column(Boolean, default=False)
    frame_processing_ms: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict] = mapped_column(JSON)  # full TelemetryEvent, for reprocessing/export


class SignalCommandRecord(Base):
    __tablename__ = "signal_commands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    intersection_id: Mapped[str] = mapped_column(String(64), index=True)
    phase: Mapped[str] = mapped_column(String(32))
    duration_s: Mapped[float] = mapped_column(Float)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=_utcnow)
    issued_by: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(256), default="")
    preempted: Mapped[bool] = mapped_column(Boolean, default=False)


class EmergencyAlertRecord(Base):
    __tablename__ = "emergency_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    intersection_id: Mapped[str] = mapped_column(String(64), index=True)
    vehicle_class: Mapped[str] = mapped_column(String(32))
    approach: Mapped[str] = mapped_column(String(8))
    confidence: Mapped[float] = mapped_column(Float)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=_utcnow)
    corridor: Mapped[dict] = mapped_column(JSON, default=list)


class IncidentAlertRecord(Base):
    __tablename__ = "incident_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    intersection_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(64))
    severity: Mapped[float] = mapped_column(Float)
    description: Mapped[str] = mapped_column(String(512))
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=_utcnow)


class IntersectionSnapshotRecord(Base):
    """Periodic per-intersection state snapshot, written by the brain service
    every decision cycle. This is what powers honest KPI/throughput queries
    (see api/routes_metrics.py) -- throughput over a window is computed as
    the *difference* in `total_discharged` between the first and last
    snapshot in that window, not estimated or guessed."""

    __tablename__ = "intersection_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    intersection_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=_utcnow)
    total_queue: Mapped[float] = mapped_column(Float)
    total_discharged: Mapped[int] = mapped_column(Integer)
    total_switches: Mapped[int] = mapped_column(Integer)
    current_phase: Mapped[str] = mapped_column(String(32))


def make_engine(dsn: str):
    if dsn.startswith("sqlite") and ":memory:" in dsn:
        # SQLite's default pooling opens a *new* connection (and therefore a
        # fresh, empty in-memory database) per checkout. StaticPool keeps a
        # single connection alive for the engine's lifetime so an in-memory
        # DB actually persists across the multiple sessions a service opens
        # -- this only matters for tests; real deployments use Postgres.
        from sqlalchemy.pool import StaticPool

        return create_engine(dsn, future=True, poolclass=StaticPool, connect_args={"check_same_thread": False})
    return create_engine(dsn, future=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def make_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
