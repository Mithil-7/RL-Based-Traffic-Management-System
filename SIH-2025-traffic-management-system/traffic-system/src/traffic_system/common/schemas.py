"""Wire-format schemas shared by every layer of the system.

These are the contracts between the edge agents (publishers), the ingestion
service (subscriber), the brain (consumer + decision maker), and the API /
dashboard (consumers of decisions). Keeping them in one module means the
edge and the brain can never silently drift out of sync on field names.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class VehicleClass(str, Enum):
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    AMBULANCE = "ambulance"
    FIRE_TRUCK = "fire_truck"
    POLICE = "police"

    @property
    def is_emergency(self) -> bool:
        return self in {VehicleClass.AMBULANCE, VehicleClass.FIRE_TRUCK, VehicleClass.POLICE}


class LaneObservation(BaseModel):
    """Per-lane counts produced by the edge CV pipeline for one detection cycle."""

    lane_id: str
    approach: str = Field(description="Compass approach, e.g. 'N', 'S', 'E', 'W'")
    vehicle_count: int = 0
    queue_length_m: float = 0.0
    avg_speed_kmh: float = 0.0
    waiting_time_s: float = 0.0
    class_counts: dict[str, int] = Field(default_factory=dict)
    emergency_vehicle_present: bool = False
    emergency_vehicle_class: VehicleClass | None = None
    emergency_confidence: float = 0.0


class TelemetryEvent(BaseModel):
    """One full observation cycle from a single intersection's edge agent."""

    intersection_id: str
    timestamp: datetime = Field(default_factory=_utcnow)
    lanes: list[LaneObservation]
    source: str = Field(default="simulated", description="'simulated' or 'raspberry_pi'")
    frame_processing_ms: float = 0.0

    @property
    def total_vehicles(self) -> int:
        return sum(lane.vehicle_count for lane in self.lanes)

    @property
    def has_emergency(self) -> bool:
        return any(lane.emergency_vehicle_present for lane in self.lanes)


class SignalPhase(str, Enum):
    """A phase is the set of approaches that get a green light simultaneously."""

    NS_THROUGH = "NS_THROUGH"
    NS_LEFT = "NS_LEFT"
    EW_THROUGH = "EW_THROUGH"
    EW_LEFT = "EW_LEFT"
    ALL_RED = "ALL_RED"


class SignalState(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class SignalCommand(BaseModel):
    """A decision emitted by the brain for one intersection."""

    intersection_id: str
    phase: SignalPhase
    duration_s: float
    issued_at: datetime = Field(default_factory=_utcnow)
    issued_by: str = Field(description="agent name: dqn_agent | coordinator | emergency_preemption")
    reason: str = ""
    preempted: bool = False


class EmergencyAlert(BaseModel):
    intersection_id: str
    vehicle_class: VehicleClass
    approach: str
    confidence: float
    detected_at: datetime = Field(default_factory=_utcnow)
    corridor: list[str] = Field(default_factory=list, description="Intersection ids on the preemption corridor")


class IncidentAlert(BaseModel):
    intersection_id: str
    kind: str = Field(description="e.g. 'stalled_vehicle', 'unusual_queue_growth', 'sensor_dropout'")
    severity: float = Field(ge=0.0, le=1.0)
    description: str
    detected_at: datetime = Field(default_factory=_utcnow)


class RouteRequest(BaseModel):
    origin_intersection_id: str
    destination_intersection_id: str
    avoid_intersection_ids: list[str] = Field(default_factory=list)


class RouteResponse(BaseModel):
    path: list[str]
    estimated_travel_time_s: float
    congestion_avoided: bool = False


class IntersectionKPIs(BaseModel):
    intersection_id: str
    window_start: datetime
    window_end: datetime
    avg_queue_vehicles: float
    avg_waiting_time_s: float
    throughput_vehicles: int
    emergency_ticks: int
    avg_frame_processing_ms: float
