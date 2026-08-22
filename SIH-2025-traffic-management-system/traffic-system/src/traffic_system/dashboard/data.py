"""All HTTP calls the dashboard makes to the API, isolated from the NiceGUI
rendering code in `app.py`. Kept separate deliberately: this module is
plain async functions returning plain dicts, so it can be unit-tested with
a real FastAPI test app and httpx, without needing a browser or a running
NiceGUI event loop to exercise the actual network/parsing logic.
"""
from __future__ import annotations

import httpx

from traffic_system.common.logging import get_logger

logger = get_logger(__name__)


class ApiClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def get_all_states(self) -> list[dict] | None:
        return await self._get("/api/intersections/state/all")

    async def get_kpis(self, minutes: int = 60) -> list[dict] | None:
        return await self._get("/api/metrics/kpis", params={"minutes": minutes})

    async def get_recent_alerts(self, minutes: int = 60) -> dict | None:
        return await self._get("/api/intersections/alerts/recent", params={"minutes": minutes})

    async def suggest_route(self, origin: str, destination: str) -> dict | None:
        try:
            r = await self._client.post(
                "/api/routes/suggest",
                json={"origin_intersection_id": origin, "destination_intersection_id": destination},
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            logger.warning("dashboard.api_request_failed", endpoint="routes/suggest", error=str(exc))
            return None

    async def _get(self, path: str, params: dict | None = None) -> list | dict | None:
        try:
            r = await self._client.get(path, params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            logger.warning("dashboard.api_request_failed", endpoint=path, error=str(exc))
            return None


def summarize_city_state(states: list[dict]) -> dict:
    """Pure function: reduces the /state/all payload into the numbers the
    KPI cards show. Separated out purely so it's trivially unit-testable."""
    total_vehicles = 0
    active_emergencies = 0
    intersections_reporting = 0

    for s in states:
        telemetry = s.get("telemetry")
        if telemetry is None:
            continue
        intersections_reporting += 1
        total_vehicles += sum(lane["vehicle_count"] for lane in telemetry["lanes"])
        if any(lane["emergency_vehicle_present"] for lane in telemetry["lanes"]):
            active_emergencies += 1

    return {
        "total_vehicles": total_vehicles,
        "active_emergencies": active_emergencies,
        "intersections_reporting": intersections_reporting,
        "intersections_total": len(states),
    }


def summarize_kpis(kpis: list[dict]) -> dict:
    """Reduces the /api/metrics/kpis payload (one IntersectionKPIs per
    intersection) into the two numbers the KPI row shows: city-wide average
    wait time and total throughput over the window. Pure function, kept
    separate from `app.py` for the same testability reason as
    `summarize_city_state`."""
    if not kpis:
        return {"avg_waiting_time_s": 0.0, "total_throughput": 0}
    avg_wait = sum(k["avg_waiting_time_s"] for k in kpis) / len(kpis)
    total_throughput = sum(k["throughput_vehicles"] for k in kpis)
    return {"avg_waiting_time_s": round(avg_wait, 1), "total_throughput": total_throughput}


PHASE_COLORS = {
    "NS_THROUGH": "#22c55e",
    "EW_THROUGH": "#3b82f6",
    "NS_LEFT": "#84cc16",
    "EW_LEFT": "#0ea5e9",
    "ALL_RED": "#f59e0b",
}
EMERGENCY_COLOR = "#ef4444"
NO_DATA_COLOR = "#9ca3af"


def marker_color(state: dict) -> str:
    telemetry = state.get("telemetry")
    if telemetry and any(lane["emergency_vehicle_present"] for lane in telemetry["lanes"]):
        return EMERGENCY_COLOR
    signal = state.get("signal")
    if signal is None:
        return NO_DATA_COLOR
    return PHASE_COLORS.get(signal["phase"], NO_DATA_COLOR)
