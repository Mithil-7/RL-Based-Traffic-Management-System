"""The control-room dashboard. Pure Python (NiceGUI), polling the FastAPI
backend on a timer -- see `dashboard/data.py` for why polling was chosen
over a browser-side WebSocket client (simplicity/robustness; the `/ws`
endpoint still exists for lower-latency integrations, e.g. a future mobile
app).

Layout: KPI row -> live map (colored by signal phase / emergency status) ->
alerts feed -> per-intersection table. Everything reacts to the same
`refresh()` timer tick, so the whole dashboard is consistent as of one
snapshot in time rather than each widget lagging independently.
"""
from __future__ import annotations

from nicegui import app as nicegui_app
from nicegui import ui

from traffic_system.common.config import get_settings
from traffic_system.common.logging import configure_logging, get_logger
from traffic_system.dashboard.data import ApiClient, marker_color, summarize_city_state, summarize_kpis

logger = get_logger(__name__)

VEHICLE_CLASS_ICONS = {"ambulance": "🚑", "fire_truck": "🚒", "police": "🚓"}


class DashboardState:
    def __init__(self, api_client: ApiClient) -> None:
        self.api = api_client
        self.states: list[dict] = []
        self.kpis: list[dict] = []
        self.alerts: dict = {"emergencies": [], "incidents": []}
        self.connected = False


def build_page(state: DashboardState) -> None:
    ui.page_title("Traffic Control Room")
    ui.add_head_html("<style>body { background: #0f172a; }</style>")

    with ui.header().classes("bg-slate-900 text-white items-center justify-between"):
        ui.label("🚦 Adaptive Traffic Management -- Control Room").classes("text-xl font-bold")
        status_badge = ui.badge("connecting...", color="grey")

    with ui.row().classes("w-full p-4 gap-4"):
        kpi_total = _kpi_card("Vehicles in city", "0", "directions_car")
        kpi_emergency = _kpi_card("Active emergencies", "0", "emergency")
        kpi_incidents = _kpi_card("Incidents (1h)", "0", "report_problem")
        kpi_reporting = _kpi_card("Intersections reporting", "0 / 0", "sensors")
        kpi_wait = _kpi_card("Avg wait time (1h)", "0s", "hourglass_empty")
        kpi_throughput = _kpi_card("Throughput (1h)", "0", "trending_up")

    with ui.row().classes("w-full p-4 gap-4 items-start"):
        with ui.card().classes("w-2/3"):
            ui.label("Live City Map").classes("text-lg font-semibold")
            city_map = ui.leaflet(center=(12.9816, 77.6046), zoom=14).classes("w-full h-96")
            markers: dict[str, object] = {}

        with ui.card().classes("w-1/3"):
            ui.label("Live Alerts").classes("text-lg font-semibold")
            alerts_container = ui.column().classes("w-full gap-1 max-h-96 overflow-y-auto")

    with ui.card().classes("w-full m-4"):
        ui.label("Intersections").classes("text-lg font-semibold")
        columns = [
            {"name": "id", "label": "Intersection", "field": "id", "align": "left"},
            {"name": "phase", "label": "Phase", "field": "phase", "align": "left"},
            {"name": "vehicles", "label": "Vehicles", "field": "vehicles", "align": "right"},
            {"name": "emergency", "label": "Emergency", "field": "emergency", "align": "center"},
            {"name": "issued_by", "label": "Decided by", "field": "issued_by", "align": "left"},
        ]
        table = ui.table(columns=columns, rows=[], row_key="id").classes("w-full")

    with ui.card().classes("w-full m-4"):
        ui.label("Route Planner").classes("text-lg font-semibold")
        with ui.row().classes("items-center gap-2"):
            origin_select = ui.select([], label="Origin").classes("w-48")
            dest_select = ui.select([], label="Destination").classes("w-48")
            route_result = ui.label("")

            async def plan_route() -> None:
                if not origin_select.value or not dest_select.value:
                    return
                result = await state.api.suggest_route(origin_select.value, dest_select.value)
                if result is None:
                    route_result.text = "Route service unavailable."
                    return
                path = " -> ".join(result["path"])
                avoided = " (rerouted around congestion)" if result["congestion_avoided"] else ""
                route_result.text = f"{path} -- ETA {result['estimated_travel_time_s']:.0f}s{avoided}"

            ui.button("Suggest route", on_click=plan_route)

    async def refresh() -> None:
        states = await state.api.get_all_states()
        kpis = await state.api.get_kpis(minutes=60)
        alerts = await state.api.get_recent_alerts(minutes=60)

        state.connected = states is not None
        status_badge.text = "live" if state.connected else "API unreachable"
        status_badge.props(f"color={'green' if state.connected else 'red'}")

        if states is None:
            return
        state.states = states

        summary = summarize_city_state(states)
        kpi_total.set_value(str(summary["total_vehicles"]))
        kpi_emergency.set_value(str(summary["active_emergencies"]))
        kpi_reporting.set_value(f"{summary['intersections_reporting']} / {summary['intersections_total']}")

        if alerts is not None:
            state.alerts = alerts
            kpi_incidents.set_value(str(len(alerts.get("incidents", []))))
            _render_alerts(alerts_container, alerts)

        if kpis is not None:
            state.kpis = kpis
            kpi_summary = summarize_kpis(kpis)
            kpi_wait.set_value(f"{kpi_summary['avg_waiting_time_s']:.0f}s")
            kpi_throughput.set_value(str(kpi_summary["total_throughput"]))

        if not origin_select.options:
            ids = sorted(s["id"] for s in states)
            origin_select.set_options(ids)
            dest_select.set_options(ids)

        for s in states:
            color = marker_color(s)
            if s["id"] not in markers:
                markers[s["id"]] = city_map.generic_layer(
                    name="circleMarker",
                    args=[[s["lat"], s["lon"]], {"radius": 12, "color": color, "fillColor": color, "fillOpacity": 0.85}],
                )
                markers[s["id"]].run_method("bindTooltip", s["name"])
            else:
                markers[s["id"]].run_method("setStyle", {"color": color, "fillColor": color})

        rows = []
        for s in states:
            telemetry, signal = s.get("telemetry"), s.get("signal")
            vehicle_total = sum(lane["vehicle_count"] for lane in telemetry["lanes"]) if telemetry else 0
            has_emergency = bool(telemetry) and any(lane["emergency_vehicle_present"] for lane in telemetry["lanes"])
            rows.append(
                {
                    "id": s["name"],
                    "phase": signal["phase"] if signal else "--",
                    "vehicles": vehicle_total,
                    "emergency": "🚨" if has_emergency else "",
                    "issued_by": signal["issued_by"] if signal else "--",
                }
            )
        table.rows = rows
        table.update()

    ui.timer(get_settings().dashboard_poll_interval_s, refresh)


def _kpi_card(label: str, initial_value: str, icon: str):
    with ui.card().classes("items-center"):
        ui.icon(icon).classes("text-3xl")
        ui.label(label).classes("text-sm text-gray-500")
        value_label = ui.label(initial_value).classes("text-2xl font-bold")
    value_label.set_value = lambda v: setattr(value_label, "text", v)
    return value_label


def _render_alerts(container, alerts: dict) -> None:
    container.clear()
    combined = [
        {"kind": "emergency", **e} for e in alerts.get("emergencies", [])
    ] + [{"kind": "incident", **i} for i in alerts.get("incidents", [])]
    combined.sort(key=lambda a: a["detected_at"], reverse=True)

    with container:
        if not combined:
            ui.label("No alerts in the last hour.").classes("text-gray-500")
        for a in combined[:20]:
            if a["kind"] == "emergency":
                icon = VEHICLE_CLASS_ICONS.get(a["vehicle_class"], "🚨")
                ui.label(f"{icon} {a['intersection_id']} -- {a['vehicle_class']} on {a['approach']} approach ({a['detected_at'][11:19]})").classes(
                    "text-red-400"
                )
            else:
                ui.label(f"⚠️ {a['intersection_id']} -- {a['kind']} (severity {a['severity']:.2f}) ({a['detected_at'][11:19]})").classes(
                    "text-amber-400"
                )


def main() -> None:
    configure_logging()
    settings = get_settings()
    api_client = ApiClient(settings.api_base_url)
    state = DashboardState(api_client)

    @ui.page("/")
    def index() -> None:
        build_page(state)

    nicegui_app.on_shutdown(api_client.close)
    ui.run(host="0.0.0.0", port=settings.dashboard_port, title="Traffic Control Room", reload=False, show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
