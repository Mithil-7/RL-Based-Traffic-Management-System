import httpx
import pytest
import pytest_asyncio

from traffic_system.common.schemas import LaneObservation, SignalCommand, SignalPhase, TelemetryEvent
from traffic_system.dashboard.data import ApiClient, marker_color, summarize_city_state, summarize_kpis


@pytest_asyncio.fixture
async def api_client(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as raw_client:
        client = ApiClient.__new__(ApiClient)
        client._client = raw_client
        yield client


@pytest.mark.asyncio
async def test_get_all_states_and_summarize(api_client, redis_state):
    event = TelemetryEvent(
        intersection_id="I_A1",
        lanes=[
            LaneObservation(
                lane_id="x", approach="N", vehicle_count=8, emergency_vehicle_present=True, emergency_vehicle_class="ambulance"
            )
        ],
    )
    redis_state.set_latest_telemetry(event)
    redis_state.set_latest_command(SignalCommand(intersection_id="I_A1", phase=SignalPhase.NS_THROUGH, duration_s=10, issued_by="dqn_agent"))

    states = await api_client.get_all_states()
    assert states is not None
    assert len(states) == 9

    summary = summarize_city_state(states)
    assert summary["total_vehicles"] == 8
    assert summary["active_emergencies"] == 1
    assert summary["intersections_reporting"] == 1


def test_marker_color_priorities_emergency_over_phase():
    state_with_emergency = {
        "telemetry": {"lanes": [{"vehicle_count": 1, "emergency_vehicle_present": True}]},
        "signal": {"phase": "NS_THROUGH"},
    }
    assert marker_color(state_with_emergency) == "#ef4444"


def test_marker_color_no_data():
    assert marker_color({"telemetry": None, "signal": None}) == "#9ca3af"


def test_marker_color_by_phase():
    state = {"telemetry": {"lanes": []}, "signal": {"phase": "EW_THROUGH"}}
    assert marker_color(state) == "#3b82f6"


def test_summarize_kpis_averages_wait_and_sums_throughput():
    kpis = [
        {"avg_waiting_time_s": 10.0, "throughput_vehicles": 50},
        {"avg_waiting_time_s": 20.0, "throughput_vehicles": 30},
    ]
    result = summarize_kpis(kpis)
    assert result == {"avg_waiting_time_s": 15.0, "total_throughput": 80}


def test_summarize_kpis_handles_empty_list():
    assert summarize_kpis([]) == {"avg_waiting_time_s": 0.0, "total_throughput": 0}


@pytest.mark.asyncio
async def test_api_client_fails_gracefully_when_unreachable():
    async with httpx.AsyncClient(base_url="http://localhost:1", timeout=0.5) as raw_client:
        client = ApiClient.__new__(ApiClient)
        client._client = raw_client
        assert await client.get_all_states() is None
        assert await client.get_kpis() is None
