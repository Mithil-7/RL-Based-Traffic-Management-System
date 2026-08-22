from traffic_system.common.schemas import LaneObservation, SignalCommand, SignalPhase, TelemetryEvent
from traffic_system.ingestion.db_models import IntersectionSnapshotRecord, TelemetryRecord


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_intersections(client, city_graph):
    r = client.get("/api/intersections")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == len(city_graph.intersection_ids)
    assert {i["id"] for i in body} == set(city_graph.intersection_ids)


def test_intersection_state_404_when_no_data(client):
    r = client.get("/api/intersections/I_A1/state")
    assert r.status_code == 404


def test_intersection_state_returns_live_data(client, redis_state):
    event = TelemetryEvent(intersection_id="I_A1", lanes=[LaneObservation(lane_id="x", approach="N", vehicle_count=7)])
    redis_state.set_latest_telemetry(event)
    cmd = SignalCommand(intersection_id="I_A1", phase=SignalPhase.NS_THROUGH, duration_s=10, issued_by="dqn_agent")
    redis_state.set_latest_command(cmd)

    r = client.get("/api/intersections/I_A1/state")
    assert r.status_code == 200
    body = r.json()
    assert body["telemetry"]["lanes"][0]["vehicle_count"] == 7
    assert body["signal"]["phase"] == "NS_THROUGH"


def test_intersection_history(client, session_factory):
    with session_factory() as session:
        session.add(
            TelemetryRecord(intersection_id="I_A1", total_vehicles=5, has_emergency=False, frame_processing_ms=10.0, payload={})
        )
        session.commit()

    r = client.get("/api/intersections/I_A1/history?minutes=60")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["total_vehicles"] == 5


def test_kpis_computes_throughput_from_snapshots(client, session_factory):
    with session_factory() as session:
        session.add(IntersectionSnapshotRecord(intersection_id="I_A1", total_queue=7, total_discharged=0, total_switches=0, current_phase="NS_THROUGH"))
        session.add(IntersectionSnapshotRecord(intersection_id="I_A1", total_queue=3, total_discharged=15, total_switches=1, current_phase="NS_THROUGH"))
        session.add(TelemetryRecord(intersection_id="I_A1", total_vehicles=5, has_emergency=False, frame_processing_ms=8.0, payload={}))
        session.commit()

    r = client.get("/api/metrics/kpis?minutes=60")
    assert r.status_code == 200
    kpis = {k["intersection_id"]: k for k in r.json()}
    assert kpis["I_A1"]["throughput_vehicles"] == 15
    assert kpis["I_A1"]["avg_queue_vehicles"] == 5.0
    assert kpis["I_A2"]["throughput_vehicles"] == 0  # no snapshots -> honestly zero, not fabricated


def test_prometheus_metrics_endpoint(client, redis_state):
    event = TelemetryEvent(intersection_id="I_A1", lanes=[LaneObservation(lane_id="x", approach="N", vehicle_count=9)])
    redis_state.set_latest_telemetry(event)

    r = client.get("/metrics")
    assert r.status_code == 200
    assert "traffic_intersection_queue_vehicles" in r.text
    assert 'intersection_id="I_A1"} 9.0' in r.text


def test_route_suggestion(client):
    r = client.post(
        "/api/routes/suggest",
        json={"origin_intersection_id": "I_A1", "destination_intersection_id": "I_A3"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == ["I_A1", "I_A2", "I_A3"]


def test_route_suggestion_unknown_intersection(client):
    r = client.post(
        "/api/routes/suggest",
        json={"origin_intersection_id": "NOPE", "destination_intersection_id": "I_A3"},
    )
    assert r.status_code == 404


def test_websocket_receives_published_updates(client, redis_state):
    with client.websocket_connect("/ws") as ws:
        redis_state.publish_update("telemetry", {"intersection_id": "I_A1", "total_vehicles": 3})
        data = ws.receive_json()
        assert data["kind"] == "telemetry"
        assert data["payload"]["intersection_id"] == "I_A1"
