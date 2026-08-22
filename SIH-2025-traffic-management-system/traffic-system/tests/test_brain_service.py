import fakeredis
import pytest

from fakes import FakeMqttClient
from traffic_system.agents.dqn_agent import IntersectionDQNAgent
from traffic_system.brain.brain_service import BrainService
from traffic_system.common.schemas import LaneObservation, TelemetryEvent, VehicleClass
from traffic_system.ingestion.redis_state import RedisState


@pytest.fixture
def redis_state():
    return RedisState(client=fakeredis.FakeStrictRedis(decode_responses=True))


@pytest.fixture
def dqn_agents(city_graph):
    return {
        iid: IntersectionDQNAgent(obs_dim=16, n_actions=2, backend="numpy", intersection_id=iid, seed=i)
        for i, iid in enumerate(city_graph.intersection_ids)
    }


def _seed_telemetry(redis_state, intersection_id, counts, emergency_on=None):
    lanes = [
        LaneObservation(
            lane_id=f"{intersection_id}-{d}",
            approach=d,
            vehicle_count=counts.get(d, 0),
            emergency_vehicle_present=(d == emergency_on),
            emergency_vehicle_class=VehicleClass.AMBULANCE if d == emergency_on else None,
        )
        for d in ("N", "S", "E", "W")
    ]
    redis_state.set_latest_telemetry(TelemetryEvent(intersection_id=intersection_id, lanes=lanes))


def test_decision_cycle_issues_one_command_per_intersection(city_graph, dqn_agents, redis_state):
    brain = BrainService(city_graph, dqn_agents, redis_state, FakeMqttClient(), decision_interval_s=5.0)
    brain.start()
    for iid in city_graph.intersection_ids:
        _seed_telemetry(redis_state, iid, {"N": 3, "S": 2, "E": 1, "W": 0})

    commands = brain.decision_cycle()
    assert set(commands.keys()) == set(city_graph.intersection_ids)
    assert len(brain.mqtt.published) == len(city_graph.intersection_ids)


def test_decision_cycle_publishes_to_redis_for_dashboard(city_graph, dqn_agents, redis_state):
    brain = BrainService(city_graph, dqn_agents, redis_state, FakeMqttClient(), decision_interval_s=5.0)
    brain.start()
    for iid in city_graph.intersection_ids:
        _seed_telemetry(redis_state, iid, {"N": 1, "S": 1, "E": 1, "W": 1})

    brain.decision_cycle()
    for iid in city_graph.intersection_ids:
        cmd = redis_state.get_latest_command(iid)
        assert cmd is not None
        assert cmd.intersection_id == iid


def test_emergency_preemption_flows_through_full_cycle(city_graph, dqn_agents, redis_state):
    brain = BrainService(city_graph, dqn_agents, redis_state, FakeMqttClient(), decision_interval_s=5.0)
    brain.start()
    for iid in city_graph.intersection_ids:
        _seed_telemetry(redis_state, iid, {"N": 0, "S": 0, "E": 0, "W": 0})
    _seed_telemetry(redis_state, "I_B2", {"N": 0, "S": 0, "E": 0, "W": 5}, emergency_on="W")

    cmd1 = brain.decision_cycle()["I_B2"]
    assert cmd1.issued_by == "emergency_preemption"
    assert cmd1.preempted is True

    # keep the emergency flag set (as CV would, until the vehicle actually clears)
    # and advance one more cycle to get past the mandatory yellow/all-red clearance.
    cmd2 = brain.decision_cycle()["I_B2"]
    assert cmd2.phase.value == "EW_THROUGH"
    assert cmd2.preempted is True


def test_incident_publishes_alert_to_redis(city_graph, dqn_agents, redis_state):
    brain = BrainService(city_graph, dqn_agents, redis_state, FakeMqttClient(), decision_interval_s=5.0)
    brain.start()
    pubsub = redis_state.subscribe_updates()

    # Simulate a stalled approach: queue keeps growing across cycles despite green time.
    for i in range(8):
        counts = {"N": 10 + i * 4, "S": 0, "E": 0, "W": 0}
        for iid in city_graph.intersection_ids:
            _seed_telemetry(redis_state, iid, counts if iid == "I_A1" else {})
        brain.decision_cycle()

    # Drain the pubsub channel and confirm at least one incident alert was published.
    # (72 signal_command messages are published before any incident message in this
    # scenario -- 9 intersections x 8 cycles -- so we need to drain generously.)
    kinds_seen = []
    for _ in range(200):
        msg = pubsub.get_message(timeout=0.01)
        if msg and msg["type"] == "message":
            import json

            kinds_seen.append(json.loads(msg["data"])["kind"])
    assert "incident" in kinds_seen
