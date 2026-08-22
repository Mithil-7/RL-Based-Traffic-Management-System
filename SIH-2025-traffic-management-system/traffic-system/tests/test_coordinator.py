from traffic_system.agents.coordinator_agent import CoordinatorAgent
from traffic_system.agents.emergency_preemption_agent import EmergencyPreemptionAgent
from traffic_system.agents.incident_detection_agent import IncidentDetectionAgent
from traffic_system.common.schemas import VehicleClass
from traffic_system.env.intersection import ApproachState, IntersectionState


def _fresh_states(city_graph):
    return {
        iid: IntersectionState(iid, approaches={d: ApproachState(lanes=2) for d in ("N", "S", "E", "W")})
        for iid in city_graph.intersection_ids
    }


def test_coordinator_overrides_spillback_risk(city_graph):
    states = _fresh_states(city_graph)
    # I_B2's S approach wants to discharge into I_A2's N approach, which is jammed.
    states["I_A2"].approaches["N"].queue = 55.0
    states["I_B2"].approaches["S"].queue = 20.0

    coordinator = CoordinatorAgent(city_graph, spillback_queue_threshold=40.0)
    final, reasons = coordinator.act({"I_B2": 0}, states)  # 0 = NS_THROUGH

    assert final["I_B2"] == 1  # forced to EW_THROUGH instead
    assert reasons["I_B2"] == "spillback_prevention"


def test_coordinator_leaves_safe_actions_untouched(city_graph):
    states = _fresh_states(city_graph)
    coordinator = CoordinatorAgent(city_graph, spillback_queue_threshold=40.0)
    final, reasons = coordinator.act({"I_B2": 0}, states)
    assert final["I_B2"] == 0
    assert "I_B2" not in reasons


def test_coordinator_does_not_override_when_both_directions_are_congested(city_graph):
    states = _fresh_states(city_graph)
    states["I_A2"].approaches["N"].queue = 55.0  # downstream for NS_THROUGH (action 0)
    states["I_B1"].approaches["E"].queue = 55.0  # downstream for EW_THROUGH (action 1)
    states["I_B3"].approaches["W"].queue = 55.0
    states["I_B2"].approaches["S"].queue = 20.0

    coordinator = CoordinatorAgent(city_graph, spillback_queue_threshold=40.0)
    final, reasons = coordinator.act({"I_B2": 0}, states)
    # Both alternatives are risky -- coordinator should defer to the DQN's original choice.
    assert final["I_B2"] == 0
    assert "I_B2" not in reasons


def test_emergency_preemption_forces_matching_phase_and_projects_corridor(city_graph):
    states = _fresh_states(city_graph)
    states["I_A1"].approaches["W"].emergency_present = True
    states["I_A1"].approaches["W"].emergency_class = VehicleClass.AMBULANCE

    agent = EmergencyPreemptionAgent(city_graph, corridor_depth=2)
    forced, bias, alerts = agent.act(states)

    assert len(alerts) == 1
    assert alerts[0].intersection_id == "I_A1"
    assert alerts[0].vehicle_class == VehicleClass.AMBULANCE
    assert forced["I_A1"].value == "EW_THROUGH"
    assert alerts[0].corridor == ["I_A2", "I_A3"]
    assert bias["I_A2"][0].value == "EW_THROUGH" and bias["I_A2"][1] > 0
    assert bias["I_A3"][0].value == "EW_THROUGH" and bias["I_A3"][1] > 0


def test_emergency_preemption_noop_when_no_emergency(city_graph):
    states = _fresh_states(city_graph)
    agent = EmergencyPreemptionAgent(city_graph)
    forced, bias, alerts = agent.act(states)
    assert forced == {}
    assert bias == {}
    assert alerts == []


def test_incident_detection_flags_stalled_growth():
    agent = IncidentDetectionAgent(window=5, min_absolute_queue=5.0)
    state = IntersectionState("I_X", approaches={d: ApproachState(lanes=2) for d in ("N", "S", "E", "W")})

    result = None
    for _ in range(6):
        state.approaches["N"].queue += 4  # growing, never discharged
        result = agent.act("I_X", state)

    assert result is not None
    assert result.kind == "unusual_queue_growth"
    assert 0.0 < result.severity <= 1.0


def test_incident_detection_ignores_normal_flow():
    agent = IncidentDetectionAgent(window=5, min_absolute_queue=5.0)
    state = IntersectionState("I_X", approaches={d: ApproachState(lanes=2) for d in ("N", "S", "E", "W")})

    result = None
    for _ in range(6):
        state.approaches["N"].queue = 3.0  # steady, low, and being served
        state.total_discharged += 3
        result = agent.act("I_X", state)

    assert result is None
