from traffic_system.agents.route_allocation_agent import RouteAllocationAgent
from traffic_system.common.schemas import RouteRequest
from traffic_system.env.intersection import ApproachState, IntersectionState


def _fresh_states(city_graph):
    return {
        iid: IntersectionState(iid, approaches={d: ApproachState(lanes=2) for d in ("N", "S", "E", "W")})
        for iid in city_graph.intersection_ids
    }


def test_route_allocation_finds_direct_path_when_uncongested(city_graph):
    states = _fresh_states(city_graph)
    agent = RouteAllocationAgent(city_graph)
    agent.update_congestion(states)

    resp = agent.act(RouteRequest(origin_intersection_id="I_A1", destination_intersection_id="I_A3"))
    assert resp.path == ["I_A1", "I_A2", "I_A3"]
    assert resp.congestion_avoided is False


def test_route_allocation_reroutes_around_direct_congestion(city_graph):
    states = _fresh_states(city_graph)
    # Jam the direction-specific queues that actually carry A1->A2->A3 traffic.
    states["I_A1"].approaches["E"].queue = 150.0
    states["I_A2"].approaches["W"].queue = 150.0
    states["I_A2"].approaches["E"].queue = 150.0
    states["I_A3"].approaches["W"].queue = 150.0

    agent = RouteAllocationAgent(city_graph)
    agent.update_congestion(states)
    resp = agent.act(RouteRequest(origin_intersection_id="I_A1", destination_intersection_id="I_A3"))

    assert resp.path != ["I_A1", "I_A2", "I_A3"]
    assert resp.congestion_avoided is True


def test_route_allocation_congestion_is_directionally_isolated(city_graph):
    """Congestion on I_A1's N approach (a grid-boundary direction with no real
    road) must not spuriously congest the real roads touching I_A1."""
    states = _fresh_states(city_graph)
    states["I_A1"].approaches["N"].queue = 150.0

    agent = RouteAllocationAgent(city_graph)
    agent.update_congestion(states)

    for u, v, data in city_graph.graph.edges(data=True):
        if "I_A1" in (u, v):
            assert data["congestion_factor"] == 1.0


def test_route_allocation_respects_avoid_list(city_graph):
    states = _fresh_states(city_graph)
    agent = RouteAllocationAgent(city_graph)
    agent.update_congestion(states)

    resp = agent.act(
        RouteRequest(
            origin_intersection_id="I_A1",
            destination_intersection_id="I_A3",
            avoid_intersection_ids=["I_A2"],
        )
    )
    assert "I_A2" not in resp.path
    assert resp.path[0] == "I_A1"
    assert resp.path[-1] == "I_A3"
