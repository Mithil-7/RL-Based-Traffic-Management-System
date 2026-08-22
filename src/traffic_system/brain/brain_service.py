"""The brain: ties every agent together into one decision cycle, run on a
fixed interval (`TRAFFIC_DECISION_INTERVAL_SECONDS`, default 5s) for every
intersection in the city graph.

Priority order applied each cycle (highest wins):

  1. Emergency Preemption (hard override) -- if a CV-detected emergency
     vehicle is present at an intersection, its phase is forced,
     unconditionally, subject only to the physical yellow/all-red clearance.
  2. Coordinator (spillback veto) -- may override the DQN's choice at an
     intersection to avoid discharging into an already-saturated
     downstream link.
  3. Corridor bias (soft nudge) -- intersections on a projected emergency
     corridor are nudged toward the matching phase, but only when doing so
     doesn't fight the coordinator's safety veto.
  4. DQN proposal -- the default, learned decision.

Every decision that reaches this point is still passed through the exact
same `SafetyLayer` used in training (min/max green, mandatory yellow +
all-red clearance) before becoming a `SignalCommand` -- this is what
guarantees training-time and production behavior can't diverge on the one
thing that actually has to be bulletproof.
"""
from __future__ import annotations

import time

from traffic_system.agents.coordinator_agent import CoordinatorAgent
from traffic_system.agents.dqn_agent import IntersectionDQNAgent
from traffic_system.agents.emergency_preemption_agent import EmergencyPreemptionAgent
from traffic_system.agents.incident_detection_agent import IncidentDetectionAgent
from traffic_system.agents.route_allocation_agent import RouteAllocationAgent
from traffic_system.common.config import get_settings
from traffic_system.common.logging import get_logger
from traffic_system.common.mqtt_client import MqttClient
from traffic_system.common.schemas import SignalCommand, SignalPhase
from traffic_system.env.city_graph import CityGraph
from traffic_system.env.intersection import ApproachState, IntersectionState, SafetyLayer
from traffic_system.env.traffic_grid_env import ACTION_PHASES, OBS_DIM, build_observation
from traffic_system.ingestion.redis_state import RedisState

logger = get_logger(__name__)

CORRIDOR_BIAS_OVERRIDE_THRESHOLD = 0.25


class BrainService:
    def __init__(
        self,
        city_graph: CityGraph,
        dqn_agents: dict[str, IntersectionDQNAgent],
        redis_state: RedisState,
        mqtt_client: MqttClient,
        session_factory=None,
        decision_interval_s: float | None = None,
    ) -> None:
        settings = get_settings()
        self.city_graph = city_graph
        self.dqn_agents = dqn_agents
        self.redis_state = redis_state
        self.mqtt = mqtt_client
        self.session_factory = session_factory
        self.decision_interval_s = decision_interval_s or settings.decision_interval_seconds

        self.safety = SafetyLayer(
            settings.min_green_seconds, settings.max_green_seconds, settings.yellow_seconds, settings.all_red_seconds
        )
        self.coordinator = CoordinatorAgent(city_graph)
        self.emergency_agent = EmergencyPreemptionAgent(city_graph)
        self.route_agent = RouteAllocationAgent(city_graph)
        self.incident_agent = IncidentDetectionAgent()

        self.states: dict[str, IntersectionState] = {
            iid: IntersectionState(iid, approaches={d: ApproachState(lanes=2) for d in ("N", "S", "E", "W")})
            for iid in city_graph.intersection_ids
        }
        self._max_green_s = settings.max_green_seconds

    def start(self) -> None:
        self.mqtt.connect()
        logger.info("brain.started", intersections=len(self.city_graph.intersection_ids))

    def _refresh_states_from_telemetry(self) -> None:
        """Pull the latest CV telemetry from Redis and update each
        intersection's in-memory queueing state. Vehicle counts are an
        absolute snapshot (how many vehicles the CV pipeline currently sees
        queued), not a delta, so we overwrite rather than accumulate.
        Discharge is *estimated* as the queue reduction on approaches that
        were actually served last cycle -- a live-telemetry approximation of
        the exact accounting the training simulator can do directly."""
        for iid, state in self.states.items():
            event = self.redis_state.get_latest_telemetry(iid)
            if event is None:
                continue
            served_last_cycle = (
                ("N", "S") if state.current_phase == SignalPhase.NS_THROUGH
                else ("E", "W") if state.current_phase == SignalPhase.EW_THROUGH
                else ()
            )
            lanes_by_approach = {lane.approach: lane for lane in event.lanes}
            for direction, approach_state in state.approaches.items():
                lane = lanes_by_approach.get(direction)
                if lane is None:
                    continue
                old_queue = approach_state.queue
                new_queue = float(lane.vehicle_count)
                if direction in served_last_cycle and new_queue < old_queue:
                    state.total_discharged += int(old_queue - new_queue)
                approach_state.queue = new_queue
                approach_state.emergency_present = lane.emergency_vehicle_present
                approach_state.emergency_class = lane.emergency_vehicle_class

    def decision_cycle(self) -> dict[str, SignalCommand]:
        self._refresh_states_from_telemetry()

        proposed_actions: dict[str, int] = {}
        for iid, state in self.states.items():
            obs = build_observation(state, self._max_green_s)
            agent = self.dqn_agents.get(iid)
            proposed_actions[iid] = agent.act(obs, explore=False) if agent is not None else 0

        final_actions, override_reasons = self.coordinator.act(proposed_actions, self.states)
        forced_phases, corridor_bias, emergency_alerts = self.emergency_agent.act(self.states)

        for iid, (preferred_phase, strength) in corridor_bias.items():
            if iid in forced_phases or iid not in final_actions:
                continue
            if strength < CORRIDOR_BIAS_OVERRIDE_THRESHOLD:
                continue
            preferred_index = ACTION_PHASES.index(preferred_phase)
            if final_actions[iid] != preferred_index:
                final_actions[iid] = preferred_index
                override_reasons[iid] = "emergency_corridor_bias"

        commands: dict[str, SignalCommand] = {}
        for iid, state in self.states.items():
            action_index = final_actions.get(iid, 0)
            forced = forced_phases.get(iid)
            was_transitioning = state.transitioning
            state = self.safety.resolve(state, ACTION_PHASES[action_index], self.decision_interval_s, forced)
            self.states[iid] = state

            issued_by = "emergency_preemption" if forced else override_reasons.get(iid, "dqn_agent")
            reason = override_reasons.get(iid, "")
            active_phase = state.pending_phase if state.transitioning and not was_transitioning else state.current_phase

            command = SignalCommand(
                intersection_id=iid,
                phase=active_phase if not state.transitioning else SignalPhase.ALL_RED,
                duration_s=self.decision_interval_s,
                issued_by=issued_by,
                reason=reason,
                preempted=forced is not None,
            )
            commands[iid] = command
            self._publish_command(command)

        self.route_agent.update_congestion(self.states)

        detected_incidents = []
        for iid, state in self.states.items():
            incident = self.incident_agent.act(iid, state)
            if incident is not None:
                detected_incidents.append(incident)
                logger.warning("brain.incident_detected", intersection_id=iid, kind=incident.kind, severity=incident.severity)
                self.redis_state.publish_update("incident", incident.model_dump(mode="json"))

        for alert in emergency_alerts:
            logger.info("brain.emergency_alert", intersection_id=alert.intersection_id, vehicle_class=alert.vehicle_class.value)
            self.redis_state.publish_update("emergency", alert.model_dump(mode="json"))

        self._persist_alerts(emergency_alerts, detected_incidents)
        self.redis_state.set_congestion_snapshot(self.city_graph.export_congestion())
        self._persist_snapshots()

        return commands

    def _persist_snapshots(self) -> None:
        """Write one row per intersection per cycle -- see
        `IntersectionSnapshotRecord` docstring for why this exists (honest,
        directly-measured throughput for the KPI API, not an estimate)."""
        if self.session_factory is None:
            return
        from traffic_system.ingestion.db_models import IntersectionSnapshotRecord

        with self.session_factory() as session:
            for iid, state in self.states.items():
                session.add(
                    IntersectionSnapshotRecord(
                        intersection_id=iid,
                        total_queue=state.total_queue(),
                        total_discharged=state.total_discharged,
                        total_switches=state.total_switches,
                        current_phase=state.current_phase.value,
                    )
                )
            session.commit()

    def _publish_command(self, command: SignalCommand) -> None:
        settings = get_settings()
        topic = f"{settings.mqtt_command_topic_prefix}/{command.intersection_id}"
        self.mqtt.publish_json(topic, command.model_dump(mode="json"))
        self.redis_state.set_latest_command(command)
        self.redis_state.publish_update("signal_command", command.model_dump(mode="json"))
        self._persist_command(command)

    def _persist_command(self, command: SignalCommand) -> None:
        if self.session_factory is None:
            return
        from traffic_system.ingestion.db_models import SignalCommandRecord

        with self.session_factory() as session:
            session.add(
                SignalCommandRecord(
                    intersection_id=command.intersection_id,
                    phase=command.phase.value,
                    duration_s=command.duration_s,
                    issued_by=command.issued_by,
                    reason=command.reason,
                    preempted=command.preempted,
                )
            )
            session.commit()

    def _persist_alerts(self, emergency_alerts, incidents) -> None:
        if self.session_factory is None:
            return
        from traffic_system.ingestion.db_models import EmergencyAlertRecord, IncidentAlertRecord

        with self.session_factory() as session:
            for alert in emergency_alerts:
                session.add(
                    EmergencyAlertRecord(
                        intersection_id=alert.intersection_id,
                        vehicle_class=alert.vehicle_class.value,
                        approach=alert.approach,
                        confidence=alert.confidence,
                        corridor=alert.corridor,
                    )
                )
            for incident in incidents:
                session.add(
                    IncidentAlertRecord(
                        intersection_id=incident.intersection_id,
                        kind=incident.kind,
                        severity=incident.severity,
                        description=incident.description,
                    )
                )
            session.commit()

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                t0 = time.monotonic()
                self.decision_cycle()
                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, self.decision_interval_s - elapsed))
        except KeyboardInterrupt:
            logger.info("brain.shutting_down")
        finally:
            self.mqtt.disconnect()


def main() -> None:
    import argparse

    from traffic_system.common.logging import configure_logging
    from traffic_system.ingestion.db_models import init_db, make_engine, make_session_factory

    configure_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the brain service (multi-agent decision loop) for every intersection.")
    parser.add_argument("--city-graph", default=str(settings.city_graph_path))
    parser.add_argument("--model-dir", default=str(settings.model_dir), help="Directory of per-intersection {id}.npz/.pt checkpoints")
    args = parser.parse_args()

    city_graph = CityGraph.load(args.city_graph)

    dqn_agents: dict[str, IntersectionDQNAgent] = {}
    for iid in city_graph.intersection_ids:
        agent = IntersectionDQNAgent(obs_dim=OBS_DIM, n_actions=len(ACTION_PHASES), intersection_id=iid)
        import os

        per_intersection_path = f"{args.model_dir}/{iid}.npz"
        shared_path = f"{args.model_dir}/shared.npz"
        if os.path.exists(per_intersection_path):
            agent.load(per_intersection_path)
            logger.info("brain.loaded_checkpoint", intersection_id=iid, path=per_intersection_path)
        elif os.path.exists(shared_path):
            agent.load(shared_path)
            logger.info("brain.loaded_shared_checkpoint", intersection_id=iid, path=shared_path)
        else:
            logger.warning("brain.no_checkpoint_found_using_untrained_agent", intersection_id=iid)
        dqn_agents[iid] = agent

    engine = make_engine(settings.postgres_dsn)
    init_db(engine)
    session_factory = make_session_factory(engine)

    service = BrainService(
        city_graph=city_graph,
        dqn_agents=dqn_agents,
        redis_state=RedisState(),
        mqtt_client=MqttClient(client_id_suffix="brain"),
        session_factory=session_factory,
    )
    service.run_forever()


if __name__ == "__main__":
    main()
