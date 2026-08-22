"""The main edge process: one instance runs per intersection (in production,
one per Raspberry Pi). Loop:

  1. Read a frame from the configured `VideoSource`.
  2. Run the configured `VehicleDetector` + `EmergencyVehicleDetector`.
  3. Assemble a `TelemetryEvent` and publish it over MQTT.
  4. Apply any `SignalCommand` most recently received from the brain (in
     simulated mode, this is what drives `SyntheticTrafficGenerator`
     forward -- a real intersection would just... physically have a green
     light, no "apply" step needed).

Swapping from simulation to real hardware is a one-line change: construct
`RaspberryPiCameraSource` instead of `SimulatedVideoSource` and
`YoloVehicleDetector` instead of `MockVehicleDetector` -- everything else
(the loop, the MQTT contract, the schemas) is identical.
"""
from __future__ import annotations

import argparse
import time

from traffic_system.common.config import get_settings
from traffic_system.common.logging import configure_logging, get_logger
from traffic_system.common.mqtt_client import MqttClient
from traffic_system.common.schemas import LaneObservation, SignalCommand, SignalPhase, TelemetryEvent, VehicleClass
from traffic_system.edge.emergency_detector import EmergencyVehicleDetector, HeuristicEmergencyDetector
from traffic_system.edge.traffic_generator import APPROACH_ORDER, SyntheticTrafficGenerator
from traffic_system.edge.vehicle_detector import MockVehicleDetector, VehicleDetector, YoloVehicleDetector
from traffic_system.edge.video_source import SimulatedVideoSource, VideoSource

logger = get_logger(__name__)


class EdgeAgent:
    def __init__(
        self,
        intersection_id: str,
        video_source: VideoSource,
        vehicle_detector: VehicleDetector,
        emergency_detector: EmergencyVehicleDetector,
        mqtt_client: MqttClient,
        traffic_generator: SyntheticTrafficGenerator | None = None,
        detection_interval_s: float = 2.0,
    ) -> None:
        self.intersection_id = intersection_id
        self.video_source = video_source
        self.vehicle_detector = vehicle_detector
        self.emergency_detector = emergency_detector
        self.mqtt = mqtt_client
        self.traffic_generator = traffic_generator  # only set in simulated mode
        self.detection_interval_s = detection_interval_s

        self._current_phase = SignalPhase.NS_THROUGH
        settings = get_settings()
        self._telemetry_topic = f"traffic/telemetry/{intersection_id}"
        self._command_topic = f"{settings.mqtt_command_topic_prefix}/{intersection_id}"

    def start(self) -> None:
        self.mqtt.connect()
        self.mqtt.subscribe_json(self._command_topic, self._on_command)
        logger.info("edge_agent.started", intersection_id=self.intersection_id)

    def _on_command(self, topic: str, payload: dict) -> None:
        try:
            command = SignalCommand.model_validate(payload)
        except Exception:
            logger.exception("edge_agent.bad_command", topic=topic)
            return
        self._current_phase = command.phase
        logger.debug("edge_agent.command_received", phase=command.phase.value, reason=command.reason)

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                self.step()
                time.sleep(self.detection_interval_s)
        except KeyboardInterrupt:
            logger.info("edge_agent.shutting_down")
        finally:
            self.video_source.release()
            self.mqtt.disconnect()

    def step(self) -> TelemetryEvent:
        if self.traffic_generator is not None:
            self.traffic_generator.tick(self._current_phase, dt=self.detection_interval_s)

        t0 = time.monotonic()
        frame = self.video_source.read()
        if frame is None:
            logger.warning("edge_agent.frame_read_failed", intersection_id=self.intersection_id)
            raise RuntimeError("Camera returned no frame")

        vehicle_detections = self.vehicle_detector.detect(frame)
        emergency_detections = self.emergency_detector.detect(frame)
        processing_ms = (time.monotonic() - t0) * 1000

        lanes = []
        for direction in APPROACH_ORDER:
            v = vehicle_detections[direction]
            e = emergency_detections.get(direction)
            emergency_present = v.emergency_present or (e.present if e else False)
            emergency_class = None
            if v.emergency_present and v.emergency_class:
                emergency_class = VehicleClass(v.emergency_class)
            elif e and e.present:
                emergency_class = e.vehicle_class

            lanes.append(
                LaneObservation(
                    lane_id=f"{self.intersection_id}-{direction}",
                    approach=direction,
                    vehicle_count=v.vehicle_count,
                    class_counts=v.class_counts,
                    emergency_vehicle_present=emergency_present,
                    emergency_vehicle_class=emergency_class,
                    emergency_confidence=max(v.emergency_confidence, e.confidence if e else 0.0),
                )
            )

        event = TelemetryEvent(
            intersection_id=self.intersection_id,
            lanes=lanes,
            source="simulated" if self.traffic_generator is not None else "raspberry_pi",
            frame_processing_ms=round(processing_ms, 2),
        )
        self.mqtt.publish_json(self._telemetry_topic, event.model_dump(mode="json"))
        return event


def build_simulated_agent(intersection_id: str, seed: int | None = None) -> EdgeAgent:
    settings = get_settings()
    generator = SyntheticTrafficGenerator(intersection_id, seed=seed)
    video_source = SimulatedVideoSource(generator)
    vehicle_detector = YoloVehicleDetector() if settings.cv_backend == "yolo" else MockVehicleDetector()
    emergency_detector = HeuristicEmergencyDetector()
    mqtt_client = MqttClient(client_id_suffix=f"edge-{intersection_id}")
    return EdgeAgent(
        intersection_id=intersection_id,
        video_source=video_source,
        vehicle_detector=vehicle_detector,
        emergency_detector=emergency_detector,
        mqtt_client=mqtt_client,
        traffic_generator=generator,
        detection_interval_s=settings.decision_interval_seconds,
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run a (simulated, by default) edge agent for one intersection.")
    parser.add_argument("--intersection-id", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    agent = build_simulated_agent(args.intersection_id, seed=args.seed)
    agent.run_forever()


if __name__ == "__main__":
    main()
