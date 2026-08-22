"""Vehicle detection backends.

`MockVehicleDetector` reads the ground-truth vehicle metadata that
`SimulatedVideoSource` attaches to each frame -- effectively a "perfect"
detector, used for local development, CI, and demos where the point is to
exercise the rest of the pipeline (MQTT, ingestion, brain, dashboard) without
needing a GPU or trained weights.

`YoloVehicleDetector` runs real Ultralytics YOLOv8 inference and buckets
detections into the four compass approaches by their position in frame
(quadrant heuristic -- replace with a calibrated per-camera ROI mapping for
a real deployment, see the module docstring note below). This is the
backend used with `RaspberryPiCameraSource` in production; it is not
exercised by this repo's test suite since it needs `pip install ultralytics`
plus downloaded weights, but the code path is real and complete.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from traffic_system.common.config import get_settings
from traffic_system.common.logging import get_logger
from traffic_system.edge.video_source import Frame

logger = get_logger(__name__)

# Ultralytics COCO class names we care about, mapped to our VehicleClass values.
COCO_VEHICLE_CLASSES = {"car": "car", "bus": "bus", "truck": "truck", "motorcycle": "motorcycle"}


@dataclass
class ApproachDetection:
    vehicle_count: int
    class_counts: dict[str, int]
    emergency_present: bool = False
    emergency_class: str | None = None
    emergency_confidence: float = 0.0


class VehicleDetector(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> dict[str, ApproachDetection]:
        """Returns one ApproachDetection per compass approach ('N','S','E','W')."""


class MockVehicleDetector(VehicleDetector):
    def detect(self, frame: Frame) -> dict[str, ApproachDetection]:
        if frame.ground_truth is None:
            raise ValueError("MockVehicleDetector requires a frame with ground_truth (i.e. a simulated frame)")
        result = {}
        for direction, gt in frame.ground_truth.items():
            result[direction] = ApproachDetection(
                vehicle_count=gt["vehicle_count"],
                class_counts=gt["class_counts"],
                emergency_present=gt["emergency_present"],
                emergency_class=gt["emergency_class"],
                emergency_confidence=0.99 if gt["emergency_present"] else 0.0,
            )
        return result


class YoloVehicleDetector(VehicleDetector):
    """Real YOLOv8 inference. NOTE on approach assignment: this quadrant
    heuristic (which quarter of the frame a detection's centroid falls in)
    is only correct for a camera mounted directly overhead, centered on the
    intersection. A real deployment should replace `_assign_approach` with
    a calibrated per-camera region-of-interest polygon set (one per
    approach, drawn once during installation) -- that calibration step is
    intentionally left as an installation-time config, not hardcoded here.
    """

    def __init__(self, weights_path: str | None = None, confidence: float | None = None) -> None:
        settings = get_settings()
        self.weights_path = weights_path or settings.yolo_weights
        self.confidence = confidence if confidence is not None else settings.yolo_confidence
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised only without ultralytics installed
            raise RuntimeError(
                "ultralytics is not installed. Run `pip install ultralytics` to use YoloVehicleDetector, "
                "or set TRAFFIC_CV_BACKEND=mock to use the simulated ground-truth detector instead."
            ) from exc
        logger.info("yolo.loading_weights", weights=self.weights_path)
        self._model = YOLO(self.weights_path)

    def _assign_approach(self, cx: float, cy: float, width: int, height: int) -> str:
        dx, dy = cx - width / 2, cy - height / 2
        if abs(dx) > abs(dy):
            return "E" if dx > 0 else "W"
        return "S" if dy > 0 else "N"

    def detect(self, frame: Frame) -> dict[str, ApproachDetection]:
        self._ensure_loaded()
        h, w = frame.image.shape[:2]
        results = self._model(frame.image, conf=self.confidence, verbose=False)[0]

        counts: dict[str, dict[str, int]] = {d: {} for d in ("N", "S", "E", "W")}
        for box in results.boxes:
            cls_name = results.names[int(box.cls[0])]
            if cls_name not in COCO_VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            approach = self._assign_approach((x1 + x2) / 2, (y1 + y2) / 2, w, h)
            mapped = COCO_VEHICLE_CLASSES[cls_name]
            counts[approach][mapped] = counts[approach].get(mapped, 0) + 1

        return {
            d: ApproachDetection(vehicle_count=sum(c.values()), class_counts=c) for d, c in counts.items()
        }
