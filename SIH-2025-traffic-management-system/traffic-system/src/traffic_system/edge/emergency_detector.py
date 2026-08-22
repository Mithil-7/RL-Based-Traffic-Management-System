"""Emergency vehicle detection.

`HeuristicEmergencyDetector` looks for the characteristic red+blue
strobe-light-bar color signature in the frame using HSV color thresholding
and connected-component analysis, then buckets any detected cluster into a
compass approach using the same quadrant heuristic as `YoloVehicleDetector`.
This is a legitimate, if simple, computer-vision technique -- it is what
`SimulatedVideoSource` renders (a strobing red/blue rectangle on the lead
vehicle when `ApproachState.emergency_present` is true) specifically so this
detector can be exercised against realistic-ish synthetic imagery in tests,
not just against ground-truth metadata.

For a real deployment, `ModelEmergencyDetector` is the documented upgrade
path: a small trained classifier (e.g. a lightweight CNN or fine-tuned
YOLO class) run on each vehicle detection's crop, which would be far more
robust than color thresholding to lighting, occlusion, and non-flashing
emergency vehicles. It's stubbed here with a clear interface so it's a
drop-in swap (`TRAFFIC_EMERGENCY_BACKEND=model`) once that model exists --
training it needs a labeled emergency-vehicle dataset this repo doesn't
ship with.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from traffic_system.common.schemas import VehicleClass
from traffic_system.edge.video_source import Frame

# HSV thresholds tuned for saturated red/blue (typical light-bar colors).
# OpenCV hue range is 0-179.
_RED_RANGES = [((0, 120, 120), (10, 255, 255)), ((170, 120, 120), (179, 255, 255))]
_BLUE_RANGE = ((100, 120, 120), (130, 255, 255))

MIN_CLUSTER_PIXELS = 12


@dataclass
class EmergencyDetection:
    present: bool
    vehicle_class: VehicleClass | None
    confidence: float


class EmergencyVehicleDetector(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> dict[str, EmergencyDetection]:
        """Returns one EmergencyDetection per compass approach."""


def _assign_quadrant(cx: float, cy: float, width: int, height: int) -> str:
    dx, dy = cx - width / 2, cy - height / 2
    if abs(dx) > abs(dy):
        return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


class HeuristicEmergencyDetector(EmergencyVehicleDetector):
    def __init__(self, min_cluster_pixels: int = MIN_CLUSTER_PIXELS) -> None:
        self.min_cluster_pixels = min_cluster_pixels

    def detect(self, frame: Frame) -> dict[str, EmergencyDetection]:
        h, w = frame.image.shape[:2]
        hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)

        red_mask = np.zeros((h, w), dtype=np.uint8)
        for lo, hi in _RED_RANGES:
            red_mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        blue_mask = cv2.inRange(hsv, np.array(_BLUE_RANGE[0]), np.array(_BLUE_RANGE[1]))

        result = {d: EmergencyDetection(present=False, vehicle_class=None, confidence=0.0) for d in ("N", "S", "E", "W")}

        for mask in (red_mask, blue_mask):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label_idx in range(1, num_labels):  # skip background label 0
                area = stats[label_idx, cv2.CC_STAT_AREA]
                if area < self.min_cluster_pixels:
                    continue
                cx, cy = centroids[label_idx]
                approach = _assign_quadrant(cx, cy, w, h)
                confidence = min(1.0, area / (self.min_cluster_pixels * 4))
                if confidence > result[approach].confidence:
                    result[approach] = EmergencyDetection(
                        present=True, vehicle_class=VehicleClass.AMBULANCE, confidence=round(confidence, 2)
                    )

        return result


class ModelEmergencyDetector(EmergencyVehicleDetector):
    """Documented upgrade path -- a trained classifier over vehicle crops.
    Not implemented: requires a labeled emergency-vehicle dataset. Wire in
    a real model here and flip TRAFFIC_EMERGENCY_BACKEND=model when ready."""

    def __init__(self, model_path: str) -> None:
        raise NotImplementedError(
            "ModelEmergencyDetector requires a trained emergency-vehicle classifier "
            "that this repo does not ship with. Use HeuristicEmergencyDetector "
            "(TRAFFIC_EMERGENCY_BACKEND=heuristic) until one is trained."
        )

    def detect(self, frame: Frame) -> dict[str, EmergencyDetection]:  # pragma: no cover
        raise NotImplementedError
