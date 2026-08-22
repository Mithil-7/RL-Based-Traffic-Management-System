"""Pluggable camera input.

`SimulatedVideoSource` renders a synthetic top-down view of one
intersection's four approaches using OpenCV drawing primitives, driven by
`SyntheticTrafficGenerator`'s queue state. It also attaches ground-truth
vehicle metadata to the frame -- this is what lets `MockVehicleDetector`
read "perfect" detections for local development/demos without a trained CV
model, while `YoloVehicleDetector` can still run real inference on the same
rendered pixels to validate the CV pipeline end-to-end before real camera
hardware is available.

`RaspberryPiCameraSource` is the documented drop-in replacement: same
`VideoSource` interface, same downstream code, only the frame source
changes. It requires `opencv-python` with V4L2/Picamera2 support and actual
hardware, so it is not exercised by this repo's test suite.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from traffic_system.edge.traffic_generator import APPROACH_ORDER, SyntheticTrafficGenerator

FRAME_SIZE = 480

# BGR colors (OpenCV convention)
COLOR_ROAD = (60, 60, 60)
COLOR_LANE_LINE = (200, 200, 200)
COLOR_VEHICLE = (0, 165, 255)
COLOR_EMERGENCY_A = (0, 0, 255)
COLOR_EMERGENCY_B = (255, 0, 0)
COLOR_BG = (30, 100, 30)


@dataclass
class Frame:
    image: np.ndarray  # BGR uint8, HxWx3
    timestamp: float = field(default_factory=time.time)
    ground_truth: dict[str, Any] | None = None  # only populated by simulated sources


class VideoSource(ABC):
    @abstractmethod
    def read(self) -> Frame | None: ...

    def release(self) -> None:  # pragma: no cover - trivial default
        pass


class SimulatedVideoSource(VideoSource):
    """Renders a synthetic intersection frame from a `SyntheticTrafficGenerator`'s
    current queue state, for demos and CV-pipeline testing without hardware."""

    def __init__(self, generator: SyntheticTrafficGenerator, frame_size: int = FRAME_SIZE) -> None:
        self.generator = generator
        self.frame_size = frame_size

    def read(self) -> Frame:
        size = self.frame_size
        img = np.full((size, size, 3), COLOR_BG, dtype=np.uint8)
        center = size // 2
        road_half_width = size // 8
        cv2.rectangle(img, (0, center - road_half_width), (size, center + road_half_width), COLOR_ROAD, -1)
        cv2.rectangle(img, (center - road_half_width, 0), (center + road_half_width, size), COLOR_ROAD, -1)
        cv2.line(img, (0, center), (size, center), COLOR_LANE_LINE, 1)
        cv2.line(img, (center, 0), (center, size), COLOR_LANE_LINE, 1)

        ground_truth: dict[str, Any] = {}
        for direction in APPROACH_ORDER:
            approach = self.generator.state.approaches[direction]
            count = int(round(approach.queue))
            self._draw_approach(img, direction, count, approach.emergency_present, center, road_half_width, size)
            ground_truth[direction] = {
                "vehicle_count": count,
                "class_counts": {"car": max(count - (1 if approach.emergency_present else 0), 0)}
                | ({"ambulance": 1} if approach.emergency_present else {}),
                "emergency_present": approach.emergency_present,
                "emergency_class": approach.emergency_class.value if approach.emergency_class else None,
            }

        return Frame(image=img, ground_truth=ground_truth)

    def _draw_approach(
        self, img: np.ndarray, direction: str, count: int, emergency: bool, center: int, half: int, size: int
    ) -> None:
        vehicle_w, vehicle_h, gap = 14, 20, 6
        max_drawn = min(count, 15)  # cap rendering so a huge queue doesn't fill the whole frame
        for i in range(max_drawn):
            offset = 30 + i * (vehicle_h + gap)
            is_lead_emergency_vehicle = emergency and i == 0
            color = COLOR_VEHICLE
            if direction == "N":
                x, y = center - half // 2, center - half - offset
            elif direction == "S":
                x, y = center + 2, center + half + offset - vehicle_h
            elif direction == "E":
                x, y = center + half + offset - vehicle_w, center - half // 2
            else:  # "W"
                x, y = center - half - offset, center + 2

            if is_lead_emergency_vehicle:
                strobe = COLOR_EMERGENCY_A if int(time.time() * 4) % 2 == 0 else COLOR_EMERGENCY_B
                color = strobe
            cv2.rectangle(img, (int(x), int(y)), (int(x + vehicle_w), int(y + vehicle_h)), color, -1)


class RaspberryPiCameraSource(VideoSource):
    """Real-hardware camera source. Requires `opencv-python` built with V4L2
    (standard USB/CSI camera) support, or `picamera2` for the native Pi
    camera module. Not exercised in this repo's tests -- wire this in when
    physical hardware is available; nothing else in the edge pipeline needs
    to change.
    """

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480) -> None:
        self._cap = cv2.VideoCapture(device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera device {device_index}")

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok:
            return None
        return Frame(image=image, ground_truth=None)

    def release(self) -> None:
        self._cap.release()
