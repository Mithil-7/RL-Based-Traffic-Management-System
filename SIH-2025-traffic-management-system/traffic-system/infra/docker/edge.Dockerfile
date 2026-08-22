# Edge agent image. Runs in "simulated" mode by default (see
# docker-compose.yml command override); for a real Raspberry Pi deployment,
# build this same image with --build-arg TARGET_ARCH=arm64 on the Pi itself
# (or cross-build with `docker buildx --platform linux/arm64`), and pass
# --video-source=picamera at runtime once RaspberryPiCameraSource is wired
# to a CLI flag for your specific hardware.
FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

# libgl1/libglib2.0-0 are OpenCV's runtime dependencies even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
        pydantic pydantic-settings paho-mqtt structlog python-dotenv tenacity \
        numpy opencv-python-headless pillow

# ultralytics + torch (real YOLO inference) are a separate, optional layer --
# uncomment when deploying with TRAFFIC_CV_BACKEND=yolo and real hardware:
# RUN pip install --no-cache-dir ultralytics torch

COPY src/ src/
COPY city_map/ city_map/
COPY scripts/ scripts/

ENTRYPOINT ["python", "-m", "traffic_system.edge.edge_agent"]
