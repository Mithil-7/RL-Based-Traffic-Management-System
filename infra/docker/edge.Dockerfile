FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
        pydantic==2.9.2 pydantic-settings==2.5.2 paho-mqtt==2.1.0 structlog==24.4.0 python-dotenv==1.0.1 tenacity==9.0.0 \
        numpy==1.26.4 opencv-python-headless==4.10.0.84 pillow==10.4.0

COPY src/ src/
COPY city_map/ city_map/
COPY scripts/ scripts/

ENTRYPOINT ["python", "-m", "traffic_system.edge.edge_agent"]

COPY src/ src/
COPY city_map/ city_map/
COPY scripts/ scripts/

ENTRYPOINT ["python", "-m", "traffic_system.edge.edge_agent"]
