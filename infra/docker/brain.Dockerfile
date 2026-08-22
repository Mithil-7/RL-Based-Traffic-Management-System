# The brain service: loads trained DQN checkpoints and runs the decision
# loop. Includes torch since this is where inference (and, via
# scripts/train_dqn.py, training) actually happens.
FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
        pydantic pydantic-settings sqlalchemy asyncpg psycopg2-binary redis \
        paho-mqtt structlog prometheus-client python-dotenv tenacity \
        gymnasium numpy networkx torch

COPY src/ src/
COPY city_map/ city_map/
COPY models/ models/

CMD ["python", "-m", "traffic_system.brain.brain_service"]
