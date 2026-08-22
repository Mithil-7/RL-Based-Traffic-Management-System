# Ingestion service: subscribes to MQTT telemetry, fans out to
# Postgres/TimescaleDB and Redis.
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
        paho-mqtt structlog python-dotenv tenacity

COPY src/ src/

CMD ["python", "-m", "traffic_system.ingestion.mqtt_subscriber"]
