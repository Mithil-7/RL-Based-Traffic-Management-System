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
        pydantic==2.9.2 pydantic-settings==2.5.2 sqlalchemy==2.0.35 asyncpg==0.29.0 psycopg2-binary==2.9.9 redis==5.0.8 \
        paho-mqtt==2.1.0 structlog==24.4.0 prometheus-client==0.20.0 python-dotenv==1.0.1 tenacity==9.0.0 \
        gymnasium==0.29.1 numpy==1.26.4 networkx==3.3 torch==2.4.1

COPY src/ src/
COPY city_map/ city_map/
COPY models/ models/

CMD ["python", "-m", "traffic_system.brain.brain_service"]
