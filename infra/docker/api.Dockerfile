# FastAPI REST + WebSocket backend.
FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# The API service never needs torch/ultralytics/opencv (those belong to the
# brain and edge images) -- installing only what it needs keeps this image
# small and its build fast.
RUN pip install --no-cache-dir \
        fastapi uvicorn[standard] websockets pydantic pydantic-settings \
        sqlalchemy asyncpg psycopg2-binary redis paho-mqtt \
        structlog prometheus-client python-dotenv tenacity

COPY src/ src/
COPY city_map/ city_map/

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "traffic_system.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
