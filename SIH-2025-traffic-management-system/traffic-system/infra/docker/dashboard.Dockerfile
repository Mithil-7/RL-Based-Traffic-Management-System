# Control-room dashboard (NiceGUI, pure Python).
FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir \
        nicegui pydantic pydantic-settings httpx structlog python-dotenv

COPY src/ src/

EXPOSE 8080
CMD ["python", "-m", "traffic_system.dashboard.app"]
