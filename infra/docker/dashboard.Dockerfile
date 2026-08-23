FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir \
        nicegui==2.7.0 pydantic==2.9.2 pydantic-settings==2.5.2 httpx==0.27.2 structlog==24.4.0 python-dotenv==1.0.1

COPY src/ src/

EXPOSE 8080
CMD ["python", "-m", "traffic_system.dashboard.app"]
