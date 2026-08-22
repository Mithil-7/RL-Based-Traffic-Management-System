"""Centralized, typed configuration.

Every tunable in the system is read from the environment (with sane
defaults for local development) so the exact same code runs unmodified in
a laptop docker-compose stack, CI, or a production cluster -- only the
environment variables change. See `.env.example` at the repo root for the
full list of variables.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="TRAFFIC_", extra="ignore", protected_namespaces=("settings_",)
    )

    # --- General ---
    env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # --- MQTT ---
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_keepalive: int = 60
    mqtt_telemetry_topic: str = "traffic/telemetry/+"
    mqtt_command_topic_prefix: str = "traffic/commands"
    mqtt_client_id_prefix: str = "traffic-system"

    # --- Postgres / TimescaleDB ---
    postgres_dsn: str = "postgresql+psycopg2://traffic:traffic@localhost:5432/traffic"
    postgres_dsn_async: str = "postgresql+asyncpg://traffic:traffic@localhost:5432/traffic"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"
    redis_state_ttl_seconds: int = 30

    # --- City map ---
    city_graph_path: Path = REPO_ROOT / "city_map" / "sample_city_graph.json"

    # --- RL brain ---
    qnet_backend: Literal["torch", "numpy", "auto"] = "auto"
    decision_interval_seconds: float = 5.0
    min_green_seconds: float = 10.0
    max_green_seconds: float = 90.0
    yellow_seconds: float = 3.0
    all_red_seconds: float = 2.0
    replay_buffer_size: int = 50_000
    batch_size: int = 64
    gamma: float = 0.97
    learning_rate: float = 5e-4
    target_sync_every_steps: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    model_dir: Path = REPO_ROOT / "models"

    # --- Computer vision ---
    cv_backend: Literal["yolo", "mock"] = "mock"
    yolo_weights: str = "yolov8n.pt"
    yolo_confidence: float = 0.35
    emergency_backend: Literal["heuristic", "model"] = "heuristic"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_allow_origins: list[str] = ["*"]

    # --- Dashboard ---
    dashboard_port: int = 8080
    api_base_url: str = "http://localhost:8000"
    dashboard_poll_interval_s: float = 3.0

    # --- Metrics ---
    prometheus_port: int = 9100


@lru_cache
def get_settings() -> Settings:
    return Settings()
