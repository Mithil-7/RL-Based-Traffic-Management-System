"""One-command local demo of the entire stack -- no Docker required.

Starts (or reuses, if already running) local Mosquitto and Redis brokers,
then launches every service (ingestion, brain, a simulated edge fleet, the
API, and the dashboard) as a subprocess, using SQLite instead of Postgres
so there's no database server to install first. This is the fastest way to
see the whole system working end to end; `docker compose up` (see
docker-compose.yml) is the deployment-shaped equivalent for anything beyond
a local demo.

Requires `mosquitto` and `redis-server` to be installed locally (both are
standard Ubuntu/Debian/Homebrew packages: `apt install mosquitto
redis-server` / `brew install mosquitto redis`). If neither is running,
this script starts them itself; it does not install them.

Usage:
    python scripts/run_full_stack_demo.py
    # then open http://localhost:8080 (dashboard) and
    # http://localhost:8000/docs (API)
    # Ctrl+C to stop everything.
"""
from __future__ import annotations

import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DB_PATH = REPO_ROOT / "demo.db"

COLORS = {
    "mosquitto": "\033[95m", "redis": "\033[95m", "ingestion": "\033[94m",
    "brain": "\033[92m", "edge-fleet": "\033[93m", "api": "\033[96m", "dashboard": "\033[91m",
}
RESET = "\033[0m"

DEMO_ENV = {
    "TRAFFIC_ENV": "development",
    "TRAFFIC_LOG_LEVEL": "INFO",
    "TRAFFIC_MQTT_HOST": "localhost",
    "TRAFFIC_MQTT_PORT": "1883",
    "TRAFFIC_REDIS_URL": "redis://localhost:6379/0",
    "TRAFFIC_POSTGRES_DSN": f"sqlite:///{DEMO_DB_PATH}",
    "TRAFFIC_QNET_BACKEND": "numpy",  # no torch required for the demo
    "TRAFFIC_CV_BACKEND": "mock",
    "TRAFFIC_API_BASE_URL": "http://localhost:8000",
}


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _ensure_broker(name: str, start_cmd: list[str], port: int, wait_s: float = 2.0) -> subprocess.Popen | None:
    if _port_open("localhost", port):
        print(f"{COLORS[name]}[{name}]{RESET} already running on port {port}, reusing it")
        return None
    if shutil.which(start_cmd[0]) is None:
        print(f"ERROR: `{start_cmd[0]}` not found. Install it (e.g. `apt install {name}`) and re-run.")
        sys.exit(1)
    print(f"{COLORS[name]}[{name}]{RESET} starting: {' '.join(start_cmd)}")
    proc = subprocess.Popen(start_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(int(wait_s / 0.1)):
        if _port_open("localhost", port):
            return proc
        time.sleep(0.1)
    print(f"ERROR: {name} did not start listening on port {port} in time.")
    sys.exit(1)


def _spawn(name: str, args: list[str]) -> subprocess.Popen:
    print(f"{COLORS[name]}[{name}]{RESET} starting: {' '.join(args)}")
    import os

    env = {**os.environ, **DEMO_ENV, "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.Popen(
        args, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )


def _stream(name: str, proc: subprocess.Popen) -> None:
    import threading

    def _pump() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            print(f"{COLORS.get(name, '')}[{name}]{RESET} {line.rstrip()}")

    threading.Thread(target=_pump, daemon=True).start()


def main() -> None:
    if DEMO_DB_PATH.exists():
        DEMO_DB_PATH.unlink()  # start each demo run from a clean database

    py = sys.executable
    owned_processes: list[subprocess.Popen] = []

    mosq = _ensure_broker("mosquitto", ["mosquitto", "-p", "1883"], 1883)
    redis_proc = _ensure_broker("redis", ["redis-server", "--port", "6379", "--save", ""], 6379)
    owned_processes += [p for p in (mosq, redis_proc) if p is not None]
    time.sleep(0.5)

    service_specs = [
        ("ingestion", [py, "-m", "traffic_system.ingestion.mqtt_subscriber"]),
        ("brain", [py, "-m", "traffic_system.brain.brain_service"]),
        ("edge-fleet", [py, str(REPO_ROOT / "scripts" / "run_edge_simulation.py")]),
        ("api", [py, "-m", "uvicorn", "traffic_system.api.main:app", "--host", "0.0.0.0", "--port", "8000"]),
        ("dashboard", [py, "-m", "traffic_system.dashboard.app"]),
    ]

    for name, args in service_specs:
        proc = _spawn(name, args)
        owned_processes.append(proc)
        _stream(name, proc)
        time.sleep(1.0)  # stagger startup so early log lines aren't a jumbled race

    print()
    print("=" * 70)
    print("  Dashboard:  http://localhost:8080")
    print("  API docs:   http://localhost:8000/docs")
    print("  Press Ctrl+C to stop every service.")
    print("=" * 70)

    def _shutdown(*_args) -> None:
        print("\nShutting down...")
        for proc in owned_processes:
            proc.terminate()
        for proc in owned_processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
