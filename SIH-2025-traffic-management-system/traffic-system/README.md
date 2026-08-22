# Adaptive Traffic Management System

A multi-agent reinforcement learning platform for real-time, computer-vision-driven
traffic signal control, emergency vehicle preemption, and dynamic route allocation.
Built for SIH 2025.

**The idea:** instead of fixed signal timings, cameras at every intersection feed a
computer vision pipeline that counts vehicles, measures queues, and detects
emergency vehicles in real time. That telemetry drives a set of reinforcement
learning and rule-based agents a per-intersection DQN, a network-wide
coordinator, an emergency preemption agent, a dynamic route allocator, and an
incident detector that decide signal phases and reroute traffic, all with a
hard safety layer (minimum/maximum green time, mandatory yellow/all-red
clearance) that no learned policy can override.

## Status

Everything described in this README has been built and verified running, including
a real end-to-end pass through actual MQTT (Mosquitto) and Redis brokers — not just
unit tests. Specifics on what's simulated vs. what's a documented hardware
integration point are in [Honest limitations](#honest-limitations) below.

```
43 passed in ~3s   (pytest, see tests/)
```

## Architecture

```
Edge agent (per intersection)          Camera sim + YOLO vehicle/emergency detection
        |  MQTT (telemetry)
        v
MQTT broker (Mosquitto)
        |
        v
Ingestion  --->  TimescaleDB (history)  +  Redis (live state cache)
                                                |
                                                v
                          Multi-agent brain (reinforcement learning)
                 DQN agent | Coordinator | Emergency preempt | Routing
                                                |
                                                v
                          REST + WebSocket API  --->  NiceGUI dashboard
                                |
                                v (MQTT, signal commands)
                          back to edge / real signal controllers
```

Full write-up, including *why* each technology was chosen and the specific
algorithms each agent uses: **[ARCHITECTURE.md](ARCHITECTURE.md)**.

### The stack, briefly

| Layer | Technology | Why |
|---|---|---|
| Computer vision | OpenCV + Ultralytics YOLOv8 | Industry-standard, real-time, pretrained on COCO vehicle classes |
| Edge transport | MQTT (Mosquitto) | Built for thousands of small, low-power, intermittently-connected clients — exactly what a city's worth of Raspberry Pis looks like |
| RL / simulation | Gymnasium + PyTorch (NumPy fallback) | Standard RL tooling; a hand-written NumPy backend means agents run without a GPU or even without torch installed, for edge inference |
| Storage | PostgreSQL/TimescaleDB + Redis | Durable time-series history vs. sub-millisecond "what's true right now" |
| API | FastAPI (REST + WebSocket) | Async, high-throughput, auto-generated OpenAPI docs |
| Dashboard | NiceGUI | Pure Python, FastAPI-native, real-time, no separate frontend build |
| Deployment | Docker Compose, GitHub Actions CI | Reproducible everywhere; see [hosting options](#hosting) below |

Everything is Python, end to end.

## Quickstart

### Option A: Docker Compose (recommended, matches production)

```bash
git clone <your-repo-url>
cd traffic-system
docker compose up --build
# first time only, after containers are healthy:
make db-hypertables
```

- Dashboard: http://localhost:8080
- API docs: http://localhost:8000/docs

### Option B: One-command local demo (no Docker, needs `mosquitto` + `redis-server` installed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or see `make setup` for the lighter dependency set
python scripts/run_full_stack_demo.py
```

Starts (or reuses) local Mosquitto/Redis, then every service as a subprocess,
using SQLite instead of Postgres. Same URLs as above. Ctrl+C stops everything.

### Just run the tests

```bash
make setup && make test
```

## Repository layout

```
city_map/                   Sample 3x3 intersection grid (swap for a real OSM extract)
infra/
  docker/                   One Dockerfile per service
  mosquitto/                Broker config
  postgres/                 TimescaleDB init + hypertable conversion
src/traffic_system/
  common/                   Config, logging, schemas, MQTT client -- shared by everything
  env/                      City graph loader, intersection queueing physics, safety layer,
                             Gymnasium training environments
  agents/                   DQN agent + pluggable NumPy/PyTorch backends, replay buffer,
                             coordinator, emergency preemption, route allocation, incident detection
  edge/                     Camera source (simulated + real-hardware-ready), CV detectors,
                             synthetic traffic generator, the edge agent loop
  ingestion/                MQTT subscriber, SQLAlchemy models, Redis state cache
  brain/                    Ties every agent into one decision cycle
  api/                      FastAPI routes (signals, routing, metrics) + WebSocket
  dashboard/                NiceGUI control-room UI
scripts/                    train_dqn.py, run_edge_simulation.py, run_full_stack_demo.py
tests/                      43 tests covering env physics, agents, brain service, API, dashboard
models/                     Checkpoints (a small pretrained starter is committed; see models/README.md)
```

## Training your own policy

A small starter checkpoint (`models/shared.npz`) ships in the repo so the stack works
immediately. To train further:

```bash
# Fast, no GPU/torch needed:
python scripts/train_dqn.py --episodes 150 --backend numpy

# Full-scale training:
pip install torch
python scripts/train_dqn.py --episodes 2000 --backend torch --steps-per-episode 360
```

See [`models/README.md`](models/README.md) for checkpoint layout and weight-sharing details.

## From simulation to real hardware

The whole edge pipeline is built behind interfaces specifically so simulation-to-hardware
is a drop-in swap, not a rewrite:

- `VideoSource`: `SimulatedVideoSource` today → `RaspberryPiCameraSource` (already implemented,
  needs a real camera to test) for a real Pi + camera.
- `VehicleDetector`: `MockVehicleDetector` (ground truth, for demos/CI) → `YoloVehicleDetector`
  (real Ultralytics inference, already implemented — just needs `pip install ultralytics`
  and downloaded weights).
- `EmergencyVehicleDetector`: `HeuristicEmergencyDetector` (real HSV color-thresholding CV,
  works today against rendered or real frames) → `ModelEmergencyDetector` (stubbed interface
  for a trained classifier, once you have labeled data).

Nothing downstream of the edge agent (MQTT contract, ingestion, brain, API, dashboard)
changes at all when you flip these switches.

## Hosting

| Option | Best for | Trade-off |
|---|---|---|
| **Render / Railway free tier** | SIH demo, judge access via a public URL | No GPU, limited scale (fine for 1-2 live intersections) |
| **Docker Compose on a VPS/laptop** | Development, training, full local demo | Bound by one machine |
| **AWS / GCP (ECS/GKE)** | A real multi-intersection city deployment | Real setup cost (IAM, VPC, IaC), $50-300+/mo |

The system is cloud-agnostic by design (12-factor config via environment variables,
stateless services, no code paths that assume a specific host) — moving from one
option to another is an infrastructure change, not a code change.

## Honest limitations

Documented in code comments at the point they matter, summarized here:

- **Signal phases**: the simulation and safety layer support NS/EW through phases;
  protected left-turn phases (`NS_LEFT`/`EW_LEFT`) exist in the schema and safety layer
  but their discharge physics isn't implemented yet — a clearly-scoped extension.
- **Emergency corridor preemption**: the current intersection is hard-forced the instant
  a CV detection fires; downstream corridor intersections get a *soft* bias, not a hard
  force, because we don't yet have a precise ETA model (distance / speed) for exactly
  when to switch them. Noted as a TODO in `emergency_preemption_agent.py`.
- **YOLO approach assignment**: real camera detections are bucketed into N/S/E/W by a
  quadrant heuristic, correct only for a centered overhead camera. A real deployment
  should replace this with a calibrated per-camera ROI polygon set.
- **Wait-time estimation**: uses an M/M/1-style queue-length/service-rate proxy, not
  per-vehicle timestamps — good enough as a training/KPI signal, not microsimulation-grade.
- **Turning traffic**: vehicles either go straight through or exit the network at a grid
  boundary; turning movements aren't separately routed in the base simulation.

None of these block the system from working end-to-end; they're the next things to
improve, not gaps that were hidden.

## License


This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
