# Architecture

This document goes deeper than the README: the exact data flow, the algorithm behind
each agent, the safety guarantees, and how this scales from a 9-intersection demo to
a real city.

## 1. Data flow, end to end

```
1. Edge agent (one process per intersection, one per Raspberry Pi in production)
   - Reads a frame (SimulatedVideoSource or RaspberryPiCameraSource)
   - Runs vehicle detection (MockVehicleDetector or YoloVehicleDetector)
   - Runs emergency detection (HeuristicEmergencyDetector, HSV color thresholding)
   - Assembles a TelemetryEvent (per-lane vehicle counts, queue estimate,
     emergency flags) and publishes it to MQTT topic `traffic/telemetry/{id}`

2. Ingestion service (one process, subscribes to `traffic/telemetry/+`)
   - Writes the event to Redis (`telemetry:{id}`, TTL'd) -- this is the
     "current state" every downstream reader uses
   - Writes the event to Postgres/TimescaleDB (`telemetry_records`) -- durable history
   - Publishes to the Redis pub/sub channel `traffic:updates` for WebSocket fan-out

3. Brain service (one process, one decision cycle every TRAFFIC_DECISION_INTERVAL_SECONDS)
   For every intersection:
   a. Refresh in-memory IntersectionState from the latest Redis telemetry
   b. Ask that intersection's DQN agent for a proposed phase (greedy, no exploration)
   c. Coordinator agent may override the proposal (spillback prevention)
   d. Emergency preemption agent may hard-force a phase (if a CV-flagged emergency
      vehicle is present) or soft-bias a downstream corridor intersection
   e. Pass the final decision through the SafetyLayer (min/max green, mandatory
      yellow + all-red clearance) -- the exact same safety layer used in training
   f. Publish the resulting SignalCommand to MQTT (`traffic/commands/{id}`) and Redis
   g. Route allocation agent refreshes live congestion weights on the city graph
   h. Incident detection agent checks this intersection's queue/discharge history
      for the "growing queue, not discharging" signature
   i. Persist a snapshot (for honest, directly-measured throughput KPIs)

4. Edge agent receives its SignalCommand over MQTT and applies it
   (in simulation: advances SyntheticTrafficGenerator; on real hardware: drives the
   actual signal controller)

5. API (FastAPI) reads current state from Redis and history from Postgres,
   serves REST endpoints and a WebSocket that relays the `traffic:updates` channel

6. Dashboard (NiceGUI) polls the API on a timer and renders the live map, KPIs,
   alerts feed, and a route planner
```

Every arrow above is a real message over a real protocol (MQTT or Redis), not an
in-process function call -- the brain, ingestion, API, and every edge agent are
independent processes that could run on entirely different machines.

## 2. Why this decomposition

Four separate services (edge, ingestion, brain, API) rather than one monolith:

- **Edge agents scale horizontally by definition** -- one per intersection, physically
  distributed. They cannot be anything but separate processes.
- **Ingestion is decoupled from the brain** so a slow database write never blocks a
  signal decision, and so ingestion can be scaled/restarted independently of the
  decision loop.
- **The brain is decoupled from the API** so a dashboard request never competes with
  the RL decision loop for CPU, and so the brain can run on a machine with different
  resource characteristics (e.g., more CPU for many DQN forward passes) than the API.

## 3. The safety layer (the one thing that has to be bulletproof)

`env/intersection.py::SafetyLayer` sits between every agent's decision and the actual
signal, in both training and production -- literally the same class, same code path.
It enforces, unconditionally:

- **Minimum green time**: a phase can't be interrupted before `TRAFFIC_MIN_GREEN_SECONDS`
  has elapsed, even if the DQN or emergency agent wants to switch sooner.
- **Maximum green time**: a phase is forced to switch after `TRAFFIC_MAX_GREEN_SECONDS`,
  even if nothing asked it to -- prevents one direction from starving another.
- **Mandatory yellow + all-red clearance**: every phase change goes through
  `TRAFFIC_YELLOW_SECONDS` + `TRAFFIC_ALL_RED_SECONDS` of no discharge. Even an
  emergency preemption goes through this -- bypassing it would be unsafe for an
  ambulance too, not just inconvenient for other traffic.

Because training (`TrafficGridEnv`) and production (`BrainService`) both construct a
`SafetyLayer` from the same settings and call `.resolve()` the same way, the policy the
DQN learned is evaluated against -- and constrained by -- the identical rules it will
actually operate under. There's no "training wheels come off in production" gap.

## 4. The agents

### 4.1 DQN agent (`agents/dqn_agent.py`) -- per-intersection signal control

- **Double DQN**: the online network selects the best next action, the target network
  evaluates it. Standard fix for DQN's overestimation bias (van Hasselt et al., 2015).
- **Dueling architecture**: separate value V(s) and advantage A(s,a) streams,
  recombined as `Q = V + (A - mean(A))`. Helps when many actions have similar value
  (e.g. short queues everywhere -- which phase runs barely matters).
- **Prioritized experience replay**: transitions are sampled proportional to their
  TD-error, so training time isn't dominated by the very common "nothing interesting
  happening" transitions at the expense of rare ones (an emergency vehicle event, a
  sudden congestion spike).
- **Reward**: `-0.25*queue - 0.05*wait + 1.0*throughput - 2.0*(switched this step) - 5.0*(emergency waiting)`
  per intersection per decision cycle -- balances clearing traffic, minimizing wait,
  discouraging flicker-switching, and prioritizing emergency vehicles even before the
  hard preemption agent kicks in.
- **Pluggable backend** (`agents/backends/`): `TorchDuelingQNetwork` (autograd, GPU-capable,
  the recommended backend for real training) or `NumpyDuelingQNetwork` (hand-written
  forward/backward pass, zero heavy dependencies). Both implement the same interface, so
  `IntersectionDQNAgent` doesn't know or care which one it's using. This exists because
  edge inference hardware, CI environments, and constrained sandboxes shouldn't need a
  multi-hundred-MB PyTorch install just to run a trained policy forward.

### 4.2 Coordinator agent (`agents/coordinator_agent.py`) -- spillback prevention

A per-intersection DQN only sees its own queues. It has no way to know whether the
road it's about to send traffic down is already saturated. The coordinator adds
exactly one thing: if serving a phase would discharge into a downstream approach
already at/above `spillback_queue_threshold`, and the *alternative* phase's downstream
isn't equally saturated, override to the alternative. This is a simplified version of
the "max-pressure" family of traffic-signal control algorithms (Varaiya, 2013) -- a
thin, auditable rule layered on top of the learned policy, not a second model.

### 4.3 Emergency preemption agent (`agents/emergency_preemption_agent.py`)

Deliberately rule-based, not learned -- this is a safety-relevant path and needs to be
100% predictable and auditable. When the edge CV pipeline flags an emergency vehicle
on an approach, this agent forces that intersection's phase immediately (still through
the SafetyLayer's yellow/all-red clearance -- see Sec 3). It also projects a *soft*
preemption corridor (the next 2 intersections the vehicle is heading toward, derived
from the city graph's direction-neighbor mapping) and returns a bias the brain service
applies to nudge -- not force -- those intersections toward the matching phase, pending
a proper ETA model (see README's Honest Limitations).

### 4.4 Route allocation agent (`agents/route_allocation_agent.py`)

Answers a different question than the signal-control agents: *which path should this
vehicle take*, not *what should this signal do*. Refreshes every road's congestion
factor each decision cycle from live queue telemetry (using the correct
direction-matched queue at each edge's endpoints -- not an intersection's total queue,
which would incorrectly congest every road touching a busy intersection regardless of
direction) and runs Dijkstra over the congestion-weighted graph. This is what
implements "the system detects congestion and allocates a specialized route" from the
original spec.

### 4.5 Incident detection agent (`agents/incident_detection_agent.py`)

Heuristic pattern detector, not learned: flags an intersection whose queue is growing
significantly faster than it's being discharged, despite receiving green time -- the
signature of a blocked lane or stalled vehicle rather than ordinary demand. Exists to
surface *diagnosis* early, before a purely reactive agent would "notice" via falling
reward, by which point the queue is already long.

## 5. Scaling to a real city

- **Weight sharing**: `scripts/train_dqn.py --share-weights` (the default) trains one
  policy against every intersection simultaneously, with every intersection's
  transitions landing in one shared replay buffer. This is what makes training
  hundreds of intersections tractable -- you cannot realistically train hundreds of
  independent networks to convergence. `--no-share-weights` is available for a small
  number of unusually-shaped intersections that warrant a specialized policy.
- **Brain service sharding**: `BrainService` takes a `CityGraph`, not a hardcoded list --
  running N brain processes, each loaded with a disjoint subset of the city graph's
  intersections (with the coordinator agent's spillback logic still correct at shard
  boundaries, since it only needs each intersection's direct neighbors' state, which is
  in the same shared Redis), is a config change, not a code change.
- **MQTT at scale**: Mosquitto handles thousands of concurrent low-traffic clients
  comfortably; at real-city scale, swap the `mosquitto` service for a managed broker
  (AWS IoT Core, HiveMQ Cloud) -- the edge agent and ingestion service code doesn't
  change, only `TRAFFIC_MQTT_HOST`/port/TLS config.
- **Storage at scale**: TimescaleDB hypertables partition automatically by time; the
  30-day retention policy on raw telemetry (`infra/postgres/convert_hypertables.sql`)
  keeps disk usage predictable regardless of city size. Redis is the bottleneck to
  watch first at real scale -- a managed Redis cluster (ElastiCache) is the first
  upgrade, not a code change.

## 6. Observability

- **Structured logging** (`structlog`) everywhere, JSON in production
  (`TRAFFIC_LOG_JSON=true`), human-readable in development.
- **Prometheus metrics endpoint** (`prometheus-client`, see `api/routes_metrics.py`)
  for request latency, decision-cycle timing, and per-agent override counts.
- **Directly-measured KPIs, not estimates**: `IntersectionSnapshotRecord` is written
  every decision cycle with `total_discharged` and `total_switches` as running
  counters; the KPI API computes throughput as the *difference* between the first and
  last snapshot in a time window -- an honest number, not a derived approximation.

## 7. Testing philosophy

43 tests, no mocked-out business logic -- the physics of `TrafficGridEnv`, the actual
learning loop of `IntersectionDQNAgent` (loss decreases, save/load round-trips
correctly), and every agent's real decision logic (spillback override, corridor
projection, congestion-aware rerouting, stalled-queue detection) are exercised
directly. External infrastructure (MQTT, Redis, Postgres) is faked in the test suite
(`fakeredis`, in-memory SQLite, a fake MQTT client in `tests/fakes.py`) specifically so
CI doesn't need real brokers -- but the system has also been run against real Mosquitto
and Redis (`scripts/run_full_stack_demo.py`) to confirm the actual wire protocol works,
not just the code against a stand-in.

## 8. What would change for a production deployment

In rough priority order:
1. Replace `MockVehicleDetector`/`HeuristicEmergencyDetector` with trained models on
   real camera data from the target city's intersections.
2. Implement the ETA-based corridor preemption (see README's Honest Limitations).
3. Add protected left-turn phase physics for intersections that need them.
4. Move MQTT to a managed, TLS-secured broker with per-device client certificates
   (see the production notes in `infra/mosquitto/mosquitto.conf`).
5. Add a proper CI/CD deploy stage (this repo's CI builds every image but doesn't push
   or deploy -- see `.github/workflows/ci.yml`).
6. Load-test the brain service's decision-cycle latency at realistic intersection
   counts and shard accordingly (see Sec 5).
