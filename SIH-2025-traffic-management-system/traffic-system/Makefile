.PHONY: setup test lint fmt train up down logs demo db-hypertables clean

VENV := .venv
PY := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

## Local (non-Docker) development

setup:  ## Create a venv and install everything needed for local dev + tests
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt || $(PIP) install \
		numpy gymnasium networkx pytest pytest-asyncio pytest-cov \
		pydantic pydantic-settings fastapi "uvicorn[standard]" websockets httpx \
		redis fakeredis sqlalchemy paho-mqtt structlog prometheus-client \
		python-dotenv tenacity ruff opencv-python-headless pillow nicegui
	@echo "Torch is NOT installed by default (large download)."
	@echo "Run 'make install-torch' for full-scale training, or use --backend numpy."

install-torch:
	$(PIP) install torch ultralytics

test:  ## Run the full test suite
	TRAFFIC_QNET_BACKEND=numpy PYTHONPATH=src $(PY) -m pytest tests/ -v

test-cov:  ## Run tests with a coverage report
	TRAFFIC_QNET_BACKEND=numpy PYTHONPATH=src $(PY) -m pytest tests/ --cov=traffic_system --cov-report=term-missing

lint:  ## Static checks (ruff)
	$(VENV)/bin/ruff check src/ tests/ scripts/

fmt:  ## Auto-format
	$(VENV)/bin/ruff format src/ tests/ scripts/
	$(VENV)/bin/ruff check --fix src/ tests/ scripts/

train:  ## Train the shared DQN policy with the lightweight numpy backend (fast, no torch needed)
	TRAFFIC_QNET_BACKEND=numpy PYTHONPATH=src $(PY) scripts/train_dqn.py --episodes 150 --backend numpy

train-torch:  ## Full training run with PyTorch (run `make install-torch` first)
	PYTHONPATH=src $(PY) scripts/train_dqn.py --episodes 2000 --backend torch --steps-per-episode 360

## Docker Compose (the full production-shaped stack)

up:  ## Build and start every service
	docker compose up --build -d

down:  ## Stop and remove all containers
	docker compose down

logs:  ## Tail logs from every service
	docker compose logs -f

db-hypertables:  ## One-time: convert Postgres tables into TimescaleDB hypertables (run after the first `make up`)
	docker compose exec -T postgres psql -U traffic -d traffic -f /docker-entrypoint-initdb.d/../../infra/postgres/convert_hypertables.sql \
		|| docker compose cp infra/postgres/convert_hypertables.sql postgres:/tmp/convert_hypertables.sql \
		&& docker compose exec -T postgres psql -U traffic -d traffic -f /tmp/convert_hypertables.sql

## Local (non-Docker) full-stack demo -- see scripts/run_full_stack_demo.py

demo:  ## Run brokers + every service in one process for a quick local demo (needs mosquitto & redis-server installed)
	TRAFFIC_QNET_BACKEND=numpy PYTHONPATH=src $(PY) scripts/run_full_stack_demo.py

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache **/__pycache__ *.db
