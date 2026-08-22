import os
import sys
from pathlib import Path

os.environ.setdefault("TRAFFIC_QNET_BACKEND", "numpy")
os.environ.setdefault("TRAFFIC_ENV", "test")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fakeredis  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from traffic_system.api.main import create_app  # noqa: E402
from traffic_system.env.city_graph import CityGraph  # noqa: E402
from traffic_system.ingestion.db_models import init_db, make_engine, make_session_factory  # noqa: E402
from traffic_system.ingestion.redis_state import RedisState  # noqa: E402


@pytest.fixture(scope="session")
def city_graph() -> CityGraph:
    return CityGraph.load(REPO_ROOT / "city_map" / "sample_city_graph.json")


@pytest.fixture
def app(city_graph):
    application = create_app()
    application.state.city_graph = city_graph
    application.state.redis_state = RedisState(client=fakeredis.FakeStrictRedis(decode_responses=True))
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    application.state.session_factory = make_session_factory(engine)
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def redis_state(app):
    return app.state.redis_state


@pytest.fixture
def session_factory(app):
    return app.state.session_factory
