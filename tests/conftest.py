import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _ensure_engine_on_state():
    """Ensure the PredictronEngine singleton is present on app.state for tests.

    The ASGITransport used by the test client does not trigger the lifespan
    context manager, so we must set the engine manually.
    """
    from predictron_engine.engine import PredictronEngine

    if not hasattr(app.state, "predictron_engine") or app.state.predictron_engine is None:
        app.state.predictron_engine = PredictronEngine()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
