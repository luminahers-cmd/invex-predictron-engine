import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _noop_evidence_collector(monkeypatch):
    """Prevent test analyses from making real network requests.

    The engine defaults to the real async EvidenceOrchestrator; tests swap
    the engine's default for a no-op orchestrator so every PredictronEngine()
    constructed during the suite produces an empty evidence bundle. The
    evidence collection layer itself is tested with MockTransport in
    tests/engine/test_evidence_collection/.
    """
    import predictron_engine.engine as engine_module
    from predictron_engine.evidence.noop import NoOpEvidenceOrchestrator

    monkeypatch.setattr(engine_module, "EvidenceOrchestrator", NoOpEvidenceOrchestrator)

@pytest.fixture(autouse=True)
def _ensure_app_state(_noop_evidence_collector):
    """Ensure the PredictronEngine singleton and startup state are present on app.state.

    The ASGITransport used by the test client does not trigger the lifespan
    context manager, so we must set the engine manually. Depends on the
    no-op collector fixture so the engine is always built with it.
    """
    from predictron_engine.engine import PredictronEngine

    if not hasattr(app.state, "predictron_engine") or app.state.predictron_engine is None:
        app.state.predictron_engine = PredictronEngine()
    if not hasattr(app.state, "startup_state") or app.state.startup_state is None:
        app.state.startup_state = "ready"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
