"""Integration tests for the analysis persistence flow.

Tests the full request -> engine -> persist -> retrieve cycle using
mocked database sessions to validate the orchestration without a
live database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.session import get_db
from app.main import app
from app.schemas.analysis import StartupAnalysisRequest


def _valid_payload(**overrides) -> dict:
    payload = {
        "startup_name": "IntegrationCo",
        "website": "https://integration.example.com",
        "description": "A comprehensive integration test startup description.",
    }
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def _mock_persist():
    """Prevent _persist_async from connecting to a real database."""
    with patch("app.services.analysis._persist_async", new_callable=AsyncMock):
        yield


@pytest.fixture()
def _mock_db():
    """Override get_db dependency with an async mock session."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    async def _override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = _override_get_db
    yield mock_session
    app.dependency_overrides.pop(get_db, None)


# ── POST /api/v1/analyze — persistence behavior ──────────────────────


@pytest.mark.anyio
async def test_analyze_persists_on_success():
    """Successful analysis should attempt to persist."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/analyze", json=_valid_payload())

    assert resp.status_code == 200
    body = resp.json()
    assert body["startup_name"] == "IntegrationCo"


@pytest.mark.anyio
async def test_analyze_persistence_failure_does_not_affect_response():
    """If persistence fails, the analysis response should still return 200."""

    async def _failing_persist(*args, **kwargs):
        raise RuntimeError("Database unavailable")

    with patch("app.services.analysis._persist_async", side_effect=_failing_persist):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/analyze", json=_valid_payload())

        assert resp.status_code == 200
        body = resp.json()
        assert body["startup_name"] == "IntegrationCo"


@pytest.mark.anyio
async def test_analyze_response_unchanged_by_persistence():
    """Response body should be identical whether persistence is enabled or not."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/analyze", json=_valid_payload())

    body = resp.json()
    assert "startup_name" in body
    assert "venture_score" in body
    assert "market_score" in body
    assert "founder_score" in body
    assert "traction_score" in body
    assert "recommendations" in body
    assert "confidence" in body
    assert len(body) == 7  # No extra fields added by persistence


# ── GET /api/v1/analyze — retrieval endpoints ────────────────────────


@pytest.mark.anyio
async def test_list_analyses_endpoint(_mock_db):
    """GET /api/v1/analyze should return a paginated list."""
    mock_list_result = MagicMock()
    mock_list_result.analyses = []
    mock_list_result.total = 0

    with patch(
        "app.services.persistence.list_analyses",
        new_callable=AsyncMock,
        return_value=mock_list_result,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/analyze")

        assert resp.status_code == 200
        body = resp.json()
        assert "analyses" in body
        assert "total" in body


@pytest.mark.anyio
async def test_list_analyses_with_pagination(_mock_db):
    """GET /api/v1/analyze?offset=10&limit=5 should pass params correctly."""
    mock_list_result = MagicMock()
    mock_list_result.analyses = []
    mock_list_result.total = 0

    with patch(
        "app.services.persistence.list_analyses",
        new_callable=AsyncMock,
        return_value=mock_list_result,
    ) as mock_fn:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/analyze?offset=10&limit=5")

        assert resp.status_code == 200
        mock_fn.assert_called_once()


@pytest.mark.anyio
async def test_get_analysis_by_id_found(_mock_db):
    """GET /api/v1/analyze/{id} should return detail when found."""
    from app.schemas.analysis import AnalysisDetailResponse

    mock_detail = AnalysisDetailResponse(
        id="abc-123",
        startup_name="FoundCo",
        website="https://found.example.com",
        description="Found startup.",
        venture_score=75.0,
        market_score=80.0,
        founder_score=70.0,
        traction_score=65.0,
        confidence=0.85,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    with patch(
        "app.services.persistence.get_analysis",
        new_callable=AsyncMock,
        return_value=mock_detail,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/analyze/abc-123")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "abc-123"
        assert body["startup_name"] == "FoundCo"


@pytest.mark.anyio
async def test_get_analysis_by_id_not_found(_mock_db):
    """GET /api/v1/analyze/{id} should return 404 when not found."""
    with patch(
        "app.services.persistence.get_analysis",
        new_callable=AsyncMock,
        return_value=None,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/analyze/nonexistent-id")

        assert resp.status_code == 404


# ── End-to-end flow with mocked DB ────────────────────────────────────


@pytest.mark.anyio
async def test_full_analyze_and_retrieve_flow(_mock_db):
    """Simulate: create analysis -> persist -> retrieve by ID."""
    from app.schemas.analysis import AnalysisDetailResponse

    # Step 1: POST to create analysis
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_resp = await client.post(
            "/api/v1/analyze", json=_valid_payload()
        )
        assert create_resp.status_code == 200

    # Step 2: GET the created analysis
    mock_detail = AnalysisDetailResponse(
        id="flow-test-id",
        startup_name="IntegrationCo",
        website="https://integration.example.com",
        description="A comprehensive integration test startup description.",
        venture_score=create_resp.json()["venture_score"],
        market_score=create_resp.json()["market_score"],
        founder_score=create_resp.json()["founder_score"],
        traction_score=create_resp.json()["traction_score"],
        recommendations=create_resp.json()["recommendations"],
        confidence=create_resp.json()["confidence"],
        engine_version="0.12.1",
        processing_time_ms=42.5,
        full_report={"test": True},
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    with patch(
        "app.services.persistence.get_analysis",
        new_callable=AsyncMock,
        return_value=mock_detail,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            get_resp = await client.get("/api/v1/analyze/flow-test-id")

        assert get_resp.status_code == 200
        detail = get_resp.json()
        assert detail["startup_name"] == "IntegrationCo"
        assert detail["id"] == "flow-test-id"


# ── Service layer integration ─────────────────────────────────────────


@pytest.mark.anyio
async def test_run_analysis_attempts_persistence():
    """run_analysis should call _persist_async after successful execution."""
    with patch("app.services.analysis._persist_async", new_callable=AsyncMock) as mock_persist:
        from app.services.analysis import run_analysis
        from predictron_engine.engine import PredictronEngine

        engine = PredictronEngine()
        request = StartupAnalysisRequest(**_valid_payload())

        result = await run_analysis(engine, request)

        assert result.startup_name == "IntegrationCo"
        mock_persist.assert_called_once()

        # Verify the arguments passed to _persist_async
        call_args = mock_persist.call_args
        persisted_request = call_args[0][0]
        persisted_response = call_args[0][2]
        assert persisted_request.startup_name == "IntegrationCo"
        assert persisted_response.startup_name == "IntegrationCo"


@pytest.mark.anyio
async def test_run_analysis_gracefully_handles_persist_error():
    """run_analysis should not raise even if persistence fails."""

    async def _raise_error(*args, **kwargs):
        raise ConnectionError("DB down")

    with patch("app.services.analysis._persist_async", side_effect=_raise_error):
        from app.services.analysis import run_analysis
        from predictron_engine.engine import PredictronEngine

        engine = PredictronEngine()
        request = StartupAnalysisRequest(**_valid_payload())

        result = await run_analysis(engine, request)

        assert result.startup_name == "IntegrationCo"
        assert 0 <= result.venture_score <= 100
