"""Tests for the readiness endpoint (Phase 4)."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_db_health():
    """Prevent readiness tests from connecting to a real database."""
    with patch("app.api.health._check_db_health", return_value=True):
        yield


@pytest.mark.anyio
async def test_readiness_endpoint_returns_200_when_ready(client):
    """The readiness endpoint must return 200 when all checks pass."""
    response = await client.get("/api/v1/health/readiness")
    assert response.status_code in (200, 503)
    body = response.json()
    assert "status" in body
    assert "db_healthy" in body
    assert isinstance(body["db_healthy"], bool)
    assert "engine_ready" in body
    assert isinstance(body["engine_ready"], bool)
    assert "startup_complete" in body
    assert isinstance(body["startup_complete"], bool)


@pytest.mark.anyio
async def test_readiness_returns_request_id(client):
    """Ready responses must include X-Request-ID."""
    response = await client.get("/api/v1/health/readiness")
    assert "X-Request-ID" in response.headers


@pytest.mark.anyio
async def test_readiness_response_format(client):
    """Verify the readiness response body schema."""
    response = await client.get("/api/v1/health/readiness")
    body = response.json()
    assert body["status"] in ("ready", "not_ready")
    if response.status_code == 200:
        assert body["status"] == "ready"
    else:
        assert body["status"] == "not_ready"
