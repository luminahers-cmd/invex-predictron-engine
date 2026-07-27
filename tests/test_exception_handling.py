"""Tests for global exception handling in Phase 1."""

import pytest
from starlette.testclient import TestClient

from app.main import app


def test_500_exception_handler_returns_json():
    """Verify that unhandled exceptions are caught by the global handler."""

    @app.get("/test-500")
    async def _trigger_500():
        raise RuntimeError("intentional test error")

    with TestClient(app, raise_server_exceptions=False) as tc:
        resp = tc.get("/test-500")
        assert resp.status_code == 500
        body = resp.json()
        assert body["code"] == "INTERNAL_ERROR"
        assert "Internal server error" in body["detail"]

    del app.routes[-1]


@pytest.mark.anyio
async def test_health_endpoint_returns_db_healthy(client):
    """Health endpoint should include the db_healthy field."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert "db_healthy" in body
    assert isinstance(body["db_healthy"], bool)


@pytest.mark.anyio
async def test_analyze_engine_is_singleton(client):
    """Multiple requests should use the same engine instance."""
    payload1 = {
        "startup_name": "Co1",
        "website": "https://co1.example.com",
        "description": "First company for singleton verification.",
    }
    payload2 = {
        "startup_name": "Co2",
        "website": "https://co2.example.com",
        "description": "Second company for singleton verification.",
    }

    resp1 = await client.post("/api/v1/analyze", json=payload1)
    resp2 = await client.post("/api/v1/analyze", json=payload2)

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["startup_name"] == "Co1"
    assert resp2.json()["startup_name"] == "Co2"
