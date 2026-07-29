import pytest


@pytest.mark.anyio
async def test_health_returns_ok(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert "version" in body
    assert isinstance(body["engine_reachable"], bool)
    assert isinstance(body["db_healthy"], bool)
    assert "startup_state" in body
    assert body["startup_state"] in ("unknown", "starting", "ready", "stopping")


@pytest.mark.anyio
async def test_health_returns_request_id(client):
    response = await client.get("/api/v1/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


@pytest.mark.anyio
async def test_health_returns_response_time(client):
    response = await client.get("/api/v1/health")
    assert "X-Response-Time" in response.headers
