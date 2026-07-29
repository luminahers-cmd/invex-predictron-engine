"""Tests for request identification (Phase 4)."""

import pytest


@pytest.mark.anyio
async def test_request_id_generated_when_missing(client):
    """Response must include X-Request-ID when client does not send one."""
    response = await client.get("/api/v1/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


@pytest.mark.anyio
async def test_request_id_propagated_from_client(client):
    """When the client sends X-Request-ID, the same value must be echoed back."""
    custom_id = "my-custom-trace-001"
    response = await client.get(
        "/api/v1/health",
        headers={"X-Request-ID": custom_id},
    )
    assert response.headers["X-Request-ID"] == custom_id


@pytest.mark.anyio
async def test_request_id_on_error_response(client):
    """Error responses must also carry X-Request-ID."""
    response = await client.get("/api/v1/analyze/some-nonexistent-id")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


@pytest.mark.anyio
async def test_response_time_header_present(client):
    """Every response must include X-Response-Time."""
    response = await client.get("/api/v1/health")
    assert "X-Response-Time" in response.headers
    ms_value = response.headers["X-Response-Time"]
    assert ms_value.endswith("ms")
    assert float(ms_value.rstrip("ms")) >= 0


@pytest.mark.anyio
async def test_request_id_unique_per_request(client):
    """Each request should get a unique request ID."""
    resp1 = await client.get("/api/v1/health")
    resp2 = await client.get("/api/v1/health")
    id1 = resp1.headers.get("X-Request-ID")
    id2 = resp2.headers.get("X-Request-ID")
    assert id1 != id2
