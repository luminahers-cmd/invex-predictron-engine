"""Tests for rate limiting (Phase 4)."""

from unittest.mock import patch

import pytest

from app.core.config import get_settings
from app.middleware.rate_limit import RateLimitMiddleware


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear rate-limit windows and apply test settings before each test."""
    RateLimitMiddleware.reset_windows()
    original = get_settings()
    with patch.object(original, "RATE_LIMIT_ENABLED", True):
        with patch.object(original, "RATE_LIMIT_REQUESTS", 3):
            with patch.object(original, "RATE_LIMIT_WINDOW_SECONDS", 60):
                yield


@pytest.mark.anyio
async def test_rate_limit_allows_normal_traffic(client):
    """Requests under the limit must succeed."""
    for _ in range(3):
        response = await client.get("/api/v1/analyze/some-id")
        assert response.status_code in (200, 401, 404)


@pytest.mark.anyio
async def test_rate_limit_returns_429_when_exceeded(client):
    """Requests beyond the limit must receive HTTP 429."""
    for i in range(5):
        response = await client.get("/api/v1/analyze/some-id")
        if i < 3:
            assert response.status_code in (200, 401, 404), f"Request {i} should pass"
        else:
            assert response.status_code == 429, f"Request {i} should be rate limited"
            body = response.json()
            assert body["code"] == "RATE_LIMIT_EXCEEDED"


@pytest.mark.anyio
async def test_rate_limit_429_response_format(client):
    """Rate limit responses must include detail and code."""
    for _ in range(4):
        await client.get("/api/v1/analyze/some-id")
    response = await client.get("/api/v1/analyze/some-id")
    assert response.status_code == 429
    body = response.json()
    assert "detail" in body
    assert body["code"] == "RATE_LIMIT_EXCEEDED"
    assert "Retry-After" in response.headers


@pytest.mark.anyio
async def test_rate_limit_excludes_health_endpoint(client):
    """The health endpoint must never be rate limited."""
    for _ in range(10):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200


@pytest.mark.anyio
async def test_rate_limit_excludes_readiness_endpoint(client):
    """The readiness endpoint must never be rate limited."""
    for _ in range(10):
        response = await client.get("/api/v1/health/readiness")
        assert response.status_code in (200, 503)


@pytest.mark.anyio
async def test_rate_limit_429_has_request_id(client):
    """Rate limited responses must still include X-Request-ID."""
    for _ in range(4):
        await client.get("/api/v1/analyze/some-id")
    response = await client.get("/api/v1/analyze/some-id")
    assert response.status_code == 429
    assert "X-Request-ID" in response.headers


@pytest.mark.anyio
async def test_rate_limit_disabled_when_not_configured(client):
    """When RATE_LIMIT_ENABLED is False, no limit is applied."""
    original = get_settings()
    with patch.object(original, "RATE_LIMIT_ENABLED", False):
        with patch.object(original, "RATE_LIMIT_REQUESTS", 1):
            for _ in range(5):
                response = await client.get("/api/v1/analyze/some-id")
                assert response.status_code in (200, 401, 404)
