"""Backwards compatibility tests.

Ensures the existing API surface is preserved and the new endpoints
do not interfere with existing health, analyze, and auth endpoints.
"""

from __future__ import annotations

import pytest


class TestExistingHealthEndpoints:
    """Verify existing health endpoints still work."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_structure(self, client):
        resp = await client.get("/api/v1/health")
        data = resp.json()
        assert "status" in data
        assert "version" in data

    @pytest.mark.asyncio
    async def test_readiness_endpoint(self, client):
        resp = await client.get("/api/v1/health/readiness")
        # Returns 200 when DB is healthy, 503 otherwise (no DB in test env)
        assert resp.status_code in (200, 503)


class TestExistingAnalyzeEndpoints:
    """Verify the existing POST /analyze endpoint still works."""

    @pytest.mark.asyncio
    async def test_analyze_returns_200(self, client):
        payload = {
            "startup_name": "LegacyCo",
            "description": "A legacy test company that builds software solutions.",
            "website_url": "https://legacy.example.com",
        }
        resp = await client.post("/api/v1/analyze", json=payload)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_analyze_structure(self, client):
        payload = {
            "startup_name": "LegacyCo",
            "description": "A legacy test company that builds software solutions.",
        }
        resp = await client.post("/api/v1/analyze", json=payload)
        data = resp.json()
        assert "startup_name" in data
        assert "venture_score" in data
        assert "confidence" in data

    @pytest.mark.asyncio
    async def test_analyze_v1_still_works(self, client):
        payload = {
            "startup_name": "V1Co",
            "description": "A v1 compatible company description for testing.",
        }
        resp = await client.post("/api/v1/analyze", json=payload)
        assert resp.status_code == 200


class TestOpenAPIPreserved:
    """Verify OpenAPI docs are still accessible."""

    @pytest.mark.asyncio
    async def test_openapi_docs(self, client):
        resp = await client.get("/docs")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_schema(self, client):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "/api/v1/analyze" in data["paths"]
        assert "/api/v1/health" in data["paths"]


class TestNewEndpointsInOpenAPI:
    """Verify new endpoints appear in the OpenAPI schema."""

    @pytest.mark.asyncio
    async def test_venture_endpoint_in_openapi(self, client):
        resp = await client.get("/openapi.json")
        data = resp.json()
        assert "/api/v1/venture" in data["paths"]

    @pytest.mark.asyncio
    async def test_portfolio_endpoint_in_openapi(self, client):
        resp = await client.get("/openapi.json")
        data = resp.json()
        assert "/api/v1/portfolio" in data["paths"]

    @pytest.mark.asyncio
    async def test_compare_endpoint_in_openapi(self, client):
        resp = await client.get("/openapi.json")
        data = resp.json()
        assert "/api/v1/compare" in data["paths"]

    @pytest.mark.asyncio
    async def test_due_diligence_endpoint_in_openapi(self, client):
        resp = await client.get("/openapi.json")
        data = resp.json()
        assert "/api/v1/due-diligence" in data["paths"]

    @pytest.mark.asyncio
    async def test_search_endpoint_in_openapi(self, client):
        resp = await client.get("/openapi.json")
        data = resp.json()
        assert "/api/v1/search" in data["paths"]

    @pytest.mark.asyncio
    async def test_batch_endpoint_in_openapi(self, client):
        resp = await client.get("/openapi.json")
        data = resp.json()
        assert "/api/v1/batch" in data["paths"]


class TestNoExistingBehaviorChanged:
    """Regression tests: existing endpoints return identical structures."""

    @pytest.mark.asyncio
    async def test_analyze_no_new_fields(self, client):
        """The legacy analyze response should not gain new fields."""
        payload = {
            "startup_name": "RegCo",
            "description": "Regression test company for backwards compatibility checks.",
        }
        resp = await client.post("/api/v1/analyze", json=payload)
        data = resp.json()
        expected_keys = {
            "startup_name", "venture_score", "market_score",
            "founder_score", "traction_score", "recommendations",
            "confidence", "id",
        }
        assert set(data.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_health_no_new_fields(self, client):
        """The legacy health response should not gain new fields."""
        resp = await client.get("/api/v1/health")
        data = resp.json()
        expected_keys = {
            "status", "version", "engine_reachable",
            "db_healthy", "startup_state",
        }
        assert set(data.keys()) == expected_keys
