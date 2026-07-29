"""Tests for OpenAPI documentation improvements (Phase 4)."""

import pytest


@pytest.mark.anyio
async def test_openapi_schema_is_accessible(client):
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/api/v1/health" in schema["paths"]
    assert "/api/v1/analyze" in schema["paths"]


@pytest.mark.anyio
async def test_health_endpoint_has_description(client):
    response = await client.get("/openapi.json")
    schema = response.json()
    health_path = schema["paths"].get("/api/v1/health", {}).get("get", {})
    assert "description" in health_path
    assert len(health_path["description"]) > 10


@pytest.mark.anyio
async def test_analyze_endpoints_have_responses(client):
    response = await client.get("/openapi.json")
    schema = response.json()
    post_ep = schema["paths"].get("/api/v1/analyze", {}).get("post", {})
    assert "responses" in post_ep
    assert "200" in post_ep["responses"]
    assert "422" in post_ep["responses"]


@pytest.mark.anyio
async def test_list_analyses_has_401_response(client):
    response = await client.get("/openapi.json")
    schema = response.json()
    get_ep = schema["paths"].get("/api/v1/analyze", {}).get("get", {})
    assert "responses" in get_ep
    assert "401" in get_ep["responses"]


@pytest.mark.anyio
async def test_security_scheme_is_registered(client):
    response = await client.get("/openapi.json")
    schema = response.json()
    components = schema.get("components", {})
    security_schemes = components.get("securitySchemes", {})
    assert "HTTPBearer" in security_schemes or any(
        v.get("scheme") == "bearer" for v in security_schemes.values()
    )
