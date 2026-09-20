"""Search API tests."""

from __future__ import annotations

import pytest


class TestSearchPost:
    """Tests for POST /api/v1/search."""

    @pytest.mark.asyncio
    async def test_search_returns_200(self, client):
        resp = await client.post("/api/v1/search", json={
            "query": "test",
            "search_type": "all",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_response_structure(self, client):
        resp = await client.post("/api/v1/search", json={"query": "test"})
        data = resp.json()
        assert "query" in data
        assert "total" in data
        assert "results" in data
        assert "offset" in data
        assert "limit" in data

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_422(self, client):
        resp = await client.post("/api/v1/search", json={"query": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_search_default_type_is_all(self, client):
        resp = await client.post("/api/v1/search", json={"query": "test"})
        data = resp.json()
        assert data["search_type"] == "all"

    @pytest.mark.asyncio
    async def test_search_company_type(self, client):
        resp = await client.post("/api/v1/search", json={
            "query": "test",
            "search_type": "company",
        })
        data = resp.json()
        assert data["search_type"] == "company"

    @pytest.mark.asyncio
    async def test_search_signal_type(self, client):
        resp = await client.post("/api/v1/search", json={
            "query": "funding",
            "search_type": "signal_type",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_industry_type(self, client):
        resp = await client.post("/api/v1/search", json={
            "query": "fintech",
            "search_type": "industry",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_country_type(self, client):
        resp = await client.post("/api/v1/search", json={
            "query": "us",
            "search_type": "country",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_knowledge_graph_node_type(self, client):
        resp = await client.post("/api/v1/search", json={
            "query": "company",
            "search_type": "knowledge_graph_node",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_pagination_offset(self, client):
        resp = await client.post("/api/v1/search", json={
            "query": "test",
            "offset": 0,
            "limit": 2,
        })
        data = resp.json()
        assert len(data["results"]) <= 2


class TestSearchGet:
    """Tests for GET /api/v1/search."""

    @pytest.mark.asyncio
    async def test_search_get_returns_200(self, client):
        resp = await client.get("/api/v1/search?q=test")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_get_response_structure(self, client):
        resp = await client.get("/api/v1/search?q=fintech")
        data = resp.json()
        assert "results" in data
        assert data["query"] == "fintech"

    @pytest.mark.asyncio
    async def test_search_get_with_type(self, client):
        resp = await client.get("/api/v1/search?q=test&search_type=company")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_get_pagination(self, client):
        resp = await client.get("/api/v1/search?q=test&offset=0&limit=5")
        data = resp.json()
        assert len(data["results"]) <= 5


class TestSearchByType:
    """Tests for GET /api/v1/search/by-type/{search_type}."""

    @pytest.mark.asyncio
    async def test_search_by_type_company(self, client):
        resp = await client.get(
            "/api/v1/search/by-type/company?q=test"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["search_type"] == "company"

    @pytest.mark.asyncio
    async def test_search_by_type_industry(self, client):
        resp = await client.get(
            "/api/v1/search/by-type/industry?q=fintech"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["search_type"] == "industry"

    @pytest.mark.asyncio
    async def test_search_by_type_signal(self, client):
        resp = await client.get(
            "/api/v1/search/by-type/signal_type?q=funding"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["search_type"] == "signal_type"

    @pytest.mark.asyncio
    async def test_search_by_type_knowledge_graph_node(self, client):
        resp = await client.get(
            "/api/v1/search/by-type/knowledge_graph_node?q=company"
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_by_type_investor(self, client):
        resp = await client.get(
            "/api/v1/search/by-type/investor?q=sequoia"
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_by_type_founder(self, client):
        resp = await client.get(
            "/api/v1/search/by-type/founder?q=john"
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_by_type_technology(self, client):
        resp = await client.get(
            "/api/v1/search/by-type/technology?q=python"
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_by_type_country(self, client):
        resp = await client.get(
            "/api/v1/search/by-type/country?q=us"
        )
        assert resp.status_code == 200
