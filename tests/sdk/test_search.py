"""Tests for the search resource."""

from __future__ import annotations

import json

import pytest

from tests.sdk.conftest import json_response


def test_search_post(make_client, search_payload) -> None:
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/search"
        body = json.loads(request.content)
        assert body["query"] == "robotics"
        assert body["search_type"] == "all"
        return json_response(200, search_payload)

    client = make_client(handler)
    result = client.search(query="robotics")
    assert result.query == "robotics"
    assert result.results[0].name == "Robotics Co"


def test_search_via_resource(make_client, search_payload) -> None:
    def handler(request):
        assert request.method == "POST"
        return json_response(200, search_payload)

    client = make_client(handler)
    result = client.search_resource.search(query="robotics")
    assert result.results[0].result_type == "company"


def test_search_resource_get(make_client, search_payload) -> None:
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/search"
        assert "q=robotics" in str(request.url.query)
        return json_response(200, search_payload)

    client = make_client(handler)
    result = client.search_resource.get(query="robotics")
    assert result.total == 1


def test_search_by_type(make_client, search_by_type_payload) -> None:
    def handler(request):
        assert request.method == "GET"
        assert "/api/v1/search/by-type/company" in request.url.path
        return json_response(200, search_by_type_payload)

    client = make_client(handler)
    result = client.search_by_type(search_type="company", query="robotics")
    assert result.search_type == "company"
    assert result.companies[0].startup_name == "Robotics Co"


def test_search_limit_bounds(make_client) -> None:
    def handler(request):
        return json_response(200, {})

    client = make_client(handler)
    with pytest.raises(ValueError):
        client.search(query="q", limit=0)
    with pytest.raises(ValueError):
        client.search(query="q", limit=200)


def test_search_results_metadata(make_client, search_payload) -> None:
    def handler(request):
        return json_response(200, search_payload)

    client = make_client(handler)
    result = client.search(query="robotics")
    assert result.offset == 0
    assert result.limit == 20
    assert result.total == 1


def test_search_payload_includes_query_and_defaults(make_client, search_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body["query"] == "robotics"
        assert body["offset"] == 0
        assert body["limit"] == 20
        return json_response(200, search_payload)

    client = make_client(handler)
    client.search(query="robotics")


def test_search_resource_repr(make_client, search_payload) -> None:
    def handler(request):
        return json_response(200, search_payload)

    client = make_client(handler)
    assert "SearchResource" in repr(client.search_resource)
