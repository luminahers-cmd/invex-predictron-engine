"""Tests for the health resource."""

from __future__ import annotations

from tests.sdk.conftest import json_response


def test_health_check(make_client, health_payload) -> None:
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/health"
        return json_response(200, health_payload)

    client = make_client(handler)
    health = client.health.check()
    assert health.status == "ok"
    assert health.version == "0.12.1"
    assert health.engine_reachable is True


def test_health_check_alias(make_client, health_payload) -> None:
    def handler(request):
        return json_response(200, health_payload)

    client = make_client(handler)
    assert client.health.health() == client.health.check()


def test_health_readiness(make_client, readiness_payload) -> None:
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/health/readiness"
        return json_response(200, readiness_payload)

    client = make_client(handler)
    readiness = client.health.readiness()
    assert readiness.status == "ready"
    assert readiness.db_healthy is True
    assert readiness.engine_ready is True


def test_client_health_status_alias(make_client, health_payload) -> None:
    def handler(request):
        return json_response(200, health_payload)

    client = make_client(handler)
    assert client.health_status().status == "ok"


def test_health_created_at_absent_ok(make_client) -> None:
    def handler(request):
        return json_response(200, {"status": "ok"})

    client = make_client(handler)
    health = client.health.check()
    assert health.db_healthy is False
