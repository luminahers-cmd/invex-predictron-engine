"""Tests for the legacy analyze resource."""

from __future__ import annotations

import json

import pytest

from predictron_sdk.models import AnalyzeRequest
from tests.sdk.conftest import json_response


def test_analyze_post(make_client, analysis_payload) -> None:
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/analyze"
        body = json.loads(request.content)
        assert body["startup_name"] == "Acme AI"
        assert body["description"].startswith("Acme AI")
        return json_response(200, analysis_payload)

    client = make_client(handler)
    analysis = client.analyze.analyze(
        startup_name="Acme AI",
        description="Acme AI builds enterprise ML tooling.",
    )
    assert analysis.id == "persisted-1"
    assert analysis.venture_score == 71.0


def test_analyze_request_model(make_client, analysis_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body["website_url"] == "https://example.com"
        return json_response(200, analysis_payload)

    client = make_client(handler)
    client.analyze.analyze(
        request=AnalyzeRequest(
            startup_name="A",
            description="A sufficiently long description.",
            website_url="https://example.com",
        )
    )


def test_analyze_fails_without_description(make_client) -> None:
    client = make_client(lambda r: json_response(200, {}))
    with pytest.raises(ValueError):
        client.analyze.analyze(startup_name="A")


def test_analyze_list_analyses(make_client, analysis_payload) -> None:
    payload = {
        "analyses": [{**analysis_payload, "created_at": "2026-01-01T00:00:00Z"}],
        "total": 1,
    }

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/analyze"
        return json_response(200, payload)

    client = make_client(handler)
    listing = client.analyze.list_analyses()
    assert listing.total == 1
    assert listing.analyses[0].venture_score == 71.0


def test_analyze_get_by_id(make_client, analysis_payload) -> None:
    detail_payload = {
        **analysis_payload,
        "website": "https://acme.example.com",
        "description": "Acme AI builds enterprise ML tooling.",
        "created_at": "2026-01-01T00:00:00Z",
        "full_report": {"scores": analysis_payload},
    }

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/analyze/persisted-1"
        return json_response(200, detail_payload)

    client = make_client(handler)
    detail = client.analyze.get("persisted-1")
    assert detail.id == "persisted-1"
    assert detail.full_report == {"scores": analysis_payload}


def test_analyze_get_params_passed(make_client) -> None:
    payload = {"analyses": [], "total": 0}

    def handler(request):
        params = dict(request.url.params)
        assert params["offset"] == "5"
        assert params["limit"] == "10"
        return json_response(200, payload)

    client = make_client(handler)
    client.analyze.list_analyses(offset=5, limit=10)


def test_analyze_payload_excludes_none(make_client, analysis_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert "website_url" not in body
        return json_response(200, analysis_payload)

    client = make_client(handler)
    client.analyze.analyze(
        startup_name="Acme AI",
        description="Acme AI builds enterprise ML tooling.",
    )
