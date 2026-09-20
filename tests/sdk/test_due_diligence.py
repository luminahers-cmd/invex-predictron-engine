"""Tests for the due diligence resource."""

from __future__ import annotations

import json

import pytest

from predictron_sdk.models import DueDiligenceRequest
from tests.sdk.conftest import json_response


def test_due_diligence_generate(make_client, due_diligence_payload) -> None:
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/due-diligence"
        body = json.loads(request.content)
        assert body["startup_name"] == "Acme AI"
        assert body["description"].startswith("Acme AI")
        return json_response(200, due_diligence_payload)

    client = make_client(handler)
    report = client.due_diligence.generate(
        startup_name="Acme AI",
        description="Acme AI builds enterprise ML tooling.",
    )
    assert report.report_id == "dd-1"
    assert len(report.risks) == 1
    assert len(report.strengths) == 1
    assert len(report.opportunities) == 1


def test_due_diligence_generate_fails_without_args(make_client) -> None:
    client = make_client(lambda r: json_response(200, {}))
    with pytest.raises(ValueError):
        client.due_diligence.generate(startup_name="Only name")


def test_due_diligence_generate_takes_request(
    make_client, due_diligence_payload
) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body["website_url"] == "https://acme.example.com"
        return json_response(200, due_diligence_payload)

    client = make_client(handler)
    client.due_diligence.generate(
        request=DueDiligenceRequest(
            startup_name="Acme AI",
            description="A sufficiently long description.",
            website_url="https://acme.example.com",
        )
    )


def test_due_diligence_report_nested_sections(make_client, due_diligence_payload) -> None:
    def handler(request):
        return json_response(200, due_diligence_payload)

    client = make_client(handler)
    report = client.due_diligence.generate(
        startup_name="Acme AI",
        description="Acme AI builds enterprise ML tooling.",
    )
    assert report.executive_summary.headline == "Strong candidate"
    assert report.strengths[0].title == "Team"
    assert report.weaknesses[0].severity == "medium"
    assert report.opportunities[0].impact == "high"
    assert report.risks[0].mitigation == "Focus niche"
    assert report.decision_trace.category == "invest"
    assert report.benchmark_context.sample_size == 120


def test_due_diligence_report_uses_appropriate_name(
    make_client, due_diligence_payload
) -> None:
    def handler(request):
        return json_response(200, due_diligence_payload)

    client = make_client(handler)
    result = client.due_diligence.report(
        startup_name="Acme AI",
        description="Acme AI builds enterprise ML tooling.",
    )
    assert isinstance(result, type(client.due_diligence.generate(
        startup_name="Acme AI",
        description="Acme AI builds enterprise ML tooling.",
    )))


def test_due_diligence_evidence_entries(make_client, due_diligence_payload) -> None:
    def handler(request):
        return json_response(200, due_diligence_payload)

    client = make_client(handler)
    report = client.due_diligence.generate(
        startup_name="Acme AI",
        description="Acme AI builds enterprise ML tooling.",
    )
    assert report.evidence[0].claim == "Revenue"
    assert report.evidence[0].trust_score == 0.9
    assert report.supporting_features == {"revenue_growth": 0.4}
    assert report.confidence == 0.78
