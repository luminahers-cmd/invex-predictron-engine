"""Tests for the comparison resource (and top-level compare method)."""

from __future__ import annotations

import json

import pytest

from tests.sdk.conftest import json_response


@pytest.mark.parametrize("as_method", [True, False])
def test_compare(make_client, comparison_payload, as_method) -> None:
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/compare"
        body = json.loads(request.content)
        assert body["company_names"] == ["Alpha", "Beta"]
        return json_response(200, comparison_payload)

    client = make_client(handler)
    result = (
        client.compare(company_names=["Alpha", "Beta"])
        if as_method
        else client.comparison.compare(company_names=["Alpha", "Beta"])
    )
    assert result.companies == ["Alpha", "Beta"]
    assert result.feature_diffs.feature_diffs[0].direction == "alpha_greater"
    assert result.overall_summary["avg_score"] == 62.5


def test_compare_too_few(make_client) -> None:
    def handler(request):
        return json_response(200, {})

    client = make_client(handler)
    with pytest.raises(ValueError):
        client.compare(company_names=["Single"])


def test_compare_too_many(make_client) -> None:
    def handler(request):
        return json_response(200, {})

    client = make_client(handler)
    with pytest.raises(ValueError):
        client.compare(company_names=[f"c{i}" for i in range(11)])


def test_compare_nested_diffs(make_client, comparison_payload) -> None:
    def handler(request):
        return json_response(200, comparison_payload)

    client = make_client(handler)
    result = client.compare(company_names=["Alpha", "Beta"])
    assert result.decision_diffs.decision_comparison[0]["startup"] == "Alpha"
    assert result.contribution_diffs.diffs[0].max_contribution == 10.0
    assert result.signal_diffs.signal_comparison == []
    assert result.benchmark_comparison.benchmark_metrics == []


def test_comparison_repr_matches_client_delegate(make_client, comparison_payload) -> None:
    def handler(request):
        return json_response(200, comparison_payload)

    client = make_client(handler)
    assert client.comparison is not None


def test_compare_supports_items_kwarg(make_client, comparison_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body["company_names"] == ["Alpha", "Beta"]
        return json_response(200, comparison_payload)

    client = make_client(handler)
    result = client.compare(company_names=["Alpha", "Beta"])
    assert result.overall_summary["score_range"] == 15.0
