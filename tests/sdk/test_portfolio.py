"""Tests for the portfolio analysis resource."""

from __future__ import annotations

import json

import pytest

from tests.sdk.conftest import json_response


def test_portfolio_analyze(make_client, portfolio_payload) -> None:
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/portfolio"
        body = json.loads(request.content)
        assert body["company_names"] == ["Alpha", "Beta"]
        return json_response(200, portfolio_payload)

    client = make_client(handler)
    analysis = client.portfolio.analyze(company_names=["Alpha", "Beta"])
    assert analysis.company_count == 2
    assert analysis.companies[0].startup_name == "Alpha"
    assert analysis.portfolio_score == 62.5
    assert analysis.concentration_risk.sector_concentration == 1.0


def test_portfolio_analyze_min_two(make_client) -> None:
    def handler(request):
        return json_response(200, {})

    client = make_client(handler)
    with pytest.raises(ValueError):
        client.portfolio.analyze(company_names=["Single"])


def test_portfolio_similarity(make_client) -> None:
    payload = {
        "companies": ["Alpha", "Beta"],
        "matrix": [[1.0, 0.3], [0.3, 1.0]],
        "method": "score_cosine",
    }

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/portfolio/similarity"
        body = json.loads(request.content)
        assert body["company_names"] == ["Alpha", "Beta"]
        return json_response(200, payload)

    client = make_client(handler)
    matrix = client.portfolio.similarity(company_names=["Alpha", "Beta"])
    assert matrix.matrix[0][1] == 0.3


def test_portfolio_descriptions_passed(make_client, portfolio_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body["descriptions"] == {"Alpha": "desc alpha"}
        return json_response(200, portfolio_payload)

    client = make_client(handler)
    client.portfolio.analyze(
        company_names=["Alpha", "Beta"],
        descriptions={"Alpha": "desc alpha"},
    )


def test_portfolio_payload_excludes_unknown_fields(make_client, portfolio_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert "portfolio_score" not in body
        return json_response(200, portfolio_payload)

    client = make_client(handler)
    client.portfolio.analyze(company_names=["Alpha", "Beta"])


def test_client_portfolio_analyze_alias(make_client, portfolio_payload) -> None:
    def handler(request):
        assert request.url.path == "/api/v1/portfolio"
        return json_response(200, portfolio_payload)

    client = make_client(handler)
    analysis = client.portfolio_analyze(company_names=["Alpha", "Beta"])
    assert analysis.company_count == 2


def test_portfolio_nested_model_population(make_client, portfolio_payload) -> None:
    def handler(request):
        return json_response(200, portfolio_payload)

    client = make_client(handler)
    analysis = client.portfolio.analyze(company_names=["Alpha", "Beta"])
    assert analysis.risk_summary.high_risk_count == 1
    assert analysis.diversification.overall_diversification == 0.17
    assert analysis.heatmap_data[0].row_label == "Alpha"
    assert analysis.sector_distribution[0].percentage == 100.0


def test_portfolio_similarity_matrix_shape(make_client) -> None:
    payload = {
        "companies": ["A", "B", "C"],
        "matrix": [
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.4],
            [0.2, 0.4, 1.0],
        ],
        "method": "score_cosine",
    }

    def handler(request):
        return json_response(200, payload)

    client = make_client(handler)
    matrix = client.portfolio.similarity(company_names=["A", "B", "C"])
    assert len(matrix.matrix) == 3
    assert matrix.matrix[2][0] == 0.2
