"""Tests for the venture analysis resource."""

from __future__ import annotations

import json

import pytest

from predictron_sdk.models import VentureRequest
from tests.sdk.conftest import json_response


def test_venture_evaluate(make_client, venture_payload) -> None:
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/venture"
        body = json.loads(request.content)
        assert body["startup_name"] == "Acme AI"
        assert body["description"].startswith("Acme AI")
        return json_response(200, venture_payload)

    client = make_client(handler)
    analysis = client.venture.evaluate(
        startup_name="Acme AI",
        description="Acme AI builds enterprise ML tooling.",
    )
    assert analysis.overall_score == 72.4
    assert analysis.overall_confidence == 0.81
    assert analysis.decision_category == "invest"
    assert [d.dimension for d in analysis.dimension_scores] == [
        "market_opportunity",
        "founder_quality",
    ]


def test_venture_evaluate_with_request_model(make_client, venture_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body["startup_name"] == "From Model"
        return json_response(200, {**venture_payload, "startup_name": "From Model"})

    client = make_client(handler)
    analysis = client.venture.evaluate(
        request=VentureRequest(
            startup_name="From Model",
            description="A sufficiently long description.",
        )
    )
    assert analysis.startup_name == "From Model"


def test_venture_explain(make_client) -> None:
    payload = {
        "company_id": "c1",
        "overall_score": 70.0,
        "verdict": "invest",
        "confidence": 0.8,
        "explanation": {"company_id": "c1", "headline": "Strong"},
        "top_strengths": ["Team"],
        "top_weaknesses": ["Burn"],
    }

    def handler(request):
        assert request.url.path == "/api/v1/venture/explain"
        return json_response(200, payload)

    client = make_client(handler)
    result = client.venture.explain(
        startup_name="A", description="A sufficiently long description."
    )
    assert result.verdict == "invest"
    assert result.explanation.headline == "Strong"


def test_venture_trace(make_client) -> None:
    payload = {
        "company_id": "c1",
        "trace_id": "t1",
        "nodes": [{"node_id": "n1", "node_type": "score", "label": "Overall"}],
        "edges": [["n1"]],
    }

    def handler(request):
        assert request.url.path == "/api/v1/venture/trace"
        return json_response(200, payload)

    client = make_client(handler)
    trace = client.venture.trace(
        startup_name="A", description="A sufficiently long description."
    )
    assert trace.trace_id == "t1"
    assert trace.nodes[0].label == "Overall"


def test_venture_features(make_client) -> None:
    payload = {
        "company_id": "c1",
        "feature_count": 1,
        "features": [
            {
                "feature_id": "f1",
                "feature_name": "revenue",
                "category": "traction",
                "value": 100000.0,
            }
        ],
    }

    def handler(request):
        assert request.url.path == "/api/v1/venture/features"
        return json_response(200, payload)

    client = make_client(handler)
    features = client.venture.features(
        startup_name="A", description="A sufficiently long description."
    )
    assert features.feature_count == 1
    assert features.features[0].value == 100000.0


def test_venture_benchmark(make_client) -> None:
    payload = {
        "startup_name": "A",
        "composite_score": 75.0,
        "benchmark_mean": 70.0,
        "benchmark_std_dev": 8.0,
        "percentile_rank": 0.8,
        "score_z_score": 0.6,
        "sample_size": 100,
        "category_distribution": {"invest": 40},
    }

    def handler(request):
        assert request.url.path == "/api/v1/venture/benchmark"
        return json_response(200, payload)

    client = make_client(handler)
    benchmark = client.venture.benchmark(
        startup_name="A", description="A sufficiently long description."
    )
    assert benchmark.composite_score == 75.0
    assert benchmark.sample_size == 100


def test_venture_knowledge_graph(make_client) -> None:
    payload = {
        "node_count": 4,
        "edge_count": 3,
        "node_type_counts": {"company": 4},
        "edge_type_counts": {"invests": 3},
        "connected_components": 1,
        "density": 0.5,
        "nodes": [{"node_id": "n1", "node_type": "company"}],
        "edges": [],
        "top_connected_nodes": [],
    }

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/venture/knowledge-graph"
        return json_response(200, payload)

    client = make_client(handler)
    graph = client.venture.knowledge_graph()
    assert graph.node_count == 4
    assert graph.nodes[0].node_id == "n1"


def test_venture_signals(make_client) -> None:
    payload = {
        "company_id": "c1",
        "signal_count": 1,
        "signals": [{"signal_id": "s1", "company_id": "c1", "signal_type": "funding"}],
        "type_counts": {"funding": 1},
    }

    def handler(request):
        assert request.url.path == "/api/v1/venture/signals/c1"
        return json_response(200, payload)

    client = make_client(handler)
    timeline = client.venture.signals("c1")
    assert timeline.signal_count == 1
    assert timeline.signals[0].signal_type == "funding"


def test_venture_signal_trends(make_client) -> None:
    payload = {
        "company_id": "c1",
        "trends": {
            "hiring": {
                "name": "hiring",
                "value": 3.0,
                "direction": "up",
                "window_days": 90,
                "explanation": "x",
                "available": True,
            }
        },
    }

    def handler(request):
        assert request.url.path == "/api/v1/venture/signals/c1/trends"
        return json_response(200, payload)

    client = make_client(handler)
    trends = client.venture.signal_trends("c1")
    assert trends.trends["hiring"].direction == "up"


def test_venture_signal_aggregation(make_client) -> None:
    payload = {"company_id": "c1", "momentum_score": 0.8}

    def handler(request):
        assert request.url.path == "/api/v1/venture/signals/c1/aggregation"
        return json_response(200, payload)

    client = make_client(handler)
    aggregation = client.venture.signal_aggregation("c1")
    assert aggregation.momentum_score == 0.8


def test_venture_requires_name_and_description(make_client) -> None:
    client = make_client(lambda r: json_response(200, {}))
    with pytest.raises(ValueError):
        client.venture.evaluate(startup_name="A")  # missing description


def test_venture_request_model_preferred_over_kwargs(make_client, venture_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body["startup_name"] == "Model Name"
        return json_response(200, {**venture_payload, "startup_name": "Model Name"})

    client = make_client(handler)
    result = client.venture.evaluate(
        startup_name="Kwarg Name (ignored)",
        description="A sufficiently long description.",
        request=VentureRequest(
            startup_name="Model Name",
            description="A sufficiently long description.",
        ),
    )
    assert result.startup_name == "Model Name"


def test_venture_payload_excludes_none(make_client, venture_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert "website_url" not in body
        return json_response(200, venture_payload)

    client = make_client(handler)
    client.venture.evaluate(
        startup_name="A", description="A sufficiently long description."
    )


def test_venture_url_encoding_in_paths(make_client) -> None:
    def handler(request):
        assert "/api/v1/venture/signals/a/b" in request.url.path
        return json_response(200, {"company_id": "a/b", "signal_count": 0})

    client = make_client(handler)
    client.venture.signals("a/b")


def test_client_evaluate_alias(make_client, venture_payload) -> None:
    def handler(request):
        return json_response(200, venture_payload)

    client = make_client(handler)
    analysis = client.evaluate(
        startup_name="Acme AI",
        description="Acme AI builds enterprise ML tooling.",
    )
    assert analysis.startup_name == "Acme AI"
