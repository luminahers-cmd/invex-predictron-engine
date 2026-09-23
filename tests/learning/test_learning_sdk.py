"""Tests for the learning resource (Phase 7 continuous intelligence)."""

from __future__ import annotations

from tests.sdk.conftest import json_response

SUMMARY_PAYLOAD = {
    "scope": "user:demo",
    "period_kind": "daily",
    "anchor_date": "2026-09-21",
    "generated_at": "2026-09-21T12:00:00Z",
    "engine_version": "7.0.0",
    "counts": {"evaluations": 10, "samples": 10, "scoreable": 10},
    "metrics": {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "ece": 0.0,
        "overconfidence": 0.0,
        "confidence.mean": 0.5,
        "confidence.bias": 0.0,
    },
    "digest": {
        "evaluation_count": 10,
        "sample_count": 10,
        "scoreable": 10,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "true_positive": 5,
        "true_negative": 5,
        "false_positive": 0,
        "false_negative": 0,
        "verdict_counts": {"correct": 10},
    },
    "calibration": {
        "expected_calibration_error": 0.0,
        "overconfidence_detected": False,
        "overconfident_bins": 0,
        "total_samples": 10,
        "bins": [],
    },
    "confidence": {
        "count": 10,
        "mean": 0.5,
        "std_dev": 0.5,
        "bias": 0.0,
        "calibrated": True,
    },
    "distributions": {"sector": {"ai": 10}},
    "knowledge": {
        "sector": [
            {
                "dimension": "sector",
                "value": "ai",
                "sample_size": 10,
                "accuracy": 1.0,
                "confidence_bias": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
                "recommendation": "maintain",
            }
        ]
    },
    "patterns": [
        {
            "dimension": "sector",
            "value": "ai",
            "samples": 10,
            "scoreable": 10,
            "true_positive": 5,
            "true_negative": 5,
            "false_positive": 0,
            "false_negative": 0,
            "accuracy": 1.0,
            "precision": 1.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "confidence": 0.5,
            "confidence_bias": 0.0,
            "recommendation": "maintain",
        }
    ],
    "observations": [
        {
            "category": "strong_accuracy",
            "dimension": "sector",
            "value": "ai",
            "metric": "accuracy",
            "metric_value": 1.0,
            "delta": None,
            "direction": "flat",
            "baseline": None,
            "sample_size": 10,
            "summary": "strong accuracy",
        }
    ],
    "recommendations": [
        {
            "kind": "maintain_confidence_weighting",
            "message": "Platform accuracy supports the current confidence weighting.",
            "severity": "info",
            "parameters": {"accuracy": 1.0},
        }
    ],
}

PATTERNS_PAYLOAD = {
    "scope": "user:demo",
    "anchor_date": "2026-09-21",
    "generated_at": "2026-09-21T12:00:00Z",
    "patterns": SUMMARY_PAYLOAD["patterns"],
}

OBSERVATIONS_PAYLOAD = {
    "scope": "user:demo",
    "anchor_date": "2026-09-21",
    "generated_at": "2026-09-21T12:00:00Z",
    "observations": SUMMARY_PAYLOAD["observations"],
}

CONFIDENCE_PAYLOAD = {
    "scope": "user:demo",
    "anchor_date": "2026-09-21",
    "generated_at": "2026-09-21T12:00:00Z",
    "confidence": SUMMARY_PAYLOAD["confidence"],
    "calibration": SUMMARY_PAYLOAD["calibration"],
}

BIAS_PAYLOAD = {
    "scope": "user:demo",
    "anchor_date": "2026-09-21",
    "generated_at": "2026-09-21T12:00:00Z",
    "metrics": SUMMARY_PAYLOAD["metrics"],
    "knowledge": SUMMARY_PAYLOAD["knowledge"],
}

RECOMMENDATIONS_PAYLOAD = {
    "scope": "user:demo",
    "anchor_date": "2026-09-21",
    "generated_at": "2026-09-21T12:00:00Z",
    "recommendations": SUMMARY_PAYLOAD["recommendations"],
}

KNOWLEDGE_PAYLOAD = {
    "dimension": "sector",
    "scope": "user:demo",
    "anchor_date": "2026-09-21",
    "generated_at": "2026-09-21T12:00:00Z",
    "entries": SUMMARY_PAYLOAD["knowledge"]["sector"],
}

REPORTS_PAYLOAD = {
    "period_kind": "daily",
    "reports": [
        {
            "snapshot_id": "snap-1",
            "scope": "repository",
            "period_kind": "daily",
            "anchor_date": "2026-09-21",
            "engine_version": "7.0.0",
            "content_hash": "abc123",
            "recorded_at": "2026-09-21T12:00:00Z",
        }
    ],
}

REPORT_DETAIL_PAYLOAD = {
    "snapshot_id": "snap-1",
    "report_id": "rep-1",
    "content_hash": "abc123",
    "recorded_at": "2026-09-21T12:00:00Z",
    "payload": SUMMARY_PAYLOAD,
}


def test_learning_summary_parses(make_client):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/learning/summary"
        return json_response(200, SUMMARY_PAYLOAD)

    client = make_client(handler)
    summary = client.learning.summary()
    assert summary.scope == "user:demo"
    assert summary.counts["evaluations"] == 10
    assert summary.metrics["accuracy"] == 1.0
    assert summary.digest.true_positive == 5
    assert summary.calibration.overconfidence_detected is False
    assert summary.confidence.mean == 0.5
    assert summary.distributions["sector"]["ai"] == 10
    assert summary.knowledge["sector"][0].value == "ai"
    assert summary.patterns[0].accuracy == 1.0
    assert summary.observations[0].category == "strong_accuracy"
    assert summary.recommendations[0].kind == "maintain_confidence_weighting"


def test_learning_patterns_and_observations(make_client):
    def handler(request):
        path = request.url.path
        assert path in ("/api/v1/learning/patterns", "/api/v1/learning/observations")
        payload = (
            PATTERNS_PAYLOAD if path.endswith("/patterns") else OBSERVATIONS_PAYLOAD
        )
        return json_response(200, payload)

    client = make_client(handler)
    patterns = client.learning.patterns()
    observations = client.learning.observations()
    assert patterns.patterns[0].dimension == "sector"
    assert observations.observations[0].metric == "accuracy"


def test_learning_confidence_and_bias(make_client):
    def handler(request):
        path = request.url.path
        payload = (
            CONFIDENCE_PAYLOAD if path.endswith("/confidence") else BIAS_PAYLOAD
        )
        return json_response(200, payload)

    client = make_client(handler)
    confidence = client.learning.confidence()
    bias = client.learning.bias()
    assert confidence.confidence.calibrated is True
    assert confidence.calibration.expected_calibration_error == 0.0
    assert bias.metrics["accuracy"] == 1.0
    assert bias.knowledge["sector"][0].sample_size == 10


def test_learning_recommendations(make_client):
    def handler(request):
        assert request.url.path == "/api/v1/learning/recommendations"
        return json_response(200, RECOMMENDATIONS_PAYLOAD)

    client = make_client(handler)
    recommendations = client.learning.recommendations()
    assert recommendations.recommendations[0].severity == "info"
    assert recommendations.recommendations[0].parameters["accuracy"] == 1.0


def test_learning_knowledge_dimension_and_shortcuts(make_client):
    for path, method in (
        ("/api/v1/learning/knowledge/sector", lambda c: c.learning.knowledge("sector")),
        ("/api/v1/learning/sector", lambda c: c.learning.sector()),
        ("/api/v1/learning/technology", lambda c: c.learning.technology()),
        ("/api/v1/learning/country", lambda c: c.learning.country()),
        ("/api/v1/learning/stage", lambda c: c.learning.stage()),
        ("/api/v1/learning/founders", lambda c: c.learning.founders()),
        (
            "/api/v1/learning/business-model",
            lambda c: c.learning.business_model(),
        ),
    ):

        def handler(request):
            assert request.url.path == path
            return json_response(200, KNOWLEDGE_PAYLOAD)

        client = make_client(handler)
        knowledge = method(client)
        assert knowledge.dimension == "sector"
        assert knowledge.entries[0].value == "ai"


def test_learning_reports_and_report_detail(make_client):
    def handler(request):
        if request.url.path == "/api/v1/learning/reports":
            assert request.url.params["period"] == "weekly"
            return json_response(200, REPORTS_PAYLOAD)
        assert request.url.path == "/api/v1/learning/reports/snap-1"
        return json_response(200, REPORT_DETAIL_PAYLOAD)

    client = make_client(handler)
    reports = client.learning.reports(period="weekly")
    assert reports.period_kind == "daily"
    assert reports.reports[0].snapshot_id == "snap-1"
    assert reports.reports[0].content_hash == "abc123"

    detail = client.learning.report("snap-1")
    assert detail.snapshot_id == "snap-1"
    assert detail.content_hash == "abc123"
    assert detail.payload["counts"]["evaluations"] == 10
