"""Tests for the monitor resource (Phase 6 continuous intelligence)."""

from __future__ import annotations

from datetime import date

from tests.sdk.conftest import json_response

SUMMARY_PAYLOAD = {
    "scope": "user:demo",
    "period_kind": "daily",
    "anchor_date": "2026-09-21",
    "generated_at": "2026-09-21T12:00:00Z",
    "counts": {"forecasts": 3, "evaluations": 2, "companies": 3},
    "metrics": {
        "metrics.accuracy": 0.5,
        "confidence.mean": 0.7,
        "calibration.ece": 0.1,
    },
    "distributions": {
        "evaluation_verdict": {
            "label": "evaluation_verdict",
            "universe": ["correct", "incorrect"],
            "counts": {"correct": 1, "incorrect": 1},
        }
    },
    "horizon_breakdown": {
        "90": {"horizon_days": 90, "forecasts": 2, "evaluated": 1, "scoreable": 1, "accuracy": 1.0}
    },
    "sector_breakdown": {
        "overall": {"sector": "overall", "evaluations": 2, "scoreable": 2, "accuracy": 0.5}
    },
    "health": {"active": 2, "resolved": 1},
}

HEALTH_PAYLOAD = {
    "as_of": "2026-09-21T12:00:00Z",
    "generated_at": "2026-09-21T12:00:00Z",
    "scope": "user:demo",
    "entries": [
        {
            "forecast_id": "f1",
            "company_id": "c1",
            "snapshot_id": "s1",
            "decision": "invest",
            "confidence": 0.8,
            "status": "active",
            "health": "overdue",
            "analysis_timestamp": "2026-08-01T12:00:00Z",
            "due_at": "2026-08-10T12:00:00Z",
            "as_of": "2026-09-21T12:00:00Z",
            "age_days": 51,
            "days_until_due": -42.0,
            "days_overdue": 42.0,
            "outcome_id": None,
            "outcome_verdict": None,
            "evaluation_status": "pending",
            "evaluation_verdict": None,
        }
    ],
    "distribution": {"active": 2, "overdue": 1},
}

DRIFT_PAYLOAD = {
    "baseline_id": "id-21",
    "comparison_id": "id-22",
    "baseline_period": "2026-09-21",
    "comparison_period": "2026-09-22",
    "signals": [
        {
            "signal": "calibration",
            "from_value": 0.05,
            "to_value": 0.2,
            "delta": 0.15,
            "magnitude": 15.0,
            "direction": "up",
            "severity": "moderate",
            "affected": True,
            "detail": {"ece_gap": 0.15},
        }
    ],
}

TRENDS_PAYLOAD = {
    "period_kind": "daily",
    "as_of": "2026-09-22",
    "trends": [
        {
            "metric": "metrics.accuracy",
            "direction": "up",
            "from_value": 0.5,
            "to_value": 0.9,
            "series": [
                {"anchor": "2026-09-21", "value": 0.5},
                {"anchor": "2026-09-22", "value": 0.9},
            ],
        }
    ],
}

REANALYSIS_PAYLOAD = {
    "as_of": "2026-09-21T12:00:00Z",
    "generated_at": "2026-09-21T12:00:00Z",
    "scope": "user:demo",
    "recommendations": [
        {
            "company_id": "c1",
            "forecast_id": "f1",
            "snapshot_id": "s1",
            "reasons": ["prediction_expired"],
            "latest_outcome_at": None,
            "latest_snapshot_at": None,
        }
    ],
}


def test_monitor_summary_parses(make_client):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/monitor/summary"
        return json_response(200, SUMMARY_PAYLOAD)

    client = make_client(handler)
    summary = client.monitor.summary()
    assert summary.counts["forecasts"] == 3
    assert summary.metrics["metrics.accuracy"] == 0.5
    assert summary.anchor_date == date(2026, 9, 21)
    verdicts = summary.distributions["evaluation_verdict"]
    assert verdicts.counts["correct"] == 1
    assert summary.horizon_breakdown["90"].forecasts == 2
    assert summary.sector_breakdown["overall"].accuracy == 0.5
    assert summary.health["resolved"] == 1


def test_monitor_rollup(make_client):
    def handler(request):
        assert request.url.path == "/api/v1/monitor/rollup"
        return json_response(200, SUMMARY_PAYLOAD)

    client = make_client(handler)
    assert client.monitor.rollup().counts["companies"] == 3


def test_monitor_health_and_filters(make_client):
    for path in ("/api/v1/monitor/health", "/api/v1/monitor/stale",
                 "/api/v1/monitor/overdue"):
        def handler(request):
            assert request.url.path == path
            return json_response(200, HEALTH_PAYLOAD)

        client = make_client(handler)
        health = client.monitor.health() if path.endswith("/health") else (
            client.monitor.stale() if path.endswith("/stale") else client.monitor.overdue()
        )
        assert health.entries[0].forecast_id == "f1"
        assert health.entries[0].health == "overdue"
        assert health.distribution["overdue"] == 1


def test_monitor_reanalysis(make_client):
    def handler(request):
        assert request.url.path == "/api/v1/monitor/reanalysis"
        return json_response(200, REANALYSIS_PAYLOAD)

    client = make_client(handler)
    reanalysis = client.monitor.reanalysis()
    assert reanalysis.recommendations[0].reasons == ["prediction_expired"]
    assert reanalysis.scope == "user:demo"


def test_monitor_trends_sends_period(make_client):
    def handler(request):
        assert request.url.path == "/api/v1/monitor/trends"
        assert request.url.params["period"] == "weekly"
        return json_response(200, TRENDS_PAYLOAD)

    client = make_client(handler)
    trends = client.monitor.trends(period="weekly")
    assert trends.trends[0].direction == "up"
    assert trends.trends[0].series[1].value == 0.9


def test_monitor_drift_defaults_to_latest(make_client):
    def handler(request):
        assert request.url.path == "/api/v1/monitor/drift"
        assert "before_id" not in request.url.params
        assert "after_id" not in request.url.params
        assert request.url.params["period"] == "daily"
        return json_response(200, DRIFT_PAYLOAD)

    client = make_client(handler)
    drift = client.monitor.drift()
    assert drift.baseline_id == "id-21"
    assert drift.signals[0].severity == "moderate"
    assert drift.signals[0].affected is True


def test_monitor_drift_with_explicit_ids(make_client):
    def handler(request):
        assert request.url.params["before_id"] == "abc"
        assert request.url.params["after_id"] == "def"
        return json_response(200, DRIFT_PAYLOAD)

    client = make_client(handler)
    client.monitor.drift(before_id="abc", after_id="def")
