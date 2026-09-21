"""Tests for the append-only file-backed monitor history + trends/drift.

Covers round-trip persistence, rotation, integrity verification failures,
ordering guarantees, and the end-to-end metrics/trend/drift path that the
monitor API and CLI depend on.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from predictron_engine.dataset.evaluation import (
    EvaluationVerdict,
    PredictionEvaluation,
    PredictionOutcomeAlignment,
)
from predictron_engine.dataset.models import DecisionLabel, PredictionSummary
from predictron_engine.dataset.outcomes import (
    OutcomeRecord,
    OutcomeStatus,
    OutcomeVerdict,
    StartupOutcome,
)
from predictron_engine.monitoring.drift import detect_monitor_drift
from predictron_engine.monitoring.history import MonitorHistory
from predictron_engine.monitoring.models import (
    ForecastRecord,
    MonitorPeriodKind,
    MonitorSnapshot,
)
from predictron_engine.monitoring.snapshot import build_monitor_snapshot
from predictron_engine.monitoring.summaries import mean_confidence
from predictron_engine.monitoring.trends import compute_trend, metric_series

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
EPSILON = 1e-6


def record(
    *,
    forecast_id: str,
    snapshot_id: str,
    confidence: float,
    outcome_id: str | None = None,
) -> ForecastRecord:
    return ForecastRecord(
        id=forecast_id,
        company_id="c1",
        snapshot_id=snapshot_id,
        decision="invest",
        confidence=confidence,
        composite_score=70.0,
        status="active",
        analysis_timestamp=NOW - timedelta(days=20),
        due_at=NOW + timedelta(days=70),
        horizon_days=90,
        outcome_id=outcome_id,
        outcome_verdict=None,
        evaluation_verdict=None,
        evaluation_created_at=None,
        latest_outcome_at=None,
        latest_snapshot_at=None,
    )


def success_evaluation(record_id: str, evaluation_id: str) -> PredictionEvaluation:
    return PredictionEvaluation(
        evaluation_id=evaluation_id,
        record_id=record_id,
        prediction=PredictionSummary(
            decision=DecisionLabel.INVEST,
            confidence=0.8,
            composite_score=70.0,
            dimension_scores={"Market": 60.0},
        ),
        outcome_record=OutcomeRecord(
            record_id=record_id,
            verdict=OutcomeVerdict.SUCCESS,
            outcome=StartupOutcome(status=OutcomeStatus.FULLY_VERIFIED, exit_type="ipo"),
        ),
        verdict=EvaluationVerdict.CORRECT,
        alignment=PredictionOutcomeAlignment.STRONG_MATCH,
        decision_match=True,
        created_at=NOW,
    )


def one_forecast(accuracy: float) -> tuple[list[ForecastRecord], list[PredictionEvaluation]]:
    return (
        [record(forecast_id="f1", snapshot_id="s1", confidence=0.8)],
        [success_evaluation("s1", "e1")],
    )


def snapshot(
    anchor: date, accuracy: float, snapshot_id: str, *, scope: str = "repository"
) -> MonitorSnapshot:
    records, evaluations = one_forecast(accuracy)
    built = build_monitor_snapshot(
        records,
        evaluations,
        anchor_date=anchor,
        period_kind=MonitorPeriodKind.DAILY,
        scope=scope,
    )
    updated = built.model_copy(
        update={"metrics": {"metrics.accuracy": accuracy}, "snapshot_id": snapshot_id}
    )
    updated.content_hash = updated.content_fingerprint()
    return updated


class TestMonitorHistory:
    def test_round_trip_preserves_content(self, tmp_path) -> None:
        history = MonitorHistory(tmp_path)
        original = snapshot(date(2026, 9, 21), 0.5, "id-1")
        history.record_snapshot(original)
        loaded = history.load_snapshot(history.snapshot_files("repository", "daily")[0])
        assert loaded.snapshot_id == original.snapshot_id
        assert loaded.metrics["metrics.accuracy"] == 0.5
        assert loaded.anchor_date == original.anchor_date
        assert loaded.engine_version == original.engine_version

    def test_deterministic_filename(self, tmp_path) -> None:
        history = MonitorHistory(tmp_path)
        path = history.snapshot_file_path("repository", MonitorPeriodKind.DAILY, date(2026, 9, 21))
        assert path.name == history.snapshot_file_path(
            "repository", MonitorPeriodKind.DAILY, date(2026, 9, 21)
        ).name

    def test_rotation_keeps_newest(self, tmp_path) -> None:
        history = MonitorHistory(tmp_path)
        for day, accuracy in ((21, 0.4), (22, 0.5), (23, 0.9)):
            history.record_snapshot(
                snapshot(date(2026, 9, day), accuracy, f"id-{day}"), keep=2
            )
        remaining = history.snapshots("repository", "daily")
        assert len(remaining) == 2
        assert [entry.anchor_date.day for entry in remaining] == [22, 23]

    def test_rotation_rejects_small_keep(self, tmp_path) -> None:
        history = MonitorHistory(tmp_path)
        with pytest.raises(ValueError):
            history.record_snapshot(snapshot(date(2026, 9, 21), 0.5, "id-1"), keep=0)

    def test_latest_snapshot_none_for_empty_scope(self, tmp_path) -> None:
        history = MonitorHistory(tmp_path)
        assert history.latest_snapshot("repository", "daily") is None

    def test_integrity_failure_raises(self, tmp_path) -> None:
        history = MonitorHistory(tmp_path)
        original = snapshot(date(2026, 9, 21), 0.5, "id-1")
        history.record_snapshot(original)
        path = history.snapshot_files("repository", "daily")[0]
        document = json.loads(path.read_text(encoding="utf-8"))
        document["snapshot"]["metrics"]["metrics.accuracy"] = 0.99
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError):
            history.load_snapshot(path)

    def test_summarize_cheap_index_ordering(self, tmp_path) -> None:
        history = MonitorHistory(tmp_path)
        history.record_snapshot(snapshot(date(2026, 9, 22), 0.6, "id-22"))
        history.record_snapshot(snapshot(date(2026, 9, 21), 0.5, "id-21"))
        summaries = history.summarize("repository", "daily")
        assert [entry.anchor_date.day for entry in summaries] == [21, 22]
        assert len(summaries) == 2


class TestMonitorTrendsAndDrift:
    def test_metrics_series_and_trend_up(self, tmp_path) -> None:
        history = MonitorHistory(tmp_path)
        history.record_snapshot(snapshot(date(2026, 9, 21), 0.5, "id-21"))
        history.record_snapshot(snapshot(date(2026, 9, 22), 0.9, "id-22"))
        loaded = history.snapshots("repository", "daily")
        series = metric_series(loaded, "metrics.accuracy")
        assert [point.value for point in series] == [0.5, 0.9]
        trend = compute_trend(loaded, "metrics.accuracy")
        assert trend is not None
        assert trend.direction.value == "up"

    def test_confidence_mean_in_series(self, tmp_path) -> None:
        history = MonitorHistory(tmp_path)
        records, evaluations = one_forecast(0.5)
        history.record_snapshot(
            build_monitor_snapshot(
                records,
                evaluations,
                anchor_date=date(2026, 9, 21),
                period_kind=MonitorPeriodKind.DAILY,
            )
        )
        series = metric_series(history.snapshots("repository", "daily"), "confidence.mean")
        assert abs(series[0].value - 0.8) <= EPSILON
        assert abs(mean_confidence(records) - 0.8) <= EPSILON

    def test_drift_between_stored_snapshots(self, tmp_path) -> None:
        history = MonitorHistory(tmp_path)
        history.record_snapshot(snapshot(date(2026, 9, 21), 0.5, "id-21"))
        history.record_snapshot(snapshot(date(2026, 9, 22), 0.9, "id-22"))
        loaded = history.snapshots("repository", "daily")
        report = detect_monitor_drift(loaded[0], loaded[1])
        assert report.baseline_id == "id-21"
        assert report.comparison_id == "id-22"
        assert len(report.signals) >= 3

    def test_scopes_are_isolated(self, tmp_path) -> None:
        history = MonitorHistory(tmp_path)
        alice = snapshot(date(2026, 9, 21), 0.5, "id-a", scope="user:alice")
        bob = snapshot(date(2026, 9, 21), 0.9, "id-b", scope="user:bob")
        history.record_snapshot(alice)
        history.record_snapshot(bob)
        assert history.latest_snapshot("user:alice", "daily").snapshot_id == "id-a"
        assert history.latest_snapshot("user:bob", "daily").snapshot_id == "id-b"
