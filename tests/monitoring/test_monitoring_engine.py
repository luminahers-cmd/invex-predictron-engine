"""Unit tests for the Phase 6 monitoring engine builders.

All analytics are deterministic: identical inputs must produce identical
outputs.  These tests exercise the pure builders (no database) and assert
the documented mathematical formulas — including the reuse of canonical
metrics (``compute_evaluation_metrics``, Sprint 8 calibration, Sprint 9
confidence drift) rather than re-implementations.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

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
from predictron_engine.intelligence.history import snapshot_from_confidence
from predictron_engine.monitoring.drift import (
    confidence_drift_between,
    detect_monitor_drift,
)
from predictron_engine.monitoring.health import (
    build_health_entries,
    derive_forecast_health,
    health_distribution,
)
from predictron_engine.monitoring.models import (
    ForecastHealth,
    ForecastRecord,
    MonitorPeriodKind,
    MonitorSnapshot,
)
from predictron_engine.monitoring.reanalysis import build_reanalysis_recommendations
from predictron_engine.monitoring.snapshot import (
    build_monitor_rolling_snapshot,
    build_monitor_snapshot,
)
from predictron_engine.monitoring.summaries import (
    calibration_summary,
    mean_confidence,
)
from predictron_engine.monitoring.trends import compute_trend, metric_series

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
ANCHOR = date(2026, 9, 21)
ANCHOR2 = date(2026, 9, 22)
EPSILON = 1e-6


def record(
    *,
    forecast_id: str,
    company_id: str = "c1",
    snapshot_id: str = "s1",
    horizon_days: int = 90,
    confidence: float = 0.7,
    decision: str = "invest",
    status: str = "active",
    outcome_id: str | None = None,
    outcome_verdict: str | None = None,
    evaluation_verdict: str | None = None,
    evaluation_created_at: datetime | None = None,
    latest_outcome_at: datetime | None = None,
    latest_snapshot_at: datetime | None = None,
    analysis_delta: int = 40,
    due_delta: int = 50,
) -> ForecastRecord:
    return ForecastRecord(
        id=forecast_id,
        company_id=company_id,
        snapshot_id=snapshot_id,
        decision=decision,
        confidence=confidence,
        composite_score=70.0,
        status=status,
        analysis_timestamp=NOW - timedelta(days=analysis_delta),
        due_at=NOW + timedelta(days=due_delta),
        horizon_days=horizon_days,
        outcome_id=outcome_id,
        outcome_verdict=outcome_verdict,
        evaluation_verdict=evaluation_verdict,
        evaluation_created_at=evaluation_created_at,
        latest_outcome_at=latest_outcome_at,
        latest_snapshot_at=latest_snapshot_at,
    )


def evaluation(
    *,
    evaluation_id: str,
    record: str,
    confidence: float,
    verdict: EvaluationVerdict,
    outcome_verdict: OutcomeVerdict,
    outcome: StartupOutcome,
    decision: DecisionLabel = DecisionLabel.INVEST,
) -> PredictionEvaluation:
    return PredictionEvaluation(
        evaluation_id=evaluation_id,
        record_id=record,
        prediction=PredictionSummary(
            decision=decision,
            confidence=confidence,
            composite_score=70.0,
            dimension_scores={"Market": 60.0, "Founder": 70.0},
        ),
        outcome_record=OutcomeRecord(
            record_id=record, verdict=outcome_verdict, outcome=outcome
        ),
        verdict=verdict,
        alignment=(
            PredictionOutcomeAlignment.STRONG_MATCH
            if verdict == EvaluationVerdict.CORRECT
            else PredictionOutcomeAlignment.MISMATCH
        ),
        decision_match=verdict == EvaluationVerdict.CORRECT,
        created_at=NOW,
    )


def success_outcome() -> StartupOutcome:
    return StartupOutcome(status=OutcomeStatus.FULLY_VERIFIED, exit_type="ipo")


def failure_outcome() -> StartupOutcome:
    return StartupOutcome(status=OutcomeStatus.FULLY_VERIFIED, shutdown=True)


# ---------------------------------------------------------------------------
# Forecast health
# ---------------------------------------------------------------------------


class TestForecastHealth:
    def test_resolved_is_highest_priority(self) -> None:
        health = derive_forecast_health(
            horizon_days=90,
            due_at=NOW + timedelta(days=50),
            analysis_timestamp=NOW - timedelta(days=40),
            resolved=True,
            as_of=NOW + timedelta(days=400),
            stale_after_days=365,
        )
        assert health is ForecastHealth.RESOLVED

    def test_overdue_when_due_passed(self) -> None:
        health = derive_forecast_health(
            horizon_days=90,
            due_at=NOW - timedelta(days=1),
            analysis_timestamp=NOW - timedelta(days=40),
            resolved=False,
            as_of=NOW,
        )
        assert health is ForecastHealth.OVERDUE

    def test_stale_after_configured_days(self) -> None:
        health = derive_forecast_health(
            horizon_days=365,
            due_at=NOW + timedelta(days=300),
            analysis_timestamp=NOW - timedelta(days=400),
            resolved=False,
            as_of=NOW,
            stale_after_days=365,
        )
        assert health is ForecastHealth.STALE

    def test_active_within_horizon(self) -> None:
        health = derive_forecast_health(
            horizon_days=90,
            due_at=NOW + timedelta(days=50),
            analysis_timestamp=NOW - timedelta(days=40),
            resolved=False,
            as_of=NOW,
        )
        assert health is ForecastHealth.ACTIVE

    def test_due_at_horizon(self) -> None:
        health = derive_forecast_health(
            horizon_days=90,
            due_at=NOW,
            analysis_timestamp=NOW - timedelta(days=90),
            resolved=False,
            as_of=NOW,
        )
        assert health is ForecastHealth.OVERDUE

    def test_build_health_entries_ordering(self) -> None:
        records = [
            record(forecast_id="f-overdue", due_delta=-5),
            record(forecast_id="f-active", due_delta=50),
            record(forecast_id="f-resolved", outcome_id="o1", due_delta=-10),
        ]
        entries = build_health_entries(records, as_of=NOW)
        due_dates = [entry.due_at for entry in entries]
        assert due_dates == sorted(due_dates)
        healths = [entry.health.value for entry in entries]
        assert healths == ["resolved", "overdue", "active"]
        distribution = health_distribution(entries)
        assert distribution == {"active": 1, "overdue": 1, "resolved": 1}

    def test_days_overdue_metrics(self) -> None:
        entries = build_health_entries(
            [record(forecast_id="f-overdue", due_delta=-5)], as_of=NOW
        )
        entry = entries[0]
        assert abs(entry.days_overdue - 5.0) <= EPSILON
        assert abs(entry.days_until_due - (-5.0)) <= EPSILON


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


class TestSummaries:
    def test_mean_confidence_exact(self) -> None:
        records = [
            record(forecast_id="f1", confidence=0.25),
            record(forecast_id="f2", confidence=0.75),
        ]
        assert mean_confidence(records) == 0.5

    def test_mean_confidence_none_when_empty(self) -> None:
        assert mean_confidence([]) is None

    def test_calibration_summary_deterministic(self) -> None:
        evaluations = [
            evaluation(
                evaluation_id="e1",
                record="s1",
                confidence=0.8,
                verdict=EvaluationVerdict.CORRECT,
                outcome_verdict=OutcomeVerdict.SUCCESS,
                outcome=success_outcome(),
            ),
            evaluation(
                evaluation_id="e2",
                record="s2",
                confidence=0.7,
                verdict=EvaluationVerdict.INCORRECT,
                outcome_verdict=OutcomeVerdict.FAILURE,
                outcome=failure_outcome(),
            ),
        ]
        report = calibration_summary(evaluations)
        assert report["total_samples"] == 2
        assert report["expected_calibration_error"] > 0.0
        assert report["maximum_calibration_error"] > 0.0
        assert report["overconfidence_detected"] is True

    def test_calibration_skips_unknown_outcomes(self) -> None:
        evaluations = [
            evaluation(
                evaluation_id="e1",
                record="s1",
                confidence=0.8,
                verdict=EvaluationVerdict.CORRECT,
                outcome_verdict=OutcomeVerdict.SUCCESS,
                outcome=success_outcome(),
            ),
            evaluation(
                evaluation_id="e2",
                record="s2",
                confidence=0.7,
                verdict=EvaluationVerdict.INCORRECT,
                outcome_verdict=OutcomeVerdict.UNKNOWN,
                outcome=StartupOutcome(status=OutcomeStatus.UNKNOWN),
            ),
        ]
        report = calibration_summary(evaluations)
        assert report["total_samples"] == 1


# ---------------------------------------------------------------------------
# Snapshot composition
# ---------------------------------------------------------------------------


class TestSnapshot:
    def _population(self) -> tuple[list[ForecastRecord], list[PredictionEvaluation]]:
        records = [
            record(forecast_id="f1", confidence=0.8, decision="invest"),
            record(forecast_id="f2", company_id="c2", snapshot_id="s2",
                   confidence=0.6, decision="watch"),
            record(
                forecast_id="f3",
                company_id="c3",
                snapshot_id="s3",
                status="resolved",
                outcome_id="o1",
                outcome_verdict="success",
                evaluation_verdict="correct",
            ),
        ]
        evaluations = [
            evaluation(
                evaluation_id="e1",
                record="s1",
                confidence=0.8,
                verdict=EvaluationVerdict.CORRECT,
                outcome_verdict=OutcomeVerdict.SUCCESS,
                outcome=success_outcome(),
            ),
            evaluation(
                evaluation_id="e2",
                record="s3",
                confidence=0.7,
                verdict=EvaluationVerdict.INCORRECT,
                outcome_verdict=OutcomeVerdict.FAILURE,
                outcome=failure_outcome(),
            ),
        ]
        return records, evaluations

    def test_snapshot_builds_deterministic_id_and_verifies(self) -> None:
        records, evaluations = self._population()
        first = build_monitor_snapshot(
            records, evaluations, anchor_date=ANCHOR, period_kind=MonitorPeriodKind.DAILY
        )
        second = build_monitor_snapshot(
            records, evaluations, anchor_date=ANCHOR, period_kind=MonitorPeriodKind.DAILY
        )
        assert first.snapshot_id == second.snapshot_id
        assert first.verify() is True

    def test_snapshot_counts_and_distributions(self) -> None:
        records, evaluations = self._population()
        snapshot = build_monitor_snapshot(
            records, evaluations, anchor_date=ANCHOR, period_kind=MonitorPeriodKind.DAILY
        )
        assert snapshot.counts["forecasts"] == 3
        assert snapshot.counts["forecasts_outcome_linked"] == 1
        assert snapshot.counts["companies"] == 3
        assert snapshot.counts["evaluations"] == 2
        assert set(snapshot.metrics) >= {"metrics.accuracy", "confidence.mean"}
        assert snapshot.metrics["metrics.accuracy"] == 0.5
        assert abs(float(snapshot.metrics["confidence.mean"]) - 0.7) <= EPSILON
        verdicts = snapshot.distributions["evaluation_verdict"]
        assert set(verdicts.universe) == {
            "correct",
            "incorrect",
            "inconclusive",
            "partially_correct",
            "unable_to_evaluate",
        }
        assert verdicts.counts["correct"] == 1
        assert verdicts.counts["incorrect"] == 1
        health = snapshot.health
        assert health.get("resolved", 0) == 1
        assert set(snapshot.horizon_breakdown) == {"90"}
        assert snapshot.horizon_breakdown["90"].forecasts == 3
        assert snapshot.sector_breakdown["overall"].evaluations == 2
        assert snapshot.distributions["decision"].total == 3

    def test_snapshot_horizon_breakdown_attribution(self) -> None:
        records = [
            record(forecast_id="f1", horizon_days=90, confidence=0.8),
            record(forecast_id="f2", horizon_days=180, confidence=0.6),
        ]
        evaluations = [
            evaluation(
                evaluation_id="e1",
                record="s1",
                confidence=0.8,
                verdict=EvaluationVerdict.CORRECT,
                outcome_verdict=OutcomeVerdict.SUCCESS,
                outcome=success_outcome(),
            )
        ]
        snapshot = build_monitor_snapshot(
            records, evaluations, anchor_date=ANCHOR, period_kind=MonitorPeriodKind.DAILY
        )
        by_horizon = snapshot.horizon_breakdown
        assert set(by_horizon) == {"90", "180"}
        assert by_horizon["90"].forecasts == 1
        assert by_horizon["90"].evaluated == 1

    def test_rolling_window_changes_aggregates(self) -> None:
        records, evaluations = self._population()
        full_metrics = build_monitor_snapshot(
            records, evaluations, anchor_date=ANCHOR, period_kind=MonitorPeriodKind.DAILY
        ).metrics
        rolled = build_monitor_rolling_snapshot(
            records, evaluations, anchor_date=ANCHOR, period_kind=MonitorPeriodKind.DAILY, window=1
        ).metrics
        assert rolled != full_metrics


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


class TestDrift:
    def _paired_snapshots(self) -> tuple[MonitorSnapshot, MonitorSnapshot]:
        def population(conf_a: float, conf_b: float, correct_b: bool) -> MonitorSnapshot:
            records = [
                record(forecast_id="f1", snapshot_id="s1", confidence=conf_a, outcome_id="o1"),
                record(forecast_id="f2", snapshot_id="s2", confidence=conf_b, outcome_id="o2"),
            ]
            evaluations = [
                evaluation(
                    evaluation_id="e1",
                    record="s1",
                    confidence=conf_a,
                    verdict=EvaluationVerdict.CORRECT,
                    outcome_verdict=OutcomeVerdict.SUCCESS,
                    outcome=success_outcome(),
                ),
                evaluation(
                    evaluation_id="e2",
                    record="s2",
                    confidence=conf_b,
                    verdict=(
                        EvaluationVerdict.CORRECT
                        if correct_b
                        else EvaluationVerdict.INCORRECT
                    ),
                    outcome_verdict=OutcomeVerdict.FAILURE,
                    outcome=failure_outcome(),
                ),
            ]
            return build_monitor_snapshot(
                records,
                evaluations,
                anchor_date=ANCHOR,
                period_kind=MonitorPeriodKind.DAILY,
            )

        before = population(0.8, 0.7, correct_b=False)
        after = population(0.9, 0.6, correct_b=True)
        return before, after

    def test_drift_accuracy_direction_and_severity(self) -> None:
        before, after = self._paired_snapshots()
        report = detect_monitor_drift(before, after)
        assert report.baseline_id == before.snapshot_id
        assert report.comparison_id == after.snapshot_id
        names = {signal.signal for signal in report.signals}
        assert "calibration" in names
        assert "confidence" in names
        assert "horizon_performance" in names
        horizon = report.signal("horizon_performance")
        assert horizon is not None
        assert horizon.delta > 0.0
        assert horizon.severity in ("minor", "moderate", "major", "low")

    def test_drift_epsilon_affects_flag(self) -> None:
        before, after = self._paired_snapshots()
        report = detect_monitor_drift(before, after)
        for signal in report.signals:
            assert signal.affected == (signal.magnitude >= 1.0)

    def test_cross_scope_raises(self) -> None:
        before, _ = self._paired_snapshots()
        other = before.model_copy(update={"scope": "user:other", "snapshot_id": "x"})
        try:
            detect_monitor_drift(before, other)
        except ValueError:
            return
        raise AssertionError("expected ValueError for cross-scope drift")

    def test_confidence_drift_reuses_canonical(self) -> None:
        a = [
            snapshot_from_confidence(70.0, 0.8, [], [], decision_confidence=0.8),
            snapshot_from_confidence(60.0, 0.6, [], [], decision_confidence=0.6),
        ]
        b = [
            snapshot_from_confidence(75.0, 0.9, [], [], decision_confidence=0.9),
            snapshot_from_confidence(65.0, 0.7, [], [], decision_confidence=0.7),
        ]
        drift = confidence_drift_between(a, b)
        expected = round((0.9 + 0.7) / 2 - (0.8 + 0.6) / 2, 4)
        assert abs(drift.decision_drift.delta - expected) <= EPSILON


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------


class TestTrends:
    def _series_snapshots(self, *, same: bool = False) -> list[MonitorSnapshot]:
        records, evaluations = self._paired_population()
        out: list[MonitorSnapshot] = []
        for anchor, conf in ((ANCHOR, 0.5), (ANCHOR2, (0.5 if same else 0.9))):
            out.append(
                build_monitor_snapshot(
                    records,
                    evaluations,
                    anchor_date=anchor,
                    period_kind=MonitorPeriodKind.DAILY,
                ).model_copy(
                    update={
                        "metrics": {"metrics.accuracy": conf},
                        "snapshot_id": f"id-{anchor}",
                    }
                )
            )
        return out

    def _paired_population(self) -> tuple[list[ForecastRecord], list[PredictionEvaluation]]:
        records = [record(forecast_id="f1", confidence=0.7)]
        evaluations = [
            evaluation(
                evaluation_id="e1",
                record="s1",
                confidence=0.7,
                verdict=EvaluationVerdict.CORRECT,
                outcome_verdict=OutcomeVerdict.SUCCESS,
                outcome=success_outcome(),
            )
        ]
        return records, evaluations

    def test_metric_series_ordered_and_skips_none(self) -> None:
        snapshots = self._series_snapshots()
        series = metric_series(snapshots, "metrics.accuracy")
        assert [point.anchor for point in series] == [ANCHOR, ANCHOR2]
        assert [point.value for point in series] == [0.5, 0.9]

    def test_compute_trend_direction(self) -> None:
        snapshots = self._series_snapshots()
        trend = compute_trend(snapshots, "metrics.accuracy")
        assert trend is not None
        assert trend.direction.value == "up"
        assert trend.from_value == 0.5
        assert trend.to_value == 0.9
        assert len(trend.series) == 2

    def test_compute_trend_flat_when_equal(self) -> None:
        snapshots = self._series_snapshots(same=True)
        trend = compute_trend(snapshots, "metrics.accuracy")
        assert trend is not None
        assert trend.direction.value == "flat"
        assert trend.from_value == 0.5
        assert trend.to_value == 0.5

    def test_single_point_is_not_a_trend(self) -> None:
        snapshot = self._series_snapshots()[0]
        assert compute_trend([snapshot], "metrics.accuracy") is None


# ---------------------------------------------------------------------------
# Re-analysis
# ---------------------------------------------------------------------------


class TestReanalysis:
    def test_reasons_and_priority(self) -> None:
        forecasts = [
            record(
                forecast_id="f-expired",
                due_delta=-5,
                status="due",
            ),
            record(
                forecast_id="f-new-outcome",
                outcome_id="o2",
                latest_outcome_at=NOW + timedelta(days=1),
            ),
            record(
                forecast_id="f-stale-evidence",
                outcome_id="o3",
                evaluation_verdict="correct",
                evaluation_created_at=NOW - timedelta(days=10),
                latest_outcome_at=NOW - timedelta(days=1),
            ),
            record(
                forecast_id="f-old",
                analysis_delta=500,
                status="active",
            ),
        ]
        recommendations = build_reanalysis_recommendations(forecasts, as_of=NOW)
        by_id = {item.forecast_id: item.reasons for item in recommendations}
        assert "prediction_expired" in by_id["f-expired"]
        assert "new_outcome_recorded" in by_id["f-new-outcome"]
        assert "evidence_changed" in by_id["f-stale-evidence"]
        assert "snapshot_age_exceeded" in by_id["f-old"]

    def test_concluded_forecast_never_recommended(self) -> None:
        forecasts = [
            record(
                forecast_id="f-concluded",
                outcome_id="o1",
                outcome_verdict="success",
                evaluation_verdict="correct",
                evaluation_created_at=NOW,
            )
        ]
        assert (
            build_reanalysis_recommendations(forecasts, as_of=NOW) == []
        )

    def test_reasons_sorted_deterministically(self) -> None:
        forecasts = [
            record(
                forecast_id="f-multi",
                due_delta=-5,
                status="late",
                analysis_delta=500,
            )
        ]
        recommendations = build_reanalysis_recommendations(forecasts, as_of=NOW)
        assert recommendations[0].reasons == [
            "prediction_expired",
            "snapshot_age_exceeded",
        ]
