"""Unit tests for the deterministic forecast lifecycle primitives."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from predictron_engine.dataset.forecast_lifecycle import (
    FORECAST_SCHEMA_VERSION,
    ForecastEventType,
    ForecastStatus,
    derive_forecast_status,
    forecast_due_at,
    forecast_earliest_due_detection,
    stable_event_id,
    stable_forecast_id,
)

ANALYSIS_AT = datetime(2025, 1, 1, tzinfo=UTC)


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


class TestStableForecastId:
    def test_deterministic(self) -> None:
        first = stable_forecast_id(
            company_id="c1", snapshot_id="s1", horizon_days=365
        )
        second = stable_forecast_id(
            company_id="c1", snapshot_id="s1", horizon_days=365
        )
        assert first == second
        assert len(first) == 64

    def test_differs_by_horizon(self) -> None:
        assert stable_forecast_id(
            company_id="c1", snapshot_id="s1", horizon_days=30
        ) != stable_forecast_id(
            company_id="c1", snapshot_id="s1", horizon_days=365
        )

    def test_differs_by_company_or_snapshot(self) -> None:
        assert stable_forecast_id(
            company_id="c1", snapshot_id="s1", horizon_days=365
        ) != stable_forecast_id(
            company_id="c2", snapshot_id="s1", horizon_days=365
        )
        assert stable_forecast_id(
            company_id="c1", snapshot_id="s1", horizon_days=365
        ) != stable_forecast_id(
            company_id="c1", snapshot_id="s2", horizon_days=365
        )


class TestStableEventId:
    def test_deterministic(self) -> None:
        first = stable_event_id(
            forecast_id="f1", event_type=ForecastEventType.REGISTERED
        )
        second = stable_event_id(
            forecast_id="f1", event_type=ForecastEventType.REGISTERED
        )
        assert first == second
        assert len(first) == 64

    def test_differs_by_type(self) -> None:
        assert stable_event_id(
            forecast_id="f1", event_type=ForecastEventType.REGISTERED
        ) != stable_event_id(forecast_id="f1", event_type=ForecastEventType.RESOLVED)

    def test_resolved_includes_outcome_for_idempotency(self) -> None:
        assert stable_event_id(
            forecast_id="f1",
            event_type=ForecastEventType.RESOLVED,
            outcome_id="o1",
        ) != stable_event_id(
            forecast_id="f1",
            event_type=ForecastEventType.RESOLVED,
            outcome_id="o2",
        )

    def test_accepts_plain_string(self) -> None:
        assert stable_event_id(
            forecast_id="f1", event_type="registered"
        ) == stable_event_id(
            forecast_id="f1", event_type=ForecastEventType.REGISTERED
        )


class TestForecastTiming:
    def test_due_at_horizon_after_analysis(self) -> None:
        due = forecast_due_at(
            analysis_timestamp=ANALYSIS_AT, horizon_days=365
        )
        assert due == datetime(2026, 1, 1, tzinfo=UTC)

    def test_earliest_due_detection_applies_tolerance(self) -> None:
        earliest = forecast_earliest_due_detection(
            analysis_timestamp=ANALYSIS_AT,
            horizon_days=365,
            tolerance_days=1,
        )
        assert earliest == datetime(2025, 12, 31, tzinfo=UTC)

    def test_due_at_normalizes_naive_analysis_timestamp(self) -> None:
        naive = datetime(2025, 1, 1)
        due = forecast_due_at(analysis_timestamp=naive, horizon_days=1)
        assert due.tzinfo is UTC


class TestDeriveForecastStatus:
    def test_active_before_horizon(self) -> None:
        status = derive_forecast_status(
            analysis_timestamp=ANALYSIS_AT,
            horizon_days=365,
            as_of=_at(2025, 6, 1),
        )
        assert status is ForecastStatus.ACTIVE

    def test_due_within_tolerance(self) -> None:
        status = derive_forecast_status(
            analysis_timestamp=ANALYSIS_AT,
            horizon_days=365,
            as_of=_at(2025, 12, 31),
        )
        assert status is ForecastStatus.DUE

    def test_due_after_horizon(self) -> None:
        status = derive_forecast_status(
            analysis_timestamp=ANALYSIS_AT,
            horizon_days=10,
            as_of=_at(2025, 2, 1),
        )
        assert status is ForecastStatus.DUE

    def test_resolved_wins_even_when_not_due(self) -> None:
        status = derive_forecast_status(
            analysis_timestamp=ANALYSIS_AT,
            horizon_days=365,
            as_of=_at(2025, 1, 2),
            resolved=True,
        )
        assert status is ForecastStatus.RESOLVED

    def test_horizon_elapsed_without_outcome_is_due_not_resolved(self) -> None:
        status = derive_forecast_status(
            analysis_timestamp=ANALYSIS_AT,
            horizon_days=365,
            as_of=d_at(2027, 1, 1),
            resolved=False,
        )
        assert status is ForecastStatus.DUE

    def test_naive_as_of_normalized_to_utc(self) -> None:
        status = derive_forecast_status(
            analysis_timestamp=ANALYSIS_AT,
            horizon_days=365,
            as_of=datetime(2025, 12, 31),
        )
        assert status is ForecastStatus.DUE


def d_at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC) + timedelta(hours=12)


def test_schema_version_constant() -> None:
    assert FORECAST_SCHEMA_VERSION == "1.0"
