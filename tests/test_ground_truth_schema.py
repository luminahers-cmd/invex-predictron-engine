"""Tests for the Phase V1 ground-truth dataset schema.

These tests lock the *shape* of the design contract only.  They create
fixture records with made-up data to exercise validation and the pure
binary-outcome derive function; they do not assert any production engine
behaviour or modify any benchmark snapshot.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from benchmarks.ground_truth.schema import (
    EngineSnapshot,
    FundingStage,
    GroundTruthRecord,
    OutcomeEvent,
    OutcomeEventKind,
    StartupStatus,
    derive_binary_outcome,
)


def _record(
    *,
    status: StartupStatus,
    exit_value: float | None = None,
    events: list[OutcomeEvent] | None = None,
) -> GroundTruthRecord:
    return GroundTruthRecord(
        startup_id="test-1",
        startup_name="Test Startup",
        analysis_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        funding_stage_at_analysis=FundingStage.SEED,
        engine_snapshot=EngineSnapshot(
            investment_decision="invest",
            overall_confidence=0.8,
            engine_version="0.0.0-test",
        ),
        actual_status=status,
        outcome_observation_date=date(2026, 1, 1),
        evaluation_horizon_days=730,
        outcome_events=events or [],
        exit_value_usd=exit_value,
    )


def test_record_round_trip_validation() -> None:
    rec = _record(status=StartupStatus.OPERATING)
    dumped = rec.model_dump()
    assert dumped["schema_version"] == 1
    assert dumped["startup_id"] == "test-1"
    # Unknown fields are tolerated by pydantic? By default extra is ignored
    # in v2; the loader preserves them, but model_dump drops them.
    assert "unused" not in dumped


def test_derive_binary_outcome_acquired_positive() -> None:
    rec = _record(status=StartupStatus.ACQUIRED, exit_value=50_000_000)
    assert derive_binary_outcome(rec) == 1


def test_derive_binary_outcome_acquired_zero_value() -> None:
    rec = _record(status=StartupStatus.ACQUIRED, exit_value=0)
    assert derive_binary_outcome(rec) == 0


def test_derive_binary_outcome_shutdown() -> None:
    assert derive_binary_outcome(_record(status=StartupStatus.SHUTDOWN)) == 0


def test_derive_binary_outcome_operating_with_arr_milestone() -> None:
    rec = _record(
        status=StartupStatus.OPERATING,
        events=[
            OutcomeEvent(
                kind=OutcomeEventKind.ARR_MILESTONE,
                occurred_at=date(2025, 6, 1),
                amount_nominal_units=2_000_000,
            )
        ],
    )
    assert derive_binary_outcome(rec) == 1


def test_derive_binary_outcome_operating_without_evidence_is_undefined() -> None:
    # Survival alone is not a success label; the metric must treat this
    # as un-scoreable, not fabricate a 0/1.
    rec = _record(status=StartupStatus.OPERATING)
    assert derive_binary_outcome(rec) is None


def test_derive_binary_outcome_unknown_is_undefined() -> None:
    assert (
        derive_binary_outcome(_record(status=StartupStatus.UNKNOWN)) is None
    )
