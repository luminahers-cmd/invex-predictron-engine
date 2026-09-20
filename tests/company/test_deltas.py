"""Tests for the Phase 3 deterministic delta and trend engines.

Exercises the pure comparison and trend-classification functions in
:mod:`app.services.company_deltas` and :mod:`app.services.company_history`:
delta calculations, identical snapshots, missing values, collection
deltas, sequence building, and trend classification.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.schemas.company_delta import DeltaKind, DeltaStatus
from app.schemas.company_history import TrendDirection
from app.schemas.company_profile import CompanyBenchmarkSummary
from app.services.company_deltas import build_delta_sequence, compare_snapshots
from app.services.company_history import (
    SnapshotView,
    classify_direction,
    trend_summary,
)
from app.services.company_protocols import CompanySnapshotRecord

REF = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def _snapshot(
    *,
    snap_id: str,
    created_at: datetime = REF,
    decision: str | None = "invest",
    confidence: float | None = 0.8,
    readiness: float | None = 60.0,
    composite: float | None = 70.0,
    dimensions: dict[str, float] | None = None,
) -> CompanySnapshotRecord:
    return CompanySnapshotRecord(
        id=snap_id,
        company_id="comp-1",
        analysis_id=f"analysis-{snap_id}",
        report_id=f"report-{snap_id}",
        decision=decision,
        confidence=confidence,
        composite_score=composite,
        readiness_score=readiness,
        dimension_scores=dimensions or {},
        created_at=created_at,
    )


def _view(
    snapshot: CompanySnapshotRecord,
    *,
    benchmark: CompanyBenchmarkSummary | None = None,
    decision_explanation: tuple[str, ...] = (),
    key_facts: tuple[str, ...] = (),
    evidence_count: int | None = None,
) -> SnapshotView:
    return SnapshotView(
        snapshot=snapshot,
        benchmark=benchmark,
        decision_explanation=decision_explanation,
        key_facts=key_facts,
        evidence_count=evidence_count,
    )


def _benchmark(percentile: float, z_score: float = 0.0) -> CompanyBenchmarkSummary:
    return CompanyBenchmarkSummary(
        composite_score=70.0,
        benchmark_mean=50.0,
        benchmark_std_dev=20.0,
        percentile_rank=percentile,
        z_score=z_score,
        sample_size=10,
    )


def _field(delta, name):
    return next(f for f in delta.fields if f.field == name)


# ── delta calculations ────────────────────────────────────────────────


class TestDeltaCalculations:
    def test_numeric_confidence_delta(self):
        earlier = _view(_snapshot(snap_id="a", confidence=0.71))
        later = _view(_snapshot(snap_id="b", confidence=0.82))

        delta = compare_snapshots(earlier, later)
        field = _field(delta, "confidence")

        assert field.delta_type == DeltaKind.NUMERIC
        assert field.status == DeltaStatus.CHANGED
        assert field.previous == 0.71
        assert field.current == 0.82
        assert field.change == pytest.approx(0.11)

    def test_numeric_change_is_signed(self):
        earlier = _view(_snapshot(snap_id="a", confidence=0.82))
        later = _view(_snapshot(snap_id="b", confidence=0.71))

        field = _field(compare_snapshots(earlier, later), "confidence")
        assert field.change == pytest.approx(-0.11)
        assert field.status == DeltaStatus.CHANGED

    def test_readiness_composite_evidences_compared(self):
        earlier = _view(
            _snapshot(snap_id="a", readiness=50.0, composite=60.0),
            evidence_count=3,
        )
        later = _view(
            _snapshot(snap_id="b", readiness=55.0, composite=72.0),
            evidence_count=5,
        )

        delta = compare_snapshots(earlier, later)
        assert _field(delta, "readiness").change == pytest.approx(5.0)
        assert _field(delta, "composite_score").change == pytest.approx(12.0)
        assert _field(delta, "evidence_count").change == pytest.approx(2.0)
        assert _field(delta, "evidence_count").status == DeltaStatus.CHANGED

    def test_decision_nominal_change(self):
        earlier = _view(_snapshot(snap_id="a", decision="invest"))
        later = _view(_snapshot(snap_id="b", decision="watch"))

        field = _field(compare_snapshots(earlier, later), "decision")
        assert field.delta_type == DeltaKind.NOMINAL
        assert field.status == DeltaStatus.CHANGED
        assert field.change == "different"

    def test_benchmark_deltas(self):
        earlier = _view(_snapshot(snap_id="a"), benchmark=_benchmark(40.0))
        later = _view(_snapshot(snap_id="b"), benchmark=_benchmark(65.0))

        delta = compare_snapshots(earlier, later)
        percentile = _field(delta, "benchmark_percentile")
        assert percentile.status == DeltaStatus.CHANGED
        assert percentile.change == pytest.approx(25.0)

    def test_reports_change_counts(self):
        earlier = _view(_snapshot(snap_id="a", confidence=0.71, decision="invest"))
        later = _view(_snapshot(snap_id="b", confidence=0.82, decision="watch"))

        delta = compare_snapshots(earlier, later)
        assert delta.changed_fields >= 2
        assert delta.unchanged_fields >= 0
        assert delta.previous_snapshot_id == "a"
        assert delta.current_snapshot_id == "b"


class TestIdenticalSnapshots:
    def test_all_fields_unchanged(self):
        earlier = _view(
            _snapshot(snap_id="a"),
            benchmark=_benchmark(50.0),
            decision_explanation=("reason A",),
            key_facts=("confidence=0.8",),
            evidence_count=4,
        )
        later = _view(
            _snapshot(snap_id="b"),
            benchmark=_benchmark(50.0),
            decision_explanation=("reason A",),
            key_facts=("confidence=0.8",),
            evidence_count=4,
        )

        delta = compare_snapshots(earlier, later)
        assert delta.changed_fields == 0
        assert delta.unchanged_fields == len(delta.fields)
        for field in delta.fields:
            assert field.status == DeltaStatus.UNCHANGED
            if field.delta_type == DeltaKind.NOMINAL:
                assert field.change == "same"

    def test_benchmark_equal_but_missing_confidence(self):
        earlier = _view(_snapshot(snap_id="a", confidence=None), benchmark=None)
        later = _view(_snapshot(snap_id="b", confidence=None), benchmark=None)

        delta = compare_snapshots(earlier, later)
        assert _field(delta, "confidence").status == DeltaStatus.MISSING
        assert _field(delta, "benchmark_percentile").status == DeltaStatus.MISSING


class TestMissingValues:
    def test_field_added(self):
        earlier = _view(_snapshot(snap_id="a", confidence=None))
        later = _view(_snapshot(snap_id="b", confidence=0.82))

        field = _field(compare_snapshots(earlier, later), "confidence")
        assert field.status == DeltaStatus.ADDED
        assert field.previous is None
        assert field.current == 0.82
        assert field.change is None

    def test_field_removed(self):
        earlier = _view(_snapshot(snap_id="a", decision="invest"))
        later = _view(_snapshot(snap_id="b", decision=None))

        field = _field(compare_snapshots(earlier, later), "decision")
        assert field.status == DeltaStatus.REMOVED
        assert field.previous == "invest"
        assert field.current is None
        assert field.change is None

    def test_field_missing_both(self):
        earlier = _view(_snapshot(snap_id="a", readiness=None))
        later = _view(_snapshot(snap_id="b", readiness=None))

        field = _field(compare_snapshots(earlier, later), "readiness")
        assert field.status == DeltaStatus.MISSING
        assert field.previous is None
        assert field.current is None

    def test_missing_benchmark_but_present_later(self):
        earlier = _view(_snapshot(snap_id="a"), benchmark=None)
        later = _view(_snapshot(snap_id="b"), benchmark=_benchmark(80.0))

        field = _field(compare_snapshots(earlier, later), "benchmark_percentile")
        assert field.status == DeltaStatus.ADDED
        assert field.current == 80.0


class TestCollectionDeltas:
    def test_collection_unchanged_when_equal(self):
        earlier = _view(
            _snapshot(snap_id="a"),
            decision_explanation=("reason A", "reason B"),
        )
        later = _view(
            _snapshot(snap_id="b"),
            decision_explanation=("reason A", "reason B"),
        )

        field = _field(compare_snapshots(earlier, later), "decision_explanation")
        assert field.status == DeltaStatus.UNCHANGED
        assert field.change == "same"

    def test_collection_changed(self):
        earlier = _view(_snapshot(snap_id="a"), key_facts=("confidence=0.8",))
        later = _view(
            _snapshot(snap_id="b"),
            key_facts=("confidence=0.8", "decision=invest"),
        )

        field = _field(compare_snapshots(earlier, later), "key_facts")
        assert field.status == DeltaStatus.CHANGED
        assert field.change == "different"

    def test_collection_missing_when_report_absent(self):
        earlier = _view(_snapshot(snap_id="a"))
        later = _view(_snapshot(snap_id="b"))

        field = _field(compare_snapshots(earlier, later), "decision_explanation")
        assert field.status == DeltaStatus.MISSING


class TestDeltaSequence:
    def test_build_sequence_links_adjacent_pairs(self):
        views = [
            _view(_snapshot(snap_id="s1", created_at=REF)),
            _view(_snapshot(snap_id="s2", created_at=REF.replace(month=2))),
            _view(_snapshot(snap_id="s3", created_at=REF.replace(month=3))),
        ]

        deltas = build_delta_sequence(views)
        assert len(deltas) == 2
        assert deltas[0].previous_snapshot_id == "s1"
        assert deltas[0].current_snapshot_id == "s2"
        assert deltas[1].previous_snapshot_id == "s2"
        assert deltas[1].current_snapshot_id == "s3"

    def test_empty_and_single_series(self):
        assert build_delta_sequence([]) == []
        assert build_delta_sequence([_view(_snapshot(snap_id="s1"))]) == []


# ── trend engine ──────────────────────────────────────────────────────


class TestClassifyDirection:
    def test_increasing(self):
        assert classify_direction([0.5, 0.6, 0.7]) == TrendDirection.INCREASING

    def test_decreasing(self):
        assert classify_direction([0.7, 0.6, 0.5]) == TrendDirection.DECREASING

    def test_stable(self):
        assert classify_direction([0.5, 0.5, 0.5]) == TrendDirection.STABLE

    def test_mixed(self):
        assert classify_direction([0.5, 0.6, 0.5]) == TrendDirection.MIXED

    def test_insufficient_series(self):
        assert classify_direction([]) == TrendDirection.INSUFFICIENT
        assert classify_direction([0.5]) == TrendDirection.INSUFFICIENT
        assert classify_direction([None, None]) == TrendDirection.INSUFFICIENT

    def test_missing_values_skipped(self):
        assert classify_direction([None, 0.5, 0.6]) == TrendDirection.INCREASING

    def test_below_epsilon_is_stable(self):
        assert classify_direction([0.5, 0.5 + 1e-12]) == TrendDirection.STABLE

    def test_flat_then_up_is_increasing(self):
        assert classify_direction([0.5, 0.5, 0.6]) == TrendDirection.INCREASING


class TestTrendSummary:
    def test_trend_summary_composition(self):
        views = [
            _view(_snapshot(snap_id="s1", confidence=0.5, readiness=60.0, composite=50.0)),
            _view(_snapshot(snap_id="s2", confidence=0.6, readiness=55.0, composite=55.0)),
            _view(_snapshot(snap_id="s3", confidence=0.66, readiness=54.0, composite=54.0)),
        ]

        trend = trend_summary(views)
        assert trend.confidence == TrendDirection.INCREASING
        assert trend.readiness == TrendDirection.DECREASING
        assert trend.composite_score == TrendDirection.MIXED
        assert trend.snapshot_count == 3

    def test_trend_summary_benchmark_percentile(self):
        views = [
            _view(
                _snapshot(snap_id="s1", composite=50.0),
                benchmark=_benchmark(30.0),
            ),
            _view(
                _snapshot(snap_id="s2", composite=60.0),
                benchmark=_benchmark(45.0),
            ),
            _view(
                _snapshot(snap_id="s3", composite=70.0),
                benchmark=_benchmark(70.0),
            ),
        ]

        trend = trend_summary(views)
        assert trend.benchmark_percentile == TrendDirection.INCREASING

    def test_trend_summary_insufficient_benchmark_without_dataset(self):
        views = [
            _view(_snapshot(snap_id="s1", composite=50.0)),
            _view(_snapshot(snap_id="s2", composite=60.0)),
        ]

        trend = trend_summary(views)
        assert trend.benchmark_percentile == TrendDirection.INSUFFICIENT

    def test_trend_summary_empty(self):
        trend = trend_summary([])
        assert trend.snapshot_count == 0
        assert trend.confidence == TrendDirection.INSUFFICIENT
