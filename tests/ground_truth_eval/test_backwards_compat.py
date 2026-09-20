"""Backwards-compatibility tests for the legacy benchmark modules.

Project E5 replaces the old ``benchmark_*`` functions with the
``ground_truth_eval`` package.  These tests pin that the legacy modules
still import and behave as before, and that the engine top-level API is
unchanged.
"""

from __future__ import annotations

import pytest


class TestLegacyModuleImports:
    @pytest.mark.parametrize(
        "module",
        [
            "benchmarks.benchmark_metrics",
            "benchmarks.benchmark_report",
            "benchmarks.benchmark_runner",
            "benchmarks.benchmark_validator",
        ],
    )
    def test_modules_still_importable(self, module: str) -> None:
        __import__(module)


class TestSnapshotCompat:
    def test_snapshot_loads(self) -> None:
        from benchmarks.benchmark_runner import load_snapshot

        snap = load_snapshot("0.13.0")
        assert snap is not None
        assert snap["engine_version"] == "0.13.0"
        assert snap["benchmark_version"] == "2.0.0"
        assert snap["total_cases"] >= 13

    def test_snapshot_categories_present(self) -> None:
        from benchmarks.benchmark_runner import load_snapshot

        snap = load_snapshot("0.13.0")
        assert snap is not None
        categories = {
            rec["category"] for r in snap["results"] for rec in r.get("recommendations", [])
        }
        assert categories >= {"due_diligence", "opportunity", "follow_up", "risk_mitigation"}


class TestGroundTruthSchemaCompat:
    def test_derive_binary_outcome_unchanged(self) -> None:
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

        record = GroundTruthRecord(
            startup_id="test",
            startup_name="Test",
            analysis_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            funding_stage_at_analysis=FundingStage.SEED,
            engine_snapshot=EngineSnapshot(engine_version="test"),
            actual_status=StartupStatus.ACQUIRED,
            outcome_observation_date=date(2026, 6, 30),
            evaluation_horizon_days=0,
            exit_value_usd=100_000_000,
            outcome_events=[
                OutcomeEvent(
                    kind=OutcomeEventKind.ACQUISITION,
                    occurred_at=date(2026, 1, 1),
                    value_currency_usd=100_000_000,
                )
            ],
        )
        assert derive_binary_outcome(record) == 1


class TestEngineAPISurface:
    def test_engine_version_exported(self) -> None:
        from predictron_engine.engine import ENGINE_VERSION

        assert isinstance(ENGINE_VERSION, str)

    def test_threshold_constants_unchanged(self) -> None:
        from predictron_engine.decision.decision_engine import _DECISION_THRESHOLDS

        mapped = {cat.value: thresh for thresh, cat in _DECISION_THRESHOLDS}
        assert mapped["strong_invest"] == 75.0
        assert mapped["invest"] == 60.0
        assert mapped["watch"] == 45.0
        assert mapped["investigate_further"] == 30.0

    def test_decision_categories_unchanged(self) -> None:
        from predictron_engine.models.report import DecisionCategory

        expected = {
            "strong_invest",
            "invest",
            "watch",
            "investigate_further",
            "pass",
        }
        assert {c.value for c in DecisionCategory} >= expected


class TestHashStability:
    def test_result_hash_sha256_shape(self, example_run) -> None:
        assert len(example_run.result_hash) == 64
        int(example_run.result_hash, 16)
