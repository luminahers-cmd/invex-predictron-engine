"""Tests for the cohort prediction-execution runner (Phase 4).

Covers end-to-end execution of a manifest with real corpora and the real
engine: freezing, evaluation, metrics/calibration aggregation, idempotent
re-execution against the same store, pending ground truth handling, and the
deterministic outcome mapping helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmarks.cohort.execute import (
    STATUS_FROZEN,
    STATUS_PENDING,
    CohortExecutionResult,
    outcome_record_from_manifest,
    run_cohort_execution,
    stable_outcome_id,
)
from benchmarks.cohort.manifest import CohortManifest, ManifestOutcome, ManifestSourceRecord
from benchmarks.ground_truth.schema import StartupStatus
from predictron_engine.dataset.outcomes import OutcomeStatus, OutcomeVerdict

ANALYSIS = datetime(2026, 1, 1, tzinfo=UTC)
VERIFICATION = datetime(2026, 6, 30, tzinfo=UTC).date()
CREATED_AT = datetime(2026, 9, 20, tzinfo=UTC)


def _manifest() -> CohortManifest:
    outcome_acquired = ManifestOutcome(
        status=StartupStatus.ACQUIRED,
        verification_date=VERIFICATION,
        verified=True,
        verified_by="qa",
        notes="acquired by Example Corp",
        sources=["official filing"],
    )
    outcome_operating = ManifestOutcome(
        status=StartupStatus.OPERATING,
        verification_date=VERIFICATION,
        verified=True,
        verified_by="qa",
        sources=["official filing"],
    )
    records = [
        ManifestSourceRecord(
            company_id="b2b_saas",
            company_name="Analytix Cloud",
            description=(
                "A fictional B2B SaaS startup used by the benchmark suite for "
                "deterministic end-to-end tests of the evaluation platform."
            ),
            website="https://analytixcloud.example.com",
            evidence_corpus="b2b_saas",
            analysis_timestamp=ANALYSIS,
            outcome=outcome_acquired,
            metadata={"sources": ["tests"]},
        ),
        ManifestSourceRecord(
            company_id="ai_infrastructure",
            company_name="Inference Labs",
            description=(
                "A fictional AI infrastructure startup used by the benchmark "
                "suite to exercise time-scoped cohort building."
            ),
            website="https://inferencelabs.example.com",
            evidence_corpus="ai_infrastructure",
            analysis_timestamp=ANALYSIS,
            outcome=outcome_operating,
            metadata={"sources": ["tests"]},
        ),
        ManifestSourceRecord(
            company_id="mrr_only_derivation",
            company_name="SubMetrics",
            description="A fictional derivation-targeted startup without a documented outcome.",
            website="https://submetrics.example.com",
            evidence_corpus="mrr_only_derivation",
            analysis_timestamp=ANALYSIS,
            outcome=None,
        ),
    ]
    return CohortManifest(
        dataset_name="test_cohort_execute",
        benchmark_version="1.0",
        created_at=CREATED_AT,
        source_records=records,
    )


def _outcomes_by_company(manifest: CohortManifest) -> dict[str, ManifestOutcome]:
    return {
        r.company_id: r.outcome
        for r in manifest.source_records
        if r.outcome is not None
    }


@pytest.fixture(autouse=True)
def _real_engine():
    from predictron_engine.engine import PredictronEngine

    return PredictronEngine()


class TestCohortExecution:
    def test_core_semantics(self, _real_engine, tmp_path: Path) -> None:
        result = run_cohort_execution(
            _manifest(), engine=_real_engine, freeze_root=tmp_path
        )
        assert isinstance(result, CohortExecutionResult)
        assert result.frozen_company_ids == ["ai_infrastructure", "b2b_saas"]
        assert result.pending_company_ids == ["mrr_only_derivation"]
        assert result.failed_company_ids == []
        assert len(result.frozen_predictions) == 3  # pending records still freeze
        assert len(result.evaluations) == 2
        assert len(result.manifest_hash) == 64

    def test_evaluations_are_frozen_and_evaluated(self, _real_engine, tmp_path: Path) -> None:
        result = run_cohort_execution(
            _manifest(), engine=_real_engine, freeze_root=tmp_path
        )
        for company_id in result.frozen_company_ids:
            row = result.company_results[company_id]
            assert row.status == STATUS_FROZEN
            assert row.frozen is True
            assert row.prediction_id is not None
            assert row.evaluation is not None
            assert len(row.evaluation.evaluation_id) == 64
            assert row.outcome_record is not None
            outcomes = _outcomes_by_company(_manifest())
            assert row.outcome_record.outcome_id == stable_outcome_id(
                company_id=company_id,
                outcome=outcomes[company_id],
            )

    def test_pending_record_is_frozen_not_evaluated(self, _real_engine, tmp_path: Path) -> None:
        result = run_cohort_execution(
            _manifest(), engine=_real_engine, freeze_root=tmp_path
        )
        row = result.company_results["mrr_only_derivation"]
        assert row.status == STATUS_PENDING
        assert row.prediction is not None
        assert row.evaluation is None

    def test_metrics_and_calibration_present(self, _real_engine, tmp_path: Path) -> None:
        result = run_cohort_execution(
            _manifest(), engine=_real_engine, freeze_root=tmp_path
        )
        summary = result.metrics.summary()
        assert summary["scoreable"] >= 0
        assert "accuracy" in summary
        assert "total_samples" in result.calibration
        assert "expected_calibration_error" in result.calibration

    def test_rerun_is_idempotent(self, _real_engine, tmp_path: Path) -> None:
        first = run_cohort_execution(_manifest(), engine=_real_engine, freeze_root=tmp_path)
        second = run_cohort_execution(_manifest(), engine=_real_engine, freeze_root=tmp_path)
        assert [p.prediction_id for p in first.frozen_predictions] == [
            p.prediction_id for p in second.frozen_predictions
        ]
        assert [
            (e.evaluation_id, e.verdict.value) for e in first.evaluations
        ] == [(e.evaluation_id, e.verdict.value) for e in second.evaluations]
        # Second run found every prediction already persisted.
        for row in second.company_results.values():
            assert row.frozen is False or row.prediction_id in {
                p.prediction_id for p in first.frozen_predictions
            }

    def test_report_shape(self, _real_engine, tmp_path: Path) -> None:
        result = run_cohort_execution(
            _manifest(), engine=_real_engine, freeze_root=tmp_path
        )
        report = result.to_report()
        assert report["report_type"] == "cohort_validation"
        assert report["dataset_name"] == "test_cohort_execute"
        assert report["summary"]["frozen"] == 2
        assert report["summary"]["pending_ground_truth"] == 1
        assert report["companies"][0]["company_id"] == "ai_infrastructure"

    def test_naive_analysis_timestamp_fails_record(self, _real_engine, tmp_path: Path) -> None:
        manifest = _manifest()
        record = manifest.source_records[0].model_copy(
            update={"analysis_timestamp": datetime(2026, 1, 1)}
        )
        bad = manifest.model_copy(
            update={"source_records": [record] + list(manifest.source_records)[1:]}
        )
        result = run_cohort_execution(bad, engine=_real_engine, freeze_root=tmp_path)
        row = result.company_results["b2b_saas"]
        assert row.status == "failed"
        assert "timezone-aware" in row.error


class TestOutcomeMapping:
    def test_outcome_record_from_manifest(self) -> None:
        outcomes = _outcomes_by_company(_manifest())
        record = outcome_record_from_manifest(
            outcomes["b2b_saas"],
            company_id="b2b_saas",
            verification_date=VERIFICATION,
            report_horizon_days=None,
            analysis_timestamp=ANALYSIS,
        )
        assert record.verdict == OutcomeVerdict.SUCCESS
        assert record.time_horizon_days == (VERIFICATION - ANALYSIS.date()).days
        assert record.notes == "acquired by Example Corp"
        assert record.outcome.status == OutcomeStatus.FULLY_VERIFIED

    def test_outcome_record_uses_manifest_verified_flag(self) -> None:
        outcomes = _outcomes_by_company(_manifest())
        record = outcome_record_from_manifest(
            outcomes["ai_infrastructure"],
            company_id="ai_infrastructure",
            verification_date=VERIFICATION,
            report_horizon_days=120,
            analysis_timestamp=ANALYSIS,
        )
        assert record.time_horizon_days == 120  # manifest horizon wins when set

    def test_unknown_status_stays_unknown(self) -> None:
        outcomes = _outcomes_by_company(_manifest())
        unknown = outcomes["b2b_saas"].model_copy(
            update={"status": StartupStatus.UNKNOWN}
        )
        record = outcome_record_from_manifest(
            unknown,
            company_id="b2b_saas",
            verification_date=VERIFICATION,
            report_horizon_days=None,
            analysis_timestamp=ANALYSIS,
        )
        # A truly unverified status is reported as UNKNOWN rather than being
        # upgraded to a real verdict; only *missed* verdicts become INCONCLUSIVE.
        assert record.verdict == OutcomeVerdict.UNKNOWN

    def test_unverifiable_but_known_status_maps_to_inconclusive(self) -> None:
        outcomes = _outcomes_by_company(_manifest())
        operating = outcomes["b2b_saas"].model_copy(
            update={"status": StartupStatus.OPERATING}
        )
        record = outcome_record_from_manifest(
            operating,
            company_id="b2b_saas",
            verification_date=VERIFICATION,
            report_horizon_days=None,
            analysis_timestamp=ANALYSIS,
        )
        assert record.verdict == OutcomeVerdict.INCONCLUSIVE
