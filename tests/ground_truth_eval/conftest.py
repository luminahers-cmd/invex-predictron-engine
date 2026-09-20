"""Shared fixtures for the ground-truth evaluation test suite."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DATASET_PATH = REPO_ROOT / "benchmarks" / "golden_datasets" / "example_v1.json"


@pytest.fixture(scope="session")
def example_dataset():
    from benchmarks.ground_truth_eval.dataset import load_golden_dataset

    return load_golden_dataset(EXAMPLE_DATASET_PATH)


@pytest.fixture(scope="session")
def example_run(example_dataset):
    """One deterministic real-engine replay of the example dataset (session-scoped)."""
    from benchmarks.ground_truth_eval.runner import BenchmarkRunner

    return BenchmarkRunner().run(example_dataset, run_id="pytest_example_run")


@pytest.fixture(scope="session")
def example_metrics(example_dataset, example_run):
    from benchmarks.ground_truth_eval.metrics import compute_metrics

    return compute_metrics(example_dataset, example_run)


@pytest.fixture
def make_report():
    """Factory building a minimal but valid engine Report for unit tests."""

    def _make(
        *,
        overall_score: float = 70.0,
        overall_confidence: float = 0.65,
        decision: str | None = "invest",
        decision_confidence: float | None = 0.55,
        composite_score: float | None = 70.0,
        rec_categories: tuple[str, ...] = ("due_diligence", "opportunity"),
        dimension_scores: dict[str, float] | None = None,
        features: dict[str, Any] | None = None,
        evidence_count: int = 3,
        observation_count: int = 2,
    ):
        from predictron_engine.decision.models import (
            ConfidenceBreakdown,
            ConfidenceLevel,
            DecisionConfidence,
            UncertaintyBreakdown,
        )
        from predictron_engine.models.extracted_features import ExtractedFeatures
        from predictron_engine.models.report import (
            ConvictionLevel,
            DecisionCategory,
            EvidenceItem,
            InvestmentDecision,
            Observation,
            Recommendation,
            Report,
            ScoreResult,
        )
        from predictron_engine.models.startup import Startup

        decision_obj = None
        if decision is not None:
            decision_obj = InvestmentDecision(
                category=DecisionCategory(decision),
                conviction=ConvictionLevel.HIGH,
                composite_score=composite_score if composite_score is not None else overall_score,
            )
        conf_obj = None
        if decision_confidence is not None:
            conf_obj = DecisionConfidence(
                confidence=decision_confidence,
                level=ConfidenceLevel.MEDIUM,
                uncertainty_score=0.1,
                breakdown=ConfidenceBreakdown(composite=0.5),
                uncertainty_breakdown=UncertaintyBreakdown(composite=0.1),
            )
        features_obj = ExtractedFeatures(**(features or {}))
        report = Report(
            startup=Startup(
                name="Test Company",
                website="https://test.example.com",
                description=(
                    "A sufficiently long description to satisfy startup normalization constraints."
                ),
            ),
            features=features_obj,
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            investment_decision=decision_obj,
            decision_confidence=conf_obj,
            recommendations=[
                Recommendation(category=c, action="act", priority="medium") for c in rec_categories
            ],
            scores=[ScoreResult(dimension=d, score=s) for d, s in (dimension_scores or {}).items()],
            evidence=[
                EvidenceItem(
                    domain="industry", category="market_size", statement="s", source="test"
                )
                for _ in range(evidence_count)
            ],
            observations=[
                Observation(dimension="market", category="a", statement="s")
                for _ in range(observation_count)
            ],
        )
        return report

    return _make


@pytest.fixture
def make_fake_engine(make_report):
    """Factory returning an engine stub whose analyze() returns a report."""

    class _FakeEngine:
        def __init__(self, report, *, raise_error: bool = False):
            self._report = report
            self._raise_error = raise_error

        def analyze(self, request, request_id=None, evidence_bundle=None):
            if self._raise_error:
                raise RuntimeError("boom")
            return self._report

    def _make(report=None, *, raise_error: bool = False):
        return _FakeEngine(report or make_report(), raise_error=raise_error)

    return _make


@pytest.fixture
def entry_builder(example_dataset):
    """Returns a callable producing a GoldenEntry variant for tests.

    Defaults to a cloned b2b_saas entry with an overridable company id and
    verified outcome, so tests can build small synthetic datasets without
    fabricating engine outputs.
    """

    def _build(
        *,
        company_id: str = "sample_company",
        status: str = "operating",
        arr_usd: float | None = 4_000_000.0,
        exit_value_usd: float | None = None,
        description: str = "A sufficiently long startup description for the request.",
        startup_name: str = "Sample Company",
        evidence_corpus: str | None = "sample_evidence",
        overall_score: float = 70.0,
        decision: str = "invest",
    ) -> Any:
        from benchmarks.ground_truth.schema import OutcomeEvent, OutcomeEventKind, StartupStatus
        from benchmarks.ground_truth_eval.models import (
            GoldenEntry,
            HistoricalEvidence,
            HistoricalPrediction,
            Provenance,
            VerifiedOutcome,
        )

        events: list[OutcomeEvent] = []
        if status == "acquired":
            events.append(
                OutcomeEvent(
                    kind=OutcomeEventKind.ACQUISITION,
                    occurred_at=datetime(2026, 3, 1, tzinfo=UTC).date(),
                    value_currency_usd=exit_value_usd,
                    description="Acquired",
                    sources=["test_sources"],
                )
            )
        elif status == "shutdown":
            events.append(
                OutcomeEvent(
                    kind=OutcomeEventKind.SHUTDOWN,
                    occurred_at=datetime(2025, 12, 1, tzinfo=UTC).date(),
                    description="Shutdown",
                    sources=["test_sources"],
                )
            )
        elif arr_usd:
            events.append(
                OutcomeEvent(
                    kind=OutcomeEventKind.ARR_MILESTONE,
                    occurred_at=datetime(2026, 4, 1, tzinfo=UTC).date(),
                    amount_nominal_units=arr_usd,
                    description="ARR milestone",
                    sources=["test_sources"],
                )
            )
        return GoldenEntry(
            company_id=company_id,
            company_name=startup_name,
            request={
                "startup_name": startup_name,
                "website": f"https://{company_id}.example.com",
                "description": description,
            },
            sector="Test Sector",
            country="us",
            stage="Series A",
            historical_prediction=HistoricalPrediction(
                overall_score=overall_score,
                overall_confidence=0.6,
                decision=decision,
            ),
            historical_evidence=HistoricalEvidence(
                corpus_name=evidence_corpus,
                sources=["test_sources"],
            ),
            verified_outcome=VerifiedOutcome(
                status=StartupStatus(status),
                verification_date=datetime(2026, 6, 30, tzinfo=UTC).date(),
                outcome_events=events,
                exit_value_usd=exit_value_usd,
                sources=["test_sources"],
            ),
            provenance=Provenance(sources=["test_sources"], is_example=True, notes="test fixture"),
        )

    return _build


@pytest.fixture
def make_run():
    """Factory building a BenchmarkRun directly from per-company output dicts."""

    def _make(
        outputs: list[dict[str, Any]],
        *,
        run_id: str = "run_test_1",
        engine_version: str = "0.0-test",
        benchmark_version: str = "1.0",
        dataset_name: str = "test_dataset",
        dataset_hash: str = "deadbeef",
    ):
        from benchmarks.ground_truth_eval.runner import BenchmarkRun, RunEntryOutput

        entries = [
            RunEntryOutput(
                company_id=str(o["company_id"]),
                success=bool(o.get("success", True)),
                error=o.get("error"),
                overall_score=o.get("overall_score"),
                overall_confidence=o.get("overall_confidence"),
                decision=o.get("decision"),
                decision_confidence=o.get("decision_confidence"),
                composite_score=o.get("composite_score"),
                recommendation_categories=[str(c) for c in o.get("recommendation_categories", [])],
                recommendation_count=int(o.get("recommendation_count", 0)),
                dimension_scores={
                    str(k): float(v) for k, v in o.get("dimension_scores", {}).items()
                },
                extracted_features={str(k): v for k, v in o.get("extracted_features", {}).items()},
                evidence_count=int(o.get("evidence_count", 0)),
                observation_count=int(o.get("observation_count", 0)),
                processing_time_ms=float(o.get("processing_time_ms", 1.0)),
            )
            for o in outputs
        ]
        from benchmarks.ground_truth_eval.runner import _result_hash

        return BenchmarkRun(
            run_id=run_id,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            engine_version=engine_version,
            benchmark_version=benchmark_version,
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            result_hash=_result_hash(entries),
            entries=entries,
        )

    return _make
