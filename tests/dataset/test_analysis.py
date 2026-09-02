"""Tests for the engine integration pipeline (Part B)."""

from __future__ import annotations

from types import SimpleNamespace

from predictron_engine.dataset.analysis import (
    AnalysisPipeline,
    AnalysisResult,
    AnalysisRun,
)
from predictron_engine.dataset.analysis_support import (
    build_request,
    extract_prediction_summary,
)
from predictron_engine.dataset.models import (
    DecisionLabel,
    PredictionSummary,
)
from predictron_engine.dataset.store import DatasetStore
from tests.dataset.conftest import make_record


def _make_fake_report(
    decision_category: str = "invest",
    overall_confidence: float = 0.8,
    overall_score: float = 72.0,
    processing_ms: float = 15.5,
) -> SimpleNamespace:
    metadata = SimpleNamespace(
        engine_version="0.12.1",
        processing_time_ms=processing_ms,
        pipeline_stages_completed=["normalize", "score"],
    )
    decision = SimpleNamespace(
        category=SimpleNamespace(value=decision_category),
        composite_score=overall_score,
    )
    score = SimpleNamespace(dimension="market", score=70.0)
    readiness = SimpleNamespace(readiness_score=80.0)
    return SimpleNamespace(
        analysis_metadata=metadata,
        investment_decision=decision,
        overall_confidence=overall_confidence,
        overall_score=overall_score,
        scores=[score],
        recommendations=["r1", "r2"],
        investment_readiness=readiness,
    )


class FakeEngine:
    """Fake PredictronEngine that returns a canned report."""

    def __init__(self, report) -> None:
        self._report = report
        self.calls = []
        self.last_request = None

    def analyze(self, request, evidence_bundle=None) -> object:
        self.calls.append(evidence_bundle)
        self.last_request = request
        return self._report


class TestAnalysisPipeline:
    def test_analyze_builds_request_and_returns_run(self) -> None:
        engine = FakeEngine(_make_fake_report())
        pipeline = AnalysisPipeline(engine)
        record = make_record()
        result = pipeline.analyze(record)

        assert isinstance(result, AnalysisResult)
        run = result.run
        assert isinstance(run, AnalysisRun)
        assert run.record_id == record.record_id
        assert run.engine_version == "0.12.1"
        assert run.processing_time_ms == 15.5
        assert run.pipeline_stages_completed == ["normalize", "score"]
        assert run.prediction.decision == DecisionLabel.INVEST
        assert run.prediction.confidence == 0.8
        assert run.prediction.composite_score == 72.0
        assert run.prediction.investment_readiness_score == 80.0
        assert run.prediction.recommendation_count == 2
        # report_summary present
        assert result.report_summary["overall_score"] == 72.0

    def test_analyze_passes_evidence_bundle(self) -> None:
        engine = FakeEngine(_make_fake_report())
        pipeline = AnalysisPipeline(engine)
        record = make_record()
        bundle = object()
        pipeline.analyze(record, evidence_bundle=bundle)
        assert engine.calls[-1] is bundle

    def test_analyze_never_mutates_record(self) -> None:
        engine = FakeEngine(_make_fake_report())
        pipeline = AnalysisPipeline(engine)
        record = make_record()
        original = record.model_dump()
        pipeline.analyze(record)
        assert record.model_dump() == original

    def test_analyze_stores_run(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record()
        store.save_record(record)
        engine = FakeEngine(_make_fake_report())
        pipeline = AnalysisPipeline(engine)
        result = pipeline.analyze(record)
        store.save_run(result.run)
        assert store.find_runs_by_record(record.record_id)
        assert store.count_runs() == 1

    def test_analyze_all(self) -> None:
        engine = FakeEngine(_make_fake_report())
        pipeline = AnalysisPipeline(engine)
        records = [make_record(), make_record(startup_name="Two")]
        results = pipeline.analyze_all(records)
        assert len(results) == 2

    def test_analyze_all_with_bundles(self) -> None:
        engine = FakeEngine(_make_fake_report())
        pipeline = AnalysisPipeline(engine)
        records = [make_record(), make_record(startup_name="Two")]
        bundles = {records[0].record_id: object()}
        pipeline.analyze_all(records, evidence_bundles=bundles)
        # first call got bundle, second got None
        assert engine.calls[0] is bundles[records[0].record_id]
        assert engine.calls[1] is None


class TestExtractHelpers:
    def test_build_request(self) -> None:
        record = make_record(website="https://acme.example.com")
        request = build_request(record)
        assert request.startup_name == record.startup_name
        assert "Acme Corp" in request.description

    def test_extract_prediction_summary(self) -> None:
        report = _make_fake_report()
        summary = extract_prediction_summary(report)
        assert isinstance(summary, PredictionSummary)
        assert summary.decision == DecisionLabel.INVEST

    def test_extract_pass_decision(self) -> None:
        report = _make_fake_report(decision_category="pass")
        summary = extract_prediction_summary(report)
        assert summary.decision == DecisionLabel.PASS

    def test_extract_unknown_category_defaults(self) -> None:
        report = _make_fake_report(decision_category="bogus")
        summary = extract_prediction_summary(report)
        assert summary.decision == DecisionLabel.WATCH
