"""Tests for the read-only benchmark accuracy view (Milestone V1.4).

The service reads the latest benchmark run from the append-only history and
never fabricates metrics; the HTTP layer 404s until a run exists.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.api import evaluation as evaluation_api
from app.schemas.evaluation import BenchmarkAccuracyResponse
from app.services.benchmark_accuracy import BenchmarkAccuracyService

RUN_CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _build_run(**overrides: object):
    from benchmarks.ground_truth_eval.metrics import (
        BaseRates,
        ConfusionMetrics,
        CoverageMetrics,
        GroundTruthMetrics,
    )
    from benchmarks.ground_truth_eval.runner import (
        BenchmarkRun,
        RunEntryOutput,
        _result_hash,
    )

    entries = [
        RunEntryOutput(
            company_id="co_a",
            success=True,
            overall_score=80.0,
            overall_confidence=0.7,
            decision="invest",
        ),
        RunEntryOutput(
            company_id="co_b",
            success=False,
            error="engine failed",
        ),
    ]
    run = BenchmarkRun(
        run_id=str(overrides.get("run_id", "run_accuracy_1")),
        created_at=RUN_CREATED_AT,
        engine_version="0.0-test",
        benchmark_version="1.0",
        dataset_name="test_dataset",
        dataset_hash="ab12",
        result_hash=_result_hash(entries),
        entries=entries,
    )
    metrics = GroundTruthMetrics(
        run_id=run.run_id,
        engine_version=run.engine_version,
        benchmark_version=run.benchmark_version,
        dataset_name=run.dataset_name,
        dataset_hash=run.dataset_hash,
        confusion=ConfusionMetrics(total=2, scoreable=1, true_positives=1),
        base_rates=BaseRates(total=1, positives=1, positive_rate=1.0),
        roc_auc=1.0,
        average_precision=1.0,
        coverage=CoverageMetrics(
            dataset_entries=2, replayed_entries=2, scoreable_entries=1
        ),
    )
    return run, metrics


class TestBenchmarkAccuracyService:
    def test_none_when_no_runs(self, tmp_path: Path) -> None:
        service = BenchmarkAccuracyService(tmp_path)
        assert service.latest() is None

    def test_reads_latest_run_and_metrics(self, tmp_path: Path) -> None:
        from benchmarks.ground_truth_eval.history import BenchmarkHistory

        run, metrics = _build_run()
        BenchmarkHistory(tmp_path).append(run, metrics=metrics.to_dict())

        response = BenchmarkAccuracyService(tmp_path).latest()
        assert isinstance(response, BenchmarkAccuracyResponse)
        assert response.run_id == run.run_id
        assert response.entry_count == 2
        assert response.successful_count == 1
        assert response.has_metrics is True
        assert response.metrics is not None
        assert response.metrics.roc_auc == 1.0
        assert response.metrics.average_precision == 1.0
        assert response.metrics.base_rate_positive == 1.0
        assert response.metrics.coverage == 0.5
        by_id = {c.company_id: c for c in response.companies}
        assert by_id["co_a"].decision == "invest"
        assert by_id["co_b"].error == "engine failed"

    def test_reads_run_without_metrics(self, tmp_path: Path) -> None:
        from benchmarks.ground_truth_eval.history import BenchmarkHistory

        run, _metrics = _build_run()
        BenchmarkHistory(tmp_path).append(run)

        response = BenchmarkAccuracyService(tmp_path).latest()
        assert response.has_metrics is False
        assert response.metrics is None

    def test_picks_latest_by_created_at(self, tmp_path: Path) -> None:
        from benchmarks.ground_truth_eval.history import BenchmarkHistory

        history = BenchmarkHistory(tmp_path)
        first, _ = _build_run(run_id="run_older")
        older = first.model_copy(
            update={"created_at": datetime(2025, 6, 1, tzinfo=UTC)}
        )
        history.append(older)
        newer, _ = _build_run(run_id="run_newer")
        history.append(newer)

        response = BenchmarkAccuracyService(tmp_path).latest()
        assert response.run_id == "run_newer"


class TestBenchmarkAccuracyEndpoint:
    async def test_404_when_no_history(self, client, monkeypatch, tmp_path: Path) -> None:
        service = BenchmarkAccuracyService(tmp_path)
        monkeypatch.setattr(evaluation_api, "BenchmarkAccuracyService", lambda: service)
        resp = await client.get("/api/v1/evaluation/benchmark")
        assert resp.status_code == 404
        assert "no benchmark run" in resp.json()["detail"].lower()

    async def test_200_with_run(self, client, monkeypatch, tmp_path: Path) -> None:
        from benchmarks.ground_truth_eval.history import BenchmarkHistory

        run, metrics = _build_run()
        BenchmarkHistory(tmp_path).append(run, metrics=metrics.to_dict())
        service = BenchmarkAccuracyService(tmp_path)
        monkeypatch.setattr(evaluation_api, "BenchmarkAccuracyService", lambda: service)

        resp = await client.get("/api/v1/evaluation/benchmark")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run.run_id
        assert data["has_metrics"] is True
        assert data["metrics"]["roc_auc"] == 1.0
        assert data["entry_count"] == 2
        assert len(data["companies"]) == 2

    async def test_200_without_metrics(self, client, monkeypatch, tmp_path: Path) -> None:
        from benchmarks.ground_truth_eval.history import BenchmarkHistory

        run, _ = _build_run()
        BenchmarkHistory(tmp_path).append(run)
        service = BenchmarkAccuracyService(tmp_path)
        monkeypatch.setattr(evaluation_api, "BenchmarkAccuracyService", lambda: service)

        resp = await client.get("/api/v1/evaluation/benchmark")
        assert resp.status_code == 200
        assert resp.json()["has_metrics"] is False
        assert resp.json()["metrics"] is None


# round-trip shape guard for the response schema
@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "dataset_name",
        "benchmark_version",
        "engine_version",
        "run_created_at",
        "dataset_hash",
        "result_hash",
        "entry_count",
        "successful_count",
        "has_metrics",
        "metrics",
        "companies",
        "generated_at",
    ],
)
def test_response_schema_fields_present(field: str) -> None:
    assert field in BenchmarkAccuracyResponse.model_fields
