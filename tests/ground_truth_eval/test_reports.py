"""Tests for deterministic report generation."""

from __future__ import annotations

import pytest

from benchmarks.ground_truth_eval.history import BenchmarkHistory
from benchmarks.ground_truth_eval.reports import (
    REPORT_KINDS,
    build_report,
)


@pytest.fixture
def example_history(tmp_path, example_run, example_metrics):
    history = BenchmarkHistory(tmp_path)
    history.append(example_run, metrics=example_metrics.to_dict())
    return history


class TestReportDispatch:
    @pytest.mark.parametrize("kind", REPORT_KINDS)
    def test_every_kind_builds(
        self,
        kind,
        example_dataset,
        example_run,
        example_metrics,
        example_history,
    ) -> None:
        kwargs = {}
        if kind in ("trend",):
            kwargs["history"] = example_history
            kwargs["dataset_name"] = example_run.dataset_name
        elif kind == "engine_drift":
            kwargs["run_a"] = example_run
            kwargs["run_b"] = example_run
        report = build_report(
            kind, dataset=example_dataset, run=example_run, metrics=example_metrics, **kwargs
        )
        assert report["report_type"] == kind
        assert report["generated_at"]
        assert isinstance(report["content"], dict)

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown report kind"):
            build_report("bogus")

    def test_metric_kind_requires_inputs(self) -> None:
        with pytest.raises(ValueError, match="dataset"):
            build_report("executive", metrics=None)

    def test_trend_requires_history(self) -> None:
        with pytest.raises(ValueError, match="history"):
            build_report("trend")

    def test_engine_drift_requires_runs(self) -> None:
        with pytest.raises(ValueError, match="run_a"):
            build_report("engine_drift")

    def test_leaderboard_requires_inputs(self) -> None:
        with pytest.raises(ValueError, match="dataset"):
            build_report("leaderboard")

    def test_metrics_passed_through(self, example_dataset, example_run, example_metrics) -> None:
        report = build_report(
            "accuracy", dataset=example_dataset, run=example_run, metrics=example_metrics
        )
        assert report["content"]["brier_score"] == example_metrics.brier


class TestEnvelope:
    def test_run_header_fields(self, example_dataset, example_run, example_metrics) -> None:
        report = build_report(
            "coverage", dataset=example_dataset, run=example_run, metrics=example_metrics
        )
        assert report["run_id"] == example_run.run_id
        assert report["engine_version"] == example_run.engine_version
        assert report["dataset_hash"] == example_run.dataset_hash

    def test_explicit_generated_at(self, example_dataset, example_run, example_metrics) -> None:
        from datetime import datetime

        stamp = datetime(2026, 1, 2, 3, 4, 5)
        report = build_report(
            "executive",
            dataset=example_dataset,
            run=example_run,
            metrics=example_metrics,
            generated_at=stamp,
        )
        assert report["generated_at"] == "2026-01-02T03:04:05"


class TestExecutive:
    def test_headline_fields(self, example_dataset, example_run, example_metrics) -> None:
        report = build_report(
            "executive", dataset=example_dataset, run=example_run, metrics=example_metrics
        )
        headline = report["content"]["headline"]
        assert headline["overall_accuracy"] == pytest.approx(0.7, abs=1e-6)
        assert headline["scoreable_samples"] == 11

    def test_verdict_bands(self, make_run, entry_builder) -> None:
        from benchmarks.ground_truth_eval.metrics import compute_metrics
        from benchmarks.ground_truth_eval.models import GoldenDataset

        def verdict_for(entries, outputs) -> str:
            dataset = GoldenDataset(dataset_name="v", benchmark_version="1.0", entries=entries)
            run = make_run(outputs, run_id="vrun")
            m = compute_metrics(dataset, run, k_values=(), num_bins=1)
            return build_report("executive", dataset=dataset, run=run, metrics=m)["content"][
                "verdict"
            ]

        confident = [
            entry_builder(company_id=f"c{i}", status="acquired", exit_value_usd=1_000_000)
            for i in range(10)
        ]
        good_run = [
            {
                "company_id": f"c{i}",
                "decision": "invest",
                "overall_score": 80.0,
                "overall_confidence": 0.9,
            }
            for i in range(10)
        ]
        assert verdict_for(confident, good_run) == "strong"

        mixed = [
            entry_builder(
                company_id=f"c{i}",
                status="acquired" if i % 2 == 0 else "shutdown",
                exit_value_usd=1_000_000 if i % 2 == 0 else None,
            )
            for i in range(10)
        ]
        mixed_run = [
            {
                "company_id": f"c{i}",
                "decision": "invest" if i % 2 == 0 else "invest",
                "overall_score": 80.0,
                "overall_confidence": 0.9,
            }
            for i in range(10)
        ]
        assert verdict_for(mixed, mixed_run) == "needs_improvement"

    def test_key_findings(self, example_dataset, example_run, example_metrics) -> None:
        report = build_report(
            "executive", dataset=example_dataset, run=example_run, metrics=example_metrics
        )
        findings = report["content"]["key_findings"]
        assert isinstance(findings, list)
        assert any("calibration error" in f for f in findings)


class TestCalibrationReport:
    def test_structure(self, example_dataset, example_run, example_metrics) -> None:
        report = build_report(
            "calibration", dataset=example_dataset, run=example_run, metrics=example_metrics
        )
        content = report["content"]
        assert "expected_calibration_error" in content
        assert "curve" in content
        assert "total_samples" in content


class TestGroupedBreakdowns:
    @pytest.mark.parametrize(
        "kind,key",
        [
            ("sectors", "unit"),
            ("countries", "unit"),
            ("stages", "unit"),
        ],
    )
    def test_grouped_structure(
        self, kind, key, example_dataset, example_run, example_metrics
    ) -> None:
        report = build_report(
            kind, dataset=example_dataset, run=example_run, metrics=example_metrics
        )
        content = report["content"]
        assert key in content
        assert isinstance(content["rows"], list)
        if content["rows"]:
            assert "accuracy" in content["rows"][0]
            assert "scoreable_count" in content["rows"][0]

    def test_rows_sorted(self, example_dataset, example_run, example_metrics) -> None:
        report = build_report(
            "sectors", dataset=example_dataset, run=example_run, metrics=example_metrics
        )
        groups = [r["group"] for r in report["content"]["rows"]]
        assert groups == sorted(groups)


class TestCoverageReport:
    def test_fields(self, example_dataset, example_run, example_metrics) -> None:
        report = build_report(
            "coverage", dataset=example_dataset, run=example_run, metrics=example_metrics
        )
        content = report["content"]
        assert content["dataset_entries"] == 13
        assert content["replayed_entries"] == 13
        assert content["replay_coverage"] == pytest.approx(1.0)


class TestTrendReport:
    def test_rows_from_history(self, example_run, example_history) -> None:
        report = build_report(
            "trend", dataset_name=example_run.dataset_name, history=example_history
        )
        assert len(report["content"]["rows"]) == 1
        row = report["content"]["rows"][0]
        assert row["accuracy"] == pytest.approx(0.7, abs=1e-6)
        assert row["run_id"] == example_run.run_id

    def test_filters_by_dataset(self, make_run, example_history, tmp_path) -> None:
        other = BenchmarkHistory(tmp_path / "second")
        other_run = make_run([{"company_id": "x"}], run_id="other", dataset_name="elsewhere")
        other.append(other_run)
        report = build_report("trend", history=other)
        assert [r["run_id"] for r in report["content"]["rows"]] == []

    def test_runs_without_metrics_omitted(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path / "third")
        history.append(make_run([{"company_id": "x"}], run_id="nomet"))
        report = build_report("trend", dataset_name="x", history=history)
        assert report["content"]["rows"] == []

    def test_missing_history_raises(self) -> None:
        with pytest.raises(ValueError, match="history"):
            build_report("trend")


class TestEngineDriftReport:
    def test_content_shape(self, example_run) -> None:
        report = build_report("engine_drift", run_a=example_run, run_b=example_run)
        content = report["content"]
        assert content["affected_company_count"] == 0
        assert len(content["signals"]) == 5
        assert "No drift detected" in content["explanation"]

    def test_run_header(self, example_run) -> None:
        report = build_report("engine_drift", run_a=example_run, run_b=example_run)
        assert report["run_id"] == example_run.run_id


class TestLeaderboard:
    def test_sorted_descending(self, example_dataset, example_run, example_metrics) -> None:
        report = build_report("leaderboard", dataset=example_dataset, run=example_run)
        rows = report["content"]["rows"]
        scores = [r["score"] for r in rows if r["score"] is not None]
        assert scores == sorted(scores, reverse=True)
        # ranking_by
        assert report["content"]["ranking_by"] == "overall_score"

    def test_row_fields(self, example_dataset, example_run) -> None:
        report = build_report("leaderboard", dataset=example_dataset, run=example_run)
        row = report["content"]["rows"][0]
        assert set(row) == {
            "company_id",
            "score",
            "confidence",
            "decision",
            "predicted_positive",
            "actual_positive",
            "is_correct",
            "sector",
            "stage",
            "country",
        }

    def test_correctness_flag(self, example_dataset, example_run) -> None:
        report = build_report("leaderboard", dataset=example_dataset, run=example_run)
        for row in report["content"]["rows"]:
            if row["predicted_positive"] is None or row["actual_positive"] is None:
                assert row["is_correct"] is None


class TestDeterminism:
    def test_report_content_ignores_generated_at(
        self, example_dataset, example_run, example_metrics
    ) -> None:
        r1 = build_report(
            "accuracy",
            dataset=example_dataset,
            run=example_run,
            metrics=example_metrics,
            generated_at=None,
        )
        r2 = build_report(
            "accuracy",
            dataset=example_dataset,
            run=example_run,
            metrics=example_metrics,
            generated_at=None,
        )
        # strip timestamp if it could differ between the two calls
        c1 = dict(r1)
        c1.pop("generated_at", None)
        c2 = dict(r2)
        c2.pop("generated_at", None)
        assert c1 == c2

    def test_json_serialisable(self, example_dataset, example_run, example_metrics) -> None:
        import json

        report = build_report(
            "executive", dataset=example_dataset, run=example_run, metrics=example_metrics
        )
        json.dumps(report)
        report2 = build_report("engine_drift", run_a=example_run, run_b=example_run)
        json.dumps(report2)
        report3 = build_report("leaderboard", dataset=example_dataset, run=example_run)
        json.dumps(report3)
