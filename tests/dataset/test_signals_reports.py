"""Tests for predictron_engine.dataset.signals.reports — Project E4."""

from __future__ import annotations

import pytest

from predictron_engine.dataset.signals.reports import (
    build_activity_heatmap,
    build_coverage_report,
    build_dataset_freshness_report,
    build_momentum_statistics,
    build_signal_dataset_report,
    build_signal_distribution,
    build_timeline_report,
    build_trend_summary_report,
)
from predictron_engine.dataset.signals.trends import TrendEngine
from tests.dataset.signals_helpers import dt, make_signal, make_timeline

AS_OF = dt("2024-12-31T00:00:00+00:00")


def _tl(signals, company="company:a"):
    return make_timeline(company, *signals)


class TestBuildTimelineReport:
    def test_structure(self) -> None:
        tl = _tl([
            make_signal(at="2024-01-01T00:00:00+00:00"),
            make_signal(at="2024-06-01T00:00:00+00:00", signal_type="ipo"),
        ])
        d = build_timeline_report(tl)
        assert d["company_id"] == "company:a"
        assert d["signal_count"] == 2
        assert d["span_days"] > 0
        assert d["first_signal_at"] is not None
        assert d["last_signal_at"] is not None
        assert d["by_type"] == {"funding_round": 1, "ipo": 1}
        assert len(d["signals"]) == 2

    def test_signal_entry_fields(self) -> None:
        tl = _tl([make_signal(at="2024-01-01T00:00:00+00:00")])
        d = build_timeline_report(tl)
        entry = d["signals"][0]
        assert "signal_id" in entry
        assert "signal_type" in entry
        assert "timestamp" in entry
        assert "confidence" in entry

    def test_empty_timeline(self) -> None:
        d = build_timeline_report(_tl([]))
        assert d["signal_count"] == 0
        assert d["first_signal_at"] is None


class TestBuildSignalDistribution:
    def test_returns_counts(self) -> None:
        tl = _tl([
            make_signal(at="2024-01-01T00:00:00+00:00", signal_type="funding_round"),
            make_signal(at="2024-01-02T00:00:00+00:00", signal_type="ipo"),
        ])
        d = build_signal_distribution([tl], expected_company_count=2)
        assert d["total_signals"] == 2
        assert d["companies_with_signals"] == 1
        assert d["expected_company_count"] == 2
        assert d["by_type"] == {"funding_round": 1, "ipo": 1}
        assert d["by_source"] == {"test": 2}
        assert d["by_year"] == {"2024": 2}
        assert d["coverage"] == 0.5
        assert d["average_confidence"] is not None


class TestBuildActivityHeatmap:
    def test_month_granularity(self) -> None:
        tl = _tl([
            make_signal(at="2024-01-01T00:00:00+00:00"),
            make_signal(at="2024-03-01T00:00:00+00:00"),
        ])
        d = build_activity_heatmap([tl], granularity="month")
        assert d["granularity"] == "month"
        assert d["periods"]["2024-01"] == 1
        assert d["periods"]["2024-02"] == 0  # zero-filled
        assert d["periods"]["2024-03"] == 1

    def test_quarter_granularity(self) -> None:
        tl = _tl([
            make_signal(at="2024-01-01T00:00:00+00:00"),
            make_signal(at="2024-09-01T00:00:00+00:00"),
        ])
        d = build_activity_heatmap([tl], granularity="quarter")
        assert d["periods"]["2024-Q1"] == 1
        assert d["periods"]["2024-Q2"] == 0  # zero-filled
        assert d["periods"]["2024-Q3"] == 1

    def test_default_granularity_month(self) -> None:
        tl = _tl([make_signal(at="2024-01-01T00:00:00+00:00")])
        assert build_activity_heatmap([tl])["granularity"] == "month"


class TestBuildMomentumStatistics:
    def test_ranking_and_distribution(self) -> None:
        tl_hot = make_timeline(
            "a", make_signal(company_id="a", at="2024-12-30T00:00:00+00:00")
        )
        tl_cold = make_timeline(
            "b", make_signal(company_id="b", at="2020-01-01T00:00:00+00:00")
        )
        d = build_momentum_statistics([tl_cold, tl_hot], as_of=AS_OF)
        assert d["company_count"] == 2
        assert d["ranking"][0]["company_id"] == "a"
        assert d["ranking"][1]["company_id"] == "b"
        assert sum(d["distribution"].values()) == 2

    def test_empty(self) -> None:
        d = build_momentum_statistics([], as_of=AS_OF)
        assert d["company_count"] == 0
        assert d["average_momentum_score"] is None
        assert d["ranking"] == []


class TestBuildDatasetFreshnessReport:
    def test_average_and_max(self) -> None:
        tl_a = make_timeline(
            "a", make_signal(company_id="a", at="2024-12-30T00:00:00+00:00")
        )
        tl_b = make_timeline(
            "b", make_signal(company_id="b", at="2024-12-01T00:00:00+00:00")
        )
        d = build_dataset_freshness_report([tl_a, tl_b], as_of=AS_OF)
        assert d["company_count"] == 2
        assert d["average_days_since_last_signal"] == pytest.approx(15.5, rel=1e-3)
        assert d["max_days_since_last_signal"] == 30.0
        assert set(d["per_company"].keys()) == {"a", "b"}

    def test_empty(self) -> None:
        d = build_dataset_freshness_report([], as_of=AS_OF)
        assert d["average_days_since_last_signal"] is None


class TestBuildCoverageReport:
    def test_missing_ids(self) -> None:
        tl = make_timeline(
            "present", make_signal(company_id="present", at="2024-01-01T00:00:00+00:00")
        )
        d = build_coverage_report(
            [tl],
            expected_company_count=3,
            expected_company_ids=["present", "missing1", "missing2"],
        )
        assert d["companies_with_signals"] == 1
        assert d["expected_company_count"] == 3
        assert d["coverage"] == pytest.approx(1 / 3, rel=1e-3)
        assert d["companies_without_signals"] == ["missing1", "missing2"]

    def test_no_expected_ids(self) -> None:
        d = build_coverage_report([], 0, [])
        assert d["coverage"] is None
        assert d["companies_without_signals"] == []


class TestBuildTrendSummaryReport:
    def test_companies_keyed(self) -> None:
        tl_a = make_timeline(
            "a", make_signal(company_id="a", at="2024-01-01T00:00:00+00:00")
        )
        d = build_trend_summary_report([tl_a], as_of=AS_OF)
        assert d["as_of"] == AS_OF.isoformat()
        assert set(d["companies"].keys()) == {"a"}
        assert "funding_velocity" in d["companies"]["a"]


class TestBuildSignalDatasetReport:
    def test_full_report_keys(self) -> None:
        tl = make_timeline(
            "a", make_signal(company_id="a", at="2024-01-01T00:00:00+00:00")
        )
        d = build_signal_dataset_report(
            [tl],
            as_of=AS_OF,
            expected_company_count=1,
            expected_company_ids=["a"],
            generated_at=AS_OF,
        )
        assert d["report_type"] == "company_signals_report"
        assert d["as_of"] == AS_OF.isoformat()
        for key in (
            "statistics", "distribution", "coverage", "momentum",
            "freshness", "heatmap", "companies", "generated_at",
        ):
            assert key in d
        assert d["generated_at"] == AS_OF.isoformat()
        assert d["companies"][0]["company_id"] == "a"

    def test_deterministic_for_fixed_asof(self) -> None:
        tl = make_timeline(
            "a",
            make_signal(company_id="a", at="2024-01-01T00:00:00+00:00"),
            make_signal(company_id="a", at="2024-03-01T00:00:00+00:00"),
        )
        d1 = build_signal_dataset_report([tl], as_of=AS_OF)
        d2 = build_signal_dataset_report([tl], as_of=AS_OF)
        assert str(d1) == str(d2)

    def test_engine_reuse(self) -> None:
        tl = make_timeline(
            "a", make_signal(company_id="a", at="2024-01-01T00:00:00+00:00")
        )
        engine = TrendEngine(as_of=AS_OF)
        d = build_trend_summary_report([tl], engine=engine)
        assert d["as_of"] == AS_OF.isoformat()
