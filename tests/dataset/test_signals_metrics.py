"""Tests for predictron_engine.dataset.signals.metrics — Project E4."""

from __future__ import annotations

import pytest

from predictron_engine.dataset.signals.metrics import (
    SignalStatistics,
    compute_signal_statistics,
)
from tests.dataset.signals_helpers import dt, make_signal, make_timeline

AS_OF = dt("2024-12-31T00:00:00+00:00")


def _company():
    return make_timeline(
        "company:a",
        make_signal(
            at="2024-01-01T00:00:00+00:00",
            signal_type="funding_round",
            confidence=0.9,
            amount_usd=1_000_000.0,
        ),
        make_signal(
            at="2024-06-01T00:00:00+00:00",
            signal_type="ipo",
            confidence=0.95,
        ),
    )


class TestComputeSignalStatistics:
    def test_totals(self) -> None:
        stats = compute_signal_statistics([_company()], as_of=AS_OF)
        assert stats.total_signals == 2
        assert stats.companies_with_signals == 1

    def test_by_type(self) -> None:
        stats = compute_signal_statistics([_company()], as_of=AS_OF)
        assert stats.by_type == {"funding_round": 1, "ipo": 1}

    def test_by_source(self) -> None:
        stats = compute_signal_statistics([_company()], as_of=AS_OF)
        assert stats.by_source == {"test": 2}

    def test_by_year_and_month(self) -> None:
        stats = compute_signal_statistics([_company()], as_of=AS_OF)
        assert stats.by_year == {"2024": 2}
        assert stats.by_month == {"2024-01": 1, "2024-06": 1}

    def test_average_confidence(self) -> None:
        stats = compute_signal_statistics([_company()], as_of=AS_OF)
        assert stats.average_confidence == pytest.approx(0.925, rel=1e-4)

    def test_confidence_histogram(self) -> None:
        stats = compute_signal_statistics([_company()], as_of=AS_OF)
        assert stats.confidence_histogram == {"0.8-1.0": 2}

    def test_funding_events_and_amount(self) -> None:
        stats = compute_signal_statistics([_company()], as_of=AS_OF)
        assert stats.funding_events == 1
        assert stats.total_amount_usd == pytest.approx(1_000_000.0, rel=1e-4)

    def test_coverage(self) -> None:
        stats = compute_signal_statistics(
            [_company()], expected_company_count=2, as_of=AS_OF
        )
        assert stats.coverage == 0.5

    def test_coverage_none_when_no_expected(self) -> None:
        stats = compute_signal_statistics([_company()], as_of=AS_OF)
        assert stats.coverage is None

    def test_no_timelines(self) -> None:
        stats = compute_signal_statistics([])
        assert stats.total_signals == 0
        assert stats.average_confidence is None

    def test_signals_per_company(self) -> None:
        stats = compute_signal_statistics(
            [
                _company(),
                make_timeline("company:b", make_signal(company_id="company:b")),
            ],
            as_of=AS_OF,
        )
        assert stats.signals_per_company == {"min": 1.0, "max": 2.0, "average": 1.5}

    def test_freshness_histogram(self) -> None:
        stats = compute_signal_statistics([_company()], as_of=AS_OF)
        assert stats.freshness_histogram == {"91-365": 1}

    def test_freshness_histogram_gt_365(self) -> None:
        tl = make_timeline(
            "company:b",
            make_signal(company_id="company:b", at="2020-01-01T00:00:00+00:00"),
        )
        stats = compute_signal_statistics([tl], as_of=AS_OF)
        assert stats.freshness_histogram == {">365": 1}

    def test_to_dict_structure(self) -> None:
        stats = compute_signal_statistics([_company()], expected_company_count=1, as_of=AS_OF)
        d = stats.to_dict()
        assert d["total_signals"] == 2
        assert d["coverage"] == 1.0
        assert "by_type" in d and "by_source" in d and "by_year" in d


class TestSignalStatisticsDefaults:
    def test_defaults(self) -> None:
        stats = SignalStatistics()
        assert stats.total_signals == 0
        assert stats.coverage is None
