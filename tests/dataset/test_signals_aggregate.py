"""Tests for predictron_engine.dataset.signals.aggregate — Project E4."""

from __future__ import annotations

import pytest

from predictron_engine.dataset.signals.aggregate import (
    aggregate_all,
    funding_cadence,
    growth_cadence,
    momentum_ranking,
    momentum_score,
    momentum_summary,
    recent_activity,
    signal_frequency,
    signal_freshness,
)
from tests.dataset.signals_helpers import dt, make_signal, make_timeline

AS_OF = "2024-12-31T00:00:00+00:00"


def _tl(signals, company="company:a"):
    return make_timeline(company, *signals)


class TestRecentActivity:
    def test_active_within_window(self) -> None:
        tl = _tl([
            make_signal(at="2023-01-15T00:00:00+00:00"),
            make_signal(at="2024-11-15T00:00:00+00:00"),
        ])
        result = recent_activity(tl, as_of=dt(AS_OF))
        assert result["is_active"] is True
        assert result["recent_signal_count"] == 1
        assert result["signal_count"] == 2

    def test_inactive_when_old(self) -> None:
        result = recent_activity(
            _tl([make_signal(at="2020-01-01T00:00:00+00:00")]),
            as_of=dt(AS_OF),
        )
        assert result["is_active"] is False
        assert result["recent_signal_count"] == 0
        assert result["recent_ratio"] == 0.0

    def test_empty_timeline(self) -> None:
        result = recent_activity(_tl([]), as_of=dt(AS_OF))
        assert result["days_since_last_signal"] is None
        assert result["recent_ratio"] == 0.0

    def test_days_since_last(self) -> None:
        result = recent_activity(
            _tl([make_signal(at="2024-12-01T00:00:00+00:00")]),
            as_of=dt("2024-12-31T00:00:00+00:00"),
        )
        assert result["days_since_last_signal"] == 30.0


class TestMomentum:
    def test_recent_scores_higher(self) -> None:
        old = momentum_score(
            _tl([make_signal(at="2024-01-01T00:00:00+00:00")]),
            as_of=dt(AS_OF),
        )
        recent = momentum_score(
            _tl([make_signal(at="2024-12-30T00:00:00+00:00")]),
            as_of=dt(AS_OF),
        )
        assert recent > old

    def test_empty_timeline_zero(self) -> None:
        assert momentum_score(_tl([])) == 0.0

    def test_confidence_scales(self) -> None:
        high = momentum_score(
            _tl([make_signal(at="2024-12-01T00:00:00+00:00", confidence=1.0)]),
            as_of=dt(AS_OF),
        )
        low = momentum_score(
            _tl([make_signal(at="2024-12-01T00:00:00+00:00", confidence=0.5)]),
            as_of=dt(AS_OF),
        )
        assert high == pytest.approx(2 * low, rel=1e-2)

    def test_deterministic(self) -> None:
        tl = _tl([
            make_signal(at="2024-01-01T00:00:00+00:00"),
            make_signal(at="2024-06-01T00:00:00+00:00"),
        ])
        assert momentum_score(tl, as_of=dt(AS_OF)) == momentum_score(
            tl, as_of=dt(AS_OF)
        )

    def test_summary_structure(self) -> None:
        summary = momentum_summary(
            _tl([make_signal(at="2024-12-01T00:00:00+00:00")]), as_of=dt(AS_OF)
        )
        assert "score" in summary
        assert "half_life_days" in summary
        assert summary["half_life_days"] == 180.0
        assert "weighted_contributions" in summary
        assert summary["signal_count"] == 1

    def test_no_future_signals_clamped(self) -> None:
        assert momentum_score(
            _tl([make_signal(at="2025-01-01T00:00:00+00:00")]),
            as_of=dt(AS_OF),
        ) == pytest.approx(1.0, rel=1e-6)


class TestFundingCadence:
    def test_two_rounds(self) -> None:
        tl = _tl([
            make_signal(
                at="2024-01-01T00:00:00+00:00",
                signal_type="funding_round",
                amount_usd=1_000_000.0,
            ),
            make_signal(
                at="2024-07-01T00:00:00+00:00",
                signal_type="funding_round",
                amount_usd=2_000_000.0,
            ),
        ])
        result = funding_cadence(tl, as_of=dt(AS_OF))
        assert result["round_count"] == 2
        assert abs(result["span_days"] - 182.0) < 5.0
        assert result["total_amount_usd"] == pytest.approx(3_000_000.0)
        assert result["avg_interval_days"] is not None

    def test_no_funding(self) -> None:
        tl = _tl([make_signal(at="2024-01-01T00:00:00+00:00", signal_type="ipo")])
        result = funding_cadence(tl)
        assert result["round_count"] == 0
        assert result["avg_interval_days"] is None
        assert result["total_amount_usd"] == 0.0

    def test_one_round(self) -> None:
        tl = _tl(
            [make_signal(at="2024-01-01T00:00:00+00:00", signal_type="funding_round")]
        )
        result = funding_cadence(tl, as_of=dt(AS_OF))
        assert result["round_count"] == 1
        assert result["avg_interval_days"] is None
        assert result["days_since_last_round"] is not None

    def test_ignores_non_numeric_amounts(self) -> None:
        tl = _tl([
            make_signal(
                at="2024-01-01T00:00:00+00:00",
                signal_type="funding_round",
                amount_usd="not-a-number",
            )
        ])
        result = funding_cadence(tl)
        assert result["total_amount_usd"] == 0.0


class TestGrowthCadence:
    def test_milestones(self) -> None:
        tl = _tl([
            make_signal(at="2024-01-01T00:00:00+00:00", signal_type="arr_milestone"),
            make_signal(
                at="2024-06-01T00:00:00+00:00", signal_type="employee_milestone"
            ),
        ])
        result = growth_cadence(tl)
        assert result["milestone_count"] == 2

    def test_none(self) -> None:
        assert growth_cadence(_tl([]))["milestone_count"] == 0


class TestSignalFrequency:
    def test_per_year(self) -> None:
        tl = _tl([
            make_signal(at="2024-01-01T00:00:00+00:00"),
            make_signal(at="2024-07-01T00:00:00+00:00"),
        ])
        result = signal_frequency(tl)
        assert result["total_signals"] == 2
        assert result["signals_per_year"] > 0
        assert result["by_type"]["funding_round"] == 2

    def test_empty(self) -> None:
        result = signal_frequency(_tl([]))
        assert result["total_signals"] == 0


class TestSignalFreshness:
    def test_bucket_0_30(self) -> None:
        result = signal_freshness(
            _tl([make_signal(at="2024-12-20T00:00:00+00:00")]), as_of=dt(AS_OF)
        )
        assert result["freshness_bucket"] == "0-30"

    def test_bucket_31_90(self) -> None:
        result = signal_freshness(
            _tl([make_signal(at="2024-10-15T00:00:00+00:00")]), as_of=dt(AS_OF)
        )
        assert result["freshness_bucket"] == "31-90"

    def test_bucket_91_365(self) -> None:
        result = signal_freshness(
            _tl([make_signal(at="2024-06-01T00:00:00+00:00")]), as_of=dt(AS_OF)
        )
        assert result["freshness_bucket"] == "91-365"

    def test_bucket_gt_365(self) -> None:
        result = signal_freshness(
            _tl([make_signal(at="2022-06-01T00:00:00+00:00")]), as_of=dt(AS_OF)
        )
        assert result["freshness_bucket"] == ">365"

    def test_empty(self) -> None:
        result = signal_freshness(_tl([]))
        assert result["freshness_bucket"] is None


class TestAggregateAll:
    def test_keys_present(self) -> None:
        tl = _tl([make_signal(at="2024-01-01T00:00:00+00:00")])
        result = aggregate_all(tl, as_of=dt(AS_OF))
        for key in (
            "recent_activity",
            "momentum",
            "funding_cadence",
            "growth_cadence",
            "signal_frequency",
            "signal_freshness",
        ):
            assert key in result


class TestMomentumRanking:
    def test_orders_highest_first(self) -> None:
        tl_active = make_timeline(
            "a", make_signal(company_id="a", at="2024-12-30T00:00:00+00:00")
        )
        tl_stale = make_timeline(
            "b", make_signal(company_id="b", at="2020-01-01T00:00:00+00:00")
        )
        ranking = momentum_ranking([tl_stale, tl_active], as_of=dt(AS_OF))
        assert ranking == ["a", "b"]

    def test_ties_broken_by_id(self) -> None:
        tl_a = make_timeline(
            "b", make_signal(company_id="b", at="2024-12-30T00:00:00+00:00")
        )
        tl_b = make_timeline(
            "a", make_signal(company_id="a", at="2024-12-30T00:00:00+00:00")
        )
        ranking = momentum_ranking([tl_a, tl_b], as_of=dt(AS_OF))
        assert ranking == ["a", "b"]
