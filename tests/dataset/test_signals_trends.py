"""Tests for predictron_engine.dataset.signals.trends — Project E4."""

from __future__ import annotations

import pytest

from predictron_engine.dataset.signals.trends import (
    TrendEngine,
    compute_decline,
    compute_funding_acceleration,
    compute_funding_velocity,
    compute_growth_consistency,
    compute_hiring_trend,
    compute_stagnation,
)
from tests.dataset.signals_helpers import dt, make_signal, make_timeline

AS_OF = dt("2024-12-31T00:00:00+00:00")


class TestFundingVelocity:
    def test_no_funding_unavailable(self) -> None:
        tl = make_timeline("a", make_signal(at="2024-01-01T00:00:00+00:00", signal_type="ipo"))
        r = compute_funding_velocity(tl, as_of=AS_OF)
        assert r.available is False
        assert r.value == 0.0
        assert r.direction == "flat"

    def test_with_amounts(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="funding_round",
            amount_usd=1_000_000.0,
        ))
        r = compute_funding_velocity(tl, as_of=AS_OF)
        assert r.available is True
        assert r.value > 0
        assert r.direction == "fundraising"
        assert "usd_per_year" in r.metrics["unit"]

    def test_without_amounts_rounds_per_year(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-06-01T00:00:00+00:00", signal_type="funding_round",
        ))
        r = compute_funding_velocity(tl, as_of=AS_OF)
        assert r.available is True
        assert r.metrics["unit"] == "rounds_per_year"
        assert r.value > 0

    def test_multiple_rounds(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="funding_round",
            amount_usd=100_000.0,
        ), make_signal(
            at="2024-07-01T00:00:00+00:00", signal_type="funding_round",
            amount_usd=200_000.0,
        ))
        r = compute_funding_velocity(tl, as_of=AS_OF)
        assert r.metrics["funding_count"] == 2
        assert r.metrics["total_amount_usd"] == pytest.approx(300_000.0, rel=1e-4)

    def test_result_to_dict(self) -> None:
        r = compute_funding_velocity(make_timeline("a"), as_of=AS_OF)
        d = r.to_dict()
        assert d["name"] == "funding_velocity"
        assert "available" in d
        assert "direction" in d


class TestFundingAcceleration:
    def test_unavailable_with_few(self) -> None:
        sig = make_signal(at="2024-01-01T00:00:00+00:00", signal_type="funding_round")
        tl = make_timeline("a", sig)
        r = compute_funding_acceleration(tl, as_of=AS_OF)
        assert r.available is False
        assert r.direction == "flat"

    def test_same_day_flat(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-12-31T00:00:00+00:00", signal_type="funding_round",
        ), make_signal(
            at="2024-12-31T12:00:00+00:00", signal_type="funding_round",
        ))
        r = compute_funding_acceleration(
            tl, as_of=dt("2024-12-31T23:00:00+00:00")
        )
        assert r.direction == "flat"
        assert r.available is True

    def test_accelerating(self) -> None:
        tl = make_timeline(
            "a",
            make_signal(at="2022-01-01T00:00:00+00:00", signal_type="funding_round"),
            make_signal(at="2023-01-01T00:00:00+00:00", signal_type="funding_round"),
            *[
                make_signal(
                    at=f"2024-{month:02d}-01T00:00:00+00:00",
                    signal_type="funding_round",
                )
                for month in (1, 2, 3, 4, 5, 6, 7, 8)
            ],
        )
        r = compute_funding_acceleration(tl, as_of=AS_OF)
        assert r.direction == "accelerating"

    def test_decelerating(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2022-01-01T00:00:00+00:00", signal_type="funding_round",
        ), make_signal(
            at="2022-03-01T00:00:00+00:00", signal_type="funding_round",
        ), make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="funding_round",
        ))
        r = compute_funding_acceleration(tl, as_of=AS_OF)
        assert r.direction == "decelerating"

    def test_deterministic(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2022-01-01T00:00:00+00:00", signal_type="funding_round",
        ), make_signal(
            at="2024-06-01T00:00:00+00:00", signal_type="funding_round",
        ))
        r1 = compute_funding_acceleration(tl, as_of=AS_OF)
        r2 = compute_funding_acceleration(tl, as_of=AS_OF)
        assert r1.value == r2.value
        assert r1.direction == r2.direction


class TestHiringTrend:
    def test_unavailable_few_snapshots(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="employee_milestone",
            count=10.0,
        ))
        r = compute_hiring_trend(tl, as_of=AS_OF)
        assert r.available is False
        assert r.direction == "flat"

    def test_upward(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="employee_milestone", count=10.0,
        ), make_signal(
            at="2024-07-01T00:00:00+00:00", signal_type="employee_milestone", count=50.0,
        ))
        r = compute_hiring_trend(tl, as_of=AS_OF)
        assert r.direction == "up"
        assert r.value > 0

    def test_downward(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="employee_milestone", count=50.0,
        ), make_signal(
            at="2024-07-01T00:00:00+00:00", signal_type="employee_milestone", count=10.0,
        ))
        r = compute_hiring_trend(tl, as_of=AS_OF)
        assert r.direction == "down"
        assert r.value < 0

    def test_flat(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="employee_milestone", count=10.0,
        ), make_signal(
            at="2024-07-01T00:00:00+00:00", signal_type="employee_milestone", count=10.0,
        ))
        r = compute_hiring_trend(tl, as_of=AS_OF)
        assert r.direction == "flat"
        assert r.value == 0.0

    def test_exact_values(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="employee_milestone", count=0.0,
        ), make_signal(
            at="2025-01-01T00:00:00+00:00", signal_type="employee_milestone", count=366.0,
        ))
        r = compute_hiring_trend(tl, as_of=AS_OF)
        assert r.value == pytest.approx(365.25, rel=1e-3)
        assert r.metrics["r_squared"] == 1.0
        assert r.metrics["points"] == 2


class TestGrowthConsistency:
    def test_insufficient_milestones(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="arr_milestone",
        ), make_signal(
            at="2024-06-01T00:00:00+00:00", signal_type="arr_milestone",
        ))
        r = compute_growth_consistency(tl)
        assert r.available is False
        assert r.direction == "stable"

    def test_consistent(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="arr_milestone",
        ), make_signal(
            at="2024-04-01T00:00:00+00:00", signal_type="arr_milestone",
        ), make_signal(
            at="2024-07-01T00:00:00+00:00", signal_type="arr_milestone",
        ))
        r = compute_growth_consistency(tl)
        assert r.direction == "consistent"
        assert r.value <= 0.6

    def test_variable(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="arr_milestone",
        ), make_signal(
            at="2024-01-10T00:00:00+00:00", signal_type="arr_milestone",
        ), make_signal(
            at="2024-12-01T00:00:00+00:00", signal_type="arr_milestone",
        ))
        r = compute_growth_consistency(tl)
        assert r.direction == "variable"
        assert r.value > 0.6


class TestStagnation:
    def test_empty_timeline_stagnant(self) -> None:
        r = compute_stagnation(make_timeline("a"), as_of=AS_OF)
        assert r.value == 1.0
        assert r.direction == "stagnant"
        assert r.available is True

    def test_active_within_window(self) -> None:
        tl = make_timeline("a", make_signal(at="2024-12-30T00:00:00+00:00"))
        r = compute_stagnation(tl, as_of=AS_OF)
        assert r.value == 0.0
        assert r.direction == "active"

    def test_stagnant_beyond_window(self) -> None:
        tl = make_timeline("a", make_signal(at="2023-01-01T00:00:00+00:00"))
        r = compute_stagnation(tl, as_of=AS_OF)
        assert r.value == 1.0
        assert r.direction == "stagnant"


class TestDecline:
    def test_healthy(self) -> None:
        tl = make_timeline("a", make_signal(at="2024-01-01T00:00:00+00:00"))
        r = compute_decline(tl, as_of=AS_OF)
        assert r.direction == "healthy"
        assert r.value == 0.0

    def test_layoffs_trigger(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="layoffs",
        ))
        r = compute_decline(tl, as_of=AS_OF)
        assert r.direction == "declining"
        assert r.value > 0

    def test_shutdown_trigger(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="shutdown",
        ))
        r = compute_decline(tl, as_of=AS_OF)
        assert r.direction == "declining"
        assert r.metrics["shutdown"] is True

    def test_multiple_indicators(self) -> None:
        tl = make_timeline(
            "a",
            make_signal(at="2024-01-01T00:00:00+00:00", signal_type="layoffs"),
            make_signal(at="2024-01-02T00:00:00+00:00", signal_type="shutdown"),
            make_signal(
                at="2024-01-01T00:00:00+00:00",
                signal_type="employee_milestone",
                count=50.0,
            ),
            make_signal(
                at="2024-07-01T00:00:00+00:00",
                signal_type="employee_milestone",
                count=10.0,
            ),
        )
        r = compute_decline(tl, as_of=AS_OF)
        assert r.value >= 2 / 3


class TestTrendEngine:
    def test_evaluate_keys(self) -> None:
        engine = TrendEngine(as_of=AS_OF)
        tl = make_timeline("a", make_signal(at="2024-01-01T00:00:00+00:00"))
        result = engine.evaluate(tl)
        assert set(result.keys()) == {
            "funding_velocity", "funding_acceleration", "hiring_trend",
            "growth_consistency", "stagnation", "decline",
        }

    def test_summarize_returns_dicts(self) -> None:
        engine = TrendEngine(as_of=AS_OF)
        result = engine.summarize(make_timeline("a"))
        for v in result.values():
            assert isinstance(v, dict)
            assert "name" in v
            assert "direction" in v

    def test_as_of_property(self) -> None:
        engine = TrendEngine(as_of=AS_OF)
        assert engine.as_of == AS_OF

    def test_deterministic(self) -> None:
        tl = make_timeline("a", make_signal(
            at="2024-01-01T00:00:00+00:00", signal_type="funding_round",
            amount_usd=100_000.0,
        ))
        e1 = TrendEngine(as_of=AS_OF)
        e2 = TrendEngine(as_of=AS_OF)
        r1 = e1.summarize(tl)
        r2 = e2.summarize(tl)
        for k in r1:
            assert r1[k]["value"] == r2[k]["value"]
