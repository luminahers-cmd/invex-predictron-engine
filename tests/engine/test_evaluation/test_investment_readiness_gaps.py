"""Tests for investment readiness gap helpers (Sprint P8C).

Covers the dimension contribution resolution and the normalized
weighted gap magnitude used to rank gap-aware recommendations.
"""

from predictron_engine.evaluation.investment_readiness import (
    dimension_contribution,
    weighted_gap_magnitude,
)
from predictron_engine.models.report import InvestmentReadiness, ScoreResult


def _score(dimension: str, value: float) -> ScoreResult:
    return ScoreResult(dimension=dimension, score=value)


class TestDimensionContribution:
    def test_prefers_readiness_contributions_over_scores(self) -> None:
        readiness = InvestmentReadiness(
            readiness_score=60.0,
            dimension_contributions={"market_opportunity": 55.0},
        )
        result = dimension_contribution(
            "market_opportunity", [_score("market_opportunity", 90.0)], readiness
        )
        assert result == 55.0

    def test_falls_back_to_score_when_contribution_absent(self) -> None:
        readiness = InvestmentReadiness(
            readiness_score=60.0,
            dimension_contributions={"product_strength": 55.0},
        )
        result = dimension_contribution(
            "market_opportunity", [_score("market_opportunity", 72.5)], readiness
        )
        assert result == 72.5

    def test_neutral_baseline_when_nothing_available(self) -> None:
        result = dimension_contribution("market_opportunity", [])
        assert result == 50.0


class TestWeightedGapMagnitude:
    def test_full_readiness_has_zero_gap(self) -> None:
        result = weighted_gap_magnitude("market_opportunity", [_score("market_opportunity", 100.0)])
        assert result == 0.0

    def test_gap_scales_by_dimension_weight(self) -> None:
        """0.20-weight dimension at 60 -> 0.4; 0.15-weight at 60 -> 0.3."""
        market = weighted_gap_magnitude("market_opportunity", [_score("market_opportunity", 60.0)])
        product = weighted_gap_magnitude("product_strength", [_score("product_strength", 60.0)])
        assert market == 0.4
        assert product == 0.3

    def test_uses_readiness_contributions(self) -> None:
        readiness = InvestmentReadiness(
            readiness_score=60.0,
            dimension_contributions={"market_opportunity": 80.0},
        )
        result = weighted_gap_magnitude(
            "market_opportunity", [_score("market_opportunity", 50.0)], readiness
        )
        assert result == 0.2

    def test_unknown_dimension_uses_default_weight(self) -> None:
        result = weighted_gap_magnitude("fundraising", [_score("fundraising", 60.0)])
        assert result == 0.1
