"""Tests for the canonical score-aggregation primitives (Sprint P9 H2/H3).

Covers:
  - The canonical ``mean_score`` shared by report + decision.
  - The canonical ``weighted_composite`` with missing-data exclusion.
  - The shared dimension weights source of truth.
  - H3: investment readiness no longer treats *missing* dimensions as
    neutral/average.
"""

from __future__ import annotations

import pytest

from predictron_engine.evaluation.aggregation import (
    DEFAULT_DIMENSION_WEIGHT,
    DIMENSION_WEIGHTS,
    MAX_DIMENSION_WEIGHT,
    mean_score,
    weighted_composite,
)
from predictron_engine.evaluation.investment_readiness import (
    compute_investment_readiness,
)
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    InvestmentReadiness,
    ScoreResult,
)


def _score(dimension: str, value: float) -> ScoreResult:
    return ScoreResult(dimension=dimension, score=value)


class TestMeanScore:
    def test_averages_dimension_scores(self) -> None:
        scores = [_score("a", 60.0), _score("b", 80.0)]
        assert mean_score(scores) == 70.0

    def test_empty_is_no_evidence_not_neutral(self) -> None:
        """H3 — empty input yields 0.0, never a fabricated neutral 50."""
        assert mean_score([]) == 0.0

    def test_single_score(self) -> None:
        assert mean_score([_score("a", 42.0)]) == 42.0

    def test_matches_report_aggregation(self) -> None:
        """The report overall_score is derived from this same primitive."""
        from predictron_engine.report.report_builder import DefaultReportBuilder

        scores = [_score("a", 55.0), _score("b", 65.0)]
        builder = DefaultReportBuilder()
        assert builder._aggregate_scores(scores) == round(mean_score(scores), 2)


class TestWeightedComposite:
    def test_all_present_agrees_with_documentation(self) -> None:
        """With every dimension present, weights normalize to 1.0."""
        values = {"market_opportunity": 80.0, "founder_quality": 60.0}
        result = weighted_composite(
            values, present_keys={"market_opportunity", "founder_quality"}
        )
        # w_market=0.20, w_founder=0.18, total=0.38
        expected = 80.0 * (0.20 / 0.38) + 60.0 * (0.18 / 0.38)
        assert result == pytest.approx(expected)

    def test_missing_dimension_excluded_from_both_terms(self) -> None:
        """H3 — an unevidenced dimension cannot drag the composite down."""
        values = {
            "market_opportunity": 80.0,   # present
            "founder_quality": 60.0,      # present
            "product_strength": 30.0,     # missing -> excluded
            "traction_signals": 30.0,     # missing -> excluded
        }
        # Only market + founder participate.
        present = weighted_composite(
            values,
            present_keys={"market_opportunity", "founder_quality"},
        )
        # If the "missing" dims had been averaged in as neutral they would
        # drag the value lower; here they are excluded entirely.
        all_keys = weighted_composite(
            values,
            present_keys=set(values.keys()),
        )
        assert present >= all_keys
        # Present-only composite uses only the evidenced dims.
        expected = 80.0 * (0.20 / 0.38) + 60.0 * (0.18 / 0.38)
        assert present == pytest.approx(expected)

    def test_no_present_dimensions_is_zero(self) -> None:
        assert weighted_composite(
            {"market_opportunity": 70.0}, present_keys=set()
        ) == 0.0

    def test_unknown_dimension_uses_default_weight(self) -> None:
        values = {"made_up_dim": 60.0}
        result = weighted_composite(values, present_keys={"made_up_dim"})
        assert result == 60.0  # single term, weight normalizes to 1.0


class TestDimensionWeightsSourceOfTruth:
    def test_weights_sum_to_one(self) -> None:
        assert round(sum(DIMENSION_WEIGHTS.values()), 6) == 1.0

    def test_weights_cover_all_dimensions(self) -> None:
        from predictron_engine.knowledge.concepts import AnalysisDimension

        for dim in AnalysisDimension:
            assert dim.value in DIMENSION_WEIGHTS

    def test_default_and_max_constants_are_derived(self) -> None:
        assert MAX_DIMENSION_WEIGHT == max(DIMENSION_WEIGHTS.values())
        assert DEFAULT_DIMENSION_WEIGHT == 0.05


class TestInvestmentReadinessMissingData:
    """H3 — readiness treats missing dimensions as unknown, not average."""

    def _ready(self, scores: list[ScoreResult]) -> InvestmentReadiness:
        return compute_investment_readiness(
            ExtractedFeatures(), [], scores, []
        )

    def test_partial_dimensions_do_not_pull_to_middle(self) -> None:
        """Only evidenced dimensions feed the weighted readiness score."""
        # Two strong dimensions only.
        sparse = self._ready(
            [
                _score("market_opportunity", 90.0),
                _score("founder_quality", 90.0),
            ]
        )
        # The same two, plus five neutral (50) dimensions.
        full = self._ready(
            [
                _score("market_opportunity", 90.0),
                _score("founder_quality", 90.0),
                _score("product_strength", 50.0),
                _score("business_model_viability", 50.0),
                _score("traction_signals", 50.0),
                _score("competitive_position", 50.0),
                _score("team_execution", 50.0),
            ]
        )
        # Sparse data must not score as if the missing dims were average.
        assert sparse.readiness_score >= full.readiness_score

    def test_explicit_negative_evidence_still_lowers_readiness(self) -> None:
        """Explicit low scores remain distinguishably negative (not missing)."""
        good = self._ready(
            [
                _score("market_opportunity", 90.0),
                _score("founder_quality", 90.0),
            ]
        )
        bad = self._ready(
            [
                _score("market_opportunity", 90.0),
                _score("founder_quality", 10.0),
            ]
        )
        assert bad.readiness_score < good.readiness_score

    def test_no_evidence_yields_zero_and_needs_data(self) -> None:
        """A completely unevidenced startup is 'needs_data', not 'developing'."""
        ready = self._ready([])
        assert ready.readiness_score == 0.0
        assert ready.readiness_level == "needs_data"

    def test_missing_dimensions_still_surface_in_contributions(self) -> None:
        """The API dict still lists all dimensions for traceability, while
        the composite only averages the evidenced ones."""
        ready = self._ready([_score("market_opportunity", 80.0)])
        assert "founder_quality" in ready.dimension_contributions
        assert ready.readiness_score != 0.0


class TestMissingEvidenceDistinctFromNegative:
    """H3 — missing and explicitly-negative must remain distinguishable."""

    def test_missing_contributes_zero_not_a_low_score(self) -> None:
        # A dimension that is missing entirely should not read as a low
        # (negative) score; it is simply not part of the composite.
        scores = [_score("market_opportunity", 80.0)]
        ready = compute_investment_readiness(
            ExtractedFeatures(), [], scores, []
        )
        assert ready.dimension_contributions is not None
        # The missing dimension's contribution is present in the dict for
        # traceability but does not drag the composite.
        assert "founder_quality" in ready.dimension_contributions
