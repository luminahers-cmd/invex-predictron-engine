"""Tests for MarketOpportunityScorer."""

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation, ScoreResult
from predictron_engine.scoring.scoring_engine import MarketOpportunityScorer


class TestMarketOpportunityScorer:
    def setup_method(self):
        self.scorer = MarketOpportunityScorer()

    def test_returns_score_result(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert isinstance(result, ScoreResult)

    def test_dimension_is_market_opportunity(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert result.dimension == "market_opportunity"

    def test_base_score_is_50(self):
        features = ExtractedFeatures(data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 50.0

    def test_score_in_valid_range(self):
        features = ExtractedFeatures(
            market_maturity="emerging",
            enterprise_orientation="enterprise",
            industry_confidence=0.9,
            market_signals=["signal1", "signal2"],
            market_characteristics=["char1"],
            geography="north_america",
        )
        result = self.scorer.score(features, [])
        assert 0.0 <= result.score <= 100.0

    def test_maturity_emerging_bonus(self):
        features = ExtractedFeatures(market_maturity="emerging", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 60.0

    def test_maturity_growth_bonus(self):
        features = ExtractedFeatures(market_maturity="growth", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 55.0

    def test_maturity_mature_penalty(self):
        features = ExtractedFeatures(market_maturity="mature", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 45.0

    def test_maturity_saturated_penalty(self):
        features = ExtractedFeatures(market_maturity="saturated", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 35.0

    def test_orientation_enterprise_bonus(self):
        features = ExtractedFeatures(enterprise_orientation="enterprise", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 53.0

    def test_orientation_hybrid_small_bonus(self):
        features = ExtractedFeatures(enterprise_orientation="hybrid", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 51.0

    def test_industry_confidence_high(self):
        features = ExtractedFeatures(industry_confidence=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_industry_confidence_low(self):
        features = ExtractedFeatures(industry_confidence=0.1)
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_market_signals_count(self):
        features = ExtractedFeatures(market_signals=["a", "b", "c"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_market_characteristics_count(self):
        features = ExtractedFeatures(market_characteristics=["a", "b"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_geography_north_america(self):
        features = ExtractedFeatures(geography="north_america", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 53.0

    def test_market_risk_penalty(self):
        features = ExtractedFeatures(market_risk=["risk1", "risk2"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_observations_affect_score(self):
        features = ExtractedFeatures()
        obs = [
            Observation(
                dimension="market_opportunity",
                category="market_context",
                statement="Strong market.",
                evidence=[],
                confidence=0.8,
                importance=0.9,
                source_rule="Test",
            )
        ]
        result = self.scorer.score(features, obs)
        assert result.score > 50.0

    def test_evidence_from_observations(self):
        features = ExtractedFeatures()
        obs = [
            Observation(
                dimension="market_opportunity",
                category="test",
                statement="Test obs.",
                evidence=[],
                confidence=0.5,
                importance=0.5,
                source_rule="Test",
            )
        ]
        result = self.scorer.score(features, obs)
        assert "Test obs." in result.evidence

    def test_data_completeness_dampens(self):
        low_dc = ExtractedFeatures(
            market_maturity="emerging", data_completeness=0.1
        )
        high_dc = ExtractedFeatures(
            market_maturity="emerging", data_completeness=0.9
        )
        low_result = self.scorer.score(low_dc, [])
        high_result = self.scorer.score(high_dc, [])
        assert low_result.score < high_result.score

    def test_rationale_mentions_dimensions(self):
        features = ExtractedFeatures(market_maturity="emerging")
        result = self.scorer.score(features, [])
        assert "signal dimensions" in result.rationale
