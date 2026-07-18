"""Tests for BusinessModelScorer."""

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation, ScoreResult
from predictron_engine.scoring.scoring_engine import BusinessModelScorer


class TestBusinessModelScorer:
    def setup_method(self):
        self.scorer = BusinessModelScorer()

    def test_returns_score_result(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert isinstance(result, ScoreResult)

    def test_dimension_is_business_model(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert result.dimension == "business_model_viability"

    def test_base_score_is_50(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert result.score == 50.0

    def test_revenue_subscription_bonus(self):
        features = ExtractedFeatures(revenue_model="subscription")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_revenue_advertising_penalty(self):
        features = ExtractedFeatures(revenue_model="advertising")
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_pricing_tiered_bonus(self):
        features = ExtractedFeatures(pricing_model="tiered")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_pricing_enterprise_bonus(self):
        features = ExtractedFeatures(pricing_model="enterprise_contract")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_acquisition_product_led_bonus(self):
        features = ExtractedFeatures(customer_acquisition_model="product_led")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_distribution_direct_bonus(self):
        features = ExtractedFeatures(distribution_model="direct")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_recurring_revenue_bonus(self):
        features = ExtractedFeatures(recurring_revenue_signal="recurring")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_recurring_revenue_transactional_penalty(self):
        features = ExtractedFeatures(recurring_revenue_signal="transactional")
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_model_maturity_mature_bonus(self):
        features = ExtractedFeatures(business_model_maturity="mature")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_model_maturity_nascent_penalty(self):
        features = ExtractedFeatures(business_model_maturity="nascent")
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_unit_economics_bonus(self):
        features = ExtractedFeatures(unit_economics_indicators=["ltv_cac", "margins"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_network_effects_bonus(self):
        features = ExtractedFeatures(network_effects_signals=["direct", "indirect"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_platform_bonus(self):
        features = ExtractedFeatures(platform_characteristics=["two_sided"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_switching_costs_bonus(self):
        features = ExtractedFeatures(switching_cost_indicators=["lock_in"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_model_confidence_high(self):
        features = ExtractedFeatures(business_model_confidence=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_business_model_risk_penalty(self):
        features = ExtractedFeatures(business_model_risk=["risk1", "risk2"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_observations_affect_score(self):
        features = ExtractedFeatures()
        obs = [
            Observation(
                dimension="business_model_viability",
                category="business_model_assessment",
                statement="Strong model.",
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
                dimension="business_model_viability",
                category="test",
                statement="BM obs.",
                evidence=[],
                confidence=0.5,
                importance=0.5,
                source_rule="Test",
            )
        ]
        result = self.scorer.score(features, obs)
        assert "BM obs." in result.evidence

    def test_score_in_valid_range(self):
        features = ExtractedFeatures(
            revenue_model="subscription",
            pricing_model="tiered",
            customer_acquisition_model="product_led",
            distribution_model="direct",
            recurring_revenue_signal="recurring",
            business_model_maturity="mature",
            unit_economics_indicators=["a", "b"],
            network_effects_signals=["n1"],
            platform_characteristics=["p1"],
            switching_cost_indicators=["s1"],
            business_model_confidence=0.9,
        )
        result = self.scorer.score(features, [])
        assert 0.0 <= result.score <= 100.0

    def test_rationale_mentions_dimensions(self):
        features = ExtractedFeatures(revenue_model="subscription")
        result = self.scorer.score(features, [])
        assert "signal dimensions" in result.rationale
