"""Tests for ProductStrengthScorer."""

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation, ScoreResult
from predictron_engine.scoring.scoring_engine import ProductStrengthScorer


class TestProductStrengthScorer:
    def setup_method(self):
        self.scorer = ProductStrengthScorer()

    def test_returns_score_result(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert isinstance(result, ScoreResult)

    def test_dimension_is_product_strength(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert result.dimension == "product_strength"

    def test_base_score_is_50(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert result.score == 50.0

    def test_product_type_platform_bonus(self):
        features = ExtractedFeatures(product_type="platform")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_product_type_infrastructure_bonus(self):
        features = ExtractedFeatures(product_type="infrastructure")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_ai_native_bonus(self):
        features = ExtractedFeatures(ai_orientation="ai_native")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_ai_enabled_bonus(self):
        features = ExtractedFeatures(ai_orientation="ai_enabled")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_non_ai_neutral(self):
        features = ExtractedFeatures(ai_orientation="non_ai")
        result = self.scorer.score(features, [])
        assert result.score == 50.0

    def test_capabilities_bonus(self):
        features = ExtractedFeatures(primary_capabilities=["cap1", "cap2", "cap3"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_feature_signals_bonus(self):
        features = ExtractedFeatures(feature_signals=["f1", "f2"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_integrations_bonus(self):
        features = ExtractedFeatures(integration_ecosystem=["int1", "int2"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_technical_complexity_high(self):
        features = ExtractedFeatures(technical_complexity="high")
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_technical_complexity_low_penalty(self):
        features = ExtractedFeatures(technical_complexity="low")
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_innovation_bonus(self):
        features = ExtractedFeatures(innovation_signals=["patent", "novel"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_defensibility_bonus(self):
        features = ExtractedFeatures(defensibility_signals=["regulatory_moat"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_scalability_bonus(self):
        features = ExtractedFeatures(scalability_indicators=["horizontal"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_differentiation_bonus(self):
        features = ExtractedFeatures(differentiation_signals=["unique"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_product_risk_penalty(self):
        features = ExtractedFeatures(product_risk=["risk1"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_technology_risk_penalty(self):
        features = ExtractedFeatures(technology_risk=["risk1"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_product_confidence_high(self):
        features = ExtractedFeatures(product_confidence=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_tech_confidence_high(self):
        features = ExtractedFeatures(technology_confidence=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_observations_affect_score(self):
        features = ExtractedFeatures()
        obs = [
            Observation(
                dimension="product_strength",
                category="product_assessment",
                statement="Strong product.",
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
                dimension="product_strength",
                category="test",
                statement="Product obs.",
                evidence=[],
                confidence=0.5,
                importance=0.5,
                source_rule="Test",
            )
        ]
        result = self.scorer.score(features, obs)
        assert "Product obs." in result.evidence

    def test_score_in_valid_range(self):
        features = ExtractedFeatures(
            product_type="platform",
            ai_orientation="ai_native",
            primary_capabilities=["a", "b", "c"],
            feature_signals=["f1", "f2"],
            integration_ecosystem=["i1"],
            technical_complexity="high",
            innovation_signals=["n1", "n2"],
            defensibility_signals=["d1"],
            scalability_indicators=["s1"],
            differentiation_signals=["u1", "u2"],
            product_confidence=0.9,
            technology_confidence=0.8,
        )
        result = self.scorer.score(features, [])
        assert 0.0 <= result.score <= 100.0

    def test_rationale_mentions_dimensions(self):
        features = ExtractedFeatures(product_type="platform")
        result = self.scorer.score(features, [])
        assert "signal dimensions" in result.rationale
