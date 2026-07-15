"""Tests for the DefaultReasoningEngine."""

from predictron_engine.models.report import Observation
from predictron_engine.reasoning.reasoning_engine import DefaultReasoningEngine


class TestDefaultReasoningEngine:
    """Unit tests for the reasoning stage."""

    def test_reason_returns_observations(self, sample_features, sample_evidence):
        engine = DefaultReasoningEngine()
        result = engine.reason(sample_features, sample_evidence)

        assert isinstance(result, list)
        assert all(isinstance(o, Observation) for o in result)

    def test_observations_have_category(self, sample_features, sample_evidence):
        engine = DefaultReasoningEngine()
        result = engine.reason(sample_features, sample_evidence)

        for obs in result:
            assert obs.category != ""

    def test_observations_have_source_rule(self, sample_features, sample_evidence):
        engine = DefaultReasoningEngine()
        result = engine.reason(sample_features, sample_evidence)

        for obs in result:
            assert obs.source_rule != ""

    def test_observations_have_importance(
        self, sample_features, sample_evidence
    ):
        engine = DefaultReasoningEngine()
        result = engine.reason(sample_features, sample_evidence)

        for obs in result:
            assert 0.0 <= obs.importance <= 1.0

    def test_industry_observation_generated(
        self, sample_features, sample_evidence
    ):
        engine = DefaultReasoningEngine()
        result = engine.reason(sample_features, sample_evidence)

        industry_obs = [
            o for o in result if "industry" in o.statement.lower()
        ]
        assert len(industry_obs) > 0

    def test_business_model_observation_generated(
        self, sample_features, sample_evidence
    ):
        engine = DefaultReasoningEngine()
        result = engine.reason(sample_features, sample_evidence)

        bm_obs = [
            o for o in result if o.category == "business_model_assessment"
        ]
        assert len(bm_obs) > 0

    def test_data_quality_observation(self, sample_features, sample_evidence):
        engine = DefaultReasoningEngine()
        result = engine.reason(sample_features, sample_evidence)

        dq_obs = [o for o in result if o.dimension == "data_quality"]
        assert len(dq_obs) >= 1

    def test_minimal_features_fewer_observations(
        self, minimal_features
    ):
        engine = DefaultReasoningEngine()
        result = engine.reason(minimal_features, [])

        assert len(result) <= 5

    def test_observation_confidence_in_range(
        self, sample_features, sample_evidence
    ):
        engine = DefaultReasoningEngine()
        result = engine.reason(sample_features, sample_evidence)

        for obs in result:
            assert 0.0 <= obs.confidence <= 1.0

    def test_evidence_enriches_observations(self, sample_features, sample_evidence):
        engine = DefaultReasoningEngine()
        result = engine.reason(sample_features, sample_evidence)

        evidence_refs = [
            o for o in result if any(e.startswith("evidence:") for e in o.evidence)
        ]
        assert len(evidence_refs) > 0

    def test_no_evidence_still_works(self, sample_features):
        engine = DefaultReasoningEngine()
        result = engine.reason(sample_features, [])

        assert isinstance(result, list)
        assert len(result) > 0

    def test_custom_rules_injected(self, sample_features, sample_evidence):
        class CustomRule:
            def evaluate(self, features, evidence):
                return [
                    Observation(
                        dimension="custom",
                        category="custom_category",
                        statement="Custom observation",
                        evidence=["custom_field"],
                        confidence=0.99,
                        importance=0.8,
                        source_rule="CustomRule",
                    )
                ]

        engine = DefaultReasoningEngine(rules=[CustomRule()])
        result = engine.reason(sample_features, sample_evidence)

        assert len(result) == 1
        assert result[0].dimension == "custom"
        assert result[0].category == "custom_category"
        assert result[0].confidence == 0.99
        assert result[0].source_rule == "CustomRule"

    def test_empty_rules_produces_no_observations(
        self, sample_features, sample_evidence
    ):
        engine = DefaultReasoningEngine(rules=[])
        result = engine.reason(sample_features, sample_evidence)

        assert result == []

    def test_competition_observations_generated(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(
            competitive_moat_indicators=["proprietary_data"],
            market_concentration="concentrated",
        )
        engine = DefaultReasoningEngine()
        result = engine.reason(features, [])
        comp_obs = [
            o for o in result if o.dimension == "competitive_position"
        ]
        assert len(comp_obs) > 0

    def test_competition_rule_in_default_rules(self):
        from predictron_engine.reasoning.rules import DEFAULT_RULES

        rule_names = [r.name for r in DEFAULT_RULES]
        assert "competition_assessment" in rule_names

    def test_rule_failure_does_not_break_others(
        self, sample_features, sample_evidence
    ):
        class FailingRule:
            def evaluate(self, features, evidence):
                raise RuntimeError("Intentional failure")

        class GoodRule:
            def evaluate(self, features, evidence):
                return [
                    Observation(
                        dimension="test",
                        category="test_cat",
                        statement="Good observation",
                        confidence=0.5,
                        source_rule="GoodRule",
                    )
                ]

        engine = DefaultReasoningEngine(rules=[FailingRule(), GoodRule()])
        result = engine.reason(sample_features, sample_evidence)

        assert len(result) == 1
        assert result[0].statement == "Good observation"
