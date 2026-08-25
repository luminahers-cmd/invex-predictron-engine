"""Tests for the explainability module."""

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    ConfidenceAssessment,
    DimensionAssessment,
    EvidenceItem,
    Observation,
    Recommendation,
)
from predictron_engine.validation.explainability import ExplanationBuilder


class TestExplanationBuilder:
    """Tests for the ExplanationBuilder."""

    def test_explain_extraction(self):
        features = ExtractedFeatures(
            industry="fintech",
            business_model="saas",
            data_completeness=0.6,
        )
        builder = ExplanationBuilder()
        result = builder.explain_extraction(features, "TestCo")
        assert result.stage == "extract"
        assert result.artifact_type == "ExtractedFeatures"
        assert "TestCo" in result.what_happened
        assert result.confidence == 0.6
        assert isinstance(result.missing_information, list)

    def test_explain_extraction_low_completeness(self):
        features = ExtractedFeatures(data_completeness=0.05)
        builder = ExplanationBuilder()
        result = builder.explain_extraction(features, "EmptyCo")
        assert result.confidence == 0.05
        assert len(result.missing_information) > 0

    def test_explain_evidence(self):
        features = ExtractedFeatures(
            industry="fintech", data_completeness=0.5
        )
        evidence = [
            EvidenceItem(
                domain="industry",
                category="market_size",
                statement="Fintech market is $500B",
                source="test",
            ),
        ]
        builder = ExplanationBuilder()
        result = builder.explain_evidence(evidence, features)
        assert result.stage == "evidence"
        assert "1 evidence items" in result.what_happened

    def test_explain_evidence_empty(self):
        features = ExtractedFeatures(data_completeness=0.5)
        builder = ExplanationBuilder()
        result = builder.explain_evidence([], features)
        assert "0 evidence items" in result.what_happened

    def test_explain_reasoning(self):
        features = ExtractedFeatures(data_completeness=0.6)
        evidence = [
            EvidenceItem(
                domain="industry",
                category="market",
                statement="Test",
                source="test",
            ),
        ]
        obs = [
            Observation(
                dimension="market_opportunity",
                category="market_context",
                statement="Market is large",
                confidence=0.7,
                importance=0.5,
                source_rule="MarketContextRule",
            ),
        ]
        builder = ExplanationBuilder()
        result = builder.explain_reasoning(obs, features, evidence)
        assert result.stage == "reason"
        assert "1 observations" in result.what_happened
        assert result.confidence == 0.7

    def test_explain_reasoning_empty(self):
        features = ExtractedFeatures(data_completeness=0.5)
        builder = ExplanationBuilder()
        result = builder.explain_reasoning([], features, [])
        assert result.confidence == 0.0

    def test_explain_evaluation(self):
        obs = [
            Observation(
                dimension="market_opportunity",
                category="test",
                statement="test",
                confidence=0.7,
                importance=0.5,
                source_rule="test",
            ),
        ]
        evidence = [
            EvidenceItem(
                domain="industry",
                category="test",
                statement="test",
                source="test",
            ),
        ]
        assessments = [
            DimensionAssessment(
                dimension="market_opportunity",
                summary="Strong market",
                rationale="Large TAM",
                confidence=0.7,
            ),
        ]
        builder = ExplanationBuilder()
        result = builder.explain_evaluation(assessments, obs, evidence)
        assert result.stage == "evaluate"
        assert result.confidence == 0.7

    def test_explain_recommendations(self):
        assessments = [
            DimensionAssessment(
                dimension="market_opportunity",
                summary="s",
                rationale="r",
                confidence=0.6,
            ),
        ]
        obs = [
            Observation(
                dimension="market_opportunity",
                category="test",
                statement="test",
                confidence=0.7,
                importance=0.5,
                source_rule="test",
            ),
        ]
        recs = [
            Recommendation(
                category="opportunity",
                action="Analyze market",
                priority="medium",
                title="Market Analysis",
                description="Do analysis",
                confidence=0.6,
                metadata={"strategy": "market"},
            ),
        ]
        builder = ExplanationBuilder()
        result = builder.explain_recommendations(recs, assessments, obs)
        assert result.stage == "recommend"
        assert "1 recommendations" in result.what_happened

    def test_explain_confidence(self):
        features = ExtractedFeatures(data_completeness=0.5)
        conf = [
            ConfidenceAssessment(
                dimension="market_opportunity",
                confidence=0.55,
                data_completeness=0.5,
            ),
            ConfidenceAssessment(
                dimension="product_strength",
                confidence=0.40,
                data_completeness=0.5,
            ),
        ]
        builder = ExplanationBuilder()
        result = builder.explain_confidence(conf, features)
        assert result.stage == "confidence"
        assert "2 dimensions" in result.what_happened

    def test_explain_confidence_with_low(self):
        features = ExtractedFeatures(data_completeness=0.2)
        conf = [
            ConfidenceAssessment(
                dimension="market_opportunity",
                confidence=0.15,
                data_completeness=0.2,
            ),
        ]
        builder = ExplanationBuilder()
        result = builder.explain_confidence(conf, features)
        assert len(result.missing_information) > 0


class TestExplainabilityDeterminism:
    """Regression: set comprehensions in explanations must be sorted."""

    def test_evidence_domains_sorted(self):
        features = ExtractedFeatures(data_completeness=0.5)
        evidence = [
            EvidenceItem(
                domain="geography", category="c",
                statement="s", source="src",
            ),
            EvidenceItem(
                domain="industry", category="c",
                statement="s", source="src",
            ),
            EvidenceItem(
                domain="business_model", category="c",
                statement="s", source="src",
            ),
        ]
        builder = ExplanationBuilder()
        result = builder.explain_evidence(evidence, features)
        assert "business_model, geography, industry" in result.what_happened

    def test_reasoning_rules_sorted(self):
        features = ExtractedFeatures(data_completeness=0.5)
        obs = [
            Observation(
                dimension="d", category="c", statement="s",
                confidence=0.7, importance=0.5, source_rule="ZebraRule",
            ),
            Observation(
                dimension="d", category="c", statement="s",
                confidence=0.7, importance=0.5, source_rule="AlphaRule",
            ),
        ]
        builder = ExplanationBuilder()
        result = builder.explain_reasoning(obs, features, [])
        assert "AlphaRule, ZebraRule" in result.upstream_contributors[0]

    def test_recommendation_categories_sorted(self):
        recs = [
            Recommendation(
                category="risk", action="a", priority="high",
            ),
            Recommendation(
                category="opportunity", action="b", priority="medium",
            ),
        ]
        builder = ExplanationBuilder()
        result = builder.explain_recommendations(recs, [], [])
        assert "opportunity, risk" in result.upstream_contributors[1]
