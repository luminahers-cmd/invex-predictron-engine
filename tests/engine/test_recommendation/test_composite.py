"""Tests for CompositeRecommendationEngine."""

import pytest

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    DimensionAssessment,
    InvestmentReadiness,
    Observation,
    Recommendation,
    ScoreResult,
)
from predictron_engine.recommendations.composite import (
    CompositeRecommendationEngine,
    _gap_aware_rank,
)
from predictron_engine.recommendations.strategies.base import (
    DomainRecommendationStrategy,
)


class TestCompositeRecommendationEngine:
    """Unit tests for the CompositeRecommendationEngine."""

    def test_default_strategies_loaded(self):
        engine = CompositeRecommendationEngine()
        assert len(engine._strategies) == 7

    def test_all_strategies_are_protocol_compliant(self):
        engine = CompositeRecommendationEngine()
        for strategy in engine._strategies:
            assert isinstance(strategy, DomainRecommendationStrategy)

    def test_empty_strategies_produces_no_recommendations(self):
        engine = CompositeRecommendationEngine(strategies=[])
        features = ExtractedFeatures(data_completeness=0.5)
        result = engine.recommend(features, [], [])
        assert result == []

    def test_recommend_returns_list(self):
        engine = CompositeRecommendationEngine()
        features = ExtractedFeatures(data_completeness=0.5)
        result = engine.recommend(features, [], [])
        assert isinstance(result, list)

    def test_custom_strategy_injected(self):
        class CustomStrategy:
            @property
            def domain(self) -> str:
                return "custom"

            def generate(self, features, observations, assessments):
                from predictron_engine.models.report import Recommendation

                return [
                    Recommendation(
                        category="custom",
                        action="Custom action",
                        priority="high",
                        rationale="Custom rationale",
                        title="Custom Title",
                        description="Custom description",
                        metadata={"strategy": "custom"},
                    )
                ]

        engine = CompositeRecommendationEngine(strategies=[CustomStrategy()])
        features = ExtractedFeatures(data_completeness=0.5)
        result = engine.recommend(features, [], [])
        assert len(result) == 1
        assert result[0].category == "custom"

    def test_multiple_strategies_combined(self):
        class Strategy1:
            @property
            def domain(self) -> str:
                return "s1"

            def generate(self, features, observations, assessments):
                from predictron_engine.models.report import Recommendation

                return [
                    Recommendation(
                        category="c1",
                        action="a1",
                        title="t1",
                        description="d1",
                        metadata={"strategy": "s1"},
                    )
                ]

        class Strategy2:
            @property
            def domain(self) -> str:
                return "s2"

            def generate(self, features, observations, assessments):
                from predictron_engine.models.report import Recommendation

                return [
                    Recommendation(
                        category="c2",
                        action="a2",
                        title="t2",
                        description="d2",
                        metadata={"strategy": "s2"},
                    ),
                    Recommendation(
                        category="c2",
                        action="a2b",
                        title="t2b",
                        description="d2b",
                        metadata={"strategy": "s2"},
                    ),
                ]

        engine = CompositeRecommendationEngine(strategies=[Strategy1(), Strategy2()])
        features = ExtractedFeatures(data_completeness=0.5)
        result = engine.recommend(features, [], [])
        assert len(result) == 3

    def test_failing_strategy_skipped_gracefully(self):
        class GoodStrategy:
            @property
            def domain(self) -> str:
                return "good"

            def generate(self, features, observations, assessments):
                from predictron_engine.models.report import Recommendation

                return [
                    Recommendation(
                        category="ok",
                        action="do something",
                        title="Do Something",
                        description="Desc",
                        metadata={"strategy": "good"},
                    )
                ]

        class BadStrategy:
            @property
            def domain(self) -> str:
                return "bad"

            def generate(self, features, observations, assessments):
                raise RuntimeError("boom")

        engine = CompositeRecommendationEngine(strategies=[BadStrategy(), GoodStrategy()])
        features = ExtractedFeatures(data_completeness=0.5)
        result = engine.recommend(features, [], [])
        assert len(result) == 1
        assert result[0].category == "ok"

    def test_assessments_passed_to_strategies(self):
        received_assessments: list = []

        class SpyStrategy:
            @property
            def domain(self) -> str:
                return "spy"

            def generate(self, features, observations, assessments):
                received_assessments.extend(assessments)
                return []

        engine = CompositeRecommendationEngine(strategies=[SpyStrategy()])
        features = ExtractedFeatures(data_completeness=0.5)
        assessments = [
            DimensionAssessment(
                dimension="d1",
                summary="s",
                rationale="r",
                confidence=0.5,
            )
        ]
        engine.recommend(features, [], [], assessments)
        assert len(received_assessments) == 1

    def test_observations_passed_to_strategies(self):
        received_obs: list = []

        class SpyStrategy:
            @property
            def domain(self) -> str:
                return "spy"

            def generate(self, features, observations, assessments):
                received_obs.extend(observations)
                return []

        engine = CompositeRecommendationEngine(strategies=[SpyStrategy()])
        features = ExtractedFeatures(data_completeness=0.5)
        observations = [
            Observation(
                dimension="d1",
                category="c",
                statement="s",
                confidence=0.5,
                importance=0.5,
                source_rule="r",
            )
        ]
        engine.recommend(features, observations, [])
        assert len(received_obs) == 1

    def test_default_strategies_domain_coverage(self):
        engine = CompositeRecommendationEngine()
        domains = {s.domain for s in engine._strategies}
        expected = {
            "market_opportunity",
            "founder_quality",
            "product_strength",
            "business_model_viability",
            "traction_signals",
            "competitive_position",
            "fundraising",
        }
        assert domains == expected


class TestCompositeIntegration:
    """Integration tests with real features and observations."""

    def test_full_features_generates_recommendations(self):
        features = ExtractedFeatures(
            industry="enterprise_saas",
            business_model="saas",
            has_revenue=True,
            funding_stage="seed",
            data_completeness=0.5,
            has_pitch_deck=True,
            founder_profile_count=2,
        )
        obs = [
            Observation(
                dimension="market_opportunity",
                category="market_context",
                statement="Enterprise SaaS market exceeds $500B.",
                confidence=0.7,
                importance=0.8,
                source_rule="MarketContextRule",
            ),
            Observation(
                dimension="business_model_viability",
                category="business_model_assessment",
                statement="Startup follows a saas business model.",
                confidence=0.6,
                importance=0.7,
                source_rule="BusinessModelContextRule",
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

        engine = CompositeRecommendationEngine()
        result = engine.recommend(features, obs, [], assessments)

        assert len(result) > 0
        for rec in result:
            assert rec.category != ""
            assert rec.title != ""
            assert 0.0 <= rec.confidence <= 1.0

    def test_minimal_features_generates_due_diligence(self):
        features = ExtractedFeatures(data_completeness=0.1)
        engine = CompositeRecommendationEngine()
        result = engine.recommend(features, [], [])

        assert len(result) > 0
        categories = {r.category for r in result}
        assert "due_diligence" in categories or "follow_up" in categories


class TestGapAwareOrdering:
    """Tests for gap-aware recommendation ordering (Sprint P8C)."""

    @staticmethod
    def _rec(
        action: str,
        priority: str = "medium",
        confidence: float = 0.5,
        domain: str = "",
    ) -> tuple[str, Recommendation]:
        rec = Recommendation(
            category="due_diligence",
            action=action,
            priority=priority,
            rationale="r",
            confidence=confidence,
        )
        return (domain, rec)

    def _readiness(self, contributions: dict[str, float]) -> InvestmentReadiness:
        overall = sum(contributions.values()) / len(contributions)
        return InvestmentReadiness(
            readiness_score=round(overall, 2),
            dimension_contributions=contributions,
        )

    def test_gap_aware_ordering_prioritizes_largest_weighted_gap(self):
        """0.20-weight market at 55 (gap 0.45) outranks 0.15-weight
        product at 75 (gap 0.1875), so market recommendation sorts first."""
        scores = [
            ScoreResult(dimension="market_opportunity", score=55.0),
            ScoreResult(dimension="product_strength", score=75.0),
        ]
        readiness = self._readiness({"market_opportunity": 55.0, "product_strength": 75.0})
        collected = [
            self._rec("product action", domain="product_strength"),
            self._rec("market action", domain="market_opportunity"),
        ]

        ordered = CompositeRecommendationEngine._order_gap_aware(collected, scores, readiness)
        assert [r.action for r in ordered] == ["market action", "product action"]

    def test_gap_beats_static_priority(self):
        """A low-priority rec on a high-gap dimension ranks above a
        high-priority rec on a near-healthy dimension."""
        scores = [
            ScoreResult(dimension="market_opportunity", score=55.0),
            ScoreResult(dimension="product_strength", score=90.0),
        ]
        readiness = self._readiness({"market_opportunity": 55.0, "product_strength": 90.0})
        collected = [
            self._rec("product high", priority="high", domain="product_strength"),
            self._rec("market low", priority="low", domain="market_opportunity"),
        ]

        ordered = CompositeRecommendationEngine._order_gap_aware(collected, scores, readiness)
        assert ordered[0].action == "market low"

    def test_readiness_contributions_override_scores(self):
        scores = [
            ScoreResult(dimension="market_opportunity", score=90.0),
            ScoreResult(dimension="product_strength", score=75.0),
        ]
        readiness = self._readiness({"market_opportunity": 55.0, "product_strength": 75.0})
        collected = [
            self._rec("product action", domain="product_strength"),
            self._rec("market action", domain="market_opportunity"),
        ]

        ordered = CompositeRecommendationEngine._order_gap_aware(collected, scores, readiness)
        assert ordered[0].action == "market action"

    def test_unknown_dimension_falls_back_to_overall_gap(self):
        """Recommendations without a canonical dimension or supporting
        assessments fall back to the overall readiness gap."""
        scores: list[ScoreResult] = []
        readiness = InvestmentReadiness(
            readiness_score=65.0,
            dimension_contributions={},
        )
        collected = [self._rec("custom rec", domain="custom")]

        [rec] = CompositeRecommendationEngine._order_gap_aware(collected, scores, readiness)
        rank = _gap_aware_rank("custom", rec, scores, readiness)
        assert rank[0] == pytest.approx(0.35)
