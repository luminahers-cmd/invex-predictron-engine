"""Tests for all domain recommendation strategies."""

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    DimensionAssessment,
    Observation,
    Recommendation,
)
from predictron_engine.recommendations.strategies.base import (
    DomainRecommendationStrategy,
)
from predictron_engine.recommendations.strategies.business_model import (
    BusinessModelStrategy,
)
from predictron_engine.recommendations.strategies.fundraising import (
    FundraisingStrategy,
)
from predictron_engine.recommendations.strategies.market import MarketStrategy
from predictron_engine.recommendations.strategies.risk import RiskStrategy
from predictron_engine.recommendations.strategies.team import TeamStrategy
from predictron_engine.recommendations.strategies.technology import (
    TechnologyStrategy,
)
from predictron_engine.recommendations.strategies.traction import TractionStrategy

ALL_STRATEGIES = [
    MarketStrategy,
    TeamStrategy,
    TechnologyStrategy,
    BusinessModelStrategy,
    TractionStrategy,
    RiskStrategy,
    FundraisingStrategy,
]


def _make_obs(dimension: str, confidence: float = 0.6) -> Observation:
    return Observation(
        dimension=dimension,
        category="test_category",
        statement="test statement",
        confidence=confidence,
        importance=0.5,
        source_rule="TestRule",
    )


def _make_assess(dimension: str, confidence: float = 0.5) -> DimensionAssessment:
    return DimensionAssessment(
        dimension=dimension,
        summary="test summary",
        rationale="test rationale",
        confidence=confidence,
    )


class TestProtocolCompliance:
    """All strategies must satisfy the DomainRecommendationStrategy protocol."""

    def test_all_strategies_satisfy_protocol(self):
        for strategy_cls in ALL_STRATEGIES:
            strategy = strategy_cls()
            assert isinstance(strategy, DomainRecommendationStrategy)

    def test_all_strategies_have_domain(self):
        for strategy_cls in ALL_STRATEGIES:
            strategy = strategy_cls()
            assert isinstance(strategy.domain, str)
            assert len(strategy.domain) > 0

    def test_all_strategies_have_generate(self):
        for strategy_cls in ALL_STRATEGIES:
            strategy = strategy_cls()
            assert callable(strategy.generate)


class TestRecommendationStructure:
    """All recommendations must have required fields."""

    FULL_FEATURES = ExtractedFeatures(
        industry="enterprise_saas",
        business_model="saas",
        has_revenue=True,
        data_completeness=0.5,
        has_pitch_deck=True,
        founder_profile_count=2,
        technology_stack=["python", "aws"],
        funding_stage="seed",
        founded_year=2020,
    )

    def test_all_strategies_produce_valid_recommendations(self):
        for strategy_cls in ALL_STRATEGIES:
            strategy = strategy_cls()
            result = strategy.generate(self.FULL_FEATURES, [], [])
            assert isinstance(result, list)
            for rec in result:
                assert isinstance(rec, Recommendation)
                assert rec.category != ""
                assert rec.action != ""
                assert rec.priority in ("high", "medium", "low")

    def test_all_recommendations_have_title(self):
        for strategy_cls in ALL_STRATEGIES:
            strategy = strategy_cls()
            result = strategy.generate(self.FULL_FEATURES, [], [])
            for rec in result:
                assert rec.title != ""

    def test_all_recommendations_have_description(self):
        for strategy_cls in ALL_STRATEGIES:
            strategy = strategy_cls()
            result = strategy.generate(self.FULL_FEATURES, [], [])
            for rec in result:
                assert rec.description != ""

    def test_all_recommendations_have_valid_confidence(self):
        for strategy_cls in ALL_STRATEGIES:
            strategy = strategy_cls()
            result = strategy.generate(self.FULL_FEATURES, [], [])
            for rec in result:
                assert 0.0 <= rec.confidence <= 1.0

    def test_all_recommendations_have_metadata(self):
        for strategy_cls in ALL_STRATEGIES:
            strategy = strategy_cls()
            result = strategy.generate(self.FULL_FEATURES, [], [])
            for rec in result:
                assert isinstance(rec.metadata, dict)
                assert "strategy" in rec.metadata


class TestMarketStrategy:
    """Tests for the MarketStrategy."""

    def test_domain(self):
        assert MarketStrategy().domain == "market_opportunity"

    def test_missing_industry_triggers_due_diligence(self):
        features = ExtractedFeatures(data_completeness=0.1)
        result = MarketStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert "Confirm Industry Classification" in titles

    def test_missing_geography_triggers_follow_up(self):
        features = ExtractedFeatures(
            industry="enterprise_saas", data_completeness=0.1
        )
        result = MarketStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert "Clarify Geographic Focus" in titles

    def test_market_observations_trigger_opportunity(self):
        features = ExtractedFeatures(
            industry="enterprise_saas",
            geography="north_america",
            data_completeness=0.5,
        )
        obs = [_make_obs("market_opportunity")]
        result = MarketStrategy().generate(features, obs, [])
        titles = [r.title for r in result]
        assert "Deep-Dive Market Analysis" in titles

    def test_no_recommendations_when_complete(self):
        features = ExtractedFeatures(
            industry="enterprise_saas",
            geography="north_america",
            data_completeness=0.8,
        )
        result = MarketStrategy().generate(features, [], [])
        assert result == []


class TestTeamStrategy:
    """Tests for the TeamStrategy."""

    def test_domain(self):
        assert TeamStrategy().domain == "founder_quality"

    def test_zero_founders_triggers_due_diligence(self):
        features = ExtractedFeatures(data_completeness=0.1)
        result = TeamStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert "Obtain Founder Information" in titles

    def test_single_founder_triggers_follow_up(self):
        features = ExtractedFeatures(
            founder_profile_count=1, data_completeness=0.3
        )
        result = TeamStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert "Expand Team Visibility" in titles

    def test_team_observations_trigger_opportunity(self):
        features = ExtractedFeatures(
            founder_profile_count=2, data_completeness=0.5
        )
        obs = [_make_obs("founder_quality")]
        result = TeamStrategy().generate(features, obs, [])
        titles = [r.title for r in result]
        assert "Team Deep-Dive Assessment" in titles


class TestTechnologyStrategy:
    """Tests for the TechnologyStrategy."""

    def test_domain(self):
        assert TechnologyStrategy().domain == "product_strength"

    def test_missing_tech_stack_triggers_due_diligence(self):
        features = ExtractedFeatures(data_completeness=0.1)
        result = TechnologyStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert "Identify Technology Stack" in titles

    def test_missing_pitch_deck_triggers_due_diligence(self):
        features = ExtractedFeatures(
            technology_stack=["python"], data_completeness=0.3
        )
        result = TechnologyStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert "Obtain Pitch Deck" in titles

    def test_tech_observations_trigger_opportunity(self):
        features = ExtractedFeatures(
            technology_stack=["python"],
            has_pitch_deck=True,
            data_completeness=0.5,
        )
        obs = [_make_obs("product_strength")]
        result = TechnologyStrategy().generate(features, obs, [])
        titles = [r.title for r in result]
        assert "Assess Technical Differentiation" in titles


class TestBusinessModelStrategy:
    """Tests for the BusinessModelStrategy."""

    def test_domain(self):
        assert BusinessModelStrategy().domain == "business_model_viability"

    def test_missing_business_model_triggers_due_diligence(self):
        features = ExtractedFeatures(data_completeness=0.1)
        result = BusinessModelStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert "Clarify Business Model" in titles

    def test_no_revenue_triggers_follow_up(self):
        features = ExtractedFeatures(
            business_model="saas", data_completeness=0.3
        )
        result = BusinessModelStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert "Validate Revenue Status" in titles

    def test_bm_observations_trigger_opportunity(self):
        features = ExtractedFeatures(
            business_model="saas",
            has_revenue=True,
            data_completeness=0.5,
        )
        obs = [_make_obs("business_model_viability")]
        result = BusinessModelStrategy().generate(features, obs, [])
        titles = [r.title for r in result]
        assert "Unit Economics Deep-Dive" in titles


class TestTractionStrategy:
    """Tests for the TractionStrategy."""

    def test_domain(self):
        assert TractionStrategy().domain == "traction_signals"

    def test_missing_funding_stage_triggers_due_diligence(self):
        features = ExtractedFeatures(data_completeness=0.1)
        result = TractionStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert "Identify Funding Stage" in titles

    def test_old_company_no_revenue_triggers_risk(self):
        features = ExtractedFeatures(
            founded_year=2018, data_completeness=0.3
        )
        result = TractionStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert "Revenue Validation for Mature Startup" in titles

    def test_traction_observations_trigger_opportunity(self):
        features = ExtractedFeatures(
            funding_stage="seed",
            has_revenue=True,
            data_completeness=0.5,
        )
        obs = [_make_obs("traction_signals")]
        result = TractionStrategy().generate(features, obs, [])
        titles = [r.title for r in result]
        assert "Traction Validation" in titles


class TestRiskStrategy:
    """Tests for the RiskStrategy."""

    def test_domain(self):
        assert RiskStrategy().domain == "competitive_position"

    def test_low_data_completeness_triggers_risk(self):
        features = ExtractedFeatures(data_completeness=0.2)
        result = RiskStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert "Improve Data Coverage" in titles

    def test_risk_observations_trigger_review(self):
        features = ExtractedFeatures(data_completeness=0.5)
        obs = [_make_obs("competitive_position")]
        result = RiskStrategy().generate(features, obs, [])
        titles = [r.title for r in result]
        assert "Risk Factor Review" in titles


class TestFundraisingStrategy:
    """Tests for the FundraisingStrategy."""

    def test_domain(self):
        assert FundraisingStrategy().domain == "fundraising"

    def test_funding_stage_triggers_benchmark(self):
        features = ExtractedFeatures(
            funding_stage="series_a", data_completeness=0.5
        )
        result = FundraisingStrategy().generate(features, [], [])
        titles = [r.title for r in result]
        assert any("Stage Benchmark Validation" in t for t in titles)

    def test_no_funding_stage_no_recommendation(self):
        features = ExtractedFeatures(data_completeness=0.5)
        result = FundraisingStrategy().generate(features, [], [])
        assert result == []

    def test_high_confidence_assessments_trigger_deep_dive(self):
        features = ExtractedFeatures(
            funding_stage="seed",
            founder_profile_count=2,
            has_pitch_deck=True,
            data_completeness=0.8,
        )
        assessments = [
            _make_assess("market_opportunity", 0.8),
            _make_assess("product_strength", 0.7),
        ]
        result = FundraisingStrategy().generate(features, [], assessments)
        titles = [r.title for r in result]
        assert "Investment Deep-Dive Readiness" in titles


class TestSupportingData:
    """Tests that recommendations correctly carry supporting data."""

    def test_market_obs_appears_in_recommendation(self):
        features = ExtractedFeatures(
            industry="enterprise_saas",
            geography="north_america",
            data_completeness=0.5,
        )
        obs = [_make_obs("market_opportunity", 0.7)]
        assess = [_make_assess("market_opportunity", 0.6)]
        result = MarketStrategy().generate(features, obs, assess)
        assert len(result) == 1
        assert len(result[0].supporting_observations) == 1
        assert result[0].supporting_observations[0].confidence == 0.7
        assert len(result[0].supporting_assessments) == 1

    def test_metadata_contains_strategy_name(self):
        features = ExtractedFeatures(data_completeness=0.1)
        result = MarketStrategy().generate(features, [], [])
        for rec in result:
            assert rec.metadata.get("strategy") == "market"

    def test_action_items_populated(self):
        features = ExtractedFeatures(
            industry="enterprise_saas",
            geography="north_america",
            data_completeness=0.5,
        )
        obs = [_make_obs("market_opportunity")]
        result = MarketStrategy().generate(features, obs, [])
        assert len(result) == 1
        assert len(result[0].action_items) > 0
