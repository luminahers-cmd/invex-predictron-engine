"""Tests for the DefaultScoringEngine with all real scorers."""

from predictron_engine.knowledge.concepts import AnalysisDimension
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import ScoreResult
from predictron_engine.scoring.scoring_engine import (
    BusinessModelScorer,
    CompetitionScorer,
    DefaultScoringEngine,
    FounderQualityScorer,
    MarketOpportunityScorer,
    ProductStrengthScorer,
    TeamExecutionScorer,
    TractionScorer,
)


class TestDefaultScoringEngineWithAllScorers:
    def test_all_dimensions_scored(self):
        engine = DefaultScoringEngine()
        features = ExtractedFeatures()
        result = engine.score(features, [])
        dims = {s.dimension for s in result}
        expected = {dim.value for dim in AnalysisDimension}
        assert dims == expected

    def test_no_dimension_returns_placeholder_rationale(self):
        engine = DefaultScoringEngine()
        features = ExtractedFeatures()
        result = engine.score(features, [])
        for score in result:
            assert "Placeholder" not in score.rationale

    def test_each_dimension_has_real_scorer(self):
        engine = DefaultScoringEngine()
        scorers = {type(s) for s in engine._scorers}
        expected = {
            MarketOpportunityScorer,
            FounderQualityScorer,
            ProductStrengthScorer,
            BusinessModelScorer,
            TractionScorer,
            CompetitionScorer,
            TeamExecutionScorer,
        }
        assert scorers == expected

    def test_market_scorer_responds_to_maturity(self):
        engine = DefaultScoringEngine()
        features_emerging = ExtractedFeatures(market_maturity="emerging")
        features_saturated = ExtractedFeatures(market_maturity="saturated")
        result_e = engine.score(features_emerging, [])
        result_s = engine.score(features_saturated, [])
        market_e = [s for s in result_e if s.dimension == "market_opportunity"][0]
        market_s = [s for s in result_s if s.dimension == "market_opportunity"][0]
        assert market_e.score > market_s.score

    def test_founder_scorer_responds_to_profiles(self):
        engine = DefaultScoringEngine()
        features_zero = ExtractedFeatures(founder_profile_count=0)
        features_many = ExtractedFeatures(founder_profile_count=3)
        result_z = engine.score(features_zero, [])
        result_m = engine.score(features_many, [])
        founder_z = [s for s in result_z if s.dimension == "founder_quality"][0]
        founder_m = [s for s in result_m if s.dimension == "founder_quality"][0]
        assert founder_m.score > founder_z.score

    def test_product_scorer_responds_to_ai(self):
        engine = DefaultScoringEngine()
        features_native = ExtractedFeatures(ai_orientation="ai_native")
        features_non = ExtractedFeatures(ai_orientation="non_ai")
        result_n = engine.score(features_native, [])
        result_x = engine.score(features_non, [])
        prod_n = [s for s in result_n if s.dimension == "product_strength"][0]
        prod_x = [s for s in result_x if s.dimension == "product_strength"][0]
        assert prod_n.score > prod_x.score

    def test_business_model_scorer_responds_to_revenue(self):
        engine = DefaultScoringEngine()
        features_sub = ExtractedFeatures(revenue_model="subscription")
        features_ad = ExtractedFeatures(revenue_model="advertising")
        result_s = engine.score(features_sub, [])
        result_a = engine.score(features_ad, [])
        bm_s = [s for s in result_s if s.dimension == "business_model_viability"][0]
        bm_a = [s for s in result_a if s.dimension == "business_model_viability"][0]
        assert bm_s.score > bm_a.score

    def test_traction_scorer_responds_to_revenue(self):
        engine = DefaultScoringEngine()
        features_rev = ExtractedFeatures(has_revenue=True)
        features_no_rev = ExtractedFeatures(has_revenue=False)
        result_r = engine.score(features_rev, [])
        result_nr = engine.score(features_no_rev, [])
        trac_r = [s for s in result_r if s.dimension == "traction_signals"][0]
        trac_nr = [s for s in result_nr if s.dimension == "traction_signals"][0]
        assert trac_r.score > trac_nr.score

    def test_team_execution_scorer_responds_to_data_completeness(self):
        engine = DefaultScoringEngine()
        features_low = ExtractedFeatures(data_completeness=0.1)
        features_high = ExtractedFeatures(data_completeness=0.9)
        result_l = engine.score(features_low, [])
        result_h = engine.score(features_high, [])
        team_l = [s for s in result_l if s.dimension == "team_execution"][0]
        team_h = [s for s in result_h if s.dimension == "team_execution"][0]
        assert team_h.score > team_l.score

    def test_risk_signals_lower_scores(self):
        engine = DefaultScoringEngine()
        features_clean = ExtractedFeatures()
        features_risky = ExtractedFeatures(
            market_risk=["risk1"],
            founder_risk=["risk2"],
            product_risk=["risk3"],
            business_model_risk=["risk4"],
            traction_risk=["risk5"],
            operational_risk=["risk6"],
        )
        result_clean = engine.score(features_clean, [])
        result_risky = engine.score(features_risky, [])
        for dim in [
            "market_opportunity",
            "founder_quality",
            "product_strength",
            "business_model_viability",
            "traction_signals",
            "team_execution",
        ]:
            clean_score = [s for s in result_clean if s.dimension == dim][0].score
            risky_score = [s for s in result_risky if s.dimension == dim][0].score
            assert risky_score <= clean_score

    def test_all_scores_in_valid_range(self):
        engine = DefaultScoringEngine()
        features = ExtractedFeatures(
            market_maturity="growth",
            enterprise_orientation="enterprise",
            industry_confidence=0.8,
            market_signals=["s1"],
            geography="north_america",
            founder_team_type="technical",
            founder_profile_count=2,
            domain_expertise_signals=["d1"],
            leadership_roles=["CEO"],
            product_type="platform",
            ai_orientation="ai_enabled",
            primary_capabilities=["a", "b"],
            revenue_model="subscription",
            pricing_model="tiered",
            recurring_revenue_signal="recurring",
            business_model_maturity="growth",
            has_revenue=True,
            funding_amount_signals=["$5M"],
            investor_signals=["VC"],
            arr_mrr_signals=["$2M ARR"],
            growth_signals=["15% MoM"],
            team_size_indicator="11-50",
            execution_signals=["e1"],
            engineering_strength="moderate",
            engineering_maturity="developing",
            data_completeness=0.85,
        )
        result = engine.score(features, [])
        for score in result:
            assert 0.0 <= score.score <= 100.0

    def test_custom_scorer_injection_still_works(self):
        class CustomScorer:
            def score(self, features, observations):
                return ScoreResult(
                    dimension="custom",
                    score=75.0,
                    rationale="custom",
                    evidence=[],
                )

        engine = DefaultScoringEngine(scorers=[CustomScorer()])
        result = engine.score(ExtractedFeatures(), [])
        assert len(result) == 1
        assert result[0].score == 75.0
        assert result[0].dimension == "custom"

    def test_empty_scorers_produces_no_scores(self):
        engine = DefaultScoringEngine(scorers=[])
        result = engine.score(ExtractedFeatures(), [])
        assert result == []

    def test_competition_scorer_still_works(self):
        engine = DefaultScoringEngine()
        features = ExtractedFeatures(
            market_concentration="fragmented",
            competitive_density="sparse",
        )
        result = engine.score(features, [])
        comp = [s for s in result if s.dimension == "competitive_position"]
        assert len(comp) == 1
        assert comp[0].score > 50.0
