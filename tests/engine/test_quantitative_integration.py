"""Tests for quantitative intelligence integration (Sprint 15).

Covers:
  - QuantitativeSignalsRule reasoning
  - TractionScorer structured metric scoring
  - TeamExecutionScorer numeric team size
  - MarketOpportunityScorer market size
  - Decision engine quantitative rationale
  - Investment readiness quantitative strengths/concerns
  - Regression tests for ARR, runway, burn, NRR/churn, customer
    concentration, funding/valuation, capital efficiency, growth
"""

from __future__ import annotations

from predictron_engine.decision.decision_engine import DefaultDecisionEngine
from predictron_engine.evaluation.evaluators.traction import (
    TractionEvaluator,
    _build_quantitative_context,
)
from predictron_engine.evaluation.investment_readiness import (
    _collect_concerns,
    _collect_strengths,
)
from predictron_engine.knowledge.concepts import AnalysisDimension
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    ConfidenceAssessment,
    DimensionAssessment,
    Observation,
    ScoreResult,
)
from predictron_engine.reasoning.reasoning_engine import DefaultReasoningEngine
from predictron_engine.reasoning.rules.quantitative_signals import (
    QuantitativeSignalsRule,
)
from predictron_engine.scoring.scoring_engine import (
    DefaultScoringEngine,
    MarketOpportunityScorer,
    TeamExecutionScorer,
    TractionScorer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _features(**kwargs) -> ExtractedFeatures:
    """Create ExtractedFeatures with overrides."""
    defaults = dict(
        industry="enterprise_saas",
        business_model="saas",
        has_revenue=True,
        has_pitch_deck=True,
        founder_profile_count=2,
        data_completeness=0.7,
        description_length=300,
    )
    defaults.update(kwargs)
    return ExtractedFeatures(**defaults)


def _obs(
    dimension: str = "traction_signals",
    category: str = "test",
    confidence: float = 0.7,
    importance: float = 0.7,
) -> Observation:
    return Observation(
        dimension=dimension,
        category=category,
        statement=f"Test observation for {category}",
        evidence=[],
        confidence=confidence,
        importance=importance,
        source_rule="TestRule",
    )


# ===================================================================
# QuantitativeSignalsRule tests
# ===================================================================


class TestQuantitativeSignalsRule:
    """Tests for the QuantitativeSignalsRule reasoning rule."""

    def test_rule_registered_in_defaults(self) -> None:
        from predictron_engine.reasoning.rules import DEFAULT_RULES

        rule_names = [r.name for r in DEFAULT_RULES]
        assert "quantitative_signals" in rule_names

    def test_no_metrics_produces_no_observations(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features()
        result = rule.evaluate(features, [])
        assert result == []

    def test_arr_observation_generated(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(arr_usd=18_500_000)
        result = rule.evaluate(features, [])

        arr_obs = [o for o in result if o.category == "quantitative_arr"]
        assert len(arr_obs) == 1
        assert "18.5M" in arr_obs[0].statement
        assert arr_obs[0].confidence >= 0.7

    def test_arr_low(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(arr_usd=500_000)
        result = rule.evaluate(features, [])

        arr_obs = [o for o in result if o.category == "quantitative_arr"]
        assert len(arr_obs) == 1
        assert "below $1M" in arr_obs[0].statement

    def test_arr_very_high(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(arr_usd=75_000_000)
        result = rule.evaluate(features, [])

        arr_obs = [o for o in result if o.category == "quantitative_arr"]
        assert "strong revenue maturity" in arr_obs[0].statement

    def test_growth_rate_observation(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(growth_rate_pct=150.0)
        result = rule.evaluate(features, [])

        growth_obs = [o for o in result if o.category == "quantitative_growth"]
        assert len(growth_obs) == 1
        assert "exceptional" in growth_obs[0].statement

    def test_growth_rate_slow(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(growth_rate_pct=10.0)
        result = rule.evaluate(features, [])

        growth_obs = [o for o in result if o.category == "quantitative_growth"]
        assert "slow growth" in growth_obs[0].statement

    def test_nrr_observation_strong(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(nrr_pct=135.0)
        result = rule.evaluate(features, [])

        nrr_obs = [o for o in result if o.category == "quantitative_nrr"]
        assert len(nrr_obs) == 1
        assert "exceptional" in nrr_obs[0].statement

    def test_nrr_observation_contraction(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(nrr_pct=85.0)
        result = rule.evaluate(features, [])

        nrr_obs = [o for o in result if o.category == "quantitative_nrr"]
        assert "contraction" in nrr_obs[0].statement
        assert nrr_obs[0].importance >= 0.7

    def test_churn_excellent(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(churn_rate_pct=1.5)
        result = rule.evaluate(features, [])

        churn_obs = [o for o in result if o.category == "quantitative_churn"]
        assert "excellent" in churn_obs[0].statement

    def test_churn_severe(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(churn_rate_pct=15.0)
        result = rule.evaluate(features, [])

        churn_obs = [o for o in result if o.category == "quantitative_churn"]
        assert "severe" in churn_obs[0].statement
        assert churn_obs[0].importance >= 0.8

    def test_runway_critical(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(runway_months=4)
        result = rule.evaluate(features, [])

        runway_obs = [o for o in result if o.category == "quantitative_runway"]
        assert "critical" in runway_obs[0].statement
        assert runway_obs[0].importance >= 0.85

    def test_runway_comfortable(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(runway_months=30)
        result = rule.evaluate(features, [])

        runway_obs = [o for o in result if o.category == "quantitative_runway"]
        assert "strong" in runway_obs[0].statement

    def test_burn_rate_high(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(burn_rate_usd=7_000_000)
        result = rule.evaluate(features, [])

        burn_obs = [o for o in result if o.category == "quantitative_burn_rate"]
        assert "very high" in burn_obs[0].statement

    def test_burn_rate_low(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(burn_rate_usd=100_000)
        result = rule.evaluate(features, [])

        burn_obs = [o for o in result if o.category == "quantitative_burn_rate"]
        assert "capital efficiency" in burn_obs[0].statement

    def test_unit_economics_excellent(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(cac_usd=500, ltv_usd=5000)
        result = rule.evaluate(features, [])

        ue_obs = [
            o for o in result if o.category == "quantitative_unit_economics"
        ]
        assert len(ue_obs) == 1
        assert "10.0x" in ue_obs[0].statement
        assert "excellent" in ue_obs[0].statement

    def test_unit_economics_unsustainable(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(cac_usd=1000, ltv_usd=500)
        result = rule.evaluate(features, [])

        ue_obs = [
            o for o in result if o.category == "quantitative_unit_economics"
        ]
        assert "unsustainable" in ue_obs[0].statement

    def test_funding_valuation_strong(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(
            funding_amount_usd=10_000_000,
            valuation_usd=100_000_000,
        )
        result = rule.evaluate(features, [])

        fv_obs = [
            o for o in result
            if o.category == "quantitative_funding_valuation"
        ]
        assert len(fv_obs) == 1
        assert "10.0x" in fv_obs[0].statement
        assert "strong value creation" in fv_obs[0].statement

    def test_funding_valuation_underwater(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(
            funding_amount_usd=50_000_000,
            valuation_usd=30_000_000,
        )
        result = rule.evaluate(features, [])

        fv_obs = [
            o for o in result
            if o.category == "quantitative_funding_valuation"
        ]
        assert "below" in fv_obs[0].statement

    def test_customer_traction_scaling(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(customer_count=2500)
        result = rule.evaluate(features, [])

        cust_obs = [
            o for o in result if o.category == "quantitative_customer_count"
        ]
        assert "established" in cust_obs[0].statement

    def test_customer_traction_very_early(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(customer_count=3)
        result = rule.evaluate(features, [])

        cust_obs = [
            o for o in result if o.category == "quantitative_customer_count"
        ]
        assert "pre-launch" in cust_obs[0].statement

    def test_team_revenue_consistency(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(
            team_size_numeric=8,
            arr_usd=15_000_000,
        )
        result = rule.evaluate(features, [])

        team_obs = [
            o for o in result
            if o.category == "quantitative_team_efficiency"
        ]
        assert len(team_obs) == 1
        assert "exceptional capital efficiency" in team_obs[0].statement

    def test_market_size_large(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(market_size_usd=500_000_000_000)
        result = rule.evaluate(features, [])

        market_obs = [
            o for o in result if o.category == "quantitative_market_size"
        ]
        assert "500B" in market_obs[0].statement
        assert "large market" in market_obs[0].statement

    def test_mrr_observation(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(mrr_usd=1_500_000)
        result = rule.evaluate(features, [])

        mrr_obs = [o for o in result if o.category == "quantitative_mrr"]
        assert len(mrr_obs) == 1
        assert "1500K" in mrr_obs[0].statement

    def test_all_metrics_combined(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(
            arr_usd=18_500_000,
            mrr_usd=1_541_667,
            growth_rate_pct=120,
            nrr_pct=135,
            churn_rate_pct=3.0,
            runway_months=24,
            burn_rate_usd=800_000,
            cac_usd=800,
            ltv_usd=6000,
            funding_amount_usd=20_000_000,
            valuation_usd=150_000_000,
            customer_count=500,
            team_size_numeric=45,
            market_size_usd=100_000_000_000,
        )
        result = rule.evaluate(features, [])

        assert len(result) >= 10
        categories = {o.category for o in result}
        assert "quantitative_arr" in categories
        assert "quantitative_mrr" in categories
        assert "quantitative_growth" in categories
        assert "quantitative_nrr" in categories
        assert "quantitative_churn" in categories
        assert "quantitative_runway" in categories
        assert "quantitative_burn_rate" in categories
        assert "quantitative_unit_economics" in categories
        assert "quantitative_funding_valuation" in categories
        assert "quantitative_customer_count" in categories
        assert "quantitative_team_efficiency" in categories
        assert "quantitative_market_size" in categories

    def test_all_observations_have_source_rule(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(arr_usd=5_000_000, growth_rate_pct=60)
        result = rule.evaluate(features, [])
        for obs in result:
            assert obs.source_rule == "QuantitativeSignalsRule"

    def test_all_observations_in_valid_range(self) -> None:
        rule = QuantitativeSignalsRule()
        features = _features(
            arr_usd=18_500_000,
            growth_rate_pct=120,
            nrr_pct=135,
            churn_rate_pct=3.0,
            runway_months=24,
            burn_rate_usd=800_000,
            cac_usd=800,
            ltv_usd=6000,
            funding_amount_usd=20_000_000,
            valuation_usd=150_000_000,
            customer_count=500,
            team_size_numeric=45,
            market_size_usd=100_000_000_000,
        )
        result = rule.evaluate(features, [])
        for obs in result:
            assert 0.0 <= obs.confidence <= 1.0
            assert 0.0 <= obs.importance <= 1.0


# ===================================================================
# TractionScorer structured metric tests
# ===================================================================


class TestTractionScorerStructured:
    """Tests for TractionScorer structured metric scoring."""

    def test_high_arr_boosts_score(self) -> None:
        scorer = TractionScorer()
        features_low = _features(arr_usd=500_000)
        features_high = _features(arr_usd=20_000_000)

        low_result = scorer.score(features_low, [])
        high_result = scorer.score(features_high, [])

        assert high_result.score > low_result.score

    def test_strong_growth_boosts_score(self) -> None:
        scorer = TractionScorer()
        features_slow = _features(growth_rate_pct=10)
        features_fast = _features(growth_rate_pct=150)

        slow_result = scorer.score(features_slow, [])
        fast_result = scorer.score(features_fast, [])

        assert fast_result.score > slow_result.score

    def test_strong_nrr_boosts_score(self) -> None:
        scorer = TractionScorer()
        features_low_nrr = _features(nrr_pct=85)
        features_high_nrr = _features(nrr_pct=135)

        low_result = scorer.score(features_low_nrr, [])
        high_result = scorer.score(features_high_nrr, [])

        assert high_result.score > low_result.score

    def test_low_churn_boosts_score(self) -> None:
        scorer = TractionScorer()
        features_high_churn = _features(churn_rate_pct=12)
        features_low_churn = _features(churn_rate_pct=1.5)

        high_result = scorer.score(features_high_churn, [])
        low_result = scorer.score(features_low_churn, [])

        assert low_result.score > high_result.score

    def test_long_runway_boosts_score(self) -> None:
        scorer = TractionScorer()
        features_short = _features(runway_months=4)
        features_long = _features(runway_months=30)

        short_result = scorer.score(features_short, [])
        long_result = scorer.score(features_long, [])

        assert long_result.score > short_result.score

    def test_many_customers_boosts_score(self) -> None:
        scorer = TractionScorer()
        features_few = _features(customer_count=5)
        features_many = _features(customer_count=5000)

        few_result = scorer.score(features_few, [])
        many_result = scorer.score(features_many, [])

        assert many_result.score > few_result.score

    def test_scores_in_valid_range(self) -> None:
        scorer = TractionScorer()
        features = _features(
            arr_usd=18_500_000,
            growth_rate_pct=120,
            nrr_pct=135,
            churn_rate_pct=3.0,
            runway_months=24,
            burn_rate_usd=800_000,
            customer_count=500,
        )
        result = scorer.score(features, [])
        assert 0.0 <= result.score <= 100.0

    def test_structured_arr_score_no_signal_list(self) -> None:
        """Structured ARR scoring works even without signal lists."""
        scorer = TractionScorer()
        features = _features(arr_usd=55_000_000)
        result = scorer.score(features, [])

        assert result.score > 50.0


# ===================================================================
# TeamExecutionScorer numeric team size tests
# ===================================================================


class TestTeamExecutionScorerNumeric:
    """Tests for TeamExecutionScorer with numeric team size."""

    def test_numeric_large_team(self) -> None:
        scorer = TeamExecutionScorer()
        features_small = _features(team_size_numeric=5)
        features_large = _features(team_size_numeric=500)

        small_result = scorer.score(features_small, [])
        large_result = scorer.score(features_large, [])

        assert large_result.score >= small_result.score

    def test_numeric_takes_precedence_over_text(self) -> None:
        scorer = TeamExecutionScorer()
        features = _features(
            team_size_numeric=150,
            team_size_indicator="1-10",
        )
        result = scorer.score(features, [])
        assert result.score >= 50.0


# ===================================================================
# MarketOpportunityScorer market size tests
# ===================================================================


class TestMarketOpportunityScorerSize:
    """Tests for MarketOpportunityScorer with market_size_usd."""

    def test_large_market_boosts_score(self) -> None:
        scorer = MarketOpportunityScorer()
        features_small = _features(market_size_usd=500_000_000)
        features_large = _features(market_size_usd=500_000_000_000)

        small_result = scorer.score(features_small, [])
        large_result = scorer.score(features_large, [])

        assert large_result.score >= small_result.score


# ===================================================================
# Decision engine quantitative rationale tests
# ===================================================================


class TestDecisionQuantitativeRationale:
    """Tests for decision engine quantitative rationale generation."""

    def test_high_arr_in_reasons_for(self) -> None:
        engine = DefaultDecisionEngine()
        features = _features(
            arr_usd=25_000_000,
            nrr_pct=125,
            growth_rate_pct=80,
            data_completeness=0.8,
        )
        scores = [
            ScoreResult(dimension=d.value, score=70.0)
            for d in AnalysisDimension
        ]
        confidence = [
            ConfidenceAssessment(
                dimension=d.value, confidence=0.7, factors=[],
                data_completeness=0.8,
            )
            for d in AnalysisDimension
        ]

        decision = engine.decide(features, [], scores, confidence)

        for_reasons = decision.rationale.primary_reasons_for
        arr_found = any("25.0M" in r for r in for_reasons)
        assert arr_found

    def test_critical_runway_in_reasons_against(self) -> None:
        engine = DefaultDecisionEngine()
        features = _features(runway_months=3, data_completeness=0.5)
        scores = [
            ScoreResult(dimension=d.value, score=50.0)
            for d in AnalysisDimension
        ]
        confidence = [
            ConfidenceAssessment(
                dimension=d.value, confidence=0.5, factors=[],
                data_completeness=0.5,
            )
            for d in AnalysisDimension
        ]

        decision = engine.decide(features, [], scores, confidence)

        against_reasons = decision.rationale.primary_reasons_against
        runway_found = any("runway" in r.lower() for r in against_reasons)
        assert runway_found

    def test_high_churn_in_reasons_against(self) -> None:
        engine = DefaultDecisionEngine()
        features = _features(churn_rate_pct=15.0, data_completeness=0.5)
        scores = [
            ScoreResult(dimension=d.value, score=50.0)
            for d in AnalysisDimension
        ]
        confidence = [
            ConfidenceAssessment(
                dimension=d.value, confidence=0.5, factors=[],
                data_completeness=0.5,
            )
            for d in AnalysisDimension
        ]

        decision = engine.decide(features, [], scores, confidence)

        against_reasons = decision.rationale.primary_reasons_against
        churn_found = any("churn" in r.lower() for r in against_reasons)
        assert churn_found

    def test_exceptional_ltv_cac_in_reasons_for(self) -> None:
        engine = DefaultDecisionEngine()
        features = _features(
            cac_usd=500, ltv_usd=5000, data_completeness=0.8,
        )
        scores = [
            ScoreResult(dimension=d.value, score=65.0)
            for d in AnalysisDimension
        ]
        confidence = [
            ConfidenceAssessment(
                dimension=d.value, confidence=0.7, factors=[],
                data_completeness=0.8,
            )
            for d in AnalysisDimension
        ]

        decision = engine.decide(features, [], scores, confidence)

        for_reasons = decision.rationale.primary_reasons_for
        ltv_cac_found = any("LTV/CAC" in r for r in for_reasons)
        assert ltv_cac_found

    def test_no_quantitative_metrics_no_quant_reasons(self) -> None:
        engine = DefaultDecisionEngine()
        features = _features(data_completeness=0.5)
        scores = [
            ScoreResult(dimension=d.value, score=50.0)
            for d in AnalysisDimension
        ]
        confidence = [
            ConfidenceAssessment(
                dimension=d.value, confidence=0.5, factors=[],
                data_completeness=0.5,
            )
            for d in AnalysisDimension
        ]

        decision = engine.decide(features, [], scores, confidence)

        all_reasons = (
            decision.rationale.primary_reasons_for
            + decision.rationale.primary_reasons_against
        )
        quant_keywords = [
            "ARR", "NRR", "churn", "LTV/CAC", "runway", "burn rate",
        ]
        for kw in quant_keywords:
            assert not any(kw in r for r in all_reasons)


# ===================================================================
# Evaluation rationale tests
# ===================================================================


class TestEvaluationQuantitativeRationale:
    """Tests for evaluation rationale including quantitative context."""

    def test_traction_evaluator_includes_quant_context(self) -> None:
        evaluator = TractionEvaluator()
        features = _features(
            arr_usd=18_500_000,
            nrr_pct=135,
            growth_rate_pct=120,
        )
        observations = [_obs(dimension="traction_signals")]
        result = evaluator.evaluate(features, observations, [])

        assert "18.5M" in result.rationale
        assert "135%" in result.rationale
        assert "120%" in result.rationale

    def test_traction_evaluator_metadata_includes_metrics(self) -> None:
        evaluator = TractionEvaluator()
        features = _features(
            arr_usd=18_500_000,
            growth_rate_pct=120,
            nrr_pct=135,
            churn_rate_pct=3.0,
            customer_count=500,
            runway_months=24,
        )
        result = evaluator.evaluate(features, [], [])

        assert result.metadata["arr_usd"] == 18_500_000
        assert result.metadata["growth_rate_pct"] == 120
        assert result.metadata["nrr_pct"] == 135
        assert result.metadata["churn_rate_pct"] == 3.0
        assert result.metadata["customer_count"] == 500
        assert result.metadata["runway_months"] == 24

    def test_quantitative_context_empty_when_no_metrics(self) -> None:
        features = _features()
        ctx = _build_quantitative_context(features)
        assert ctx == ""

    def test_quantitative_context_with_metrics(self) -> None:
        features = _features(arr_usd=18_500_000, nrr_pct=135)
        ctx = _build_quantitative_context(features)
        assert "18.5M" in ctx
        assert "135%" in ctx
        assert ctx.startswith(" Quantitative metrics:")


# ===================================================================
# Investment readiness quantitative tests
# ===================================================================


class TestInvestmentReadinessQuantitative:
    """Tests for investment readiness with quantitative strengths/concerns."""

    def test_high_arr_is_strength(self) -> None:
        features = _features(arr_usd=25_000_000)
        strengths = _collect_strengths([], [], {}, features)
        arr_found = any("25.0M" in s for s in strengths)
        assert arr_found

    def test_high_nrr_is_strength(self) -> None:
        features = _features(nrr_pct=125)
        strengths = _collect_strengths([], [], {}, features)
        nrr_found = any("125%" in s for s in strengths)
        assert nrr_found

    def test_critical_runway_is_concern(self) -> None:
        features = _features(runway_months=4)
        concerns = _collect_concerns([], [], {}, features)
        runway_found = any("runway" in c.lower() for c in concerns)
        assert runway_found

    def test_high_churn_is_concern(self) -> None:
        features = _features(churn_rate_pct=12)
        concerns = _collect_concerns([], [], {}, features)
        churn_found = any("churn" in c.lower() for c in concerns)
        assert churn_found

    def test_unsustainable_ltv_cac_is_concern(self) -> None:
        features = _features(cac_usd=1000, ltv_usd=500)
        concerns = _collect_concerns([], [], {}, features)
        ltv_found = any("LTV/CAC" in c for c in concerns)
        assert ltv_found


# ===================================================================
# Regression tests: combined scenarios
# ===================================================================


class TestRegressionScenarios:
    """Regression tests covering common real-world scenarios."""

    def test_saas_strong_metrics(self) -> None:
        """A strong SaaS company with good metrics."""
        features = _features(
            arr_usd=18_500_000,
            mrr_usd=1_541_667,
            growth_rate_pct=135,
            nrr_pct=130,
            churn_rate_pct=3.0,
            runway_months=24,
            burn_rate_usd=800_000,
            cac_usd=800,
            ltv_usd=6000,
            funding_amount_usd=20_000_000,
            valuation_usd=150_000_000,
            customer_count=500,
            team_size_numeric=45,
            market_size_usd=100_000_000_000,
        )

        rule = QuantitativeSignalsRule()
        observations = rule.evaluate(features, [])
        assert len(observations) >= 10

        scorer = TractionScorer()
        score_result = scorer.score(features, observations)
        assert score_result.score >= 60.0

    def test_early_stage_low_metrics(self) -> None:
        """An early stage startup with limited metrics."""
        features = _features(
            arr_usd=200_000,
            customer_count=5,
            runway_months=8,
            team_size_numeric=4,
            data_completeness=0.3,
        )

        rule = QuantitativeSignalsRule()
        observations = rule.evaluate(features, [])

        scorer = TractionScorer()
        score_result = scorer.score(features, observations)
        assert 0.0 <= score_result.score <= 100.0

    def test_high_burn_short_runway(self) -> None:
        """A company with high burn and short runway."""
        features = _features(
            arr_usd=5_000_000,
            burn_rate_usd=6_000_000,
            runway_months=3,
            data_completeness=0.5,
        )

        rule = QuantitativeSignalsRule()
        observations = rule.evaluate(features, [])

        run_obs = [
            o for o in observations
            if o.category == "quantitative_runway"
        ]
        assert len(run_obs) == 1
        assert "critical" in run_obs[0].statement

        burn_obs = [
            o for o in observations
            if o.category == "quantitative_burn_rate"
        ]
        assert len(burn_obs) == 1
        assert "very high" in burn_obs[0].statement

    def test_negative_ltv_cac(self) -> None:
        """A company losing money per customer."""
        features = _features(cac_usd=2000, ltv_usd=800)

        rule = QuantitativeSignalsRule()
        observations = rule.evaluate(features, [])

        ue_obs = [
            o for o in observations
            if o.category == "quantitative_unit_economics"
        ]
        assert "unsustainable" in ue_obs[0].statement
        assert ue_obs[0].importance >= 0.8

    def test_full_pipeline_with_quantitative_metrics(self) -> None:
        """Integration test: run reasoning through decision with metrics."""
        features = _features(
            arr_usd=12_000_000,
            growth_rate_pct=80,
            nrr_pct=120,
            churn_rate_pct=4.0,
            runway_months=18,
            burn_rate_usd=500_000,
            customer_count=300,
            team_size_numeric=30,
            funding_amount_usd=15_000_000,
            valuation_usd=80_000_000,
            market_size_usd=50_000_000_000,
            data_completeness=0.7,
        )

        engine = DefaultReasoningEngine()
        observations = engine.reason(features, [])

        quant_obs = [
            o for o in observations
            if o.category.startswith("quantitative_")
        ]
        assert len(quant_obs) >= 8

        scorer = DefaultScoringEngine()
        scores = scorer.score(features, observations)

        traction_score = [
            s for s in scores
            if s.dimension == AnalysisDimension.TRACTION_SIGNALS.value
        ]
        assert len(traction_score) == 1
        assert traction_score[0].score >= 50.0

        decision_engine = DefaultDecisionEngine()
        confidence = [
            ConfidenceAssessment(
                dimension=d.value,
                confidence=0.65,
                factors=[],
                data_completeness=0.7,
            )
            for d in AnalysisDimension
        ]
        assessments = [
            DimensionAssessment(
                dimension=d.value,
                summary=f"Summary for {d.value}",
                rationale=f"Rationale for {d.value}",
                confidence=0.65,
                score=s.score,
            )
            for d, s in zip(AnalysisDimension, scores)
        ]

        decision = decision_engine.decide(
            features, observations, scores, confidence, assessments,
        )
        assert 0.0 <= decision.composite_score <= 100.0
