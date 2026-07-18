"""Tests for TractionScorer."""

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation, ScoreResult
from predictron_engine.scoring.scoring_engine import TractionScorer


class TestTractionScorer:
    def setup_method(self):
        self.scorer = TractionScorer()

    def test_returns_score_result(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert isinstance(result, ScoreResult)

    def test_dimension_is_traction_signals(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert result.dimension == "traction_signals"

    def test_base_score_is_50(self):
        features = ExtractedFeatures(has_revenue=True, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 58.0

    def test_has_revenue_true_bonus(self):
        features = ExtractedFeatures(has_revenue=True, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 58.0

    def test_has_revenue_false_penalty(self):
        features = ExtractedFeatures(has_revenue=False, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 45.0

    def test_has_revenue_none_penalty(self):
        features = ExtractedFeatures(has_revenue=None, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 47.0

    def test_funding_signals_bonus(self):
        features = ExtractedFeatures(funding_amount_signals=["$12M Series A"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_investor_signals_bonus(self):
        features = ExtractedFeatures(has_revenue=True, investor_signals=["Tier 1 VC"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_grants_bonus(self):
        features = ExtractedFeatures(has_revenue=True, grants_accelerator_signals=["YC"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_revenue_signals_bonus(self):
        features = ExtractedFeatures(has_revenue=True, revenue_amount_signals=["$2.8M ARR"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_arr_mrr_bonus(self):
        features = ExtractedFeatures(arr_mrr_signals=["$4.2M ARR"])
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_customer_count_bonus(self):
        features = ExtractedFeatures(has_revenue=True, customer_count_signals=["85 enterprise clients"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_active_users_bonus(self):
        features = ExtractedFeatures(has_revenue=True, active_user_signals=["100K MAU"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_enterprise_customers_bonus(self):
        features = ExtractedFeatures(has_revenue=True, enterprise_customer_signals=["Fortune 500"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_paying_customers_bonus(self):
        features = ExtractedFeatures(has_revenue=True, paying_customer_signals=["paying"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_retention_bonus(self):
        features = ExtractedFeatures(has_revenue=True, retention_signals=["140% NRR"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_growth_bonus(self):
        features = ExtractedFeatures(has_revenue=True, growth_signals=["20% MoM"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_expansion_bonus(self):
        features = ExtractedFeatures(has_revenue=True, expansion_signals=["new_market"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_milestones_bonus(self):
        features = ExtractedFeatures(has_revenue=True, milestone_signals=["100th_customer"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_partnerships_bonus(self):
        features = ExtractedFeatures(has_revenue=True, partnership_signals=["strategic"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_engagement_bonus(self):
        features = ExtractedFeatures(has_revenue=True, engagement_signals=["daily_usage"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_traction_risk_penalty(self):
        features = ExtractedFeatures(traction_risk=["churn"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_observations_affect_score(self):
        features = ExtractedFeatures()
        obs = [
            Observation(
                dimension="traction_signals",
                category="traction",
                statement="Strong traction.",
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
                dimension="traction_signals",
                category="test",
                statement="Traction obs.",
                evidence=[],
                confidence=0.5,
                importance=0.5,
                source_rule="Test",
            )
        ]
        result = self.scorer.score(features, obs)
        assert "Traction obs." in result.evidence

    def test_score_in_valid_range(self):
        features = ExtractedFeatures(
            has_revenue=True,
            funding_amount_signals=["$12M", "$5M"],
            investor_signals=["Tier 1 VC", "Angel"],
            revenue_amount_signals=["$2.8M ARR"],
            arr_mrr_signals=["$4.2M ARR"],
            customer_count_signals=["85 clients"],
            retention_signals=["140% NRR"],
            growth_signals=["20% MoM"],
            data_completeness=0.9,
        )
        result = self.scorer.score(features, [])
        assert 0.0 <= result.score <= 100.0

    def test_rationale_mentions_dimensions(self):
        features = ExtractedFeatures(has_revenue=True)
        result = self.scorer.score(features, [])
        assert "signal dimensions" in result.rationale
