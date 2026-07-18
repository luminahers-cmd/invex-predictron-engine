"""Tests for TeamExecutionScorer."""

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation, ScoreResult
from predictron_engine.scoring.scoring_engine import TeamExecutionScorer


class TestTeamExecutionScorer:
    def setup_method(self):
        self.scorer = TeamExecutionScorer()

    def test_returns_score_result(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert isinstance(result, ScoreResult)

    def test_dimension_is_team_execution(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert result.dimension == "team_execution"

    def test_base_score_is_50(self):
        features = ExtractedFeatures(data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 58.0

    def test_team_size_medium_bonus(self):
        features = ExtractedFeatures(team_size_indicator="11-50", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_team_size_large_bonus(self):
        features = ExtractedFeatures(team_size_indicator="201-1000", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_hiring_signals_bonus(self):
        features = ExtractedFeatures(hiring_signals=["hiring_eng", "hiring_sales"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_hiring_growth_bonus(self):
        features = ExtractedFeatures(hiring_growth_signals=["doubling_team"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_execution_signals_bonus(self):
        features = ExtractedFeatures(execution_signals=["arr_revenue", "customer_count"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_engineering_strong_bonus(self):
        features = ExtractedFeatures(engineering_strength="strong", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_engineering_weak_penalty(self):
        features = ExtractedFeatures(engineering_strength="weak")
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_engineering_maturity_sophisticated(self):
        features = ExtractedFeatures(engineering_maturity="sophisticated", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_engineering_maturity_nascent_penalty(self):
        features = ExtractedFeatures(engineering_maturity="nascent")
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_infrastructure_enterprise_grade(self):
        features = ExtractedFeatures(infrastructure_maturity="enterprise_grade", data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_infrastructure_early_penalty(self):
        features = ExtractedFeatures(infrastructure_maturity="early")
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_open_source_bonus(self):
        features = ExtractedFeatures(open_source_signals=["community"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_dev_tooling_bonus(self):
        features = ExtractedFeatures(developer_tooling_signals=["ci_cd"], data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_recency_young_bonus(self):
        features = ExtractedFeatures(founded_year=2024, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_recency_old_penalty(self):
        features = ExtractedFeatures(founded_year=2010)
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_data_completeness_high_bonus(self):
        features = ExtractedFeatures(data_completeness=0.95)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_data_completeness_low_penalty(self):
        features = ExtractedFeatures(data_completeness=0.1)
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_operational_risk_penalty(self):
        features = ExtractedFeatures(operational_risk=["risk1", "risk2"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_hiring_risk_penalty(self):
        features = ExtractedFeatures(hiring_risk=["talent_shortage"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_funding_risk_penalty(self):
        features = ExtractedFeatures(funding_risk=["low_runway"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_scaling_risk_penalty(self):
        features = ExtractedFeatures(scaling_risk=["bottleneck"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_security_risk_penalty(self):
        features = ExtractedFeatures(security_risk=["vulnerability"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_compliance_risk_penalty(self):
        features = ExtractedFeatures(compliance_risk=["gdpr"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_observations_affect_score(self):
        features = ExtractedFeatures(data_completeness=1.0)
        obs = [
            Observation(
                dimension="team_execution",
                category="execution",
                statement="Strong execution.",
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
                dimension="team_execution",
                category="test",
                statement="Exec obs.",
                evidence=[],
                confidence=0.5,
                importance=0.5,
                source_rule="Test",
            )
        ]
        result = self.scorer.score(features, obs)
        assert "Exec obs." in result.evidence

    def test_score_in_valid_range(self):
        features = ExtractedFeatures(
            team_size_indicator="51-200",
            hiring_signals=["h1", "h2"],
            hiring_growth_signals=["g1"],
            execution_signals=["e1", "e2", "e3"],
            engineering_strength="strong",
            engineering_maturity="established",
            infrastructure_maturity="mature",
            open_source_signals=["os1"],
            developer_tooling_signals=["dt1"],
            founded_year=2022,
            data_completeness=0.9,
        )
        result = self.scorer.score(features, [])
        assert 0.0 <= result.score <= 100.0

    def test_score_clamped_at_100(self):
        features = ExtractedFeatures(
            team_size_indicator="1000+",
            hiring_signals=["h1", "h2", "h3", "h4"],
            hiring_growth_signals=["g1", "g2", "g3", "g4"],
            execution_signals=["e1", "e2", "e3", "e4", "e5", "e6"],
            engineering_strength="strong",
            engineering_maturity="sophisticated",
            infrastructure_maturity="enterprise_grade",
            open_source_signals=["os1", "os2", "os3", "os4"],
            developer_tooling_signals=["dt1", "dt2", "dt3", "dt4"],
            founded_year=2024,
            data_completeness=1.0,
        )
        result = self.scorer.score(features, [])
        assert result.score == 100.0

    def test_rationale_mentions_dimensions(self):
        features = ExtractedFeatures(team_size_indicator="11-50")
        result = self.scorer.score(features, [])
        assert "signal dimensions" in result.rationale
