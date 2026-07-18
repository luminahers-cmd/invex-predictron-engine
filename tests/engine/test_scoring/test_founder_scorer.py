"""Tests for FounderQualityScorer."""

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation, ScoreResult
from predictron_engine.scoring.scoring_engine import FounderQualityScorer


class TestFounderQualityScorer:
    def setup_method(self):
        self.scorer = FounderQualityScorer()

    def test_returns_score_result(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert isinstance(result, ScoreResult)

    def test_dimension_is_founder_quality(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert result.dimension == "founder_quality"

    def test_base_score_is_50(self):
        features = ExtractedFeatures(founder_profile_count=1, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score == 48.0

    def test_team_type_technical_bonus(self):
        features = ExtractedFeatures(founder_team_type="technical", founder_profile_count=1, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_team_type_mixed(self):
        features = ExtractedFeatures(founder_team_type="mixed", founder_profile_count=1, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_profile_count_zero_penalty(self):
        features = ExtractedFeatures(founder_profile_count=0)
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_profile_count_two_bonus(self):
        features = ExtractedFeatures(founder_profile_count=2)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_domain_expertise_bonus(self):
        features = ExtractedFeatures(domain_expertise_signals=["signal1", "signal2"], founder_profile_count=1, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_serial_founder_bonus(self):
        features = ExtractedFeatures(serial_founder_indicators=["previous_exit"], founder_profile_count=1, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_leadership_roles_bonus(self):
        features = ExtractedFeatures(leadership_roles=["CEO", "CTO"], founder_profile_count=1, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_advisor_bonus(self):
        features = ExtractedFeatures(advisor_mentions=["advisor1", "advisor2"], founder_profile_count=1, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_market_fit_bonus(self):
        features = ExtractedFeatures(founder_market_fit_signals=["fit1", "fit2"], founder_profile_count=1, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_engineering_strong_bonus(self):
        features = ExtractedFeatures(engineering_strength="strong", founder_profile_count=1, data_completeness=1.0)
        result = self.scorer.score(features, [])
        assert result.score > 50.0

    def test_engineering_weak_penalty(self):
        features = ExtractedFeatures(engineering_strength="weak")
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_founder_risk_penalty(self):
        features = ExtractedFeatures(founder_risk=["solo_founder"])
        result = self.scorer.score(features, [])
        assert result.score < 50.0

    def test_observations_affect_score(self):
        features = ExtractedFeatures(founder_profile_count=1, data_completeness=1.0)
        obs = [
            Observation(
                dimension="founder_quality",
                category="team_assessment",
                statement="Strong team.",
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
                dimension="founder_quality",
                category="test",
                statement="Team obs.",
                evidence=[],
                confidence=0.5,
                importance=0.5,
                source_rule="Test",
            )
        ]
        result = self.scorer.score(features, obs)
        assert "Team obs." in result.evidence

    def test_score_in_valid_range(self):
        features = ExtractedFeatures(
            founder_team_type="technical",
            founder_profile_count=3,
            domain_expertise_signals=["d1", "d2"],
            serial_founder_indicators=["s1", "s2"],
            leadership_roles=["CEO", "CTO", "COO"],
            advisor_mentions=["a1", "a2"],
            founder_market_fit_signals=["f1", "f2", "f3"],
            engineering_strength="strong",
        )
        result = self.scorer.score(features, [])
        assert 0.0 <= result.score <= 100.0

    def test_score_clamped_at_100(self):
        features = ExtractedFeatures(
            founder_team_type="technical",
            founder_profile_count=3,
            domain_expertise_signals=["d1", "d2", "d3", "d4", "d5"],
            serial_founder_indicators=["s1", "s2", "s3"],
            leadership_roles=["CEO", "CTO", "COO"],
            advisor_mentions=["a1", "a2", "a3"],
            founder_market_fit_signals=["f1", "f2", "f3", "f4"],
            engineering_strength="strong",
            data_completeness=1.0,
        )
        result = self.scorer.score(features, [])
        assert result.score <= 100.0

    def test_rationale_mentions_dimensions(self):
        features = ExtractedFeatures(founder_team_type="technical")
        result = self.scorer.score(features, [])
        assert "signal dimensions" in result.rationale
