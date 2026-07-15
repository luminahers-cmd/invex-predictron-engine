"""Tests for CompetitionScorer and updated scoring engine."""

from predictron_engine.knowledge.concepts import AnalysisDimension
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation, ScoreResult
from predictron_engine.scoring.scoring_engine import (
    CompetitionScorer,
    DefaultScoringEngine,
)


class TestCompetitionScorer:
    def setup_method(self):
        self.scorer = CompetitionScorer()

    def test_returns_score_result(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert isinstance(result, ScoreResult)

    def test_dimension_is_competitive_position(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert result.dimension == "competitive_position"

    def test_base_score_is_50(self):
        features = ExtractedFeatures()
        result = self.scorer.score(features, [])
        assert result.score == 50.0

    def test_score_in_valid_range(self):
        features = ExtractedFeatures(
            market_concentration="fragmented",
            competitive_density="sparse",
            competitive_moat_indicators=["proprietary_data", "patent_portfolio"],
            switching_cost_signals=["a", "b", "c"],
            network_effect_competition="strong_network_effects",
            barriers_to_entry=["regulatory", "capital"],
            open_source_competition=["tool1"],
            differentiation_signals=["unique_feature"],
        )
        result = self.scorer.score(features, [])
        assert 0.0 <= result.score <= 100.0

    def test_concentration_fragmented_bonus(self):
        features = ExtractedFeatures(market_concentration="fragmented")
        result = self.scorer.score(features, [])
        assert result.score == 55.0

    def test_concentration_concentrated_penalty(self):
        features = ExtractedFeatures(market_concentration="concentrated")
        result = self.scorer.score(features, [])
        assert result.score == 42.0

    def test_concentration_dominated_penalty(self):
        features = ExtractedFeatures(market_concentration="dominated")
        result = self.scorer.score(features, [])
        assert result.score == 35.0

    def test_concentration_moderately_concentrated_neutral(self):
        features = ExtractedFeatures(
            market_concentration="moderately_concentrated"
        )
        result = self.scorer.score(features, [])
        assert result.score == 50.0

    def test_density_sparse_bonus(self):
        features = ExtractedFeatures(competitive_density="sparse")
        result = self.scorer.score(features, [])
        assert result.score == 55.0

    def test_density_dense_penalty(self):
        features = ExtractedFeatures(competitive_density="dense")
        result = self.scorer.score(features, [])
        assert result.score == 44.0

    def test_density_hyper_competitive_penalty(self):
        features = ExtractedFeatures(competitive_density="hyper_competitive")
        result = self.scorer.score(features, [])
        assert result.score == 38.0

    def test_moats_single(self):
        features = ExtractedFeatures(
            competitive_moat_indicators=["proprietary_data"]
        )
        result = self.scorer.score(features, [])
        assert result.score == 53.0

    def test_moats_multiple(self):
        features = ExtractedFeatures(
            competitive_moat_indicators=[
                "proprietary_data",
                "patent_portfolio",
                "strong_network_effects",
            ]
        )
        result = self.scorer.score(features, [])
        assert result.score == 59.0

    def test_switching_costs(self):
        features = ExtractedFeatures(
            switching_cost_signals=["a", "b"]
        )
        result = self.scorer.score(features, [])
        assert result.score == 56.0

    def test_network_strong(self):
        features = ExtractedFeatures(
            network_effect_competition="strong_network_effects"
        )
        result = self.scorer.score(features, [])
        assert result.score == 62.0

    def test_network_moderate(self):
        features = ExtractedFeatures(
            network_effect_competition="moderate_network_effects"
        )
        result = self.scorer.score(features, [])
        assert result.score == 55.0

    def test_network_none(self):
        features = ExtractedFeatures(
            network_effect_competition="no_network_effects"
        )
        result = self.scorer.score(features, [])
        assert result.score == 47.0

    def test_barriers(self):
        features = ExtractedFeatures(
            barriers_to_entry=["a", "b", "c"]
        )
        result = self.scorer.score(features, [])
        assert result.score == 59.0

    def test_open_source_penalty(self):
        features = ExtractedFeatures(
            open_source_competition=["tool1"]
        )
        result = self.scorer.score(features, [])
        assert result.score == 47.5

    def test_differentiation_bonus(self):
        features = ExtractedFeatures(
            differentiation_signals=["unique_feature"]
        )
        result = self.scorer.score(features, [])
        assert result.score == 52.5

    def test_all_signals_combined(self):
        features = ExtractedFeatures(
            market_concentration="fragmented",
            competitive_density="sparse",
            competitive_moat_indicators=["proprietary_data", "patent_portfolio"],
            switching_cost_signals=["a", "b", "c"],
            network_effect_competition="strong_network_effects",
            barriers_to_entry=["regulatory", "capital"],
            differentiation_signals=["unique_feature"],
        )
        result = self.scorer.score(features, [])
        assert result.score >= 80.0
        assert result.score <= 100.0

    def test_worst_case_score_very_low(self):
        features = ExtractedFeatures(
            market_concentration="dominated",
            competitive_density="hyper_competitive",
            network_effect_competition="no_network_effects",
            open_source_competition=["a", "b", "c", "d", "d", "d"],
        )
        result = self.scorer.score(features, [])
        assert result.score <= 15.0

    def test_score_clamped_at_100(self):
        features = ExtractedFeatures(
            market_concentration="fragmented",
            competitive_density="sparse",
            competitive_moat_indicators=[
                "proprietary_data",
                "patent_portfolio",
                "strong_network_effects",
                "regulatory_advantage",
            ],
            switching_cost_signals=["a", "b", "c", "d"],
            network_effect_competition="strong_network_effects",
            barriers_to_entry=["a", "b", "c", "d"],
            differentiation_signals=["a", "b", "c", "d"],
        )
        result = self.scorer.score(features, [])
        assert result.score == 100.0

    def test_observations_affect_score(self):
        features = ExtractedFeatures()
        obs = [
            Observation(
                dimension="competitive_position",
                category="moat_analysis",
                statement="Strong moat detected.",
                evidence=[],
                confidence=0.8,
                importance=0.9,
                source_rule="Test",
            )
        ]
        result = self.scorer.score(features, obs)
        assert result.score > 50.0

    def test_rationale_mentions_dimensions(self):
        features = ExtractedFeatures(
            market_concentration="fragmented",
            competitive_moat_indicators=["proprietary_data"],
        )
        result = self.scorer.score(features, [])
        assert "signal dimensions" in result.rationale

    def test_evidence_from_observations(self):
        features = ExtractedFeatures()
        obs = [
            Observation(
                dimension="competitive_position",
                category="test",
                statement="Test observation.",
                evidence=[],
                confidence=0.5,
                importance=0.5,
                source_rule="Test",
            )
        ]
        result = self.scorer.score(features, obs)
        assert "Test observation." in result.evidence


class TestDefaultScoringEngineWithCompetition:
    def test_competition_scorer_is_used(self):
        engine = DefaultScoringEngine()
        features = ExtractedFeatures(market_concentration="dominated")
        result = engine.score(features, [])
        comp = [s for s in result if s.dimension == "competitive_position"]
        assert len(comp) == 1
        assert comp[0].score < 50.0

    def test_all_dimensions_scored(self):
        engine = DefaultScoringEngine()
        features = ExtractedFeatures()
        result = engine.score(features, [])
        dims = {s.dimension for s in result}
        expected = {dim.value for dim in AnalysisDimension}
        assert dims == expected
