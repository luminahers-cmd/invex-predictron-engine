"""Tests for CompetitionAssessmentRule."""

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.reasoning.rules.competition_assessment import (
    CompetitionAssessmentRule,
)


class TestCompetitionAssessmentRule:
    def setup_method(self):
        self.rule = CompetitionAssessmentRule()

    def test_name(self):
        assert self.rule.name == "competition_assessment"

    def test_returns_list(self):
        features = ExtractedFeatures()
        result = self.rule.evaluate(features, [])
        assert isinstance(result, list)

    def test_all_observations_have_correct_dimension(self):
        features = ExtractedFeatures(
            competitive_moat_indicators=["proprietary_data"],
            market_concentration="fragmented",
            switching_cost_signals=["integration_lock_in"],
            barriers_to_entry=["regulatory"],
            open_source_competition=["linux_alternative"],
        )
        result = self.rule.evaluate(features, [])
        for obs in result:
            assert obs.dimension == "competitive_position"

    def test_all_observations_have_source_rule(self):
        features = ExtractedFeatures(
            competitive_moat_indicators=["proprietary_data"]
        )
        result = self.rule.evaluate(features, [])
        for obs in result:
            assert obs.source_rule == "CompetitionAssessmentRule"

    def test_moat_strong_indicators(self):
        features = ExtractedFeatures(
            competitive_moat_indicators=[
                "proprietary_data",
                "patent_portfolio",
            ]
        )
        result = self.rule.evaluate(features, [])
        moat_obs = [o for o in result if o.category == "moat_analysis"]
        assert len(moat_obs) == 1
        assert "Strong competitive moat" in moat_obs[0].statement
        assert moat_obs[0].importance == 0.8

    def test_moat_moderate_indicators(self):
        features = ExtractedFeatures(
            competitive_moat_indicators=["brand_recognition"]
        )
        result = self.rule.evaluate(features, [])
        moat_obs = [o for o in result if o.category == "moat_analysis"]
        assert len(moat_obs) == 1
        assert "Moderate competitive moat" in moat_obs[0].statement
        assert moat_obs[0].importance == 0.4

    def test_moat_unknown_signals(self):
        features = ExtractedFeatures(
            competitive_moat_indicators=["some_unknown_moat"]
        )
        result = self.rule.evaluate(features, [])
        moat_obs = [o for o in result if o.category == "moat_analysis"]
        assert len(moat_obs) == 1
        assert "none matched" in moat_obs[0].statement

    def test_no_moat_no_moat_observation(self):
        features = ExtractedFeatures()
        result = self.rule.evaluate(features, [])
        assert not any(o.category == "moat_analysis" for o in result)

    def test_concentration_fragmented(self):
        features = ExtractedFeatures(market_concentration="fragmented")
        result = self.rule.evaluate(features, [])
        conc_obs = [o for o in result if o.category == "market_concentration"]
        assert len(conc_obs) == 1
        assert "differentiation" in conc_obs[0].statement.lower()
        assert conc_obs[0].importance == 0.5

    def test_concentration_concentrated(self):
        features = ExtractedFeatures(market_concentration="concentrated")
        result = self.rule.evaluate(features, [])
        conc_obs = [o for o in result if o.category == "market_concentration"]
        assert len(conc_obs) == 1
        assert conc_obs[0].importance == 0.4

    def test_concentration_dominated(self):
        features = ExtractedFeatures(market_concentration="dominated")
        result = self.rule.evaluate(features, [])
        conc_obs = [o for o in result if o.category == "market_concentration"]
        assert len(conc_obs) == 1
        assert conc_obs[0].importance == 0.7

    def test_concentration_with_density(self):
        features = ExtractedFeatures(
            market_concentration="fragmented",
            competitive_density="dense",
        )
        result = self.rule.evaluate(features, [])
        conc_obs = [o for o in result if o.category == "market_concentration"]
        assert "dense" in conc_obs[0].statement

    def test_no_concentration_no_observation(self):
        features = ExtractedFeatures()
        result = self.rule.evaluate(features, [])
        assert not any(o.category == "market_concentration" for o in result)

    def test_switching_costs_strong(self):
        features = ExtractedFeatures(
            switching_cost_signals=["a", "b", "c"]
        )
        result = self.rule.evaluate(features, [])
        sw_obs = [o for o in result if o.category == "switching_costs"]
        assert len(sw_obs) == 1
        assert "Strong switching cost" in sw_obs[0].statement
        assert sw_obs[0].importance == 0.7

    def test_switching_costs_moderate(self):
        features = ExtractedFeatures(switching_cost_signals=["a"])
        result = self.rule.evaluate(features, [])
        sw_obs = [o for o in result if o.category == "switching_costs"]
        assert len(sw_obs) == 1
        assert "Moderate switching cost" in sw_obs[0].statement
        assert sw_obs[0].importance == 0.4

    def test_no_switching_costs_no_observation(self):
        features = ExtractedFeatures()
        result = self.rule.evaluate(features, [])
        assert not any(o.category == "switching_costs" for o in result)

    def test_entry_difficulty_many_barriers(self):
        features = ExtractedFeatures(
            barriers_to_entry=["a", "b", "c"]
        )
        result = self.rule.evaluate(features, [])
        entry_obs = [o for o in result if o.category == "entry_difficulty"]
        assert len(entry_obs) == 1
        assert "Multiple barriers" in entry_obs[0].statement
        assert entry_obs[0].importance == 0.6

    def test_entry_difficulty_few_barriers(self):
        features = ExtractedFeatures(barriers_to_entry=["a"])
        result = self.rule.evaluate(features, [])
        entry_obs = [o for o in result if o.category == "entry_difficulty"]
        assert len(entry_obs) == 1
        assert "Some barriers" in entry_obs[0].statement
        assert entry_obs[0].importance == 0.3

    def test_no_barriers_no_observation(self):
        features = ExtractedFeatures()
        result = self.rule.evaluate(features, [])
        assert not any(o.category == "entry_difficulty" for o in result)

    def test_open_source_risk(self):
        features = ExtractedFeatures(
            open_source_competition=["linux_tool", "github_project"]
        )
        result = self.rule.evaluate(features, [])
        oss_obs = [o for o in result if o.category == "open_source_risk"]
        assert len(oss_obs) == 1
        assert "2 signals" in oss_obs[0].statement

    def test_no_open_source_no_observation(self):
        features = ExtractedFeatures()
        result = self.rule.evaluate(features, [])
        assert not any(o.category == "open_source_risk" for o in result)

    def test_empty_features_returns_empty(self):
        features = ExtractedFeatures()
        result = self.rule.evaluate(features, [])
        assert result == []

    def test_multiple_signals_produce_multiple_observations(self):
        features = ExtractedFeatures(
            competitive_moat_indicators=["proprietary_data", "patent_portfolio"],
            market_concentration="concentrated",
            switching_cost_signals=["a", "b"],
            barriers_to_entry=["x", "y", "z"],
            open_source_competition=["tool"],
        )
        result = self.rule.evaluate(features, [])
        categories = {o.category for o in result}
        assert "moat_analysis" in categories
        assert "market_concentration" in categories
        assert "switching_costs" in categories
        assert "entry_difficulty" in categories
        assert "open_source_risk" in categories

    def test_evidence_refs_included_when_evidence_present(self):
        features = ExtractedFeatures(
            competitive_moat_indicators=["proprietary_data"]
        )
        evidence = [
            EvidenceItem(
                domain="competition",
                category="moat",
                statement="Moat detected.",
                source="test",
            )
        ]
        result = self.rule.evaluate(features, evidence)
        moat_obs = [o for o in result if o.category == "moat_analysis"]
        assert any(e.startswith("evidence:") for e in moat_obs[0].evidence)

    def test_feature_refs_included(self):
        features = ExtractedFeatures(
            competitive_moat_indicators=["proprietary_data"]
        )
        result = self.rule.evaluate(features, [])
        moat_obs = [o for o in result if o.category == "moat_analysis"]
        assert any(e.startswith("feature:") for e in moat_obs[0].evidence)

    def test_unknown_concentration_ignored(self):
        features = ExtractedFeatures(market_concentration="unknown")
        result = self.rule.evaluate(features, [])
        assert not any(o.category == "market_concentration" for o in result)
