"""Tests for all individual reasoning rules."""

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation
from predictron_engine.reasoning.rules.business_model_context import (
    BusinessModelContextRule,
)
from predictron_engine.reasoning.rules.data_quality import DataQualityRule
from predictron_engine.reasoning.rules.market_context import MarketContextRule
from predictron_engine.reasoning.rules.risk_indicator import RiskIndicatorRule
from predictron_engine.reasoning.rules.stage_expectation import (
    StageExpectationRule,
)
from predictron_engine.reasoning.rules.team_assessment import TeamAssessmentRule
from predictron_engine.reasoning.rules.technology_context import (
    TechnologyContextRule,
)


class TestMarketContextRule:
    def test_produces_observation_with_industry_evidence(
        self, industry_only_features, rich_evidence
    ):
        rule = MarketContextRule()
        result = rule.evaluate(industry_only_features, rich_evidence)

        assert len(result) >= 1
        assert all(isinstance(o, Observation) for o in result)

    def test_industry_observation_dimension(
        self, industry_only_features, rich_evidence
    ):
        rule = MarketContextRule()
        result = rule.evaluate(industry_only_features, rich_evidence)

        dims = {o.dimension for o in result}
        assert "market_opportunity" in dims

    def test_empty_without_industry(self, minimal_features):
        rule = MarketContextRule()
        result = rule.evaluate(minimal_features, [])

        assert result == []

    def test_source_rule_populated(
        self, industry_only_features, rich_evidence
    ):
        rule = MarketContextRule()
        result = rule.evaluate(industry_only_features, rich_evidence)

        for obs in result:
            assert obs.source_rule == "MarketContextRule"

    def test_evidence_refs_present(
        self, industry_only_features, rich_evidence
    ):
        rule = MarketContextRule()
        result = rule.evaluate(industry_only_features, rich_evidence)

        for obs in result:
            assert len(obs.evidence) > 0


class TestBusinessModelContextRule:
    def test_produces_observation(self, rich_features, rich_evidence):
        rule = BusinessModelContextRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert len(result) == 1
        assert result[0].category == "business_model_assessment"

    def test_empty_without_business_model(
        self, minimal_features, rich_evidence
    ):
        rule = BusinessModelContextRule()
        result = rule.evaluate(minimal_features, rich_evidence)

        assert result == []

    def test_includes_industry_context(self, rich_features, rich_evidence):
        rule = BusinessModelContextRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert "fintech" in result[0].statement

    def test_source_rule_populated(self, rich_features, rich_evidence):
        rule = BusinessModelContextRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert result[0].source_rule == "BusinessModelContextRule"


class TestStageExpectationRule:
    def test_produces_observation(self, rich_features, rich_evidence):
        rule = StageExpectationRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert len(result) == 1
        assert result[0].category == "stage_assessment"

    def test_empty_without_stage(self, minimal_features, rich_evidence):
        rule = StageExpectationRule()
        result = rule.evaluate(minimal_features, rich_evidence)

        assert result == []

    def test_references_stage(self, rich_features, rich_evidence):
        rule = StageExpectationRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert "seed" in result[0].statement

    def test_source_rule_populated(self, rich_features, rich_evidence):
        rule = StageExpectationRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert result[0].source_rule == "StageExpectationRule"


class TestTechnologyContextRule:
    def test_produces_observation(self, rich_features, rich_evidence):
        rule = TechnologyContextRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert len(result) == 1
        assert result[0].category == "technology_assessment"

    def test_empty_without_tech_stack(self, minimal_features, rich_evidence):
        rule = TechnologyContextRule()
        result = rule.evaluate(minimal_features, rich_evidence)

        assert result == []

    def test_references_tech_count(self, rich_features, rich_evidence):
        rule = TechnologyContextRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert "3" in result[0].statement

    def test_source_rule_populated(self, rich_features, rich_evidence):
        rule = TechnologyContextRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert result[0].source_rule == "TechnologyContextRule"


class TestTeamAssessmentRule:
    def test_produces_observation_with_founders(
        self, rich_features, rich_evidence
    ):
        rule = TeamAssessmentRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert len(result) >= 1
        founder_obs = [
            o for o in result if "founder" in o.statement.lower()
        ]
        assert len(founder_obs) == 1

    def test_empty_with_no_team_data(self, minimal_features, rich_evidence):
        rule = TeamAssessmentRule()
        result = rule.evaluate(minimal_features, rich_evidence)

        assert result == []

    def test_source_rule_populated(self, rich_features, rich_evidence):
        rule = TeamAssessmentRule()
        result = rule.evaluate(rich_features, rich_evidence)

        for obs in result:
            assert obs.source_rule == "TeamAssessmentRule"


class TestDataQualityRule:
    def test_high_completeness(self, rich_features, rich_evidence):
        rule = DataQualityRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert len(result) == 1
        assert "high" in result[0].statement.lower()

    def test_low_completeness(self, minimal_features, rich_evidence):
        rule = DataQualityRule()
        result = rule.evaluate(minimal_features, rich_evidence)

        assert len(result) == 1
        assert "low" in result[0].statement.lower()

    def test_dimension_is_data_quality(self, rich_features, rich_evidence):
        rule = DataQualityRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert result[0].dimension == "data_quality"
        assert result[0].category == "data_assessment"

    def test_always_produces_observation(self, rich_evidence):
        rule = DataQualityRule()
        features_empty = ExtractedFeatures()
        result = rule.evaluate(features_empty, rich_evidence)

        assert len(result) == 1

    def test_source_rule_populated(self, rich_features, rich_evidence):
        rule = DataQualityRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert result[0].source_rule == "DataQualityRule"


class TestRiskIndicatorRule:
    def test_no_pitch_deck_detected(self, minimal_features, rich_evidence):
        rule = RiskIndicatorRule()
        result = rule.evaluate(minimal_features, rich_evidence)

        pitch_risks = [
            o for o in result if "pitch deck" in o.statement.lower()
        ]
        assert len(pitch_risks) == 1

    def test_no_founders_detected(self, minimal_features, rich_evidence):
        rule = RiskIndicatorRule()
        result = rule.evaluate(minimal_features, rich_evidence)

        founder_risks = [
            o for o in result if "founder" in o.statement.lower()
        ]
        assert len(founder_risks) == 1

    def test_low_data_risk(self, minimal_features, rich_evidence):
        rule = RiskIndicatorRule()
        result = rule.evaluate(minimal_features, rich_evidence)

        data_risks = [
            o for o in result if "completeness" in o.statement.lower()
        ]
        assert len(data_risks) == 1

    def test_rich_features_fewer_risks(self, rich_features, rich_evidence):
        rule = RiskIndicatorRule()
        result = rule.evaluate(rich_features, rich_evidence)

        assert len(result) == 0

    def test_category_is_risk_indicator(
        self, minimal_features, rich_evidence
    ):
        rule = RiskIndicatorRule()
        result = rule.evaluate(minimal_features, rich_evidence)

        for obs in result:
            assert obs.category == "risk_indicator"

    def test_source_rule_populated(self, minimal_features, rich_evidence):
        rule = RiskIndicatorRule()
        result = rule.evaluate(minimal_features, rich_evidence)

        for obs in result:
            assert obs.source_rule == "RiskIndicatorRule"
