"""Tests for the CompositeReasoner."""

from predictron_engine.models.report import Observation
from predictron_engine.reasoning.composite import CompositeReasoner


class TestCompositeReasoner:
    """Unit tests for the composite reasoner orchestrator."""

    def test_returns_observations_list(
        self, rich_features, rich_evidence
    ):
        from predictron_engine.reasoning.rules import DEFAULT_RULES

        reasoner = CompositeReasoner(list(DEFAULT_RULES))
        result = reasoner.reason(rich_features, rich_evidence)

        assert isinstance(result, list)
        assert all(isinstance(o, Observation) for o in result)

    def test_aggregates_from_multiple_rules(
        self, rich_features, rich_evidence
    ):
        from predictron_engine.reasoning.rules import DEFAULT_RULES

        reasoner = CompositeReasoner(list(DEFAULT_RULES))
        result = reasoner.reason(rich_features, rich_evidence)

        source_rules = {o.source_rule for o in result}
        assert len(source_rules) > 1

    def test_rule_count_property(self):
        from predictron_engine.reasoning.rules import DEFAULT_RULES

        reasoner = CompositeReasoner(list(DEFAULT_RULES))
        assert reasoner.rule_count == len(DEFAULT_RULES)

    def test_failing_rule_does_not_break_others(
        self, rich_features, rich_evidence
    ):
        class FailingRule:
            def evaluate(self, features, evidence):
                raise RuntimeError("boom")

        class GoodRule:
            def evaluate(self, features, evidence):
                return [
                    Observation(
                        dimension="test",
                        category="test",
                        statement="ok",
                        source_rule="GoodRule",
                    )
                ]

        reasoner = CompositeReasoner([FailingRule(), GoodRule()])
        result = reasoner.reason(rich_features, rich_evidence)

        assert len(result) == 1
        assert result[0].source_rule == "GoodRule"

    def test_empty_rules_returns_empty(
        self, rich_features, rich_evidence
    ):
        reasoner = CompositeReasoner([])
        result = reasoner.reason(rich_features, rich_evidence)

        assert result == []

    def test_observations_have_all_required_fields(
        self, rich_features, rich_evidence
    ):
        from predictron_engine.reasoning.rules import DEFAULT_RULES

        reasoner = CompositeReasoner(list(DEFAULT_RULES))
        result = reasoner.reason(rich_features, rich_evidence)

        for obs in result:
            assert obs.dimension != ""
            assert obs.category != ""
            assert obs.statement != ""
            assert 0.0 <= obs.confidence <= 1.0
            assert 0.0 <= obs.importance <= 1.0
            assert obs.source_rule != ""

    def test_custom_rules_only(
        self, rich_features, rich_evidence
    ):
        class OnlyOne:
            def evaluate(self, features, evidence):
                return [
                    Observation(
                        dimension="custom",
                        category="custom",
                        statement="only one",
                        source_rule="OnlyOne",
                    )
                ]

        reasoner = CompositeReasoner([OnlyOne()])
        result = reasoner.reason(rich_features, rich_evidence)

        assert len(result) == 1
        assert result[0].source_rule == "OnlyOne"
