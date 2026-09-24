"""Tests for the extensible rule engine."""

from __future__ import annotations

import pytest

from predictron_engine.research import (
    DEFAULT_RULES,
    TOPIC_REGISTRY,
    EvidenceStatus,
    MissingFoundersRule,
    MissingFundingRule,
    MissingMarketRule,
    MissingPricingRule,
    MissingTechnologyRule,
    PlannerInput,
    ResearchPlanner,
    ResearchTopic,
    RuleRegistrationError,
    UnknownTopicError,
    evaluate_rules,
    register_rule,
    registered_rules,
    research_rules,
)
from predictron_engine.research.rules import CoverageRule
from predictron_engine.research.topics import all_topic_ids

SEMANTIC_RULES = {
    "founders": MissingFoundersRule,
    "funding": MissingFundingRule,
    "pricing": MissingPricingRule,
    "market": MissingMarketRule,
    "technology": MissingTechnologyRule,
}


def _input(known: tuple[str, ...] = (), partial: tuple[str, ...] = ()) -> PlannerInput:
    return PlannerInput(
        company_name="Acme",
        known_topics=known,
        partial_topics=partial,
    )


def _topic(topic_id: str = "founders") -> ResearchTopic:
    return ResearchTopic(
        topic_id=topic_id,
        name=topic_id.title(),
        description=f"Assess {topic_id}.",
        importance=0.9,
        prediction_impact=0.9,
        freshness_requirement_days=365,
        freshness_sensitivity=0.5,
        source_categories=("linkedin",),
    )


@pytest.fixture(autouse=True)
def _clean_extension_registry() -> None:
    """Extension rules must never leak across tests."""
    yield
    from predictron_engine.research import rules as _rules_module

    _rules_module._EXTENSIONS.clear()  # type: ignore[attr-defined]


class TestSemanticRules:
    """The named rules cover the spec's example mappings."""

    def test_missing_topic_creates_gap(self) -> None:
        for topic_id, rule_type in SEMANTIC_RULES.items():
            covered = tuple(t for t in all_topic_ids() if t != topic_id)
            gaps = evaluate_rules(_input(known=covered), rules=(rule_type(),))
            assert [gap.topic_id for gap in gaps] == [topic_id]
            assert gaps[0].evidence_status is EvidenceStatus.NONE

    def test_rule_suppressed_when_known(self) -> None:
        assert MissingFoundersRule().evaluate(_input(known=("founders",))) is None

    def test_rule_reports_partial_evidence(self) -> None:
        gap = MissingFundingRule().evaluate(_input(partial=("funding",)))
        assert gap is not None
        assert gap.evidence_status is EvidenceStatus.PARTIAL


class TestBuiltinRules:
    """The built-in registry is complete and deterministic."""

    def test_default_rules_cover_all_topics(self) -> None:
        assert {rule.topic_id for rule in DEFAULT_RULES} == set(all_topic_ids())

    def test_empty_input_gaps_all_topics(self) -> None:
        gaps = evaluate_rules(_input())
        assert tuple(gap.topic_id for gap in gaps) == all_topic_ids()
        assert all(gap.evidence_status is EvidenceStatus.NONE for gap in gaps)

    def test_known_topics_produce_no_gaps(self) -> None:
        assert evaluate_rules(_input(known=all_topic_ids())) == ()

    def test_covers_every_single_missing_topic(self) -> None:
        for missing in all_topic_ids():
            covered = tuple(t for t in all_topic_ids() if t != missing)
            gaps = evaluate_rules(_input(known=covered))
            assert [gap.topic_id for gap in gaps] == [missing]

    def test_duplicate_rules_deduplicated(self) -> None:
        rules = (MissingFoundersRule(), CoverageRule(_topic("founders")))
        gaps = evaluate_rules(_input(), rules=rules)
        assert tuple(gap.topic_id for gap in gaps) == ("founders",)


class TestExtensionRegistry:
    """Rules are extensible without changing planner logic."""

    def test_register_rule_appears_in_registry(self) -> None:
        register_rule(CoverageRule(_topic("partnerships")))
        assert any(r.topic_id == "partnerships" for r in registered_rules())
        assert any(r.topic_id == "partnerships" for r in research_rules())

    def test_duplicate_registration_rejected(self) -> None:
        register_rule(CoverageRule(_topic("founders")))
        with pytest.raises(RuleRegistrationError):
            register_rule(CoverageRule(_topic("founders")))

    def test_non_rule_registration_rejected(self) -> None:
        with pytest.raises(RuleRegistrationError):
            register_rule(object())  # type: ignore[arg-type]

    def test_registered_rule_overrides_builtin(self) -> None:
        class SuppressingRule(CoverageRule):
            def evaluate(self, planner_input: PlannerInput) -> None:  # type: ignore[override]
                return None

        register_rule(SuppressingRule(_topic("founders")))
        gaps = evaluate_rules(_input())
        assert "founders" not in {gap.topic_id for gap in gaps}

    def test_external_topic_rule_requires_custom_topics(self) -> None:
        register_rule(CoverageRule(_topic("unicorns")))
        with pytest.raises(UnknownTopicError):
            ResearchPlanner().create_plan(_input())

    def test_external_topic_plans_with_custom_registry(self) -> None:
        custom_topic = _topic("unicorns")
        registry = {**dict(TOPIC_REGISTRY), "unicorns": custom_topic}
        register_rule(CoverageRule(custom_topic))
        plan = ResearchPlanner(topics=registry).create_plan(_input())
        assert plan.task_by_topic("unicorns") is not None
        assert plan.topics_researched == tuple(
            sorted(all_topic_ids() + ("unicorns",))
        )
