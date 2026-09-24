"""Tests for the SourceDiscovery engine.

Covers plan-to-source mapping, deterministic ranking, serialization,
empty plans, multiple tasks, dependency preservation, invalid inputs,
custom registries, and the fallback rule set.
"""

from __future__ import annotations

import json

import pytest

from predictron_engine.research import (
    PlannerInput,
    ResearchPlan,
    ResearchPlanner,
    ResearchSource,
    SourceCategory,
    SourceDiscovery,
    SourceDiscoveryPlan,
    SourceRegistry,
    all_topic_ids,
)
from predictron_engine.research.exceptions import InvalidDiscoveryInputError
from predictron_engine.research.source_registry import SourceDiscoveryRule

ALL_TOPIC_IDS = all_topic_ids()


def _all_known_input() -> PlannerInput:
    return PlannerInput(company_name="Acme", known_topics=ALL_TOPIC_IDS)


def _plan_for(company_name: str = "Acme", **kwargs: object) -> ResearchPlan:
    return ResearchPlanner().plan_for_company(
        company_name,
        **kwargs,  # type: ignore[arg-type]
    )


class TestDiscoveryBasics:
    """A plan maps cleanly to a serializable discovery plan."""

    def test_discover_produces_plan(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        discovered = SourceDiscovery().discover(plan)
        assert isinstance(discovered, SourceDiscoveryPlan)
        assert discovered.plan_id == plan.plan_id
        assert discovered.company_name == "Acme"
        assert discovered.schema_version == "1.0.0"

    def test_every_task_receives_sources(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        discovered = SourceDiscovery().discover(plan)
        assert len(discovered.recommendations) == len(plan.tasks)
        for recommendation in discovered.recommendations:
            assert recommendation.sources
            assert recommendation.has_recommendations()

    def test_recommendations_match_tasks(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        discovered = SourceDiscovery().discover(plan)
        for task in plan.tasks:
            recommendation = discovered.recommendation_for(task.topic_id)
            assert recommendation is not None
            assert recommendation.task_id == task.task_id
            assert recommendation.topic_id == task.topic_id

    def test_all_topics_covered(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        discovered = SourceDiscovery().discover(plan)
        assert set(discovered.topics_covered) == set(ALL_TOPIC_IDS)


class TestRankingIntegration:
    """Discovery output is consistent with the deterministic ranker."""

    def test_sources_sorted_by_score_desc(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        discovered = SourceDiscovery().discover(plan)
        for recommendation in discovered.recommendations:
            totals = [entry.score.total for entry in recommendation.sources]
            assert totals == sorted(totals, reverse=True)
            assert [entry.rank for entry in recommendation.sources] == list(
                range(1, len(totals) + 1)
            )

    def test_expected_sources_rank_for_funding(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(
            PlannerInput(
                company_name="Acme",
                known_topics=tuple(t for t in ALL_TOPIC_IDS if t != "funding"),
            )
        )
        discovered = SourceDiscovery().discover(plan)
        identifiers = discovered.source_identifiers_for("funding")
        assert "crunchbase" in identifiers
        assert "sec_edgar" in identifiers
        assert "press_releases" in identifiers


class TestDeterminism:
    """The same plan always yields the same discovery plan."""

    def test_two_runs_equal(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        first = SourceDiscovery().discover(plan)
        second = SourceDiscovery().discover(plan)
        assert first == second
        assert first.to_dict() == second.to_dict()
        assert SourceDiscovery().discover(plan) == first


class TestSerialization:
    """SourceDiscoveryPlan is fully serializable."""

    def test_roundtrip_and_json(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        discovered = SourceDiscovery().discover(plan)
        assert SourceDiscoveryPlan.from_dict(discovered.to_dict()) == discovered
        raw = json.dumps(discovered.to_dict())
        restored = SourceDiscoveryPlan.from_dict(json.loads(raw))
        assert restored == discovered
        assert restored.sources_for("funding") == discovered.sources_for(
            "funding"
        )

    def test_empty_plan_roundtrip(self) -> None:
        plan = _plan_for(known_topics=ALL_TOPIC_IDS)
        discovered = SourceDiscovery().discover(plan)
        assert discovered.recommendations == ()
        assert discovered.topics_covered == ()
        json.dumps(discovered.to_dict())
        assert SourceDiscoveryPlan.from_dict(discovered.to_dict()) == discovered


class TestEmptyAndPartialPlans:
    """Empty and subset plans behave correctly."""

    def test_empty_plan_has_no_recommendations(self) -> None:
        discovered = SourceDiscovery().discover(_plan_for(known_topics=ALL_TOPIC_IDS))
        assert discovered.recommendations == ()
        assert discovered.recommendation_for("funding") is None
        assert discovered.sources_for("funding") == ()

    def test_single_topic_plan(self) -> None:
        known = tuple(t for t in ALL_TOPIC_IDS if t != "funding")
        discovered = SourceDiscovery().discover(_plan_for(known_topics=known))
        assert len(discovered.recommendations) == 1
        assert discovered.topics_covered == ("funding",)

    def test_multiple_tasks_preserve_plan_order(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        discovered = SourceDiscovery().discover(plan)
        assert [r.topic_id for r in discovered.recommendations] == [
            task.topic_id for task in plan.tasks
        ]


class TestDependencyPreservation:
    """Discovery does not disturb the planner's task ordering or deps."""

    def test_source_tasks_match_plan_tasks(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        discovered = SourceDiscovery().discover(plan)
        assert discovered.source_tasks == plan.tasks

    def test_dependencies_survive_discovery(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        discovered = SourceDiscovery().discover(plan)
        for task in discovered.source_tasks:
            for dependency in task.dependency_ids:
                order = {t.topic_id: t.execution_order for t in discovered.source_tasks}
                assert order[dependency] < task.execution_order


class TestInvalidInputs:
    """Malformed input raises deterministic errors."""

    def test_non_plan_rejected(self) -> None:
        discovery = SourceDiscovery()
        for invalid in (None, "Acme", PlannerInput(company_name="Acme")):
            with pytest.raises(InvalidDiscoveryInputError):
                discovery.discover(invalid)  # type: ignore[arg-type]


class TestCustomRegistries:
    """Custom catalogs/rules drive discovery without logic changes."""

    def test_custom_source_and_rule(self) -> None:
        source = ResearchSource(
            identifier="quantum_lens",
            display_name="Quantum Lens",
            source_category=SourceCategory.RESEARCH_PAPERS,
            trust_score=0.9,
            freshness_score=0.5,
            coverage_score=0.8,
            relative_cost=0.1,
            supports_structured_data=True,
            preferred_topics=("technology",),
        )
        registry = SourceRegistry(
            sources=(source,),
            rules=(
                SourceDiscoveryRule(
                    topic_id="technology", source_identifiers=("quantum_lens",)
                ),
            ),
        )
        plan = ResearchPlanner().create_plan(
            PlannerInput(
                company_name="Acme",
                known_topics=tuple(t for t in ALL_TOPIC_IDS if t != "technology"),
            )
        )
        discovered = SourceDiscovery(registry=registry).discover(plan)
        identifiers = discovered.source_identifiers_for("technology")
        assert identifiers[0] == "quantum_lens"
        assert len(identifiers) == 1

    def test_unknown_topic_uses_fallback(self) -> None:
        registry = SourceRegistry(
            sources=(
                _single_source("solo_news", category=SourceCategory.NEWS),
                _single_source("solo_github"),
            ),
            rules=(
                SourceDiscoveryRule(
                    topic_id="other", source_identifiers=("solo_github",)
                ),
            ),
        )
        plan = ResearchPlanner().create_plan(
            PlannerInput(
                company_name="Acme",
                known_topics=tuple(t for t in ALL_TOPIC_IDS if t != "market"),
            )
        )
        # "market" has no rule in this registry, so the fallback is used.
        discovered = SourceDiscovery(registry=registry).discover(plan)
        identifiers = discovered.source_identifiers_for("market")
        assert identifiers  # fallback guarantees at least one source
        assert set(identifiers) == {"solo_github", "solo_news"}

    def test_fallback_is_deterministic(self) -> None:
        registry = SourceRegistry(sources=(_single_source("solo"),), rules=())
        plan = _plan_for(known_topics=tuple(t for t in ALL_TOPIC_IDS if t != "funding"))
        first = SourceDiscovery(registry=registry).discover(plan)
        second = SourceDiscovery(registry=registry).discover(plan)
        assert first == second


class TestConvenience:
    """discover_for_company matches the two-step public API."""

    def test_discover_for_company(self) -> None:
        direct = SourceDiscovery().discover(
            ResearchPlanner().create_plan(
                PlannerInput(company_name="Acme", website="https://acme.io")
            )
        )
        convenience = SourceDiscovery().discover_for_company(
            "Acme", website="https://acme.io"
        )
        assert direct == convenience
        assert convenience.company_name == "Acme"
        assert len(convenience.recommendations) > 0


def _single_source(
    identifier: str, category: SourceCategory = SourceCategory.GITHUB
) -> ResearchSource:
    return ResearchSource(
        identifier=identifier,
        display_name=identifier.title(),
        source_category=category,
        trust_score=0.7,
        freshness_score=0.6,
        coverage_score=0.6,
        relative_cost=0.1,
        supports_structured_data=True,
        preferred_topics=("technology",),
    )
