"""Tests for the deterministic ResearchPlanner.

Covers planner creation, gap detection, rule-driven tasks, priority
ordering, dependency ordering, determinism, serialization, and input
validation.
"""

from __future__ import annotations

import json

import pytest

from predictron_engine.research import (
    EvidenceStatus,
    InvalidPlannerInputError,
    PlannerInput,
    ResearchPlan,
    ResearchPlanner,
    ResearchPriority,
    all_topic_ids,
    recommend_topics,
)
from predictron_engine.research.rules import MissingFoundersRule

ALL_TOPICS = all_topic_ids()
DEPENDENCY_PAIRS = (
    ("founders", "team"),
    ("product", "market"),
    ("product", "customers"),
    ("customers", "traction"),
    ("technology", "competition"),
)
MISSING_TOPICS = ("founders", "funding", "pricing", "market", "technology")


def _known_except(excluded: str) -> tuple[str, ...]:
    return tuple(t for t in ALL_TOPICS if t != excluded)


class TestPlannerCreation:
    """Construction with defaults and custom configuration."""

    def test_default_construction(self, planner: ResearchPlanner) -> None:
        assert planner.create_plan(PlannerInput(company_name="Acme")) is not None

    def test_custom_rules_scope_plan(self) -> None:
        planner = ResearchPlanner(rules=(MissingFoundersRule(),))
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        assert {task.topic_id for task in plan.tasks} == {"founders"}


class TestBasicPlanning:
    """Company-only and company+website inputs."""

    def test_empty_input_plans_all_topics(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        assert len(plan.tasks) == len(ALL_TOPICS)
        assert len(plan.gaps) == len(ALL_TOPICS)
        assert {task.topic_id for task in plan.tasks} == set(ALL_TOPICS)
        assert plan.company_name == "Acme"
        assert plan.schema_version == "1.0.0"
        assert plan.plan_id.startswith("plan_")
        assert plan.available_source_categories == ("web_search",)

    def test_company_with_website(self, planner: ResearchPlanner) -> None:
        with_website = planner.create_plan(
            PlannerInput(company_name="Acme", website="https://acme.io")
        )
        bare = planner.create_plan(PlannerInput(company_name="Acme"))
        assert "company_website" in with_website.available_source_categories
        assert with_website.task_by_topic("product").priority_score > bare.task_by_topic(
            "product"
        ).priority_score

    def test_task_metadata_populated(self, planner: ResearchPlanner) -> None:
        task = planner.create_plan(PlannerInput(company_name="Acme")).task_by_topic(
            "market"
        )
        assert task is not None
        assert task.title == "Market research"
        assert task.description
        assert task.source_categories
        assert task.freshness_requirement_days >= 1


class TestKnowledgeGapDetection:
    """Missing topics become tasks; known topics do not."""

    def test_missing_topic_task_created(
        self, planner: ResearchPlanner
    ) -> None:
        for excluded in MISSING_TOPICS:
            plan = planner.create_plan(
                PlannerInput(
                    company_name="Acme", known_topics=_known_except(excluded)
                )
            )
            assert plan.task_by_topic(excluded) is not None
            assert len(plan.tasks) == 1

    def test_all_known_produces_empty_plan(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(
            PlannerInput(company_name="Acme", known_topics=ALL_TOPICS)
        )
        assert plan.tasks == ()
        assert plan.gaps == ()

    def test_partial_topic_creates_partial_gap(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(
            PlannerInput(
                company_name="Acme",
                known_topics=_known_except("market"),
                partial_topics=("market",),
            )
        )
        assert len(plan.gaps) == 1
        assert plan.gaps[0].evidence_status is EvidenceStatus.PARTIAL


class TestDependencyOrdering:
    """Executable order respects the dependency DAG."""

    def _order_map(self, plan: ResearchPlan) -> dict[str, int]:
        return {task.topic_id: task.execution_order for task in plan.tasks}

    def test_prerequisite_before_dependent(
        self, planner: ResearchPlanner
    ) -> None:
        order = self._order_map(planner.create_plan(PlannerInput(company_name="Acme")))
        for prerequisite, dependent in DEPENDENCY_PAIRS:
            assert order[prerequisite] < order[dependent]
        assert order["founders"] < order["team"] < order["hiring"]
        assert order["product"] < order["customers"] < order["traction"]

    def test_every_task_after_its_dependencies(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        order = self._order_map(plan)
        for task in plan.tasks:
            for dependency in task.dependency_ids:
                assert order[dependency] < order[task.topic_id]

    def test_execution_order_sequential(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        assert [t.execution_order for t in plan.tasks] == list(
            range(1, len(plan.tasks) + 1)
        )

    def test_dependencies_only_reference_planned_topics(
        self, planner: ResearchPlanner
    ) -> None:
        plan = planner.create_plan(
            PlannerInput(
                company_name="Acme",
                known_topics=tuple(t for t in ALL_TOPICS if t not in ("founders", "team")),
            )
        )
        assert plan.task_by_topic("team").dependency_ids == ("founders",)


class TestPriorityOrdering:
    """Priority ranks are deterministic and consistent."""

    def test_priority_ranks_sequential_unique(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        ranks = sorted(task.priority_rank for task in plan.tasks)
        assert ranks == list(range(1, len(plan.tasks) + 1))

    def test_priority_ordered_tasks_sorted_desc(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        scores = [task.priority_score for task in plan.priority_ordered_tasks()]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_match_score_ordering(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        by_score = sorted(
            plan.tasks,
            key=lambda t: (-t.priority_score, ALL_TOPICS.index(t.topic_id)),
        )
        for position, task in enumerate(by_score, start=1):
            assert task.priority_rank == position

    def test_priority_band_assigned(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        assert all(isinstance(t.priority, ResearchPriority) for t in plan.tasks)


class TestDeterminism:
    """The same input always produces the same plan."""

    def test_two_runs_equal_and_plan_id_stable(
        self, planner: ResearchPlanner
    ) -> None:
        first = planner.create_plan(PlannerInput(company_name="Acme"))
        second = planner.create_plan(PlannerInput(company_name="Acme"))
        assert first == second
        assert first.to_dict() == second.to_dict()
        assert first.plan_id == second.plan_id
        assert ResearchPlanner().create_plan(
            PlannerInput(company_name="Acme")
        ) == first

    def test_plan_id_changes_with_input(self, planner: ResearchPlanner) -> None:
        base = planner.create_plan(PlannerInput(company_name="Acme"))
        changed = planner.create_plan(
            PlannerInput(company_name="Acme", known_topics=("founders",))
        )
        assert base.plan_id != changed.plan_id


class TestSerialization:
    """ResearchPlan is fully serializable."""

    def test_roundtrip_and_json(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(PlannerInput(company_name="Acme"))
        assert ResearchPlan.from_dict(plan.to_dict()) == plan
        raw = json.dumps(plan.to_dict())
        assert ResearchPlan.from_dict(json.loads(raw)) == plan
        restored = ResearchPlan.from_dict(plan.to_dict())
        original = plan.task_by_topic("market")
        copy = restored.task_by_topic("market")
        assert original is not None and copy is not None
        assert copy.priority is original.priority
        assert copy.priority_score == original.priority_score

    def test_empty_plan_roundtrip(self, planner: ResearchPlanner) -> None:
        plan = planner.create_plan(
            PlannerInput(company_name="Acme", known_topics=ALL_TOPICS)
        )
        json.dumps(plan.to_dict())
        assert ResearchPlan.from_dict(plan.to_dict()) == plan


class TestInvalidInputs:
    """Malformed inputs raise deterministic errors."""

    def test_invalid_inputs_rejected(self, planner: ResearchPlanner) -> None:
        with pytest.raises(InvalidPlannerInputError):
            PlannerInput(company_name="  ")
        for invalid in (
            None,
            PlannerInput(company_name="Acme", known_topics=("bogus",)),
        ):
            with pytest.raises(InvalidPlannerInputError):
                planner.create_plan(invalid)  # type: ignore[arg-type]


class TestConvenience:
    """Convenience entry points behave like create_plan."""

    def test_plan_for_company(self, planner: ResearchPlanner) -> None:
        plan = planner.plan_for_company("Acme", website="https://acme.io")
        assert plan.company_name == "Acme"
        assert "company_website" in plan.available_source_categories

    def test_recommend_topics(self) -> None:
        assert set(recommend_topics(PlannerInput(company_name="Acme"))) == set(
            ALL_TOPICS
        )
        assert "founders" not in recommend_topics(
            PlannerInput(company_name="Acme", known_topics=("founders",))
        )
