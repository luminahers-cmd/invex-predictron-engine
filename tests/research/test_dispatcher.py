"""Tests for the deterministic Dispatcher and its value objects.

Covers registry-only resolution, task-to-collector pairing in executable
order, unresolved tasks, empty plans, invalid inputs, and serialization
of the dispatch artifacts.
"""

from __future__ import annotations

import json

import pytest

from predictron_engine.research import (
    COLLECTOR_REGISTRY,
    CollectorDispatch,
    CollectorRegistry,
    Dispatcher,
    DispatchPlan,
    EvidenceCollector,
    EvidenceReference,
    PlannerInput,
    ResearchPlan,
    ResearchPriority,
    ResearchTask,
)
from predictron_engine.research.exceptions import (
    CollectorNotFound,
    InvalidCollectionInputError,
)
from predictron_engine.research.models import Evidence


def _task(topic_id: str, task_id: str | None = None) -> ResearchTask:
    return ResearchTask(
        task_id=task_id or f"research_{topic_id}",
        topic_id=topic_id,
        title=f"{topic_id} research",
        description="Research.",
        priority=ResearchPriority.HIGH,
        priority_score=60.0,
        importance=0.8,
        prediction_impact=0.9,
        freshness_requirement_days=90,
        dependency_ids=(),
        source_categories=(),
        priority_rank=1,
        execution_order=1,
    )


class _FundingCollector(EvidenceCollector):
    collector_id = "funding_collector"
    display_name = "Funding"
    supported_topics = ("funding",)

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple:
        del task, company_name, website
        return (
            Evidence(
                task_id="research_funding",
                topic_id="funding",
                collector_id="funding_collector",
                category="funding_rounds",
                claim="Recorded funding.",
                confidence=0.9,
                reference=EvidenceReference(
                    source_category="crunchbase",
                    source_identifier="crunchbase",
                ),
            ),
        )


class _ProductCollector(EvidenceCollector):
    collector_id = "product_collector"
    display_name = "Product"
    supported_topics = ("product",)

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple:
        del task, company_name, website
        return ()


def _registry() -> CollectorRegistry:
    return CollectorRegistry((_FundingCollector(), _ProductCollector()))


def _plan(tasks: tuple[ResearchTask, ...]) -> ResearchPlan:
    return ResearchPlan(
        schema_version="1.0.0",
        plan_id=f"plan_{len(tasks)}",
        company_name="Acme",
        input=PlannerInput(company_name="Acme"),
        gaps=(),
        tasks=tasks,
        topics_researched=tuple(task.topic_id for task in tasks),
        available_source_categories=("web_search",),
    )


class TestDispatch:
    """Tasks pair with collectors through the registry, in order."""

    def test_dispatch_pairs_tasks_to_collectors(self) -> None:
        plan = Dispatcher(_registry()).dispatch(
            (_task("funding"), _task("product"))
        )
        assert plan.task_ids == ("research_funding", "research_product")
        assert plan.topics_dispatched == ("funding", "product")
        assert plan.collector_ids == (
            "funding_collector",
            "product_collector",
        )
        assert plan.is_complete()

    def test_dispatch_resolves_only_through_registry(self) -> None:
        plan = Dispatcher(_registry()).dispatch((_task("funding"),))
        assert plan.dispatches[0].collector.metadata().collector_id == (
            "funding_collector"
        )

    def test_dispatch_preserves_input_order(self) -> None:
        plan = Dispatcher(_registry()).dispatch(
            (_task("product"), _task("funding"))
        )
        assert plan.task_ids == ("research_product", "research_funding")

    def test_unresolved_tasks_reported(self) -> None:
        plan = Dispatcher(_registry()).dispatch(
            (_task("funding"), _task("team"), _task("product"))
        )
        assert plan.is_complete() is False
        assert plan.unresolved == ("research_team",)
        assert plan.task_ids == ("research_funding", "research_product")

    def test_dispatch_for_task(self) -> None:
        plan = Dispatcher(_registry()).dispatch(
            (_task("funding"), _task("product"))
        )
        entry = plan.dispatch_for_task("research_funding")
        assert entry is not None
        assert entry.task.topic_id == "funding"
        assert plan.dispatch_for_task("missing") is None

    def test_empty_tasks_produce_empty_plan(self) -> None:
        plan = Dispatcher(_registry()).dispatch(())
        assert plan.dispatches == ()
        assert plan.unresolved == ()
        assert plan.is_complete()

    def test_non_task_entry_rejected(self) -> None:
        with pytest.raises(InvalidCollectionInputError):
            Dispatcher(_registry()).dispatch(("funding",))  # type: ignore[list-item]

    def test_plan_convenience(self) -> None:
        plan = _plan((_task("funding"), _task("team")))
        dispatched = Dispatcher(_registry()).dispatch_plan(plan)
        assert dispatched.unresolved == ("research_team",)
        assert dispatched.task_ids == ("research_funding",)

    def test_plan_convenience_rejects_non_plan(self) -> None:
        with pytest.raises(InvalidCollectionInputError):
            Dispatcher(_registry()).dispatch_plan("Acme")  # type: ignore[arg-type]


class TestDispatchPlanModel:
    """DispatchPlan validates its payload."""

    def test_entries_must_be_collector_dispatches(self) -> None:
        with pytest.raises(InvalidCollectionInputError):
            DispatchPlan(dispatches=(_task("funding"),))  # type: ignore[arg-type]

    def test_unresolved_deduped_preserving_order(self) -> None:
        plan = DispatchPlan(unresolved=("a", "b", "a"))
        assert plan.unresolved == ("a", "b")

    def test_defaults(self) -> None:
        plan = DispatchPlan()
        assert plan.dispatches == ()
        assert plan.unresolved == ()
        assert plan.task_ids == ()
        assert plan.collector_ids == ()


class TestSerialization:
    """Dispatch artifacts serialize and restore losslessly."""

    def test_dispatch_plan_roundtrip_with_registry(self) -> None:
        registry = _registry()
        plan = Dispatcher(registry).dispatch(
            (_task("funding"), _task("product"), _task("team"))
        )
        restored = DispatchPlan.from_dict(
            json.loads(json.dumps(plan.to_dict())), registry=registry
        )
        assert restored == plan
        assert restored.unresolved == ("research_team",)

    def test_dispatch_plan_roundtrip_without_registry(self) -> None:
        plan = Dispatcher(COLLECTOR_REGISTRY).dispatch(
            (_task("funding"), _task("team"))
        )
        restored = DispatchPlan.from_dict(plan.to_dict())
        assert restored == plan

    def test_collector_dispatch_roundtrip(self) -> None:
        registry = _registry()
        plan = Dispatcher(registry).dispatch((_task("funding"),))
        dispatch = plan.dispatches[0]
        restored = CollectorDispatch.from_dict(dispatch.to_dict(), registry=registry)
        assert restored == dispatch

    def test_unknown_collector_restore_raises(self) -> None:
        plan = Dispatcher(_registry()).dispatch((_task("funding"),))
        dispatch = plan.dispatches[0]
        data = dispatch.to_dict()
        with pytest.raises(CollectorNotFound):
            CollectorDispatch.from_dict(data, registry=CollectorRegistry())

    def test_malformed_dict_raises(self) -> None:
        with pytest.raises(ValueError):
            DispatchPlan.from_dict({"dispatches": "bad", "unresolved": []})
