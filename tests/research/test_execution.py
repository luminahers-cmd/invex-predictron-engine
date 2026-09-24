"""Tests for the EvidenceCollectionEngine — Phase 8 Sprint 3.

Covers the public API, deterministic ordering and aggregation, status
derivation, dependency-order execution, partial coverage, error tiers,
integration with the planner, and result serialization.
"""

from __future__ import annotations

import json

import pytest

from predictron_engine.research import (
    CollectorRegistry,
    Evidence,
    EvidenceCollector,
    EvidenceReference,
    PlannerInput,
    ResearchPlan,
    ResearchPlanner,
    ResearchPriority,
    ResearchTask,
)
from predictron_engine.research.exceptions import (
    CollectorExecutionError,
    InvalidCollectionInputError,
    UnsupportedTopic,
)
from predictron_engine.research.execution import EvidenceCollectionEngine
from predictron_engine.research.models import (
    CollectionResult,
    CollectionStatus,
)


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


def _plan(*tasks: ResearchTask) -> ResearchPlan:
    return ResearchPlan(
        schema_version="1.0.0",
        plan_id=f"plan_{len(tasks)}",
        company_name="Acme",
        input=PlannerInput(company_name="Acme"),
        gaps=(),
        tasks=tuple(tasks),
        topics_researched=tuple(task.topic_id for task in tasks),
        available_source_categories=("web_search",),
    )


def _reference() -> EvidenceReference:
    return EvidenceReference(
        source_category="web_search",
        source_identifier="placeholder",
    )


class _GoodCollector(EvidenceCollector):
    collector_id = "good_collector"
    display_name = "Good"
    supported_topics = ("product",)

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]:
        return (
            Evidence(
                task_id=task.task_id,
                topic_id="product",
                collector_id="good_collector",
                category="product_fit",
                claim=f"Product claim for {company_name}.",
                confidence=0.8,
                reference=_reference(),
            ),
        )


class _TeamCollector(EvidenceCollector):
    collector_id = "team_collector"
    display_name = "Team"
    supported_topics = ("team",)

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]:
        return (
            Evidence(
                task_id=task.task_id,
                topic_id="team",
                collector_id="team_collector",
                category="team_profile",
                claim=f"Team for {company_name}.",
                confidence=0.7,
                reference=_reference(),
            ),
        )


class _UnsupportedCollector(EvidenceCollector):
    """Raises the designed UnsupportedTopic signal while collecting."""

    collector_id = "unsupported_collector"
    display_name = "Unsupported"
    supported_topics = ("funding",)

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]:
        raise UnsupportedTopic(f"cannot collect topic {task.topic_id!r}")


class _NonEvidenceCollector(EvidenceCollector):
    """Returns garbage to violate the evidence contract."""

    collector_id = "garbage_collector"
    display_name = "Garbage"
    supported_topics = ("funding",)

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]:
        return ("junk",)  # type: ignore[return-value]


class _CrashCollector(EvidenceCollector):
    """Crashes with an unexpected exception."""

    collector_id = "crash_collector"
    display_name = "Crash"
    supported_topics = ("funding",)

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]:
        raise ValueError("boom")


class _WrappedCrashCollector(EvidenceCollector):
    """Raises CollectorExecutionError directly."""

    collector_id = "wrapped_crash_collector"
    display_name = "Wrapped"
    supported_topics = ("funding",)

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]:
        raise CollectorExecutionError("wrapped crash")


class TestPublicApi:
    """The documented engine surface behaves as specified."""

    def test_single_topic_plan_succeeds(self) -> None:
        engine = EvidenceCollectionEngine(CollectorRegistry((_GoodCollector(),)))
        result = engine.execute(_plan(_task("product")))
        assert result.status is CollectionStatus.SUCCESS
        assert result.executed_tasks == ("research_product",)
        assert result.failed_tasks == ()
        assert result.unresolved_tasks == ()
        assert len(result.collection.evidence) == 1

    def test_custom_collector_can_cover_uncovered_topic(self) -> None:
        engine = EvidenceCollectionEngine(CollectorRegistry((_TeamCollector(),)))
        result = engine.execute(_plan(_task("team")))
        assert result.status is CollectionStatus.SUCCESS
        assert result.executed_tasks == ("research_team",)
        assert result.collection.has_topic("team")

    def test_invalid_plan_rejected(self) -> None:
        engine = EvidenceCollectionEngine()
        with pytest.raises(InvalidCollectionInputError):
            engine.execute("Acme")  # type: ignore[arg-type]
        with pytest.raises(InvalidCollectionInputError):
            engine.execute(None)  # type: ignore[arg-type]

    def test_execute_for_company_returns_result(self) -> None:
        result = EvidenceCollectionEngine().execute_for_company("Acme")
        assert isinstance(result, CollectionResult)
        assert result.company_name == "Acme"
        assert result.executed_tasks
        assert len(result.collection.evidence) == 3 * len(result.executed_tasks)

    def test_execute_for_company_accepts_known_topics(self) -> None:
        result = EvidenceCollectionEngine().execute_for_company(
            "Acme", known_topics=("funding",)
        )
        assert "research_funding" not in result.executed_tasks
        assert len(result.executed_tasks) == 14
        assert len(result.collection.evidence) == 42


class TestDefaultPlanExecution:
    """The full default plan produces a deterministic partial result."""

    def test_full_plan_runs_all_supported_topics(self) -> None:
        plan = ResearchPlanner().plan_for_company("Acme")
        result = EvidenceCollectionEngine().execute(plan)
        assert result.status is CollectionStatus.PARTIAL
        assert len(result.executed_tasks) == 15
        assert len(result.collection.evidence) == 45

    def test_full_plan_reports_uncovered_pair(self) -> None:
        plan = ResearchPlanner().plan_for_company("Acme")
        result = EvidenceCollectionEngine().execute(plan)
        assert set(result.failed_tasks) == {
            "research_team",
            "research_business_model",
        }
        assert set(result.unresolved_tasks) == {
            "research_team",
            "research_business_model",
        }

    def test_execution_follows_plan_order(self) -> None:
        plan = ResearchPlanner().plan_for_company("Acme")
        result = EvidenceCollectionEngine().execute(plan)
        expected = [
            task.task_id
            for task in plan.tasks
            if task.task_id in result.executed_tasks
        ]
        assert list(result.executed_tasks) == expected

    def test_evidence_is_aggregated_and_unique(self) -> None:
        plan = ResearchPlanner().plan_for_company("Acme")
        result = EvidenceCollectionEngine().execute(plan)
        evidence_ids = [item.evidence_id for item in result.collection.evidence]
        assert len(evidence_ids) == len(set(evidence_ids)) == 45


class TestStatusDerivation:
    """CollectionStatus is derived from task outcomes."""

    def test_empty_plan_is_empty(self) -> None:
        result = EvidenceCollectionEngine(CollectorRegistry((_GoodCollector(),))).execute(
            _plan()
        )
        assert result.status is CollectionStatus.EMPTY
        assert result.executed_tasks == ()
        assert result.failed_tasks == ()
        assert len(result.collection.evidence) == 0

    def test_all_unresolved_is_failed(self) -> None:
        engine = EvidenceCollectionEngine(CollectorRegistry(()))
        result = engine.execute(
            _plan(_task("team", "research_team"), _task("pricing", "research_pricing"))
        )
        assert result.status is CollectionStatus.FAILED
        assert result.executed_tasks == ()
        assert set(result.failed_tasks) == {"research_team", "research_pricing"}

    def test_all_unresolved_with_empty_registry(self) -> None:
        engine = EvidenceCollectionEngine(CollectorRegistry())
        result = engine.execute(_plan(_task("funding")))
        assert result.status is CollectionStatus.FAILED
        assert result.unresolved_tasks == ("research_funding",)


class TestErrorTiers:
    """Designed signals degrade gracefully; crashes surface."""

    def test_unsupported_topic_recorded_as_failed(self) -> None:
        engine = EvidenceCollectionEngine(
            CollectorRegistry((_UnsupportedCollector(), _GoodCollector()))
        )
        result = engine.execute(_plan(_task("funding"), _task("product")))
        assert result.status is CollectionStatus.PARTIAL
        assert result.executed_tasks == ("research_product",)
        assert result.failed_tasks == ("research_funding",)
        assert len(result.collection.evidence) == 1

    def test_invalid_evidence_recorded_as_failed(self) -> None:
        engine = EvidenceCollectionEngine(
            CollectorRegistry((_NonEvidenceCollector(), _GoodCollector()))
        )
        result = engine.execute(_plan(_task("funding"), _task("product")))
        assert result.status is CollectionStatus.PARTIAL
        assert result.executed_tasks == ("research_product",)
        assert result.failed_tasks == ("research_funding",)

    def test_crash_surfaces_as_collector_execution_error(self) -> None:
        engine = EvidenceCollectionEngine(
            CollectorRegistry((_CrashCollector(), _GoodCollector()))
        )
        with pytest.raises(CollectorExecutionError) as exc_info:
            engine.execute(_plan(_task("funding"), _task("product")))
        assert "crash_collector" in str(exc_info.value)

    def test_wrapped_crash_passes_through(self) -> None:
        engine = EvidenceCollectionEngine(
            CollectorRegistry((_WrappedCrashCollector(),))
        )
        with pytest.raises(CollectorExecutionError) as exc_info:
            engine.execute(_plan(_task("funding")))
        assert "wrapped crash" in str(exc_info.value)

    def test_unknown_topic_resolution_never_reaches_execution(self) -> None:
        engine = EvidenceCollectionEngine(CollectorRegistry(()))
        result = engine.execute(_plan(_task("funding"), _task("product")))
        assert result.status is CollectionStatus.FAILED
        assert result.unresolved_tasks == ("research_funding", "research_product")


class TestDeterminism:
    """Identical plans and registries give identical results."""

    def test_repeated_execution_is_identical(self) -> None:
        engine = EvidenceCollectionEngine(CollectorRegistry((_GoodCollector(),)))
        first = engine.execute(_plan(_task("product")))
        second = engine.execute(_plan(_task("product")))
        assert first == second
        assert first.collection_id == second.collection_id

    def test_two_engines_agree(self) -> None:
        plan = ResearchPlanner().plan_for_company("Acme")
        left = EvidenceCollectionEngine().execute(plan)
        right = EvidenceCollectionEngine().execute(plan)
        assert left == right
        assert [e.evidence_id for e in left.collection.evidence] == [
            e.evidence_id for e in right.collection.evidence
        ]

    def test_result_serialization_roundtrip(self) -> None:
        result = EvidenceCollectionEngine().execute_for_company("Acme")
        restored = CollectionResult.from_dict(
            json.loads(json.dumps(result.to_dict()))
        )
        assert restored == result

    def test_collection_id_is_stable_across_roundtrips(self) -> None:
        result = EvidenceCollectionEngine().execute_for_company("Acme")
        restored = CollectionResult.from_dict(result.to_dict())
        assert restored.collection_id == result.collection_id
        assert restored.collection_id.startswith("collection_")
