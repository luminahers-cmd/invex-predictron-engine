"""Deterministic Research Planner — Phase 8 Sprint 1.

The planner is the orchestration layer for every future research
pipeline: it decides *what information is missing, what research should
be performed, and in what order*.

The planner is pure:

* no HTTP requests, search APIs, browser automation, or scraping
* no LLMs or external services
* no background jobs or randomness

The same input always produces the same plan (deterministic priority,
deterministic dependency ordering, deterministic serialization).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from predictron_engine.research.exceptions import (
    DependencyCycleError,
    InvalidPlannerInputError,
    UnknownTopicError,
)
from predictron_engine.research.models import (
    PLAN_SCHEMA_VERSION,
    KnowledgeGap,
    PlannerInput,
    ResearchPlan,
    ResearchTask,
    ResearchTopic,
)
from predictron_engine.research.priorities import (
    compute_priority_score,
    priority_from_score,
)
from predictron_engine.research.rules import (
    ResearchRule,
    evaluate_rules,
    research_rules,
)
from predictron_engine.research.topics import TOPIC_REGISTRY

# Source categories the planner treats as already available.
_AVAILABLE_SOURCE_CATEGORIES: tuple[str, ...] = ("web_search",)
_WEBSITE_SOURCE_CATEGORY: str = "company_website"


class ResearchPlanner:
    """Deterministic planning engine for the research layer.

    Parameters
    ----------
    rules:
        Optional rule set used for gap detection.  When ``None`` the
        registered rule set is used.
    topics:
        Optional topic registry overriding the default taxonomy.  Keys
        must equal ``topic.topic_id`` for every value.
    """

    def __init__(
        self,
        rules: Sequence[ResearchRule] | None = None,
        topics: Mapping[str, ResearchTopic] | None = None,
    ) -> None:
        self._rules: tuple[ResearchRule, ...] = (
            tuple(rules) if rules is not None else research_rules()
        )
        if topics is None:
            self._topics: Mapping[str, ResearchTopic] = TOPIC_REGISTRY
        else:
            registry = dict(topics)
            for topic in registry.values():
                if topic.topic_id not in registry:
                    raise ValueError(
                        f"topic key mismatch for: {topic.topic_id!r}"
                    )
            self._topics = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_plan(self, planner_input: PlannerInput) -> ResearchPlan:
        """Produce a complete :class:`ResearchPlan` for ``planner_input``."""
        if not isinstance(planner_input, PlannerInput):
            raise InvalidPlannerInputError(
                "create_plan expects a PlannerInput instance"
            )
        self._validate_known_topics(planner_input)
        gaps = evaluate_rules(planner_input, rules=self._rules)
        tasks = self._build_tasks(planner_input, gaps)
        ordered = self._executable_order(tasks)
        ranked = self._assign_priority_ranks(ordered)
        plan_id = self._compute_plan_id(planner_input, gaps, ranked)
        return ResearchPlan(
            schema_version=PLAN_SCHEMA_VERSION,
            plan_id=plan_id,
            company_name=planner_input.company_name,
            input=planner_input,
            gaps=gaps,
            tasks=ranked,
            topics_researched=tuple(sorted(gap.topic_id for gap in gaps)),
            available_source_categories=self._available_sources(planner_input),
        )

    def plan_for_company(
        self,
        company_name: str,
        *,
        website: str = "",
        description: str = "",
        known_topics: Sequence[str] = (),
        partial_topics: Sequence[str] = (),
        prediction_horizon_days: int = 365,
    ) -> ResearchPlan:
        """Convenience wrapper building a plan from plain fields."""
        return self.create_plan(
            PlannerInput(
                company_name=company_name,
                website=website,
                description=description,
                known_topics=tuple(known_topics),
                partial_topics=tuple(partial_topics),
                prediction_horizon_days=prediction_horizon_days,
            )
        )

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------

    def _validate_known_topics(self, planner_input: PlannerInput) -> None:
        """Reject planner inputs referencing topics outside the registry."""
        referenced = set(planner_input.known_topics) | set(
            planner_input.partial_topics
        )
        unknown = sorted(referenced - set(self._topics))
        if unknown:
            raise InvalidPlannerInputError(
                f"input references unknown topics: {unknown}"
            )

    def _resolve_topic(self, topic_id: str) -> ResearchTopic:
        topic = self._topics.get(topic_id)
        if topic is None:
            raise UnknownTopicError(f"unknown research topic: {topic_id!r}")
        return topic

    def _topic_order(self, topic_id: str) -> int:
        for position, known in enumerate(self._topics):
            if known == topic_id:
                return position
        raise UnknownTopicError(f"unknown research topic: {topic_id!r}")

    def _build_tasks(
        self,
        planner_input: PlannerInput,
        gaps: tuple[KnowledgeGap, ...],
    ) -> tuple[ResearchTask, ...]:
        """Build an unranked task per gap, in deterministic rule order."""
        selected = {gap.topic_id for gap in gaps}
        tasks: list[ResearchTask] = []
        for gap in gaps:
            topic = self._resolve_topic(gap.topic_id)
            topic_order = self._topic_order
            dependency_ids = tuple(
                sorted(topic.dependencies, key=topic_order)
            )
            dependency_factor = self._dependency_factor(gap.topic_id, selected)
            score = compute_priority_score(
                topic,
                gap.evidence_status,
                dependency_factor=dependency_factor,
                prediction_horizon_days=planner_input.prediction_horizon_days,
                website_present=bool(planner_input.website),
            )
            tasks.append(
                ResearchTask(
                    task_id=f"research_{gap.topic_id}",
                    topic_id=gap.topic_id,
                    title=f"{topic.name} research",
                    description=topic.description,
                    priority=priority_from_score(score),
                    priority_score=score,
                    importance=topic.importance,
                    prediction_impact=topic.prediction_impact,
                    freshness_requirement_days=(
                        topic.freshness_requirement_days
                    ),
                    dependency_ids=(
                        tuple(dep for dep in dependency_ids if dep in selected)
                    ),
                    source_categories=topic.source_categories,
                    priority_rank=0,
                    execution_order=0,
                )
            )
        return tuple(tasks)

    def _dependency_factor(
        self,
        topic_id: str,
        selected: set[str],
    ) -> float:
        """Return 1.0 when the topic unlocks another selected topic."""
        for candidate in selected:
            if candidate == topic_id:
                continue
            topic = self._resolve_topic(candidate)
            if topic_id in topic.dependencies:
                return 1.0
        return 0.0

    def _assign_priority_ranks(
        self,
        tasks: tuple[ResearchTask, ...],
    ) -> tuple[ResearchTask, ...]:
        """Assign deterministic 1-based priority ranks to tasks."""
        ordered = sorted(
            tasks,
            key=lambda t: (-t.priority_score, self._topic_order(t.topic_id)),
        )
        rank_by_topic = {
            task.topic_id: position + 1
            for position, task in enumerate(ordered)
        }
        return tuple(
            replace(task, priority_rank=rank_by_topic[task.topic_id])
            for task in tasks
        )

    def _executable_order(
        self,
        tasks: tuple[ResearchTask, ...],
    ) -> tuple[ResearchTask, ...]:
        """Resolve a deterministic executable order respecting dependencies.

        Uses a Kahn-style topological sort.  When multiple tasks are
        ready, the highest-priority task (score desc, taxonomy position
        asc) is scheduled first.
        """
        by_topic = {task.topic_id: task for task in tasks}
        in_degree = {
            task.topic_id: len(task.dependency_ids) for task in tasks
        }
        dependents: dict[str, list[str]] = {
            task.topic_id: [] for task in tasks
        }
        for task in tasks:
            for dependency in task.dependency_ids:
                dependents[dependency].append(task.topic_id)

        remaining: set[str] = set(by_topic)
        ordered: list[str] = []

        def _pick() -> str:
            ready = [
                topic_id
                for topic_id in remaining
                if in_degree[topic_id] == 0
            ]
            if not ready:
                raise DependencyCycleError(
                    "research task dependencies form a cycle"
                )
            return min(
                ready,
                key=lambda tid: (
                    -by_topic[tid].priority_score,
                    self._topic_order(tid),
                ),
            )

        while remaining:
            topic_id = _pick()
            ordered.append(topic_id)
            remaining.discard(topic_id)
            for dependent in dependents[topic_id]:
                in_degree[dependent] -= 1

        return tuple(
            replace(by_topic[topic_id], execution_order=position + 1)
            for position, topic_id in enumerate(ordered)
        )

    @staticmethod
    def _available_sources(planner_input: PlannerInput) -> tuple[str, ...]:
        """Return the source categories available for planning."""
        if planner_input.website:
            return (_WEBSITE_SOURCE_CATEGORY,) + _AVAILABLE_SOURCE_CATEGORIES
        return _AVAILABLE_SOURCE_CATEGORIES

    @staticmethod
    def _compute_plan_id(
        planner_input: PlannerInput,
        gaps: tuple[KnowledgeGap, ...],
        tasks: tuple[ResearchTask, ...],
    ) -> str:
        """Return a stable content-hash identifier for the plan."""
        payload = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "company_name": planner_input.company_name,
            "website": planner_input.website,
            "description": planner_input.description,
            "prediction_horizon_days": planner_input.prediction_horizon_days,
            "known_topics": sorted(planner_input.known_topics),
            "partial_topics": sorted(planner_input.partial_topics),
            "gaps": [gap.topic_id for gap in gaps],
            "tasks": [task.task_id for task in tasks],
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"plan_{digest[:16]}"


def recommend_topics(planner_input: PlannerInput) -> tuple[str, ...]:
    """Return the topic identifiers the planner would research.

    Convenience entry point for callers that only need the topic set,
    e.g. feature flags or dashboards.  Pure and deterministic.
    """
    return ResearchPlanner().create_plan(planner_input).topics_researched
