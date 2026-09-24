"""Source Discovery engine — Phase 8 Sprint 2.

:class:`SourceDiscovery` consumes a :class:`ResearchPlan` produced by the
Research Planner (Sprint 1) and maps every :class:`ResearchTask` to a
ranked set of candidate research sources.

Responsibilities:

* Inspect each research task in a plan.
* Look up candidate sources via the source registry's discovery rules.
* Rank the candidates deterministically.
* Produce a fully serializable :class:`SourceDiscoveryPlan`.

This layer is **metadata only**: no scraping, no HTTP requests, no web
search, no browser automation, and no authentication.  The same plan
always produces the same discovery plan.
"""

from __future__ import annotations

from predictron_engine.research.exceptions import InvalidDiscoveryInputError
from predictron_engine.research.models import (
    SOURCE_SCHEMA_VERSION,
    ResearchPlan,
    ResearchTask,
    SourceDiscoveryPlan,
    SourceRecommendation,
)
from predictron_engine.research.planner import ResearchPlanner
from predictron_engine.research.source_ranker import rank_sources
from predictron_engine.research.source_registry import (
    SOURCE_REGISTRY,
    SourceRegistry,
)

# Deterministic fallback candidates used when a task's topic has no rule.
# Only identifiers present in the registry are used; when none are present
# the registry catalog (in canonical order) becomes the fallback.
_FALLBACK_SOURCE_IDS: tuple[str, ...] = (
    "official_website",
    "crunchbase",
    "news",
    "sec_edgar",
    "github",
    "linkedin",
)


class SourceDiscovery:
    """Deterministic source selection layer for research plans.

    Parameters
    ----------
    registry:
        Optional :class:`SourceRegistry` overriding the default catalog
        and rules.  When ``None`` the built-in registry is used.
    """

    def __init__(self, registry: SourceRegistry | None = None) -> None:
        self._registry = registry if registry is not None else SOURCE_REGISTRY
        self._order = {
            source.identifier: self._registry.source_index(source.identifier)
            for source in self._registry.all_sources()
        }
        self._fallback = tuple(
            self._registry.get_source(identifier)
            for identifier in _FALLBACK_SOURCE_IDS
            if self._registry.has_source(identifier)
        )
        if not self._fallback:
            self._fallback = self._registry.all_sources()

    def discover(self, plan: ResearchPlan) -> SourceDiscoveryPlan:
        """Map every task in ``plan`` to ranked candidate sources.

        Raises
        ------
        InvalidDiscoveryInputError:
            When ``plan`` is not a :class:`ResearchPlan`.
        """
        if not isinstance(plan, ResearchPlan):
            raise InvalidDiscoveryInputError(
                "discover expects a ResearchPlan instance"
            )
        recommendations = tuple(
            self._recommendation_for(task) for task in plan.tasks
        )
        return SourceDiscoveryPlan(
            schema_version=SOURCE_SCHEMA_VERSION,
            plan_id=plan.plan_id,
            company_name=plan.company_name,
            source_tasks=plan.tasks,
            recommendations=recommendations,
        )

    def discover_for_company(
        self,
        company_name: str,
        *,
        website: str = "",
        description: str = "",
        known_topics: tuple[str, ...] = (),
        partial_topics: tuple[str, ...] = (),
        prediction_horizon_days: int = 365,
    ) -> SourceDiscoveryPlan:
        """Convenience: plan and then discover for a company in one call."""
        plan = ResearchPlanner().plan_for_company(
            company_name,
            website=website,
            description=description,
            known_topics=known_topics,
            partial_topics=partial_topics,
            prediction_horizon_days=prediction_horizon_days,
        )
        return self.discover(plan)

    def _recommendation_for(self, task: ResearchTask) -> SourceRecommendation:
        """Build the ranked recommendation for one research task."""
        candidates = self._registry.candidate_sources(task.topic_id)
        if not candidates:
            candidates = self._fallback
        ranked = rank_sources(
            candidates,
            task.topic_id,
            freshness_requirement_days=task.freshness_requirement_days,
            registry_order=self._order,
        )
        return SourceRecommendation(
            task_id=task.task_id,
            topic_id=task.topic_id,
            sources=ranked,
        )
