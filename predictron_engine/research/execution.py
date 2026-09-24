"""Deterministic Evidence Collection Engine — Phase 8 Sprint 3.

:class:`EvidenceCollectionEngine` executes a :class:`ResearchPlan`
produced by the Research Planner (Sprint 1) and dispatches its tasks to
collectors resolved through the :class:`CollectorRegistry` (never to
concrete classes).

Responsibilities:

* Accept a :class:`ResearchPlan` and dispatch its tasks in order.
* Execute every dispatched task through its collector.
* Respect the plan's dependency ordering (tasks are executed in the
  plan's executable order; concurrency is a later-sprint concern).
* Aggregate the produced evidence into an :class:`EvidenceCollection`.
* Return a deterministic :class:`CollectionResult`.

Execution is strictly sequential and deterministic: the same plan and
registry always produce an identical result.  No randomness, no network,
no LLMs — collectors in this sprint return placeholder evidence only.
"""

from __future__ import annotations

import hashlib
import json

from predictron_engine.research.dispatcher import (
    CollectorDispatch,
    Dispatcher,
)
from predictron_engine.research.exceptions import (
    CollectorExecutionError,
    InvalidCollectionInputError,
    InvalidEvidence,
    UnsupportedTopic,
)
from predictron_engine.research.models import (
    COLLECTION_SCHEMA_VERSION,
    CollectionResult,
    CollectionStatus,
    Evidence,
    EvidenceCollection,
    ResearchPlan,
    ResearchTask,
)
from predictron_engine.research.planner import ResearchPlanner
from predictron_engine.research.registry import (
    COLLECTOR_REGISTRY,
    CollectorRegistry,
)


class EvidenceCollectionEngine:
    """Deterministic executor of research plans.

    Parameters
    ----------
    registry:
        Optional :class:`CollectorRegistry` used for task resolution.
        When ``None`` the shared registry is used.
    dispatcher:
        Optional :class:`Dispatcher`.  When ``None`` a dispatcher bound
        to the resolved registry is created.
    """

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        dispatcher: Dispatcher | None = None,
    ) -> None:
        self._registry = registry if registry is not None else COLLECTOR_REGISTRY
        self._dispatcher = (
            dispatcher if dispatcher is not None else Dispatcher(self._registry)
        )

    @property
    def registry(self) -> CollectorRegistry:
        """Return the registry this engine resolves collectors through."""
        return self._registry

    @property
    def dispatcher(self) -> Dispatcher:
        """Return the dispatcher this engine uses."""
        return self._dispatcher

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, plan: ResearchPlan) -> CollectionResult:
        """Execute ``plan`` and return a deterministic collection result.

        Unsupported topics and invalid evidence are recorded as failed
        tasks so the run still returns a (partial) result.  A collector
        that crashes hard surfaces as :class:`CollectorExecutionError`.

        Raises
        ------
        InvalidCollectionInputError:
            When ``plan`` is not a :class:`ResearchPlan`.
        CollectorExecutionError:
            When a collector fails unexpectedly while executing a task.
        """
        if not isinstance(plan, ResearchPlan):
            raise InvalidCollectionInputError(
                "execute expects a ResearchPlan instance"
            )
        dispatch_plan = self._dispatcher.dispatch(plan.tasks)

        collected: list[Evidence] = []
        executed: list[str] = []
        unresolved = set(dispatch_plan.unresolved)
        for dispatch in dispatch_plan.dispatches:
            task_id = dispatch.task.task_id
            if task_id in unresolved:
                continue
            produced = self._execute_dispatch(dispatch, plan)
            if produced is None:
                continue
            collected.extend(produced)
            executed.append(task_id)

        executed_set = set(executed)
        failed: list[str] = []
        for task in plan.tasks:
            if task.task_id not in executed_set:
                failed.append(task.task_id)

        collection = EvidenceCollection(
            evidence=tuple(collected),
            topic_order=tuple(task.topic_id for task in plan.tasks),
        )
        status = self._status_for(plan.tasks, executed, failed)
        collection_id = self._collection_id(
            plan, executed, failed, collection
        )
        return CollectionResult(
            schema_version=COLLECTION_SCHEMA_VERSION,
            plan_id=plan.plan_id,
            company_name=plan.company_name,
            collection_id=collection_id,
            status=status,
            collection=collection,
            executed_tasks=tuple(executed),
            failed_tasks=tuple(failed),
            unresolved_tasks=tuple(dispatch_plan.unresolved),
        )

    def execute_for_company(
        self,
        company_name: str,
        *,
        website: str = "",
        description: str = "",
        known_topics: tuple[str, ...] = (),
        partial_topics: tuple[str, ...] = (),
        prediction_horizon_days: int = 365,
    ) -> CollectionResult:
        """Convenience: plan and collect for a company in one call."""
        plan = ResearchPlanner().plan_for_company(
            company_name,
            website=website,
            description=description,
            known_topics=known_topics,
            partial_topics=partial_topics,
            prediction_horizon_days=prediction_horizon_days,
        )
        return self.execute(plan)

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------

    def _execute_dispatch(
        self,
        dispatch: CollectorDispatch,
        plan: ResearchPlan,
    ) -> tuple[Evidence, ...] | None:
        """Collect evidence for one dispatch, or ``None`` on failure.

        Two error tiers keep execution deterministic:

        * **Designed signals** — :class:`UnsupportedTopic` and
          :class:`InvalidEvidence` raised while collecting are recorded
          as a failed task so the run can produce a partial result.
        * **Crashes** — any other exception (including a
          :class:`CollectorExecutionError`) is surfaced to the caller
          wrapped uniformly as :class:`CollectorExecutionError`.

        A collector that returns a non-Evidence object is a hard
        contract violation and always raises :class:`InvalidEvidence`.
        """
        task = dispatch.task
        collector = dispatch.collector
        try:
            produced = collector.collect(
                task,
                company_name=plan.company_name,
                website=plan.input.website,
            )
            items = tuple(produced)
            for item in items:
                if not isinstance(item, Evidence):
                    raise InvalidEvidence(
                        f"collector {collector.metadata().collector_id!r} "
                        f"returned a non-Evidence object for topic "
                        f"{task.topic_id!r}"
                    )
        except (UnsupportedTopic, InvalidEvidence):
            return None
        except CollectorExecutionError:
            raise
        except Exception as exc:
            raise CollectorExecutionError(
                f"collector {collector.metadata().collector_id!r} failed "
                f"while collecting topic {task.topic_id!r}"
            ) from exc
        return items

    @staticmethod
    def _status_for(
        tasks: tuple[ResearchTask, ...],
        executed: list[str],
        failed: list[str],
    ) -> CollectionStatus:
        """Derive the deterministic collection status."""
        if not tasks:
            return CollectionStatus.EMPTY
        if not failed:
            return CollectionStatus.SUCCESS
        if not executed:
            return CollectionStatus.FAILED
        return CollectionStatus.PARTIAL

    @staticmethod
    def _collection_id(
        plan: ResearchPlan,
        executed: list[str],
        failed: list[str],
        collection: EvidenceCollection,
    ) -> str:
        """Return a stable content-hash identifier for the result."""
        payload = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "plan_id": plan.plan_id,
            "executed_tasks": executed,
            "failed_tasks": failed,
            "evidence_ids": [
                item.evidence_id for item in collection.evidence
            ],
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"collection_{digest[:16]}"
