"""Deterministic dispatcher — Phase 8 Sprint 3.

:class:`Dispatcher` pairs every :class:`ResearchTask` in a plan with the
collector responsible for it.  The dispatcher knows no concrete
collector class — it resolves strictly through the
:class:`~predictron_engine.research.registry.CollectorRegistry`.

Tasks whose topic has no registered collector are reported as
``unresolved`` instead of aborting the dispatch, so an execution engine
can produce a deterministic partial result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from predictron_engine.research.collector import EvidenceCollector
from predictron_engine.research.exceptions import (
    CollectorNotFound,
    InvalidCollectionInputError,
)
from predictron_engine.research.models import ResearchPlan, ResearchTask
from predictron_engine.research.registry import (
    COLLECTOR_REGISTRY,
    CollectorRegistry,
)


@dataclass(frozen=True)
class CollectorDispatch:
    """A task paired with the collector that will execute it."""

    task: ResearchTask
    collector: EvidenceCollector

    def __post_init__(self) -> None:
        if not isinstance(self.task, ResearchTask):
            raise InvalidCollectionInputError(
                "task must be a ResearchTask instance"
            )
        if not isinstance(self.collector, EvidenceCollector):
            raise InvalidCollectionInputError(
                "collector must be an EvidenceCollector instance"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "task": self.task.to_dict(),
            "collector": self.collector.metadata().to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        registry: CollectorRegistry | None = None,
    ) -> CollectorDispatch:
        """Deserialize from :meth:`to_dict` output.

        The collector is resolved back through ``registry`` (the shared
        registry when omitted).
        """
        collector_id = _require_collector_id(data)
        resolved = (
            registry if registry is not None else COLLECTOR_REGISTRY
        ).get(collector_id)
        if resolved is None:
            raise CollectorNotFound(
                f"cannot restore dispatch for unknown collector: "
                f"{collector_id!r}"
            )
        return cls(
            task=ResearchTask.from_dict(_require_task_dict(data)),
            collector=resolved,
        )


@dataclass(frozen=True)
class DispatchPlan:
    """The ordered result of dispatching a plan's tasks.

    ``dispatches`` preserves the input task order and resolve through the
    registry; ``unresolved`` lists the task identifiers that could not be
    matched to any collector (in input order).
    """

    dispatches: tuple[CollectorDispatch, ...] = ()
    unresolved: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for dispatch in self.dispatches:
            if not isinstance(dispatch, CollectorDispatch):
                raise InvalidCollectionInputError(
                    "dispatches entries must be CollectorDispatch instances"
                )
        object.__setattr__(
            self, "unresolved", tuple(dict.fromkeys(self.unresolved))
        )

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return the dispatched task identifiers, in input order."""
        return tuple(dispatch.task.task_id for dispatch in self.dispatches)

    @property
    def topics_dispatched(self) -> tuple[str, ...]:
        """Return the dispatched topic identifiers, in input order."""
        return tuple(
            dispatch.task.topic_id for dispatch in self.dispatches
        )

    @property
    def collector_ids(self) -> tuple[str, ...]:
        """Return the resolved collector identifiers, in input order."""
        return tuple(
            dispatch.collector.metadata().collector_id
            for dispatch in self.dispatches
        )

    def is_complete(self) -> bool:
        """Return whether every task was matched to a collector."""
        return not self.unresolved

    def dispatch_for_task(self, task_id: str) -> CollectorDispatch | None:
        """Return the dispatch for ``task_id``, or ``None`` when absent."""
        for dispatch in self.dispatches:
            if dispatch.task.task_id == task_id:
                return dispatch
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "dispatches": [dispatch.to_dict() for dispatch in self.dispatches],
            "unresolved": list(self.unresolved),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        registry: CollectorRegistry | None = None,
    ) -> DispatchPlan:
        """Deserialize from :meth:`to_dict` output."""
        entries = data.get("dispatches", [])
        if not isinstance(entries, list):
            raise ValueError("expected list for 'dispatches'")
        dispatches = tuple(
            CollectorDispatch.from_dict(entry, registry=registry)
            for entry in entries
        )
        return cls(
            dispatches=dispatches,
            unresolved=tuple(_require_unresolved(data)),
        )


class Dispatcher:
    """Resolves research tasks to collectors through a registry.

    Parameters
    ----------
    registry:
        Optional :class:`CollectorRegistry` to resolve through.  When
        ``None`` the shared registry is used.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry if registry is not None else COLLECTOR_REGISTRY

    @property
    def registry(self) -> CollectorRegistry:
        """Return the registry this dispatcher resolves through."""
        return self._registry

    def dispatch(self, tasks: Sequence[ResearchTask]) -> DispatchPlan:
        """Pair ``tasks`` with collectors, in the given (executable) order.

        Raises
        ------
        InvalidCollectionInputError:
            When any entry is not a :class:`ResearchTask`.
        """
        dispatches: list[CollectorDispatch] = []
        unresolved: list[str] = []
        for task in tasks:
            if not isinstance(task, ResearchTask):
                raise InvalidCollectionInputError(
                    "dispatch expects ResearchTask entries"
                )
            try:
                collector = self._registry.resolve(task.topic_id)
            except CollectorNotFound:
                unresolved.append(task.task_id)
            else:
                dispatches.append(
                    CollectorDispatch(task=task, collector=collector)
                )
        return DispatchPlan(
            dispatches=tuple(dispatches),
            unresolved=tuple(unresolved),
        )

    def dispatch_plan(self, plan: ResearchPlan) -> DispatchPlan:
        """Convenience: dispatch every task in a :class:`ResearchPlan`."""
        if not isinstance(plan, ResearchPlan):
            raise InvalidCollectionInputError(
                "dispatch_plan expects a ResearchPlan instance"
            )
        return self.dispatch(plan.tasks)


def _require_collector_id(data: Mapping[str, object]) -> str:
    """Read the nested collector identifier from a serialized dispatch."""
    collector = data.get("collector")
    if not isinstance(collector, dict):
        raise ValueError("expected dict for 'collector'")
    collector_id = collector.get("collector_id")
    if not isinstance(collector_id, str) or not collector_id:
        raise ValueError("expected collector_id string")
    return collector_id


def _require_task_dict(data: Mapping[str, object]) -> dict[str, object]:
    """Read and validate the nested task dictionary."""
    task = data.get("task")
    if not isinstance(task, dict):
        raise ValueError("expected dict for 'task'")
    return task


def _require_unresolved(data: Mapping[str, object]) -> list[str]:
    """Read and validate the unresolved task-id list."""
    value = data.get("unresolved", [])
    if not isinstance(value, list):
        raise ValueError("expected list for 'unresolved'")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("expected list of strings for 'unresolved'")
        result.append(item)
    return result
