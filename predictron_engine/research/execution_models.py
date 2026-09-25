"""Immutable execution value objects — Phase 8 Sprint 5.

This module holds every model the orchestration layer needs *around* the
research pipeline: the execution request, per-step records, domain
statistics, operational metrics, coverage, confidence distribution,
warnings, errors, and timing metadata.

Design constraints honoured by every model here:

* **Immutable** — frozen dataclasses only; no setters, no mutable
  collections (tuples throughout).
* **Serializable** — a JSON-ready :meth:`to_dict` paired with a strict
  :meth:`from_dict` for every single model, so an execution round-trips
  losslessly.
* **Deterministic** — every derived value is a pure function of the
  stored fields.  No wall-clock, no randomness, and no dictionary
  iteration order leaking into output.
* **No mutation of upstream artifacts** — the models hold *references* to
  the plan / discovery / collection / validation objects produced by the
  earlier sprints and never rewrite them.

Wall-clock measurement is deliberately quarantined in
:class:`ExecutionTiming` and is **off by default**, which is what keeps
the default execution path fully deterministic.

The private ``_require_*`` deserialization helpers below intentionally
mirror the ones in :mod:`predictron_engine.research.models` and
:mod:`predictron_engine.research.validation_models`; keeping them local is
the established convention in this package and keeps the four Sprint 5
modules free of cross-imports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum

from predictron_engine.research.dispatcher import DispatchPlan
from predictron_engine.research.exceptions import (
    InvalidExecutionInputError,
    InvalidExecutionModelError,
)
from predictron_engine.research.execution_state import (
    ACTIVE_STATES,
    ExecutionState,
)
from predictron_engine.research.models import PlannerInput
from predictron_engine.research.validation_models import (
    EvidenceConfidence,
    ValidationSummary,
    content_digest,
)

#: Serialization schema version for the execution value objects.
EXECUTION_SCHEMA_VERSION = "1.0.0"

#: Confidence score used when a run produced no validated evidence.
EMPTY_MEAN_CONFIDENCE = 0.0

#: Coverage ratio used when a plan contained no topics.
EMPTY_COVERAGE_RATIO = 0.0

#: Decimal places used when rounding derived ratios and means.
RATIO_PRECISION = 6


# ----------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionPolicy:
    """Immutable, fully serialized execution policy.

    Parameters
    ----------
    run_validation:
        Whether the Evidence Validation Engine runs.  When ``False`` the
        validation step is recorded as ``SKIPPED`` — a configuration
        decision, not a failure — and the run still reaches ``COMPLETED``.
    fail_fast:
        Whether a stage error aborts the run by re-raising.  The default is
        ``False``, which is exactly what enables partial completion: stage
        errors are recorded on the result and the run keeps going.
    measure_duration:
        Whether wall-clock duration metadata is captured.  The default is
        ``False`` so the default execution path stays deterministic; when
        enabled, :class:`ExecutionTiming` carries real timestamps and
        per-step durations.
    """

    run_validation: bool = True
    fail_fast: bool = False
    measure_duration: bool = False

    def __post_init__(self) -> None:
        for name in ("run_validation", "fail_fast", "measure_duration"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise InvalidExecutionModelError(
                    f"{name} must be a bool for ExecutionPolicy"
                )

    @property
    def allow_partial(self) -> bool:
        """Return whether stage errors are recorded instead of aborting."""
        return not self.fail_fast

    @property
    def is_deterministic(self) -> bool:
        """Return whether this policy guarantees deterministic output."""
        return not self.measure_duration

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "run_validation": self.run_validation,
            "fail_fast": self.fail_fast,
            "measure_duration": self.measure_duration,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecutionPolicy:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            run_validation=_require_bool(data, "run_validation"),
            fail_fast=_require_bool(data, "fail_fast"),
            measure_duration=_require_bool(data, "measure_duration"),
        )


# ----------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------


class ExecutionStepStatus(str, Enum):
    """Deterministic outcome of a single pipeline step.

    * ``COMPLETED`` — the step produced an artifact.
    * ``SKIPPED`` — the step was not applicable (e.g. validation disabled).
    * ``FAILED`` — the step raised and contributed at least one error.
    """

    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"

    @classmethod
    def from_value(cls, value: object) -> ExecutionStepStatus:
        """Restore a status from its serialized ``value`` string.

        Raises
        ------
        InvalidExecutionModelError:
            When ``value`` is not a string or is not a known status.  This
            keeps :meth:`ExecutionStep.from_dict` from leaking ``ValueError``.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise InvalidExecutionModelError(
                "expected string for 'status' when restoring "
                "ExecutionStepStatus"
            )
        try:
            return cls(value)
        except ValueError as exc:
            raise InvalidExecutionModelError(
                f"unknown ExecutionStepStatus value: {value!r}"
            ) from exc


@dataclass(frozen=True)
class ExecutionStep:
    """The record of one pipeline step within an execution.

    Parameters
    ----------
    name:
        Stable step identifier, e.g. ``"planning"`` or ``"validation"``.
    state:
        The :class:`ExecutionState` the run was in for this step.
    status:
        Whether the step produced a result, was skipped, or errored.
    order:
        Zero-based position of the step within the run's step list.
    artifact_id:
        Identifier of the artifact the step produced (plan id, collection
        id, validation id, ...).  Empty when nothing was produced.
    detail:
        Short human-readable, deterministic description.
    warning_count:
        Number of warnings the step contributed.
    error_count:
        Number of errors the step contributed.
    """

    name: str
    state: ExecutionState
    status: ExecutionStepStatus
    order: int
    artifact_id: str = ""
    detail: str = ""
    warning_count: int = 0
    error_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidExecutionModelError(
                "name must be a non-empty string for ExecutionStep"
            )
        if not isinstance(self.state, ExecutionState):
            raise InvalidExecutionModelError(
                "state must be an ExecutionState for ExecutionStep"
            )
        if not isinstance(self.status, ExecutionStepStatus):
            raise InvalidExecutionModelError(
                "status must be an ExecutionStepStatus for ExecutionStep"
            )
        for field_name in ("order", "warning_count", "error_count"):
            _require_non_negative_int(
                getattr(self, field_name), field_name, "ExecutionStep"
            )
        if self.status is ExecutionStepStatus.FAILED and not self.error_count:
            raise InvalidExecutionModelError(
                "a failed ExecutionStep must record at least one error"
            )
        if self.status is ExecutionStepStatus.COMPLETED and not (
            self.artifact_id or self.detail
        ):
            raise InvalidExecutionModelError(
                "a completed ExecutionStep must record an artifact or a detail"
            )
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "artifact_id", self.artifact_id.strip())
        object.__setattr__(self, "detail", self.detail.strip())

    @property
    def is_completed(self) -> bool:
        """Return whether the step produced a result."""
        return self.status is ExecutionStepStatus.COMPLETED

    @property
    def is_skipped(self) -> bool:
        """Return whether the step was skipped."""
        return self.status is ExecutionStepStatus.SKIPPED

    @property
    def is_failed(self) -> bool:
        """Return whether the step errored."""
        return self.status is ExecutionStepStatus.FAILED

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "name": self.name,
            "state": self.state.value,
            "status": self.status.value,
            "order": self.order,
            "artifact_id": self.artifact_id,
            "detail": self.detail,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecutionStep:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            name=_require_str(data, "name"),
            state=ExecutionState.from_value(data.get("state")),
            status=ExecutionStepStatus.from_value(data.get("status")),
            order=_require_int(data, "order"),
            artifact_id=_require_str(data, "artifact_id", default=""),
            detail=_require_str(data, "detail", default=""),
            warning_count=_require_int(data, "warning_count"),
            error_count=_require_int(data, "error_count"),
        )


# ----------------------------------------------------------------------
# Warnings and errors
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionWarning:
    """One deterministic, non-fatal observation about a run.

    Parameters
    ----------
    code:
        Stable machine-readable code, e.g. ``"unresolved_tasks"``.
    message:
        Human-readable description.
    step:
        Name of the step that raised the warning.
    detail:
        Optional deterministic supporting detail (task ids, counts, ...).
    """

    code: str
    message: str
    step: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        for field_name in ("code", "message"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidExecutionModelError(
                    f"{field_name} must be a non-empty string for "
                    "ExecutionWarning"
                )
        object.__setattr__(self, "code", self.code.strip())
        object.__setattr__(self, "message", self.message.strip())
        object.__setattr__(self, "step", self.step.strip())
        object.__setattr__(self, "detail", self.detail.strip())

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "step": self.step,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecutionWarning:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            code=_require_str(data, "code"),
            message=_require_str(data, "message"),
            step=_require_str(data, "step", default=""),
            detail=_require_str(data, "detail", default=""),
        )


@dataclass(frozen=True)
class ExecutionError:
    """One deterministic, fatal observation about a run.

    Parameters
    ----------
    code:
        Stable machine-readable code, e.g. ``"planning_failed"``.
    message:
        Human-readable description.
    step:
        Name of the step that errored.
    exception_type:
        Class name of the originating exception, so callers can branch on
        the failure kind without importing the raising module.
    detail:
        Optional deterministic supporting detail.
    """

    code: str
    message: str
    step: str = ""
    exception_type: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        for field_name in ("code", "message"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidExecutionModelError(
                    f"{field_name} must be a non-empty string for ExecutionError"
                )
        object.__setattr__(self, "code", self.code.strip())
        object.__setattr__(self, "message", self.message.strip())
        object.__setattr__(self, "step", self.step.strip())
        object.__setattr__(self, "exception_type", self.exception_type.strip())
        object.__setattr__(self, "detail", self.detail.strip())

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "step": self.step,
            "exception_type": self.exception_type,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecutionError:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            code=_require_str(data, "code"),
            message=_require_str(data, "message"),
            step=_require_str(data, "step", default=""),
            exception_type=_require_str(data, "exception_type", default=""),
            detail=_require_str(data, "detail", default=""),
        )


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class StepTiming:
    """Wall-clock duration attributed to one step, in milliseconds.

    Only populated when :attr:`ExecutionPolicy.measure_duration` is
    enabled; the default unmeasured block carries no entries at all.
    """

    step: str
    duration_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.step, str) or not self.step.strip():
            raise InvalidExecutionModelError(
                "step must be a non-empty string for StepTiming"
            )
        object.__setattr__(self, "step", self.step.strip())
        object.__setattr__(
            self,
            "duration_ms",
            _require_non_negative_float(self.duration_ms, "duration_ms"),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {"step": self.step, "duration_ms": self.duration_ms}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> StepTiming:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            step=_require_str(data, "step"),
            duration_ms=_require_float(data, "duration_ms"),
        )


@dataclass(frozen=True)
class ExecutionTiming:
    """Optional wall-clock metadata for one execution.

    With the default policy (``measure_duration=False``) every field is a
    constant and ``measured`` is ``False``, so the default execution path
    stays byte-for-byte deterministic.  The timing block is excluded from
    the report's content hash, so enabling measurement never changes the
    ``report_id``.
    """

    measured: bool = False
    started_at: str = ""
    finished_at: str = ""
    total_duration_ms: float = 0.0
    step_durations: tuple[StepTiming, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.measured, bool):
            raise InvalidExecutionModelError(
                "measured must be a bool for ExecutionTiming"
            )
        for entry in self.step_durations:
            if not isinstance(entry, StepTiming):
                raise InvalidExecutionModelError(
                    "step_durations entries must be StepTiming instances"
                )
        object.__setattr__(self, "started_at", self.started_at.strip())
        object.__setattr__(self, "finished_at", self.finished_at.strip())
        object.__setattr__(
            self,
            "total_duration_ms",
            _require_non_negative_float(
                self.total_duration_ms, "total_duration_ms"
            ),
        )

    @classmethod
    def unmeasured(cls) -> ExecutionTiming:
        """Return the deterministic, empty timing block."""
        return cls()

    def duration_for(self, step: str) -> float:
        """Return the measured duration of ``step`` in milliseconds."""
        for entry in self.step_durations:
            if entry.step == step:
                return entry.duration_ms
        return 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "measured": self.measured,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_duration_ms": self.total_duration_ms,
            "step_durations": [entry.to_dict() for entry in self.step_durations],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecutionTiming:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            measured=_require_bool(data, "measured"),
            started_at=_require_str(data, "started_at", default=""),
            finished_at=_require_str(data, "finished_at", default=""),
            total_duration_ms=_require_float(data, "total_duration_ms"),
            step_durations=tuple(
                StepTiming.from_dict(item)
                for item in _require_list_of_dicts(data, "step_durations")
            ),
        )


# ----------------------------------------------------------------------
# Dispatch projection
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchOutcome:
    """A serializable projection of a :class:`DispatchPlan`.

    The live :class:`DispatchPlan` holds resolved collector *instances*,
    which cannot be meaningfully deserialized without a collector
    registry.  The orchestration layer therefore reports the dispatch as
    identifiers only, keeping the execution result round-trippable.
    """

    task_ids: tuple[str, ...] = ()
    collector_ids: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("task_ids", "collector_ids", "unresolved"):
            values = tuple(getattr(self, field_name))
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise InvalidExecutionModelError(
                        f"{field_name} entries must be non-empty strings"
                    )
            object.__setattr__(self, field_name, values)
        if len(self.task_ids) != len(self.collector_ids):
            raise InvalidExecutionModelError(
                "task_ids and collector_ids must be the same length"
            )

    @property
    def is_complete(self) -> bool:
        """Return whether every planned task resolved to a collector."""
        return not self.unresolved

    @property
    def dispatched_count(self) -> int:
        """Return how many tasks resolved to a collector."""
        return len(self.task_ids)

    @property
    def unresolved_count(self) -> int:
        """Return how many tasks were left unresolved."""
        return len(self.unresolved)

    @property
    def distinct_collectors(self) -> tuple[str, ...]:
        """Return the distinct collector identifiers, sorted."""
        return tuple(sorted(set(self.collector_ids)))

    def collector_for(self, task_id: str) -> str | None:
        """Return the collector id dispatched for ``task_id`` or ``None``."""
        for candidate, collector_id in zip(
            self.task_ids, self.collector_ids, strict=True
        ):
            if candidate == task_id:
                return collector_id
        return None

    @classmethod
    def from_dispatch(cls, dispatch: DispatchPlan) -> DispatchOutcome:
        """Project a live :class:`DispatchPlan` onto its identifiers."""
        return cls(
            task_ids=dispatch.task_ids,
            collector_ids=dispatch.collector_ids,
            unresolved=dispatch.unresolved,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "task_ids": list(self.task_ids),
            "collector_ids": list(self.collector_ids),
            "unresolved": list(self.unresolved),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DispatchOutcome:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            task_ids=tuple(_require_str_list(data, "task_ids")),
            collector_ids=tuple(_require_str_list(data, "collector_ids")),
            unresolved=tuple(_require_str_list(data, "unresolved")),
        )


# ----------------------------------------------------------------------
# Domain statistics and operational metrics
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionStatistics:
    """Deterministic domain counters for one execution.

    Every field is a plain count derived from the pipeline artifacts, so
    the whole block is trivially comparable across runs.
    """

    topics_planned: int = 0
    topics_covered: int = 0
    tasks_planned: int = 0
    tasks_dispatched: int = 0
    tasks_executed: int = 0
    tasks_failed: int = 0
    tasks_unresolved: int = 0
    sources_discovered: int = 0
    distinct_sources: int = 0
    collectors_executed: int = 0
    distinct_collectors: int = 0
    evidence_collected: int = 0
    evidence_validated: int = 0

    def __post_init__(self) -> None:
        for entry in fields(self):
            _require_non_negative_int(
                getattr(self, entry.name), entry.name, "ExecutionStatistics"
            )

    @property
    def topics_researched(self) -> int:
        """Return how many topics the plan asked the pipeline to research."""
        return self.topics_planned

    @property
    def is_complete(self) -> bool:
        """Return whether every planned task executed successfully."""
        return self.tasks_planned > 0 and not (
            self.tasks_failed or self.tasks_unresolved
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "topics_planned": self.topics_planned,
            "topics_covered": self.topics_covered,
            "tasks_planned": self.tasks_planned,
            "tasks_dispatched": self.tasks_dispatched,
            "tasks_executed": self.tasks_executed,
            "tasks_failed": self.tasks_failed,
            "tasks_unresolved": self.tasks_unresolved,
            "sources_discovered": self.sources_discovered,
            "distinct_sources": self.distinct_sources,
            "collectors_executed": self.collectors_executed,
            "distinct_collectors": self.distinct_collectors,
            "evidence_collected": self.evidence_collected,
            "evidence_validated": self.evidence_validated,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecutionStatistics:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            topics_planned=_require_int(data, "topics_planned"),
            topics_covered=_require_int(data, "topics_covered"),
            tasks_planned=_require_int(data, "tasks_planned"),
            tasks_dispatched=_require_int(data, "tasks_dispatched"),
            tasks_executed=_require_int(data, "tasks_executed"),
            tasks_failed=_require_int(data, "tasks_failed"),
            tasks_unresolved=_require_int(data, "tasks_unresolved"),
            sources_discovered=_require_int(data, "sources_discovered"),
            distinct_sources=_require_int(data, "distinct_sources"),
            collectors_executed=_require_int(data, "collectors_executed"),
            distinct_collectors=_require_int(data, "distinct_collectors"),
            evidence_collected=_require_int(data, "evidence_collected"),
            evidence_validated=_require_int(data, "evidence_validated"),
        )


@dataclass(frozen=True)
class ExecutionMetrics:
    """Deterministic operational counters for one execution.

    Deliberately distinct from :class:`ExecutionStatistics`: the statistics
    block counts *research domain* work (topics, sources, collectors,
    evidence) while the metrics block counts *pipeline* work (steps,
    warnings, errors, validation findings, and the achieved coverage
    ratio).
    """

    steps_total: int = 0
    steps_completed: int = 0
    steps_skipped: int = 0
    steps_failed: int = 0
    warnings: int = 0
    errors: int = 0
    validation_issues: int = 0
    validation_conflicts: int = 0
    validation_duplicates: int = 0
    coverage_ratio: float = EMPTY_COVERAGE_RATIO

    def __post_init__(self) -> None:
        for entry in fields(self):
            value = getattr(self, entry.name)
            if entry.name == "coverage_ratio":
                object.__setattr__(
                    self,
                    entry.name,
                    _round_ratio(_require_unit_interval(value, entry.name)),
                )
            else:
                _require_non_negative_int(value, entry.name, "ExecutionMetrics")

    @property
    def is_clean(self) -> bool:
        """Return whether the run recorded neither warnings nor errors."""
        return not self.warnings and not self.errors

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "steps_total": self.steps_total,
            "steps_completed": self.steps_completed,
            "steps_skipped": self.steps_skipped,
            "steps_failed": self.steps_failed,
            "warnings": self.warnings,
            "errors": self.errors,
            "validation_issues": self.validation_issues,
            "validation_conflicts": self.validation_conflicts,
            "validation_duplicates": self.validation_duplicates,
            "coverage_ratio": self.coverage_ratio,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecutionMetrics:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            steps_total=_require_int(data, "steps_total"),
            steps_completed=_require_int(data, "steps_completed"),
            steps_skipped=_require_int(data, "steps_skipped"),
            steps_failed=_require_int(data, "steps_failed"),
            warnings=_require_int(data, "warnings"),
            errors=_require_int(data, "errors"),
            validation_issues=_require_int(data, "validation_issues"),
            validation_conflicts=_require_int(data, "validation_conflicts"),
            validation_duplicates=_require_int(data, "validation_duplicates"),
            coverage_ratio=_require_float(data, "coverage_ratio"),
        )


@dataclass(frozen=True)
class ExecutionCoverage:
    """Deterministic topic-coverage breakdown for one execution.

    ``planned_topics`` preserves the plan's canonical topic order and
    ``covered_topics`` preserves the collection's topic order, so
    ``uncovered_topics`` is a stable, reproducible list.
    """

    planned_topics: tuple[str, ...] = ()
    covered_topics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "planned_topics", _clean_topics(self.planned_topics))
        object.__setattr__(
            self, "covered_topics", _clean_topics(self.covered_topics)
        )

    @property
    def uncovered_topics(self) -> tuple[str, ...]:
        """Return planned topics that produced no collected evidence."""
        covered = set(self.covered_topics)
        return tuple(
            topic_id
            for topic_id in self.planned_topics
            if topic_id not in covered
        )

    @property
    def planned_count(self) -> int:
        """Return how many topics the plan asked to research."""
        return len(self.planned_topics)

    @property
    def covered_count(self) -> int:
        """Return how many planned topics produced evidence."""
        covered = set(self.covered_topics)
        return len(
            [topic_id for topic_id in self.planned_topics if topic_id in covered]
        )

    @property
    def ratio(self) -> float:
        """Return covered / planned, or ``0.0`` when nothing was planned."""
        planned = self.planned_count
        if planned == 0:
            return EMPTY_COVERAGE_RATIO
        return _round_ratio(self.covered_count / planned)

    @property
    def is_complete(self) -> bool:
        """Return whether every planned topic produced evidence."""
        return bool(self.planned_topics) and not self.uncovered_topics

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "planned_topics": list(self.planned_topics),
            "covered_topics": list(self.covered_topics),
            "uncovered_topics": list(self.uncovered_topics),
            "planned_count": self.planned_count,
            "covered_count": self.covered_count,
            "ratio": self.ratio,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecutionCoverage:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            planned_topics=tuple(_require_str_list(data, "planned_topics")),
            covered_topics=tuple(_require_str_list(data, "covered_topics")),
        )


@dataclass(frozen=True)
class ConfidenceDistribution:
    """Deterministic distribution of validated confidence bands.

    ``mean_score`` is the arithmetic mean of the per-evidence confidence
    scores, rounded to :data:`RATIO_PRECISION` decimals so floating-point
    noise can never make two otherwise identical runs compare unequal.
    """

    high: int = 0
    medium: int = 0
    low: int = 0
    unknown: int = 0
    mean_score: float = EMPTY_MEAN_CONFIDENCE

    def __post_init__(self) -> None:
        for field_name in ("high", "medium", "low", "unknown"):
            _require_non_negative_int(
                getattr(self, field_name), field_name, "ConfidenceDistribution"
            )
        object.__setattr__(
            self,
            "mean_score",
            _round_ratio(_require_unit_interval(self.mean_score, "mean_score")),
        )

    @property
    def total(self) -> int:
        """Return how many evidence items were banded."""
        return self.high + self.medium + self.low + self.unknown

    @property
    def is_empty(self) -> bool:
        """Return whether no evidence was banded."""
        return self.total == 0

    def count_for(self, band: EvidenceConfidence) -> int:
        """Return the count for one :class:`EvidenceConfidence` band."""
        if band is EvidenceConfidence.HIGH:
            return self.high
        if band is EvidenceConfidence.MEDIUM:
            return self.medium
        if band is EvidenceConfidence.LOW:
            return self.low
        return self.unknown

    def distribution(self) -> dict[str, int]:
        """Return band counts keyed by band value, best band first."""
        return {band.value: self.count_for(band) for band in EvidenceConfidence}

    def top_band(self) -> EvidenceConfidence:
        """Return the most common band, breaking ties best-to-worst.

        The enum is declared best-to-worst, so iterating it in order and
        keeping a strict ``>`` comparison resolves every tie to the
        stronger band.
        """
        best = EvidenceConfidence.UNKNOWN
        for band in EvidenceConfidence:
            if self.count_for(band) > self.count_for(best):
                best = band
        return best

    @classmethod
    def empty(cls) -> ConfidenceDistribution:
        """Return the deterministic, empty distribution."""
        return cls()

    @classmethod
    def from_validations(
        cls,
        validations: tuple[ValidationSummary, ...] = (),
    ) -> ConfidenceDistribution:
        """Derive the distribution from validation summaries, purely.

        A single summary is the normal input; additional summaries are
        merged band-tally-wise in the given order, which keeps the result
        deterministic for a given sequence.
        """
        tallies = {band: 0 for band in EvidenceConfidence}
        total_score = 0.0
        total_items = 0
        for summary in validations:
            for validation in summary.validations:
                tallies[validation.confidence] += 1
                total_score += validation.confidence_score
                total_items += 1
        mean = _round_ratio(total_score / total_items) if total_items else (
            EMPTY_MEAN_CONFIDENCE
        )
        return cls(
            high=tallies[EvidenceConfidence.HIGH],
            medium=tallies[EvidenceConfidence.MEDIUM],
            low=tallies[EvidenceConfidence.LOW],
            unknown=tallies[EvidenceConfidence.UNKNOWN],
            mean_score=mean,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "high": self.high,
            "medium": self.medium,
            "low": self.low,
            "unknown": self.unknown,
            "total": self.total,
            "mean_score": self.mean_score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ConfidenceDistribution:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            high=_require_int(data, "high"),
            medium=_require_int(data, "medium"),
            low=_require_int(data, "low"),
            unknown=_require_int(data, "unknown"),
            mean_score=_require_float(data, "mean_score"),
        )


# ----------------------------------------------------------------------
# The execution request
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchExecution:
    """The immutable request behind one orchestrated research run.

    This is the *input* side of the execution layer: the planner input,
    the policy that governs the run, and the states the run is expected to
    visit.  It is produced before any stage runs and never changes
    afterwards, which is what makes ``execution_id`` a stable content
    hash of the request.

    Parameters
    ----------
    schema_version:
        Execution serialization schema version.
    execution_id:
        Stable content hash of the request.
    planner_input:
        The :class:`PlannerInput` handed to the Research Planner.
    policy:
        The :class:`ExecutionPolicy` governing the run.
    requested_states:
        Active states the run is expected to visit, in canonical pipeline
        order.  Always contains ``PLANNING``; contains ``VALIDATION``
        unless the policy disabled validation.
    """

    schema_version: str
    execution_id: str
    planner_input: PlannerInput
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    requested_states: tuple[ExecutionState, ...] = ACTIVE_STATES

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise InvalidExecutionModelError(
                "schema_version must not be empty for ResearchExecution"
            )
        if not isinstance(self.execution_id, str) or not self.execution_id.strip():
            raise InvalidExecutionModelError(
                "execution_id must not be empty for ResearchExecution"
            )
        if not isinstance(self.planner_input, PlannerInput):
            raise InvalidExecutionInputError(
                "planner_input must be a PlannerInput instance"
            )
        if not isinstance(self.policy, ExecutionPolicy):
            raise InvalidExecutionModelError(
                "policy must be an ExecutionPolicy for ResearchExecution"
            )
        object.__setattr__(
            self, "schema_version", self.schema_version.strip()
        )
        object.__setattr__(self, "execution_id", self.execution_id.strip())
        states = tuple(self.requested_states)
        if not states:
            raise InvalidExecutionModelError(
                "requested_states must not be empty for ResearchExecution"
            )
        positions: list[int] = []
        for state in states:
            if not isinstance(state, ExecutionState) or not state.is_active:
                raise InvalidExecutionModelError(
                    "requested_states entries must be active ExecutionStates"
                )
            positions.append(ACTIVE_STATES.index(state))
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            raise InvalidExecutionModelError(
                "requested_states must be unique and in canonical pipeline order"
            )
        if ExecutionState.PLANNING not in states:
            raise InvalidExecutionModelError(
                "requested_states must include the PLANNING state"
            )
        object.__setattr__(self, "requested_states", states)

    @classmethod
    def create(
        cls,
        planner_input: PlannerInput,
        *,
        policy: ExecutionPolicy | None = None,
    ) -> ResearchExecution:
        """Build a request for ``planner_input`` and derive its identifier.

        ``execution_id`` is a content hash over the schema version, the
        planner input, and the policy, so two runs with the same request
        always share an identifier.
        """
        if not isinstance(planner_input, PlannerInput):
            raise InvalidExecutionInputError(
                "create expects a PlannerInput instance"
            )
        resolved_policy = policy if policy is not None else ExecutionPolicy()
        states = (
            ACTIVE_STATES
            if resolved_policy.run_validation
            else tuple(
                state
                for state in ACTIVE_STATES
                if state is not ExecutionState.VALIDATION
            )
        )
        payload = cls._identity_payload(planner_input, resolved_policy)
        execution_id = f"exec_{content_digest(payload)}"
        return cls(
            schema_version=EXECUTION_SCHEMA_VERSION,
            execution_id=execution_id,
            planner_input=planner_input,
            policy=resolved_policy,
            requested_states=states,
        )

    @staticmethod
    def _identity_payload(
        planner_input: PlannerInput,
        policy: ExecutionPolicy,
    ) -> dict[str, object]:
        """Return the canonical payload hashed into ``execution_id``."""
        return {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "planner_input": planner_input.to_dict(),
            "policy": policy.to_dict(),
        }

    @property
    def company_name(self) -> str:
        """Return the company under research."""
        return self.planner_input.company_name

    def visits(self, state: ExecutionState) -> bool:
        """Return whether the run is expected to visit ``state``."""
        return state in self.requested_states

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "company_name": self.company_name,
            "planner_input": self.planner_input.to_dict(),
            "policy": self.policy.to_dict(),
            "requested_states": [state.value for state in self.requested_states],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResearchExecution:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        raw_states = _require_str_list(data, "requested_states")
        states: list[ExecutionState] = []
        for value in raw_states:
            state = ExecutionState.from_value(value)
            states.append(state)
        return cls(
            schema_version=_require_str(data, "schema_version"),
            execution_id=_require_str(data, "execution_id"),
            planner_input=PlannerInput.from_dict(
                _require_dict(data, "planner_input")
            ),
            policy=ExecutionPolicy.from_dict(_require_dict(data, "policy")),
            requested_states=tuple(states),
        )


# ----------------------------------------------------------------------
# Deserialization helpers
# ----------------------------------------------------------------------


def _require_str(
    data: Mapping[str, object],
    key: str,
    *,
    default: str | None = None,
) -> str:
    """Read and validate a string field from a serialized dictionary."""
    value = data.get(key, default)
    if not isinstance(value, str):
        raise InvalidExecutionModelError(f"expected string for '{key}'")
    return value


def _require_int(data: Mapping[str, object], key: str) -> int:
    """Read and validate an int field from a serialized dictionary."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidExecutionModelError(f"expected int for '{key}'")
    return value


def _require_float(data: Mapping[str, object], key: str) -> float:
    """Read and validate a number field from a serialized dictionary."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidExecutionModelError(f"expected number for '{key}'")
    return float(value)


def _require_bool(data: Mapping[str, object], key: str) -> bool:
    """Read and validate a bool field from a serialized dictionary."""
    value = data.get(key)
    if not isinstance(value, bool):
        raise InvalidExecutionModelError(f"expected bool for '{key}'")
    return value


def _require_str_list(data: Mapping[str, object], key: str) -> list[str]:
    """Read and validate a string-list field."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise InvalidExecutionModelError(f"expected list for '{key}'")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise InvalidExecutionModelError(
                f"expected list of strings for '{key}'"
            )
        result.append(item)
    return result


def _require_dict(data: Mapping[str, object], key: str) -> dict[str, object]:
    """Read and validate a nested dictionary field."""
    value = data.get(key)
    if not isinstance(value, dict):
        raise InvalidExecutionModelError(f"expected dict for '{key}'")
    return value


def _require_list_of_dicts(
    data: Mapping[str, object], key: str
) -> list[dict[str, object]]:
    """Read and validate a list-of-dicts field."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise InvalidExecutionModelError(f"expected list for '{key}'")
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise InvalidExecutionModelError(
                f"expected list of dicts for '{key}'"
            )
        result.append(item)
    return result


def _require_non_negative_int(value: object, field_name: str, owner: str) -> None:
    """Validate that ``value`` is a non-negative ``int``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidExecutionModelError(
            f"{field_name} must be an int for {owner}"
        )
    if value < 0:
        raise InvalidExecutionModelError(
            f"{field_name} must be non-negative for {owner}"
        )


def _require_non_negative_float(value: object, field_name: str) -> float:
    """Validate that ``value`` is a finite, non-negative number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidExecutionModelError(f"{field_name} must be a number")
    number = float(value)
    if number < 0.0:
        raise InvalidExecutionModelError(f"{field_name} must be non-negative")
    return number


def _require_unit_interval(value: object, field_name: str) -> float:
    """Validate that ``value`` is a number within the closed unit interval."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidExecutionModelError(f"{field_name} must be a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise InvalidExecutionModelError(
            f"{field_name} must be within [0, 1]"
        )
    return number


def _clean_topics(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return ``values`` de-duplicated, order preserved, whitespace stripped."""
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise InvalidExecutionModelError(
                "topic identifiers must be non-empty strings"
            )
        stripped = value.strip()
        if stripped not in cleaned:
            cleaned.append(stripped)
    return tuple(cleaned)


def _round_ratio(value: float) -> float:
    """Round a derived ratio so float noise cannot break determinism."""
    return round(value, RATIO_PRECISION)


__all__ = [
    "EMPTY_COVERAGE_RATIO",
    "EMPTY_MEAN_CONFIDENCE",
    "EXECUTION_SCHEMA_VERSION",
    "RATIO_PRECISION",
    "ConfidenceDistribution",
    "DispatchOutcome",
    "ExecutionCoverage",
    "ExecutionError",
    "ExecutionMetrics",
    "ExecutionPolicy",
    "ExecutionStatistics",
    "ExecutionStep",
    "ExecutionStepStatus",
    "ExecutionTiming",
    "ExecutionWarning",
    "ResearchExecution",
    "StepTiming",
]
