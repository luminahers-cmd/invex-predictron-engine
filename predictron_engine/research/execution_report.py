"""Deterministic research execution report — Phase 8 Sprint 5.

This module owns everything a caller needs to *understand* one
orchestrated run, without holding the (potentially large) pipeline
artifacts themselves:

* :class:`ValidationOutcome` — a compact, serializable projection of the
  Evidence Validation Engine's :class:`ValidationSummary`.
* :class:`ExecutionSummary` — statistics, metrics, coverage, confidence
  distribution, and the validation outcome, bundled.
* :class:`ResearchReport` — the final, content-addressed report.
* :class:`ResearchExecutionResult` — the public return type of
  :meth:`~predictron_engine.research.orchestrator.ResearchOrchestrator.execute`:
  the stage artifacts plus the report derived from them.
* :func:`build_research_report` — the single pure function that derives a
  report from a run's artifacts.

Determinism
-----------
:func:`build_research_report` is a pure function of the artifacts it is
given.  The report's ``report_id`` is a content hash over the
*deterministic* payload only — the optional wall-clock timing block is
excluded — so a run produces the same identifier whether or not duration
measurement is enabled.

The full :class:`ResearchExecutionResult` is also deterministic, with the
single deliberate exception of :attr:`ExecutionTiming` when
``measure_duration`` is enabled.  That is why timing is opt-in and lives in
its own block rather than being spread across the report.

This module intentionally keeps its own private ``_require_*``
deserialization helpers, mirroring the convention established by
:mod:`predictron_engine.research.models` and
:mod:`predictron_engine.research.validation_models`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

from predictron_engine.research.exceptions import InvalidExecutionModelError
from predictron_engine.research.execution_models import (
    EMPTY_MEAN_CONFIDENCE,
    RATIO_PRECISION,
    ConfidenceDistribution,
    DispatchOutcome,
    ExecutionCoverage,
    ExecutionError,
    ExecutionMetrics,
    ExecutionPolicy,
    ExecutionStatistics,
    ExecutionStep,
    ExecutionTiming,
    ExecutionWarning,
    ResearchExecution,
)
from predictron_engine.research.execution_state import (
    ExecutionState,
    ExecutionStateTransition,
)
from predictron_engine.research.models import (
    CollectionResult,
    ResearchPlan,
    SourceDiscoveryPlan,
)
from predictron_engine.research.validation_models import (
    ValidationStatus,
    ValidationSummary,
    content_digest,
)

#: Serialization schema version for the execution result and report.
REPORT_SCHEMA_VERSION = "1.0.0"

#: Status recorded when the Evidence Validation Engine did not run.
NOT_VALIDATED_STATUS = ValidationStatus.SKIPPED.value


# ----------------------------------------------------------------------
# Validation projection
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationOutcome:
    """A compact, serializable projection of one validation run.

    Parameters
    ----------
    performed:
        Whether the Evidence Validation Engine actually ran and produced a
        summary.  ``False`` for a disabled validation step and for a run
        whose collection stage errored.
    status:
        The :class:`ValidationStatus` value of the run, or
        ``"skipped"`` when validation did not run.
    validation_id:
        The validation summary's content hash (empty when not performed).
    passed / warned / failed / skipped:
        Per-status evidence item counters.
    issues:
        Total validation issues raised across all evidence items.
    conflicts:
        Total evidence conflicts detected.
    duplicates:
        Total duplicate evidence items detected.
    topics_covered:
        How many topics the validation run covered.
    mean_confidence:
        Mean confidence score across validated evidence, rounded to
        :data:`~predictron_engine.research.execution_models.RATIO_PRECISION`
        decimals.  ``0.0`` when nothing was validated.
    """

    performed: bool = False
    status: str = NOT_VALIDATED_STATUS
    validation_id: str = ""
    passed: int = 0
    warned: int = 0
    failed: int = 0
    skipped: int = 0
    issues: int = 0
    conflicts: int = 0
    duplicates: int = 0
    topics_covered: int = 0
    mean_confidence: float = EMPTY_MEAN_CONFIDENCE

    def __post_init__(self) -> None:
        if not isinstance(self.performed, bool):
            raise InvalidExecutionModelError(
                "performed must be a bool for ValidationOutcome"
            )
        if not isinstance(self.status, str) or not self.status.strip():
            raise InvalidExecutionModelError(
                "status must be a non-empty string for ValidationOutcome"
            )
        if not isinstance(self.validation_id, str):
            raise InvalidExecutionModelError(
                "validation_id must be a string for ValidationOutcome"
            )
        for entry in fields(self):
            if entry.name in (
                "performed",
                "status",
                "validation_id",
                "mean_confidence",
            ):
                continue
            value = getattr(self, entry.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidExecutionModelError(
                    f"{entry.name} must be an int for ValidationOutcome"
                )
            if value < 0:
                raise InvalidExecutionModelError(
                    f"{entry.name} must be non-negative for ValidationOutcome"
                )
        if not isinstance(self.mean_confidence, int | float) or isinstance(
            self.mean_confidence, bool
        ):
            raise InvalidExecutionModelError(
                "mean_confidence must be a number for ValidationOutcome"
            )
        if not 0.0 <= float(self.mean_confidence) <= 1.0:
            raise InvalidExecutionModelError(
                "mean_confidence must be within [0, 1] for ValidationOutcome"
            )
        object.__setattr__(self, "status", self.status.strip())
        object.__setattr__(self, "validation_id", self.validation_id.strip())
        object.__setattr__(
            self,
            "mean_confidence",
            round(float(self.mean_confidence), RATIO_PRECISION),
        )

    @property
    def total(self) -> int:
        """Return how many evidence items the validation run examined."""
        return self.passed + self.warned + self.failed + self.skipped

    @property
    def has_failures(self) -> bool:
        """Return whether any validated evidence item failed."""
        return self.failed > 0

    @property
    def is_clean(self) -> bool:
        """Return whether validation ran and found nothing at all."""
        return self.performed and not (
            self.issues or self.conflicts or self.duplicates
        )

    @classmethod
    def from_summary(
        cls,
        summary: ValidationSummary | None,
    ) -> ValidationOutcome:
        """Project a :class:`ValidationSummary` onto its report projection.

        Passing ``None`` yields the deterministic "not validated" outcome,
        which is what a disabled or failed validation stage produces.
        """
        if summary is None:
            return cls()
        counts = summary.counts
        total = counts.total
        mean = (
            sum(
                validation.confidence_score
                for validation in summary.validations
            )
            / total
            if total
            else EMPTY_MEAN_CONFIDENCE
        )
        return cls(
            performed=True,
            status=summary.status.value,
            validation_id=summary.validation_id,
            passed=counts.passed,
            warned=counts.warned,
            failed=counts.failed,
            skipped=counts.skipped,
            issues=counts.issues,
            conflicts=counts.conflicts,
            duplicates=counts.duplicates,
            topics_covered=counts.topics_covered,
            mean_confidence=mean,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "performed": self.performed,
            "status": self.status,
            "validation_id": self.validation_id,
            "passed": self.passed,
            "warned": self.warned,
            "failed": self.failed,
            "skipped": self.skipped,
            "total": self.total,
            "issues": self.issues,
            "conflicts": self.conflicts,
            "duplicates": self.duplicates,
            "topics_covered": self.topics_covered,
            "mean_confidence": self.mean_confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ValidationOutcome:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            performed=_require_bool(data, "performed"),
            status=_require_str(data, "status"),
            validation_id=_require_str(data, "validation_id", default=""),
            passed=_require_int(data, "passed"),
            warned=_require_int(data, "warned"),
            failed=_require_int(data, "failed"),
            skipped=_require_int(data, "skipped"),
            issues=_require_int(data, "issues"),
            conflicts=_require_int(data, "conflicts"),
            duplicates=_require_int(data, "duplicates"),
            topics_covered=_require_int(data, "topics_covered"),
            mean_confidence=_require_float(data, "mean_confidence"),
        )


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionSummary:
    """The deterministic digest of one execution.

    Bundles the four reporting views the pipeline produces:

    * :class:`ExecutionStatistics` — domain counters (topics, sources,
      collectors, evidence),
    * :class:`ExecutionMetrics` — operational counters (steps, warnings,
      errors, validation findings, coverage ratio),
    * :class:`ExecutionCoverage` — which planned topics produced evidence,
    * :class:`ConfidenceDistribution` — the validated confidence bands,
    * :class:`ValidationOutcome` — the validation run's headline numbers.
    """

    statistics: ExecutionStatistics = field(default_factory=ExecutionStatistics)
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    coverage: ExecutionCoverage = field(default_factory=ExecutionCoverage)
    confidence: ConfidenceDistribution = field(
        default_factory=ConfidenceDistribution
    )
    validation: ValidationOutcome = field(default_factory=ValidationOutcome)

    def __post_init__(self) -> None:
        expected = {
            "statistics": ExecutionStatistics,
            "metrics": ExecutionMetrics,
            "coverage": ExecutionCoverage,
            "confidence": ConfidenceDistribution,
            "validation": ValidationOutcome,
        }
        for field_name, model_type in expected.items():
            if not isinstance(getattr(self, field_name), model_type):
                raise InvalidExecutionModelError(
                    f"{field_name} must be a {model_type.__name__} for "
                    "ExecutionSummary"
                )

    @property
    def is_clean(self) -> bool:
        """Return whether the run recorded no warnings and no errors."""
        return self.metrics.is_clean

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "statistics": self.statistics.to_dict(),
            "metrics": self.metrics.to_dict(),
            "coverage": self.coverage.to_dict(),
            "confidence": self.confidence.to_dict(),
            "validation": self.validation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecutionSummary:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            statistics=ExecutionStatistics.from_dict(
                _require_dict(data, "statistics")
            ),
            metrics=ExecutionMetrics.from_dict(_require_dict(data, "metrics")),
            coverage=ExecutionCoverage.from_dict(_require_dict(data, "coverage")),
            confidence=ConfidenceDistribution.from_dict(
                _require_dict(data, "confidence")
            ),
            validation=ValidationOutcome.from_dict(
                _require_dict(data, "validation")
            ),
        )


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchReport:
    """The final, content-addressed report for one research execution.

    Parameters
    ----------
    schema_version:
        Report serialization schema version.
    report_id:
        Stable content hash over the deterministic report payload.  The
        wall-clock :class:`ExecutionTiming` block is deliberately excluded
        from the hash, so enabling duration measurement never changes it.
    execution_id:
        Identifier of the :class:`ResearchExecution` this report describes.
    company_name:
        Company under research.
    plan_id:
        Identifier of the underlying research plan (empty when planning
        failed).
    state:
        The terminal :class:`ExecutionState` of the run.
    summary:
        The bundled statistics / metrics / coverage / confidence /
        validation digest.
    steps:
        Every :class:`ExecutionStep`, in execution order.
    warnings:
        Non-fatal observations, in the order the pipeline raised them.
    errors:
        Fatal stage errors, in the order the pipeline raised them.
    timing:
        Wall-clock metadata, empty unless duration measurement is enabled.
    """

    schema_version: str
    report_id: str
    execution_id: str
    company_name: str
    plan_id: str
    state: ExecutionState
    summary: ExecutionSummary
    steps: tuple[ExecutionStep, ...] = ()
    warnings: tuple[ExecutionWarning, ...] = ()
    errors: tuple[ExecutionError, ...] = ()
    timing: ExecutionTiming = field(default_factory=ExecutionTiming)

    def __post_init__(self) -> None:
        for field_name in ("schema_version", "report_id", "execution_id", "company_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidExecutionModelError(
                    f"{field_name} must be a non-empty string for ResearchReport"
                )
        if not isinstance(self.plan_id, str):
            raise InvalidExecutionModelError(
                "plan_id must be a string for ResearchReport"
            )
        if not isinstance(self.state, ExecutionState):
            raise InvalidExecutionModelError(
                "state must be an ExecutionState for ResearchReport"
            )
        if not isinstance(self.summary, ExecutionSummary):
            raise InvalidExecutionModelError(
                "summary must be an ExecutionSummary for ResearchReport"
            )
        if not isinstance(self.timing, ExecutionTiming):
            raise InvalidExecutionModelError(
                "timing must be an ExecutionTiming for ResearchReport"
            )
        for field_name, model_type in (
            ("steps", ExecutionStep),
            ("warnings", ExecutionWarning),
            ("errors", ExecutionError),
        ):
            entries = tuple(getattr(self, field_name))
            for entry in entries:
                if not isinstance(entry, model_type):
                    raise InvalidExecutionModelError(
                        f"{field_name} entries must be {model_type.__name__} "
                        "instances for ResearchReport"
                    )
            object.__setattr__(self, field_name, entries)
        orders = [step.order for step in self.steps]
        if orders != sorted(orders) or len(set(orders)) != len(orders):
            raise InvalidExecutionModelError(
                "ResearchReport steps must have unique, ascending order values"
            )
        names = [step.name for step in self.steps]
        if len(set(names)) != len(names):
            raise InvalidExecutionModelError(
                "ResearchReport step names must be unique"
            )

    # -- Convenience views -------------------------------------------

    @property
    def statistics(self) -> ExecutionStatistics:
        """Return the domain counters."""
        return self.summary.statistics

    @property
    def metrics(self) -> ExecutionMetrics:
        """Return the operational counters."""
        return self.summary.metrics

    @property
    def coverage(self) -> ExecutionCoverage:
        """Return the topic-coverage breakdown."""
        return self.summary.coverage

    @property
    def confidence(self) -> ConfidenceDistribution:
        """Return the validated confidence distribution."""
        return self.summary.confidence

    @property
    def validation(self) -> ValidationOutcome:
        """Return the validation headline numbers."""
        return self.summary.validation

    @property
    def is_successful(self) -> bool:
        """Return whether the run reached ``COMPLETED`` with no errors."""
        return self.state is ExecutionState.COMPLETED and not self.errors

    @property
    def is_deterministic(self) -> bool:
        """Return whether every field of this report is input-derived."""
        return not self.timing.measured

    @property
    def duration_ms(self) -> float:
        """Return the total measured duration in milliseconds."""
        return self.timing.total_duration_ms

    def has_errors(self) -> bool:
        """Return whether the run recorded at least one fatal error."""
        return bool(self.errors)

    def has_warnings(self) -> bool:
        """Return whether the run recorded at least one warning."""
        return bool(self.warnings)

    @property
    def warning_codes(self) -> tuple[str, ...]:
        """Return the distinct warning codes, in first-raised order."""
        return _distinct_codes(self.warnings)

    @property
    def error_codes(self) -> tuple[str, ...]:
        """Return the distinct error codes, in first-raised order."""
        return _distinct_codes(self.errors)

    def step(self, name: str) -> ExecutionStep | None:
        """Return the :class:`ExecutionStep` named ``name`` or ``None``."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    @property
    def topic_count(self) -> int:
        """Return how many topics the run researched."""
        return self.statistics.topics_researched

    @property
    def evidence_count(self) -> int:
        """Return how much evidence the run collected."""
        return self.statistics.evidence_collected

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "execution_id": self.execution_id,
            "company_name": self.company_name,
            "plan_id": self.plan_id,
            "state": self.state.value,
            "summary": self.summary.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "errors": [error.to_dict() for error in self.errors],
            "timing": self.timing.to_dict(),
            "is_successful": self.is_successful,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResearchReport:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            schema_version=_require_str(data, "schema_version"),
            report_id=_require_str(data, "report_id"),
            execution_id=_require_str(data, "execution_id"),
            company_name=_require_str(data, "company_name"),
            plan_id=_require_str(data, "plan_id", default=""),
            state=ExecutionState.from_value(data.get("state")),
            summary=ExecutionSummary.from_dict(_require_dict(data, "summary")),
            steps=tuple(
                ExecutionStep.from_dict(item)
                for item in _require_list_of_dicts(data, "steps")
            ),
            warnings=tuple(
                ExecutionWarning.from_dict(item)
                for item in _require_list_of_dicts(data, "warnings")
            ),
            errors=tuple(
                ExecutionError.from_dict(item)
                for item in _require_list_of_dicts(data, "errors")
            ),
            timing=ExecutionTiming.from_dict(_require_dict(data, "timing")),
        )


# ----------------------------------------------------------------------
# Report construction
# ----------------------------------------------------------------------


def build_research_report(
    *,
    execution: ResearchExecution,
    state: ExecutionState,
    steps: tuple[ExecutionStep, ...],
    plan: ResearchPlan | None,
    discovery: SourceDiscoveryPlan | None,
    dispatch: DispatchOutcome | None,
    collection: CollectionResult | None,
    validation: ValidationSummary | None,
    warnings: tuple[ExecutionWarning, ...],
    errors: tuple[ExecutionError, ...],
    timing: ExecutionTiming | None = None,
) -> ResearchReport:
    """Derive the deterministic :class:`ResearchReport` for one run.

    This is a pure function: given the same artifacts it always produces
    the same report, down to ``report_id``.  It never mutates the supplied
    artifacts — the plan, discovery plan, collection result, and
    validation summary are read-only inputs, and the returned report holds
    only identifiers and counts derived from them.

    Parameters mirror the pipeline stages; every one of them is optional
    because a failed stage simply contributes no artifact, which is what
    makes partial-completion reports possible.
    """
    if not isinstance(execution, ResearchExecution):
        raise InvalidExecutionModelError(
            "execution must be a ResearchExecution instance"
        )
    resolved_timing = timing if timing is not None else ExecutionTiming.unmeasured()
    resolved_steps = _require_entries(steps, ExecutionStep, "steps")
    resolved_warnings = _require_entries(warnings, ExecutionWarning, "warnings")
    resolved_errors = _require_entries(errors, ExecutionError, "errors")

    coverage = _coverage_for(plan, collection)
    confidence = (
        ConfidenceDistribution.from_validations((validation,))
        if validation is not None
        else ConfidenceDistribution.empty()
    )
    validation_outcome = ValidationOutcome.from_summary(validation)
    statistics = _statistics_for(
        plan=plan,
        discovery=discovery,
        dispatch=dispatch,
        collection=collection,
        validation=validation,
        coverage=coverage,
    )
    metrics = _metrics_for(
        steps=resolved_steps,
        warnings=resolved_warnings,
        errors=resolved_errors,
        validation=validation_outcome,
        coverage=coverage,
    )
    summary = ExecutionSummary(
        statistics=statistics,
        metrics=metrics,
        coverage=coverage,
        confidence=confidence,
        validation=validation_outcome,
    )
    plan_id = plan.plan_id if plan is not None else ""
    report_id = _report_id(
        execution=execution,
        state=state,
        plan_id=plan_id,
        summary=summary,
        steps=resolved_steps,
        warnings=resolved_warnings,
        errors=resolved_errors,
    )
    return ResearchReport(
        schema_version=REPORT_SCHEMA_VERSION,
        report_id=report_id,
        execution_id=execution.execution_id,
        company_name=execution.company_name,
        plan_id=plan_id,
        state=state,
        summary=summary,
        steps=resolved_steps,
        warnings=resolved_warnings,
        errors=resolved_errors,
        timing=resolved_timing,
    )


def _require_entries[Entry](
    entries: tuple[Entry, ...],
    model_type: type[Entry],
    field_name: str,
) -> tuple[Entry, ...]:
    """Validate that every entry of ``entries`` is a ``model_type``.

    Guards the derivation against a caller that assembles a bookkeeping
    tuple by hand, so a malformed entry surfaces as a domain error rather
    than an ``AttributeError`` from deep inside the counters.
    """
    resolved = tuple(entries)
    for entry in resolved:
        if not isinstance(entry, model_type):
            raise InvalidExecutionModelError(
                f"{field_name} entries must be {model_type.__name__} instances"
            )
    return resolved


def _coverage_for(
    plan: ResearchPlan | None,
    collection: CollectionResult | None,
) -> ExecutionCoverage:
    """Derive the topic-coverage breakdown from the plan and collection."""
    planned = (
        tuple(task.topic_id for task in plan.tasks) if plan is not None else ()
    )
    covered = collection.topics_covered if collection is not None else ()
    return ExecutionCoverage(planned_topics=planned, covered_topics=covered)


def _statistics_for(
    *,
    plan: ResearchPlan | None,
    discovery: SourceDiscoveryPlan | None,
    dispatch: DispatchOutcome | None,
    collection: CollectionResult | None,
    validation: ValidationSummary | None,
    coverage: ExecutionCoverage,
) -> ExecutionStatistics:
    """Derive the domain counters from the pipeline artifacts.

    ``collectors_executed`` counts collector *invocations that produced
    evidence*.  Because the dispatcher binds exactly one collector per
    task, it equals :attr:`tasks_executed` by construction; it is reported
    separately because "collectors executed" is the number operators ask
    about, and :attr:`distinct_collectors` is the number of distinct
    collector implementations involved.
    """
    source_identifiers: set[str] = set()
    sources_discovered = 0
    if discovery is not None:
        for recommendation in discovery.recommendations:
            sources_discovered += len(recommendation.sources)
            source_identifiers.update(recommendation.source_identifiers())
    unresolved = (
        len(dispatch.unresolved)
        if dispatch is not None
        else (len(collection.unresolved_tasks) if collection is not None else 0)
    )
    executed = (
        len(collection.executed_tasks) if collection is not None else 0
    )
    return ExecutionStatistics(
        topics_planned=coverage.planned_count,
        topics_covered=coverage.covered_count,
        tasks_planned=len(plan.tasks) if plan is not None else 0,
        tasks_dispatched=(
            dispatch.dispatched_count if dispatch is not None else 0
        ),
        tasks_executed=executed,
        tasks_failed=(
            len(collection.failed_tasks) if collection is not None else 0
        ),
        tasks_unresolved=unresolved,
        sources_discovered=sources_discovered,
        distinct_sources=len(source_identifiers),
        collectors_executed=executed,
        distinct_collectors=(
            len(dispatch.distinct_collectors) if dispatch is not None else 0
        ),
        evidence_collected=(
            collection.evidence_count if collection is not None else 0
        ),
        evidence_validated=(
            validation.counts.total if validation is not None else 0
        ),
    )


def _metrics_for(
    *,
    steps: tuple[ExecutionStep, ...],
    warnings: tuple[ExecutionWarning, ...],
    errors: tuple[ExecutionError, ...],
    validation: ValidationOutcome,
    coverage: ExecutionCoverage,
) -> ExecutionMetrics:
    """Derive the operational counters from the run's bookkeeping."""
    return ExecutionMetrics(
        steps_total=len(steps),
        steps_completed=sum(1 for step in steps if step.is_completed),
        steps_skipped=sum(1 for step in steps if step.is_skipped),
        steps_failed=sum(1 for step in steps if step.is_failed),
        warnings=len(warnings),
        errors=len(errors),
        validation_issues=validation.issues,
        validation_conflicts=validation.conflicts,
        validation_duplicates=validation.duplicates,
        coverage_ratio=coverage.ratio,
    )


def _report_id(
    *,
    execution: ResearchExecution,
    state: ExecutionState,
    plan_id: str,
    summary: ExecutionSummary,
    steps: tuple[ExecutionStep, ...],
    warnings: tuple[ExecutionWarning, ...],
    errors: tuple[ExecutionError, ...],
) -> str:
    """Return the stable content hash identifying the report.

    The payload deliberately omits :class:`ExecutionTiming`: a report is
    identified by *what happened*, not by *how long it took*.
    """
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "execution_id": execution.execution_id,
        "company_name": execution.company_name,
        "plan_id": plan_id,
        "state": state.value,
        "summary": summary.to_dict(),
        "steps": [step.to_dict() for step in steps],
        "warnings": [warning.to_dict() for warning in warnings],
        "errors": [error.to_dict() for error in errors],
    }
    return f"report_{content_digest(payload)}"


def _distinct_codes(entries: tuple[ExecutionWarning | ExecutionError, ...]) -> tuple[str, ...]:
    """Return the distinct codes of ``entries`` in first-raised order."""
    codes: list[str] = []
    for entry in entries:
        if entry.code not in codes:
            codes.append(entry.code)
    return tuple(codes)


# ----------------------------------------------------------------------
# Execution result
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchExecutionResult:
    """The complete outcome of one orchestrated research execution.

    This is what :meth:`ResearchOrchestrator.execute` returns.  It holds
    the stage artifacts produced by the pipeline (or ``None`` for a stage
    that never produced one) alongside the :class:`ResearchReport` derived
    from them.

    Every stage artifact is an *existing* object from the earlier sprints
    and is stored by reference — the orchestrator never rewrites, repairs,
    or reorders a plan, a collection, or a validation summary.

    Parameters
    ----------
    execution:
        The :class:`ResearchExecution` request that produced this result.
    state:
        The terminal :class:`ExecutionState` of the run.
    steps:
        One :class:`ExecutionStep` per pipeline step, in execution order.
    transitions:
        Every validated hop through the state machine, in order.
    plan:
        The research plan, or ``None`` when planning errored.
    discovery:
        The source discovery plan, or ``None`` when discovery errored.
    dispatch:
        The dispatch projection, or ``None`` when dispatch errored.
    collection:
        The collection result, or ``None`` when collection errored.
    validation:
        The validation summary, or ``None`` when validation was skipped or
        errored.
    report:
        The deterministic report derived from the fields above.
    timing:
        Wall-clock metadata, empty unless duration measurement is enabled.
    """

    execution: ResearchExecution
    state: ExecutionState
    steps: tuple[ExecutionStep, ...]
    transitions: tuple[ExecutionStateTransition, ...]
    report: ResearchReport
    plan: ResearchPlan | None = None
    discovery: SourceDiscoveryPlan | None = None
    dispatch: DispatchOutcome | None = None
    collection: CollectionResult | None = None
    validation: ValidationSummary | None = None
    timing: ExecutionTiming = field(default_factory=ExecutionTiming)

    def __post_init__(self) -> None:
        if not isinstance(self.execution, ResearchExecution):
            raise InvalidExecutionModelError(
                "execution must be a ResearchExecution instance"
            )
        if not isinstance(self.state, ExecutionState):
            raise InvalidExecutionModelError(
                "state must be an ExecutionState for ResearchExecutionResult"
            )
        if not isinstance(self.report, ResearchReport):
            raise InvalidExecutionModelError(
                "report must be a ResearchReport for ResearchExecutionResult"
            )
        if not isinstance(self.timing, ExecutionTiming):
            raise InvalidExecutionModelError(
                "timing must be an ExecutionTiming for ResearchExecutionResult"
            )
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(
            self, "transitions", tuple(self.transitions)
        )
        for step in self.steps:
            if not isinstance(step, ExecutionStep):
                raise InvalidExecutionModelError(
                    "steps entries must be ExecutionStep instances"
                )
        for transition in self.transitions:
            if not isinstance(transition, ExecutionStateTransition):
                raise InvalidExecutionModelError(
                    "transitions entries must be ExecutionStateTransition "
                    "instances"
                )
        sequences = [entry.sequence for entry in self.transitions]
        if sequences != sorted(sequences):
            raise InvalidExecutionModelError(
                "transitions must be ordered by ascending sequence"
            )
        if not self.transitions or self.transitions[-1].target is not self.state:
            raise InvalidExecutionModelError(
                "the final recorded transition must target the result state"
            )

    # -- Delegated views --------------------------------------------

    @property
    def execution_id(self) -> str:
        """Return the execution identifier."""
        return self.execution.execution_id

    @property
    def report_id(self) -> str:
        """Return the report identifier."""
        return self.report.report_id

    @property
    def company_name(self) -> str:
        """Return the company under research."""
        return self.execution.company_name

    @property
    def policy(self) -> ExecutionPolicy:
        """Return the policy that governed the run."""
        return self.execution.policy

    @property
    def warnings(self) -> tuple[ExecutionWarning, ...]:
        """Return the run's warnings."""
        return self.report.warnings

    @property
    def errors(self) -> tuple[ExecutionError, ...]:
        """Return the run's fatal errors."""
        return self.report.errors

    @property
    def warning_codes(self) -> tuple[str, ...]:
        """Return the distinct warning codes, in first-raised order."""
        return self.report.warning_codes

    @property
    def error_codes(self) -> tuple[str, ...]:
        """Return the distinct error codes, in first-raised order."""
        return self.report.error_codes

    @property
    def duration_ms(self) -> float:
        """Return the total measured duration in milliseconds."""
        return self.report.duration_ms

    @property
    def statistics(self) -> ExecutionStatistics:
        """Return the run's domain counters."""
        return self.report.statistics

    @property
    def metrics(self) -> ExecutionMetrics:
        """Return the run's operational counters."""
        return self.report.metrics

    @property
    def coverage(self) -> ExecutionCoverage:
        """Return the run's topic coverage."""
        return self.report.coverage

    @property
    def confidence(self) -> ConfidenceDistribution:
        """Return the run's confidence distribution."""
        return self.report.confidence

    @property
    def validation_outcome(self) -> ValidationOutcome:
        """Return the run's validation headline numbers."""
        return self.report.validation

    # -- Derived state ----------------------------------------------

    def succeeded(self) -> bool:
        """Return whether the run completed with no fatal stage error."""
        return self.state is ExecutionState.COMPLETED and not self.errors

    def has_errors(self) -> bool:
        """Return whether the run recorded at least one fatal error."""
        return bool(self.errors)

    def has_warnings(self) -> bool:
        """Return whether the run recorded at least one warning."""
        return bool(self.warnings)

    @property
    def is_partial(self) -> bool:
        """Return whether the run completed while missing some work.

        ``True`` means the pipeline itself finished (``COMPLETED``) but at
        least one planned task failed or was left unresolved.  This is the
        normal, supported outcome for a partially researched company.
        """
        return self.state is ExecutionState.COMPLETED and not (
            self.statistics.is_complete
        )

    @property
    def is_deterministic(self) -> bool:
        """Return whether every field of this result is input-derived."""
        return not self.timing.measured

    def step(self, name: str) -> ExecutionStep | None:
        """Return the :class:`ExecutionStep` named ``name`` or ``None``."""
        return self.report.step(name)

    def state_path(self) -> tuple[ExecutionState, ...]:
        """Return the states the run visited, in visit order.

        The first element is the state the run started in
        (``NOT_STARTED``); the last is :attr:`state`.
        """
        path = [entry.source for entry in self.transitions]
        path.append(self.transitions[-1].target)
        return tuple(path)

    def evidence(self) -> tuple[object, ...]:
        """Return the collected evidence items in deterministic order.

        Returns an empty tuple when the collection stage never produced a
        result.  The items are the untouched :class:`Evidence` objects from
        the Evidence Collection Engine.
        """
        if self.collection is None:
            return ()
        return self.collection.collection.items()

    def to_dict(self) -> dict[str, object]:
        """Serialize to a fully JSON-ready dictionary."""
        return {
            "execution": self.execution.to_dict(),
            "state": self.state.value,
            "steps": [step.to_dict() for step in self.steps],
            "transitions": [
                transition.to_dict() for transition in self.transitions
            ],
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "discovery": (
                self.discovery.to_dict() if self.discovery is not None else None
            ),
            "dispatch": (
                self.dispatch.to_dict() if self.dispatch is not None else None
            ),
            "collection": (
                self.collection.to_dict() if self.collection is not None else None
            ),
            "validation": (
                self.validation.to_dict() if self.validation is not None else None
            ),
            "report": self.report.to_dict(),
            "timing": self.timing.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResearchExecutionResult:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        plan_data = _optional_dict(data, "plan")
        discovery_data = _optional_dict(data, "discovery")
        dispatch_data = _optional_dict(data, "dispatch")
        collection_data = _optional_dict(data, "collection")
        validation_data = _optional_dict(data, "validation")
        return cls(
            execution=ResearchExecution.from_dict(
                _require_dict(data, "execution")
            ),
            state=ExecutionState.from_value(data.get("state")),
            steps=tuple(
                ExecutionStep.from_dict(item)
                for item in _require_list_of_dicts(data, "steps")
            ),
            transitions=tuple(
                ExecutionStateTransition.from_dict(item)
                for item in _require_list_of_dicts(data, "transitions")
            ),
            report=ResearchReport.from_dict(_require_dict(data, "report")),
            plan=ResearchPlan.from_dict(plan_data) if plan_data else None,
            discovery=(
                SourceDiscoveryPlan.from_dict(discovery_data)
                if discovery_data
                else None
            ),
            dispatch=(
                DispatchOutcome.from_dict(dispatch_data) if dispatch_data else None
            ),
            collection=(
                CollectionResult.from_dict(collection_data)
                if collection_data
                else None
            ),
            validation=(
                ValidationSummary.from_dict(validation_data)
                if validation_data
                else None
            ),
            timing=ExecutionTiming.from_dict(_require_dict(data, "timing")),
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


def _require_dict(data: Mapping[str, object], key: str) -> dict[str, object]:
    """Read and validate a nested dictionary field."""
    value = data.get(key)
    if not isinstance(value, dict):
        raise InvalidExecutionModelError(f"expected dict for '{key}'")
    return value


def _optional_dict(
    data: Mapping[str, object],
    key: str,
) -> dict[str, object] | None:
    """Read a nested dictionary field that may be absent or ``None``."""
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvalidExecutionModelError(
            f"expected dict or null for '{key}'"
        )
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


__all__ = [
    "NOT_VALIDATED_STATUS",
    "REPORT_SCHEMA_VERSION",
    "ExecutionSummary",
    "ResearchExecutionResult",
    "ResearchReport",
    "ValidationOutcome",
    "build_research_report",
]
