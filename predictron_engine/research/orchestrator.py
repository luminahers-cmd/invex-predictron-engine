"""Deterministic research orchestrator — Phase 8 Sprint 5.

:class:`ResearchOrchestrator` is the execution layer that runs the whole
research pipeline end to end and is the production entry point every
future autonomous capability builds on::

    ResearchPlanner
        -> SourceDiscovery
        -> CollectorDispatcher
        -> EvidenceCollectionEngine
        -> EvidenceValidationEngine
        -> ResearchReport

Public API
----------
::

    orchestrator = ResearchOrchestrator()
    result = orchestrator.execute(planner_input)

Every component is *reused*, never reimplemented: the orchestrator holds
no planning logic, no source-ranking logic, no collection logic, and no
validation logic.  It only sequences the existing deterministic components
and records what happened.

Determinism guarantees
----------------------
The orchestrator is deterministic by construction:

* **No threading, no async, no background workers, no scheduling.**  The
  pipeline is a straight-line sequence of calls; stages run strictly in
  order, one at a time.
* **No retries.**  A stage runs exactly once.  A failure is recorded, not
  re-attempted.
* **No LLM reasoning and no crawling.**  All external access is delegated
  to the injected components, which are themselves metadata-only.
* **No mutation of upstream artifacts.**  Plans, discovery plans,
  collection results, and validation summaries are read-only inputs and
  are stored by reference on the result.
* **Stable ordering everywhere.**  Warnings, errors, steps, and
  transitions are appended in a fixed pipeline order; every derived count
  iterates a sorted or plan-ordered collection.
* **No wall-clock by default.**  Duration metadata is opt-in via
  :attr:`ExecutionPolicy.measure_duration`, and it is excluded from the
  report's content hash, so the default path is byte-for-byte stable.

Error handling
--------------
Two tiers, mirroring the collection engine's own convention:

* **Stage errors** — any exception escaping a component is recorded as an
  :class:`~predictron_engine.research.execution_models.ExecutionError`
  and the run continues with the stages whose inputs still exist.  This is
  what makes partial completion the default.
* **Fatal input errors** — a ``planner_input`` that is not a
  :class:`PlannerInput` raises
  :class:`~predictron_engine.research.exceptions.InvalidExecutionInputError`,
  because there is no execution to report at all.

Aborting is opt-in: construct the orchestrator with
``ExecutionPolicy(fail_fast=True)`` and the first stage error is re-raised
instead of recorded.

Which stages still run after a failure is deterministic and derived purely
from artifact availability: planning gates discovery, dispatch, and
collection; collection gates validation.  A discovery failure therefore
does **not** prevent collection, because the Evidence Collection Engine
consumes the *plan*, not the source recommendations.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from predictron_engine.research.dispatcher import Dispatcher
from predictron_engine.research.exceptions import InvalidExecutionInputError
from predictron_engine.research.execution import EvidenceCollectionEngine
from predictron_engine.research.execution_models import (
    DispatchOutcome,
    ExecutionError,
    ExecutionPolicy,
    ExecutionStep,
    ExecutionStepStatus,
    ExecutionTiming,
    ExecutionWarning,
    ResearchExecution,
    StepTiming,
)
from predictron_engine.research.execution_report import (
    ResearchExecutionResult,
    build_research_report,
)
from predictron_engine.research.execution_state import (
    ExecutionState,
    ExecutionStateTransition,
    resolve_final_state,
    transition_state,
)
from predictron_engine.research.models import (
    CollectionResult,
    PlannerInput,
    ResearchPlan,
    SourceDiscoveryPlan,
)
from predictron_engine.research.planner import ResearchPlanner
from predictron_engine.research.source_discovery import SourceDiscovery
from predictron_engine.research.validation import EvidenceValidationEngine
from predictron_engine.research.validation_models import (
    ValidationStatus,
    ValidationSummary,
    content_digest,
)

#: Stable step names, in execution order.
STEP_PLANNING = "planning"
STEP_DISCOVERY = "discovery"
STEP_DISPATCH = "dispatch"
STEP_COLLECTION = "collection"
STEP_VALIDATION = "validation"

#: Every step the pipeline can record, in execution order.
PIPELINE_STEPS: tuple[str, ...] = (
    STEP_PLANNING,
    STEP_DISCOVERY,
    STEP_DISPATCH,
    STEP_COLLECTION,
    STEP_VALIDATION,
)

# -- Warning codes (non-fatal observations) ---------------------------

WARNING_EMPTY_PLAN = "empty_plan"
WARNING_NO_SOURCES = "no_sources_discovered"
WARNING_UNRESOLVED_TASKS = "unresolved_tasks"
WARNING_COLLECTOR_FAILURES = "collector_failures"
WARNING_NO_EVIDENCE = "no_evidence_collected"
WARNING_VALIDATION_SKIPPED = "validation_skipped"
WARNING_VALIDATION_WARNED = "validation_warned"
WARNING_VALIDATION_FAILED = "validation_failed"
WARNING_VALIDATION_CONFLICTS = "validation_conflicts"
WARNING_DUPLICATE_EVIDENCE = "duplicate_evidence"

# -- Error codes (fatal stage failures) -------------------------------

ERROR_PLANNING_FAILED = "planning_failed"
ERROR_DISCOVERY_FAILED = "discovery_failed"
ERROR_DISPATCH_FAILED = "dispatch_failed"
ERROR_COLLECTION_FAILED = "collection_failed"
ERROR_VALIDATION_FAILED = "validation_error"

#: Matches the memory addresses ``repr`` embeds, so recorded error text
#: stays deterministic even for exceptions that stringify an object.
_ADDRESS_PATTERN = re.compile(r"0x[0-9a-fA-F]+")


def _stable_text(value: object) -> str:
    """Return a deterministic, address-free string for ``value``."""
    return _ADDRESS_PATTERN.sub("0xADDR", str(value).strip())


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(UTC).isoformat()


class ResearchOrchestrator:
    """Deterministic coordinator of the complete research pipeline.

    Parameters
    ----------
    planner:
        Optional :class:`ResearchPlanner`.  Defaults to a planner over the
        built-in rule registry.
    source_discovery:
        Optional :class:`SourceDiscovery`.  Defaults to discovery over the
        built-in source registry.
    dispatcher:
        Optional :class:`Dispatcher`.  When omitted, the dispatcher of the
        resolved collection engine is used, so the orchestrator can never
        report a dispatch that the collection engine would not reproduce.
    collection_engine:
        Optional :class:`EvidenceCollectionEngine`.  Defaults to an engine
        bound to the resolved dispatcher.
    validation_engine:
        Optional :class:`EvidenceValidationEngine`.  Defaults to an engine
        over the shared validator registry.
    policy:
        Optional :class:`ExecutionPolicy`.  Defaults to the deterministic,
        partial-completion-friendly default.

    Wiring note
    -----------
    ``dispatcher`` and ``collection_engine`` are reconciled rather than
    used independently: the collection engine always resolves collectors
    through the dispatcher the orchestrator reports, which removes the
    possibility of an orchestrator whose reported dispatch disagrees with
    the dispatch that actually executed.
    """

    def __init__(
        self,
        planner: ResearchPlanner | None = None,
        source_discovery: SourceDiscovery | None = None,
        dispatcher: Dispatcher | None = None,
        collection_engine: EvidenceCollectionEngine | None = None,
        validation_engine: EvidenceValidationEngine | None = None,
        policy: ExecutionPolicy | None = None,
    ) -> None:
        self._planner = planner if planner is not None else ResearchPlanner()
        self._source_discovery = (
            source_discovery if source_discovery is not None else SourceDiscovery()
        )
        self._validation_engine = (
            validation_engine
            if validation_engine is not None
            else EvidenceValidationEngine()
        )
        self._policy = policy if policy is not None else ExecutionPolicy()
        self._dispatcher, self._collection_engine = _reconcile_pipeline(
            dispatcher, collection_engine
        )

    # ------------------------------------------------------------------
    # Component accessors
    # ------------------------------------------------------------------

    @property
    def planner(self) -> ResearchPlanner:
        """Return the Research Planner this orchestrator drives."""
        return self._planner

    @property
    def source_discovery(self) -> SourceDiscovery:
        """Return the Source Discovery engine this orchestrator drives."""
        return self._source_discovery

    @property
    def dispatcher(self) -> Dispatcher:
        """Return the dispatcher used for the reported dispatch."""
        return self._dispatcher

    @property
    def collection_engine(self) -> EvidenceCollectionEngine:
        """Return the Evidence Collection Engine this orchestrator drives."""
        return self._collection_engine

    @property
    def validation_engine(self) -> EvidenceValidationEngine:
        """Return the Evidence Validation Engine this orchestrator drives."""
        return self._validation_engine

    @property
    def policy(self) -> ExecutionPolicy:
        """Return the policy governing every run."""
        return self._policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, planner_input: PlannerInput) -> ResearchExecutionResult:
        """Execute the complete research pipeline for ``planner_input``.

        Runs, in order: planning, discovery, dispatch, collection, and
        validation; then derives the report and returns the full result.

        The same ``planner_input`` and the same injected components always
        produce an identical result, including its ``execution_id`` and
        ``report_id``.

        Raises
        ------
        InvalidExecutionInputError:
            When ``planner_input`` is not a :class:`PlannerInput`.  There is
            no execution to report, so this fails regardless of policy.
        Exception:
            When :attr:`ExecutionPolicy.fail_fast` is set, the first stage
            error is re-raised unchanged instead of being recorded.
        """
        if not isinstance(planner_input, PlannerInput):
            raise InvalidExecutionInputError(
                "execute expects a PlannerInput instance"
            )
        return self._run_pipeline(planner_input)

    def execute_for_company(
        self,
        company_name: str,
        *,
        website: str = "",
        description: str = "",
        known_topics: tuple[str, ...] = (),
        partial_topics: tuple[str, ...] = (),
        prediction_horizon_days: int = 365,
    ) -> ResearchExecutionResult:
        """Convenience: build a :class:`PlannerInput` and execute it.

        Mirrors the ``*_for_company`` helpers on the individual components
        so callers do not have to assemble planner input by hand.
        """
        return self.execute(
            PlannerInput(
                company_name=company_name,
                website=website,
                description=description,
                known_topics=known_topics,
                partial_topics=partial_topics,
                prediction_horizon_days=prediction_horizon_days,
            )
        )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        planner_input: PlannerInput,
    ) -> ResearchExecutionResult:
        """Run every stage and assemble the immutable result."""
        execution = ResearchExecution.create(planner_input, policy=self._policy)
        run = _Run(execution)
        measure = self._policy.measure_duration
        started_at = _now_iso() if measure else ""
        started_perf = time.perf_counter() if measure else 0.0

        plan = self._stage_planning(run, planner_input)
        discovery = self._stage_discovery(run, plan) if plan is not None else None
        dispatch = self._stage_dispatch(run, plan) if plan is not None else None
        collection = (
            self._stage_collection(run, plan) if plan is not None else None
        )
        validation = (
            self._stage_validation(run, collection)
            if collection is not None and execution.visits(ExecutionState.VALIDATION)
            else None
        )
        state = run.settle()

        timing = self._timing(
            run,
            started_at=started_at,
            finished_at=_now_iso() if measure else "",
            total_duration_ms=(
                (time.perf_counter() - started_perf) * 1000.0
                if measure
                else 0.0
            ),
        )
        report = build_research_report(
            execution=execution,
            state=state,
            steps=tuple(run.steps),
            plan=plan,
            discovery=discovery,
            dispatch=dispatch,
            collection=collection,
            validation=validation,
            warnings=tuple(run.warnings),
            errors=tuple(run.errors),
            timing=timing,
        )
        return ResearchExecutionResult(
            execution=execution,
            state=state,
            steps=tuple(run.steps),
            transitions=tuple(run.transitions),
            report=report,
            plan=plan,
            discovery=discovery,
            dispatch=dispatch,
            collection=collection,
            validation=validation,
            timing=timing,
        )

    # -- Stage 1: planning -------------------------------------------

    def _stage_planning(
        self,
        run: _Run,
        planner_input: PlannerInput,
    ) -> ResearchPlan | None:
        """Run the Research Planner, or record the failure and return ``None``."""
        run.enter(ExecutionState.PLANNING)
        run.begin_step(STEP_PLANNING, ExecutionState.PLANNING)
        try:
            with self._measure(run, STEP_PLANNING):
                plan = self._planner.create_plan(planner_input)
        except Exception as exc:
            self._fail_stage(
                run,
                step=STEP_PLANNING,
                code=ERROR_PLANNING_FAILED,
                exc=exc,
                summary="research planning failed",
            )
            return None
        if not plan.tasks:
            run.warn(
                WARNING_EMPTY_PLAN,
                "the planner produced no research tasks",
                STEP_PLANNING,
            )
        run.end_step(
            ExecutionStepStatus.COMPLETED,
            artifact_id=plan.plan_id,
            detail=(
                f"{len(plan.tasks)} task(s) across "
                f"{len(plan.topics_researched)} topic(s)"
            ),
        )
        return plan

    # -- Stage 2: discovery -------------------------------------------

    def _stage_discovery(
        self,
        run: _Run,
        plan: ResearchPlan,
    ) -> SourceDiscoveryPlan | None:
        """Run Source Discovery over ``plan``, recording any failure."""
        run.enter(ExecutionState.DISCOVERY)
        run.begin_step(STEP_DISCOVERY, ExecutionState.DISCOVERY)
        try:
            with self._measure(run, STEP_DISCOVERY):
                discovery = self._source_discovery.discover(plan)
        except Exception as exc:
            self._fail_stage(
                run,
                step=STEP_DISCOVERY,
                code=ERROR_DISCOVERY_FAILED,
                exc=exc,
                summary="source discovery failed",
            )
            return None
        recommendations = len(discovery.recommendations)
        ranked = sum(len(entry.sources) for entry in discovery.recommendations)
        if recommendations == 0:
            run.warn(
                WARNING_NO_SOURCES,
                "source discovery produced no recommendations",
                STEP_DISCOVERY,
            )
        run.end_step(
            ExecutionStepStatus.COMPLETED,
            artifact_id=_discovery_artifact_id(discovery),
            detail=f"{recommendations} recommendation(s), {ranked} ranked source(s)",
        )
        return discovery

    # -- Stage 3: dispatch --------------------------------------------

    def _stage_dispatch(
        self,
        run: _Run,
        plan: ResearchPlan,
    ) -> DispatchOutcome | None:
        """Resolve the plan's tasks to collectors and project the result.

        Also opens the ``COLLECTION`` state: dispatch and collection share
        it, because resolving a task to a collector and running that
        collector are two views of the same stage.
        """
        run.enter(ExecutionState.COLLECTION)
        run.begin_step(STEP_DISPATCH, ExecutionState.COLLECTION)
        try:
            with self._measure(run, STEP_DISPATCH):
                dispatched = self._dispatcher.dispatch(plan.tasks)
        except Exception as exc:
            self._fail_stage(
                run,
                step=STEP_DISPATCH,
                code=ERROR_DISPATCH_FAILED,
                exc=exc,
                summary="collector dispatch failed",
            )
            return None
        outcome = DispatchOutcome.from_dispatch(dispatched)
        if outcome.unresolved:
            run.warn(
                WARNING_UNRESOLVED_TASKS,
                f"{outcome.unresolved_count} task(s) had no registered collector",
                STEP_DISPATCH,
                detail=",".join(outcome.unresolved),
            )
        run.end_step(
            ExecutionStepStatus.COMPLETED,
            artifact_id=_dispatch_artifact_id(outcome),
            detail=(
                f"{outcome.dispatched_count} task(s) dispatched to "
                f"{len(outcome.distinct_collectors)} collector(s)"
            ),
        )
        return outcome

    # -- Stage 4: collection ------------------------------------------

    def _stage_collection(
        self,
        run: _Run,
        plan: ResearchPlan,
    ) -> CollectionResult | None:
        """Run the Evidence Collection Engine over ``plan``.

        The ``COLLECTION`` state was already opened by
        :meth:`_stage_dispatch`; both stages always run together.
        """
        run.begin_step(STEP_COLLECTION, ExecutionState.COLLECTION)
        try:
            with self._measure(run, STEP_COLLECTION):
                collection = self._collection_engine.execute(plan)
        except Exception as exc:
            self._fail_stage(
                run,
                step=STEP_COLLECTION,
                code=ERROR_COLLECTION_FAILED,
                exc=exc,
                summary="evidence collection failed",
            )
            return None
        if collection.failed_tasks:
            run.warn(
                WARNING_COLLECTOR_FAILURES,
                f"{len(collection.failed_tasks)} task(s) produced no evidence",
                STEP_COLLECTION,
                detail=",".join(collection.failed_tasks),
            )
        if collection.evidence_count == 0:
            run.warn(
                WARNING_NO_EVIDENCE,
                "evidence collection produced no evidence",
                STEP_COLLECTION,
            )
        run.end_step(
            ExecutionStepStatus.COMPLETED,
            artifact_id=collection.collection_id,
            detail=(
                f"{collection.evidence_count} evidence item(s) from "
                f"{len(collection.executed_tasks)} task(s) "
                f"[{collection.status.value}]"
            ),
        )
        return collection

    # -- Stage 5: validation ------------------------------------------

    def _stage_validation(
        self,
        run: _Run,
        collection: CollectionResult,
    ) -> ValidationSummary | None:
        """Validate the collected evidence and record the outcome.

        Only reached when the run's request includes the ``VALIDATION``
        state; a run configured without validation never enters this state,
        records no validation step, and short-circuits straight from
        ``COLLECTION`` to its terminal state.
        """
        run.enter(ExecutionState.VALIDATION)
        run.begin_step(STEP_VALIDATION, ExecutionState.VALIDATION)
        try:
            with self._measure(run, STEP_VALIDATION):
                validation = self._validation_engine.validate(
                    collection.collection
                )
        except Exception as exc:
            self._fail_stage(
                run,
                step=STEP_VALIDATION,
                code=ERROR_VALIDATION_FAILED,
                exc=exc,
                summary="evidence validation failed",
            )
            return None
        counts = validation.counts
        self._warn_on_validation(run, validation)
        run.end_step(
            ExecutionStepStatus.COMPLETED,
            artifact_id=validation.validation_id,
            detail=f"{counts.total} item(s) validated [{validation.status.value}]",
        )
        return validation

    @staticmethod
    def _warn_on_validation(
        run: _Run,
        validation: ValidationSummary,
    ) -> None:
        """Record the deterministic, content-level validation findings.

        These are *warnings*, not errors: the validation engine ran to
        completion and produced a summary, so the pipeline itself succeeded
        even when the evidence it produced is untrustworthy.
        """
        counts = validation.counts
        if validation.status is ValidationStatus.SKIPPED:
            run.warn(
                WARNING_VALIDATION_SKIPPED,
                "validation found no evidence to check",
                STEP_VALIDATION,
            )
        elif validation.status is ValidationStatus.FAILED:
            run.warn(
                WARNING_VALIDATION_FAILED,
                f"{counts.failed} evidence item(s) failed validation",
                STEP_VALIDATION,
            )
        elif validation.status is ValidationStatus.WARNED:
            run.warn(
                WARNING_VALIDATION_WARNED,
                f"{counts.warned} evidence item(s) raised validation warnings",
                STEP_VALIDATION,
            )
        if counts.conflicts:
            run.warn(
                WARNING_VALIDATION_CONFLICTS,
                f"{counts.conflicts} evidence conflict(s) detected",
                STEP_VALIDATION,
            )
        if counts.duplicates:
            run.warn(
                WARNING_DUPLICATE_EVIDENCE,
                f"{counts.duplicates} duplicate evidence item(s) detected",
                STEP_VALIDATION,
            )

    # ------------------------------------------------------------------
    # Failure and timing helpers
    # ------------------------------------------------------------------

    def _fail_stage(
        self,
        run: _Run,
        *,
        step: str,
        code: str,
        exc: Exception,
        summary: str,
    ) -> None:
        """Record a stage failure, close its step, and honour fail-fast.

        Stages call this from their ``except`` block and then
        ``return None``; the recorded error is what turns the run's final
        state into ``FAILED``.
        """
        run.fail(code, f"{summary}: {_stable_text(exc)}", step, exc)
        run.end_step(
            ExecutionStepStatus.FAILED,
            detail=f"{summary} ({type(exc).__name__})",
        )
        if self._policy.fail_fast:
            raise exc

    @contextmanager
    def _measure(self, run: _Run, step: str) -> Iterator[None]:
        """Time one stage when duration measurement is enabled.

        A no-op otherwise, so the default path never reads the wall clock.
        """
        if not self._policy.measure_duration:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            run.record_duration(
                step, (time.perf_counter() - started) * 1000.0
            )

    @staticmethod
    def _timing(
        run: _Run,
        *,
        started_at: str,
        finished_at: str,
        total_duration_ms: float,
    ) -> ExecutionTiming:
        """Assemble the timing block, empty unless measurement is enabled."""
        if not run.execution.policy.measure_duration:
            return ExecutionTiming.unmeasured()
        return ExecutionTiming(
            measured=True,
            started_at=started_at,
            finished_at=finished_at,
            total_duration_ms=total_duration_ms,
            step_durations=tuple(run.durations),
        )


# ----------------------------------------------------------------------
# Run bookkeeping
# ----------------------------------------------------------------------


@dataclass
class _Run:
    """Mutable bookkeeping for exactly one execution.

    This is the orchestrator's only mutable state.  It exists purely while
    a run is in flight and is never part of any returned model: every
    public value it accumulates is copied into an immutable model by
    :func:`build_research_report` and
    :class:`~predictron_engine.research.execution_report.ResearchExecutionResult`.

    Ordering is append-only and driven by the fixed pipeline order, which
    is what makes ``steps``, ``transitions``, ``warnings``, and ``errors``
    deterministic without any sorting.
    """

    execution: ResearchExecution
    state: ExecutionState = ExecutionState.NOT_STARTED
    steps: list[ExecutionStep] = field(default_factory=list)
    transitions: list[ExecutionStateTransition] = field(default_factory=list)
    warnings: list[ExecutionWarning] = field(default_factory=list)
    errors: list[ExecutionError] = field(default_factory=list)
    durations: list[StepTiming] = field(default_factory=list)
    states: dict[str, ExecutionState] = field(default_factory=dict)
    _warning_counts: dict[str, int] = field(default_factory=dict)
    _error_counts: dict[str, int] = field(default_factory=dict)
    _open_step: str = ""

    # -- State machine -----------------------------------------------

    def enter(self, target: ExecutionState) -> None:
        """Record a validated hop to ``target`` and adopt it as current.

        Raises
        ------
        InvalidExecutionStateTransitionError:
            When the hop is not part of the declared state machine.  This
            is an orchestrator bug, not a run failure, so it propagates.
        """
        previous = self.state
        self.state = transition_state(previous, target)
        self.transitions.append(
            ExecutionStateTransition(
                sequence=len(self.transitions),
                source=previous,
                target=self.state,
            )
        )

    def settle(self) -> ExecutionState:
        """Enter the run's terminal state and return it.

        The derivation is pure: any recorded stage error yields ``FAILED``,
        and every other run yields ``COMPLETED``.  A run configured without
        validation settles straight from ``COLLECTION``, a transition the
        state machine explicitly allows.
        """
        final = resolve_final_state(has_errors=bool(self.errors))
        self.enter(final)
        return final

    # -- Steps --------------------------------------------------------

    def begin_step(self, name: str, state: ExecutionState) -> None:
        """Open a step and seed its warning and error counters."""
        if self._open_step:
            raise RuntimeError(
                f"step {self._open_step!r} is still open"
            )
        self._open_step = name
        self.states[name] = state
        self._warning_counts.setdefault(name, 0)
        self._error_counts.setdefault(name, 0)

    def end_step(
        self,
        status: ExecutionStepStatus,
        *,
        artifact_id: str = "",
        detail: str = "",
    ) -> ExecutionStep:
        """Close the open step, folding in its warning and error counts.

        Counts are attributed by step *name*, not by whether the step
        happens to be open, so a stage may raise a content-level warning
        either before or after closing its own step and still be counted.
        """
        if not self._open_step:
            raise RuntimeError("no step is open")
        name = self._open_step
        step = ExecutionStep(
            name=name,
            state=self.states[name],
            status=status,
            order=len(self.steps),
            artifact_id=artifact_id,
            detail=detail,
            warning_count=self._warning_counts.get(name, 0),
            error_count=self._error_counts.get(name, 0),
        )
        self.steps.append(step)
        self._open_step = ""
        return step

    # -- Warnings and errors -----------------------------------------

    def warn(
        self,
        code: str,
        message: str,
        step: str,
        detail: str = "",
    ) -> None:
        """Record one non-fatal observation against ``step``."""
        self.warnings.append(
            ExecutionWarning(
                code=code,
                message=message,
                step=step,
                detail=detail,
            )
        )
        self._warning_counts[step] = self._warning_counts.get(step, 0) + 1

    def fail(
        self,
        code: str,
        message: str,
        step: str,
        exc: BaseException,
    ) -> None:
        """Record one fatal stage error against ``step``."""
        self.errors.append(
            ExecutionError(
                code=code,
                message=message,
                step=step,
                exception_type=type(exc).__name__,
            )
        )
        self._error_counts[step] = self._error_counts.get(step, 0) + 1

    # -- Timing -------------------------------------------------------

    def record_duration(self, step: str, duration_ms: float) -> None:
        """Record a measured stage duration, ignoring duplicates."""
        for entry in self.durations:
            if entry.step == step:
                return
        self.durations.append(StepTiming(step=step, duration_ms=duration_ms))


def _reconcile_pipeline(
    dispatcher: Dispatcher | None,
    collection_engine: EvidenceCollectionEngine | None,
) -> tuple[Dispatcher, EvidenceCollectionEngine]:
    """Bind a dispatcher and a collection engine to one shared registry.

    Returns the pair the orchestrator will use.  An explicitly supplied
    collection engine keeps precedence, because it carries the collector
    registry a caller deliberately installed; otherwise an explicitly
    supplied dispatcher is used to build a matching engine.  With neither
    supplied, both are created bound to the same fresh dispatcher.

    This guarantees the orchestrator's reported
    :class:`~predictron_engine.research.execution_models.DispatchOutcome`
    is exactly the dispatch the collection engine performs.
    """
    if collection_engine is not None:
        engine_dispatcher = getattr(collection_engine, "dispatcher", None)
        return (
            engine_dispatcher
            if isinstance(engine_dispatcher, Dispatcher)
            else (dispatcher if dispatcher is not None else Dispatcher()),
            collection_engine,
        )
    if dispatcher is not None:
        return dispatcher, EvidenceCollectionEngine(dispatcher=dispatcher)
    resolved = Dispatcher()
    return resolved, EvidenceCollectionEngine(dispatcher=resolved)


def _discovery_artifact_id(discovery: SourceDiscoveryPlan) -> str:
    """Return a stable content hash identifying a discovery artifact.

    :class:`SourceDiscoveryPlan` carries no identifier of its own, so the
    orchestrator derives one from the plan id plus the ranked
    recommendations.  The digest is deterministic, which keeps the report's
    step artifacts stable across runs.
    """
    payload = {
        "plan_id": discovery.plan_id,
        "recommendations": [
            recommendation.to_dict()
            for recommendation in discovery.recommendations
        ],
    }
    return f"discovery_{content_digest(payload)}"


def _dispatch_artifact_id(outcome: DispatchOutcome) -> str:
    """Return a stable content hash identifying a dispatch artifact."""
    payload = {
        "task_ids": list(outcome.task_ids),
        "collector_ids": list(outcome.collector_ids),
        "unresolved": list(outcome.unresolved),
    }
    return f"dispatch_{content_digest(payload)}"


__all__ = [
    "ERROR_COLLECTION_FAILED",
    "ERROR_DISCOVERY_FAILED",
    "ERROR_DISPATCH_FAILED",
    "ERROR_PLANNING_FAILED",
    "ERROR_VALIDATION_FAILED",
    "PIPELINE_STEPS",
    "STEP_COLLECTION",
    "STEP_DISCOVERY",
    "STEP_DISPATCH",
    "STEP_PLANNING",
    "STEP_VALIDATION",
    "WARNING_COLLECTOR_FAILURES",
    "WARNING_DUPLICATE_EVIDENCE",
    "WARNING_EMPTY_PLAN",
    "WARNING_NO_EVIDENCE",
    "WARNING_NO_SOURCES",
    "WARNING_UNRESOLVED_TASKS",
    "WARNING_VALIDATION_CONFLICTS",
    "WARNING_VALIDATION_FAILED",
    "WARNING_VALIDATION_SKIPPED",
    "WARNING_VALIDATION_WARNED",
    "ResearchOrchestrator",
]
