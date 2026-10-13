"""The deterministic research intelligence loop — Phase 8 Sprint 7.

:class:`ResearchLoop` closes the circuit the Sprint 5 pipeline left open.
One iteration is:

    plan → discover → dispatch → collect → validate
        → write to research memory
        → analyse residual gaps
        → decide whether to iterate again

and the loop repeats that until it is finished, provably done, or out of
budget.  The loop is the **only** component that iterates; the
:class:`~predictron_engine.research.orchestrator.ResearchOrchestrator` stays
a straight-line single pass, and all gap reasoning belongs to
:class:`~predictron_engine.research.residual.ResidualGapEngine`.  Neither is
reimplemented here — the loop only decides *when* to call them again.

Why iteration is bounded
------------------------
An unbounded self-refining loop can spin forever.  This one cannot, because
every path out is deterministic and enumerated in
:class:`~predictron_engine.research.iteration_models.TerminationReason`:

* a fatal stage error stops the run immediately, because retrying a failed
  pass would repeat the same failure;
* meeting the policy's thresholds with no blocking gaps completes the
  research;
* meeting the thresholds finishes as a *partial* completion when the policy
  allows it, even with gaps still open;
* finding no *actionable* gap converges: another pass could not improve
  anything.  A gap caused by a missing collector (``UNRESOLVED``) is
  blocking but not actionable, because the collector registry does not change
  between passes;
* a pass whose blocking gaps are identical to the previous pass's converges,
  since it learned nothing another pass would not also fail to learn;
* an unchanged revised planner input converges for the same reason, and is
  the backstop that makes an oscillating planner terminate;
* exhausting ``max_iterations`` stops the run.

Convergence is therefore never a hope — it is a reachable, named state.

Every iteration reduces uncertainty or stops
--------------------------------------------
This is the loop's core promise, and it is enforced structurally.  To
continue, a pass must both leave an *actionable* gap and either change the
blocking gap set or revise the planner input.  An iteration that changes
nothing is the definition of wasted work, so the loop treats it as
convergence.  The convergence check compares gaps by ``(topic, kind)``
identity rather than by ``task_id``, so re-ranking or renaming a task for the
same still-broken topic is not mistaken for progress.

Nothing in here reads a clock, draws a random number, or makes a network
call of its own: iteration counts, thresholds, and gap identity are all
functions of the artifacts each pass produced.  Wall-clock time enters only
through the memory write's caller-supplied ``stored_at``.  The same
``PlannerInput``, components, and policy therefore always produce the same
:class:`~predictron_engine.research.iteration_models.ResearchIterationHistory`
— the property that makes the loop replayable.

Public API
----------
::

    loop = ResearchLoop(policy=IterationPolicy(max_iterations=3))
    history = loop.run(planner_input)
    metrics = LoopMetrics.from_history(history)
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from predictron_engine.research.exceptions import InvalidIterationInputError
from predictron_engine.research.execution_report import ResearchExecutionResult
from predictron_engine.research.execution_state import ExecutionState
from predictron_engine.research.iteration_models import (
    GapKind,
    IterationPolicy,
    IterationRecord,
    LoopDecision,
    LoopMetrics,
    MemoryContext,
    MemoryUpdate,
    ResearchIterationHistory,
    ResidualAnalysis,
    ResidualGap,
    TerminationReason,
)
from predictron_engine.research.memory import ResearchMemory
from predictron_engine.research.models import (
    CollectionResult,
    Evidence,
    PlannerInput,
    ResearchPlan,
)
from predictron_engine.research.ports import ResearchExecutorPort
from predictron_engine.research.residual import ResidualGapEngine
from predictron_engine.research.validation_models import ValidationSummary

#: Stable label used in record explanations for each termination reason, so a
#: history reads the same way every time without embedding volatile detail.
STOP_EXPLANATIONS: dict[TerminationReason, str] = {
    TerminationReason.RESEARCH_COMPLETE: (
        "every planned topic met the policy thresholds and no blocking "
        "residual gap remains"
    ),
    TerminationReason.PARTIAL_COMPLETION_ALLOWED: (
        "the policy thresholds were met and the policy allows finishing "
        "while residual gaps remain"
    ),
    TerminationReason.CONVERGED: (
        "residual gaps remain but none of them can be acted on, so another "
        "iteration would repeat the same work"
    ),
    TerminationReason.MAX_ITERATIONS_REACHED: (
        "the iteration budget was spent"
    ),
    TerminationReason.EXECUTION_FAILED: (
        "the pass recorded a fatal stage error and would repeat it"
    ),
}

#: Matches the memory addresses ``repr`` embeds, so a recorded memory error
#: stays deterministic even when the exception text stringifies an object.
_ADDRESS_PATTERN = re.compile(r"0x[0-9a-fA-F]+")

#: Gap kinds another iteration can plausibly close, derived from the single
#: definition on :attr:`GapKind.is_actionable` so the loop and the residual
#: analysis can never disagree about what "actionable" means.  ``UNRESOLVED``
#: is blocking but not actionable (the collector registry does not change
#: between passes) and ``STALE`` is not even blocking; everything else is.
ACTIONABLE_GAP_KINDS: frozenset[GapKind] = frozenset(
    kind for kind in GapKind if kind.is_actionable
)


class ResearchLoop:
    """Iterate the research pipeline until it converges or runs out of budget.

    Parameters
    ----------
    orchestrator:
        The pipeline runner.  Defaults to a fresh
        :class:`~predictron_engine.research.orchestrator.ResearchOrchestrator`,
        constructed lazily so importing this module never builds collectors.
        Anything satisfying :class:`~predictron_engine.research.ports.ResearchExecutorPort`
        structurally is accepted, so custom executors can be injected without
        subclassing the orchestrator.
    policy:
        Bounds and thresholds for the loop.  Defaults to
        :class:`~predictron_engine.research.iteration_models.IterationPolicy`,
        whose ``max_iterations`` of ``1`` reproduces the Sprint 5 single-pass
        behaviour exactly — opting into iteration is an explicit decision.
    memory:
        Optional research memory.  Written after each pass and read back to
        build the next pass's context, but only when
        :attr:`IterationPolicy.use_research_memory` is set.  Memory narrows
        future work; it never invents facts about the current one.
    gap_engine:
        The residual gap analysis.  Defaults to a
        :class:`~predictron_engine.research.residual.ResidualGapEngine`
        configured with ``policy``, so gap thresholds and loop thresholds
        cannot drift apart.
    ages:
        Optional caller-supplied ``topic_id -> age in days`` map used for
        freshness.  Deliberately not read from a clock: staleness stays a
        comparison of integers the caller already had.

    Notes
    -----
    The loop keeps no state between :meth:`run` calls — every attribute is
    injected and immutable — so one instance can run any number of
    companies, and two runs with the same inputs are indistinguishable.
    """

    def __init__(
        self,
        orchestrator: ResearchExecutorPort | None = None,
        *,
        policy: IterationPolicy | None = None,
        memory: ResearchMemory | None = None,
        gap_engine: ResidualGapEngine | None = None,
        ages: Mapping[str, int] | None = None,
    ) -> None:
        self._policy = _require_policy(policy)
        self._memory = memory
        self._gap_engine = (
            gap_engine
            if gap_engine is not None
            else ResidualGapEngine(policy=self._policy)
        )
        self._ages = dict(ages) if ages else {}
        self._orchestrator = orchestrator

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def policy(self) -> IterationPolicy:
        """Return the bounds governing every run."""
        return self._policy

    @property
    def memory(self) -> ResearchMemory | None:
        """Return the research memory this loop reads and writes."""
        return self._memory

    @property
    def gap_engine(self) -> ResidualGapEngine:
        """Return the residual gap engine driving every decision."""
        return self._gap_engine

    @property
    def orchestrator(self) -> ResearchExecutorPort:
        """Return the pipeline runner, constructing the default on first use."""
        if self._orchestrator is None:
            from predictron_engine.research.orchestrator import (
                ResearchOrchestrator,
            )

            self._orchestrator = ResearchOrchestrator()
        return self._orchestrator

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def run(self, planner_input: PlannerInput) -> ResearchIterationHistory:
        """Research until done, converged, or out of budget.

        Parameters
        ----------
        planner_input:
            The company to research.  Iteration 1 runs this input as given;
            every later iteration runs the input the previous iteration's
            residual analysis revised.

        Returns
        -------
        ResearchIterationHistory
            Every iteration's record, the final residual analysis, and why
            the loop stopped.

        Raises
        ------
        InvalidIterationInputError:
            When ``planner_input`` is not a :class:`PlannerInput`.
        """
        current = _require_planner_input(planner_input)
        records: list[IterationRecord] = []
        analysis: ResidualAnalysis | None = None
        previous: ResidualAnalysis | None = None
        context: MemoryContext | None = None
        iteration = 0

        while True:
            iteration += 1
            result = self._execute(current)
            memory_update = self._write_memory(result)
            context = self._build_context(
                company_name=current.company_name,
                plan=result.plan,
                validation=result.validation,
            )
            analysis = self._gap_engine.analyze(
                iteration=iteration,
                validation=result.validation,
                plan=result.plan,
                collection=result.collection,
                dispatch=result.dispatch,
                errors=result.errors,
                context=context,
            )
            records.append(
                self._record(
                    iteration=iteration,
                    planner_input=current,
                    result=result,
                    analysis=analysis,
                    memory_update=memory_update,
                )
            )
            decision = self._decide(
                iteration=iteration,
                analysis=analysis,
                result=result,
            )
            if decision.should_stop:
                return ResearchIterationHistory(
                    iterations=tuple(records),
                    termination=decision.reason,
                    policy=self._policy,
                    analysis=analysis,
                    memory_context=context,
                )
            if previous is not None and not _made_progress(previous, analysis):
                return ResearchIterationHistory(
                    iterations=tuple(records),
                    termination=TerminationReason.CONVERGED,
                    policy=self._policy,
                    analysis=analysis,
                    memory_context=context,
                )
            previous = analysis
            revised = self._gap_engine.revise_planner_input(
                current, analysis, context=context
            )
            if revised.to_dict() == current.to_dict():
                return ResearchIterationHistory(
                    iterations=tuple(records),
                    termination=TerminationReason.CONVERGED,
                    policy=self._policy,
                    analysis=analysis,
                    memory_context=context,
                )
            current = revised

    def run_for_company(
        self,
        company_name: str,
        *,
        website: str = "",
        description: str = "",
        known_topics: tuple[str, ...] = (),
        partial_topics: tuple[str, ...] = (),
        prediction_horizon_days: int = 365,
    ) -> ResearchIterationHistory:
        """Convenience: build a :class:`PlannerInput` and run the loop.

        Mirrors ``execute_for_company`` on the orchestrator so callers do not
        have to assemble planner input by hand.
        """
        return self.run(
            PlannerInput(
                company_name=company_name,
                website=website,
                description=description,
                known_topics=known_topics,
                partial_topics=partial_topics,
                prediction_horizon_days=prediction_horizon_days,
            )
        )

    def metrics(self, history: ResearchIterationHistory) -> LoopMetrics:
        """Return the deterministic metrics for a completed run."""
        return LoopMetrics.from_history(history)

    # ------------------------------------------------------------------
    # One iteration
    # ------------------------------------------------------------------

    def _execute(self, planner_input: PlannerInput) -> ResearchExecutionResult:
        """Run one full pipeline pass and return its result."""
        orchestrator = self.orchestrator
        if not isinstance(orchestrator, ResearchExecutorPort):
            raise InvalidIterationInputError(
                "the orchestrator must expose a callable execute(planner_input)"
            )
        execute = orchestrator.execute
        if not callable(execute):
            raise InvalidIterationInputError(
                "the orchestrator must expose a callable execute(planner_input)"
            )
        result = execute(planner_input)
        if not isinstance(result, ResearchExecutionResult):
            raise InvalidIterationInputError(
                "the orchestrator's execute must return a "
                "ResearchExecutionResult"
            )
        return result

    def _write_memory(self, result: ResearchExecutionResult) -> MemoryUpdate:
        """Record one pass in research memory, tolerating a failed write.

        A memory write that raises is recorded as
        :attr:`MemoryUpdate.applied` ``False`` rather than propagated: it
        degrades the *next* iteration, not the one that just completed.  That
        is the same partial-completion contract the orchestrator follows for
        stage failures, and it keeps the loop deterministic — the failure is
        part of the recorded history, not a surprise raised at the caller.
        """
        if self._memory is None or not self._policy.use_research_memory:
            return MemoryUpdate()
        try:
            write = self._memory.add_report(
                result.report,
                evidence=_collected_evidence(result.collection),
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            return MemoryUpdate(applied=False, error=_stable_text(exc))
        return MemoryUpdate(
            applied=True,
            reports_written=1,
            evidence_stored=len(write.evidence_ids),
            duplicates=len(write.duplicate_evidence_ids),
            entities_created=len(write.created_entity_ids),
            entities_updated=len(write.updated_entity_ids),
            relationships_created=len(write.created_relationship_ids),
            history_entries=len(write.history_sequences),
            new_evidence_ids=write.evidence_ids,
        )

    def _build_context(
        self,
        *,
        company_name: str,
        plan: ResearchPlan | None,
        validation: ValidationSummary | None,
    ) -> MemoryContext | None:
        """Return the memory-derived context for one pass, or ``None``.

        Returns ``None`` when memory is not in play, which tells the residual
        engine there is nothing to say about freshness rather than pretending
        every topic is fresh.
        """
        if self._memory is None or not self._policy.use_research_memory:
            return None
        return self._gap_engine.build_memory_context(
            company_name=company_name,
            plan=plan,
            validation=validation,
            memory=self._memory,
            ages=self._ages,
        )

    def _record(
        self,
        *,
        iteration: int,
        planner_input: PlannerInput,
        result: ResearchExecutionResult,
        analysis: ResidualAnalysis,
        memory_update: MemoryUpdate,
    ) -> IterationRecord:
        """Fold one pass into the immutable record history keeps."""
        plan = result.plan
        completed = set(analysis.completed_tasks)
        unresolved = set(analysis.unresolved_tasks) | set(analysis.failed_tasks)
        return IterationRecord(
            iteration=iteration,
            planner_input=planner_input,
            plan_id=plan.plan_id if plan is not None else "",
            planned_tasks=tuple(task.task_id for task in plan.tasks) if plan else (),
            planned_task_topics=(
                plan.topics_researched if plan is not None else ()
            ),
            completed_tasks=tuple(
                task.task_id
                for task in plan.tasks
                if task.task_id in completed
            )
            if plan
            else (),
            unresolved_tasks=tuple(
                task.task_id
                for task in plan.tasks
                if task.task_id in unresolved
            )
            if plan
            else (),
            new_evidence_ids=memory_update.new_evidence_ids,
            evidence_count=_evidence_count(result.collection),
            topics_completed=analysis.completed_topics,
            topics_incomplete=analysis.topics_with_gaps,
            confidence=analysis.mean_confidence,
            coverage_ratio=analysis.coverage_ratio,
            residual_gaps=analysis.gaps,
            memory_update=memory_update,
            state=result.state,
            steps=result.steps,
            warnings=result.warnings,
            errors=result.errors,
        )

    # ------------------------------------------------------------------
    # The decision
    # ------------------------------------------------------------------

    def _decide(
        self,
        *,
        iteration: int,
        analysis: ResidualAnalysis,
        result: ResearchExecutionResult,
    ) -> LoopDecision:
        """Return the deterministic answer to "should another pass run?".

        The order matters and encodes the priorities:

        1. a fatal error stops first — retrying repeats the failure;
        2. completion is checked before the budget, so a finished run reports
           ``RESEARCH_COMPLETE`` rather than looking budget-exhausted;
        3. allowed partial completion is next;
        4. no *actionable* gap means there is nothing left another pass could
           improve, which is convergence rather than failure.  A gap caused by
           a missing collector (``UNRESOLVED``) is blocking but not actionable
           and counts here as convergence;
        5. only then does the budget apply.
        """
        if result.state is ExecutionState.FAILED or result.errors:
            return LoopDecision.stop(
                TerminationReason.EXECUTION_FAILED,
                STOP_EXPLANATIONS[TerminationReason.EXECUTION_FAILED],
            )
        if analysis.is_complete and self._policy.stop_when_complete:
            return LoopDecision.stop(
                TerminationReason.RESEARCH_COMPLETE,
                STOP_EXPLANATIONS[TerminationReason.RESEARCH_COMPLETE],
            )
        if analysis.thresholds_met and self._policy.allow_partial_completion:
            return LoopDecision.stop(
                TerminationReason.PARTIAL_COMPLETION_ALLOWED,
                STOP_EXPLANATIONS[TerminationReason.PARTIAL_COMPLETION_ALLOWED],
            )
        if not _actionable_gaps(analysis):
            return LoopDecision.stop(
                TerminationReason.CONVERGED,
                STOP_EXPLANATIONS[TerminationReason.CONVERGED],
            )
        if not self._policy.allows_iteration(iteration):
            return LoopDecision.stop(
                TerminationReason.MAX_ITERATIONS_REACHED,
                STOP_EXPLANATIONS[TerminationReason.MAX_ITERATIONS_REACHED],
            )
        return LoopDecision.continue_loop(
            f"{len(_actionable_gaps(analysis))} actionable residual gap(s) "
            f"remain after iteration {iteration}"
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _require_policy(policy: IterationPolicy | None) -> IterationPolicy:
    """Return the policy to use, rejecting anything that is not one."""
    if policy is None:
        return IterationPolicy()
    if not isinstance(policy, IterationPolicy):
        raise InvalidIterationInputError("policy must be an IterationPolicy")
    return policy


def _require_planner_input(planner_input: object) -> PlannerInput:
    """Return ``planner_input`` if it is a real one, else raise."""
    if not isinstance(planner_input, PlannerInput):
        raise InvalidIterationInputError(
            "run expects a PlannerInput instance"
        )
    return planner_input


def _collected_evidence(
    collection: CollectionResult | None,
) -> tuple[Evidence, ...]:
    """Return the evidence items one pass collected."""
    if collection is None:
        return ()
    return collection.collection.items()


def _evidence_count(collection: CollectionResult | None) -> int:
    """Return how many evidence items one pass collected."""
    if collection is None:
        return 0
    return collection.evidence_count


def _actionable_gaps(analysis: ResidualAnalysis) -> tuple[ResidualGap, ...]:
    """Return the blocking gaps another iteration could plausibly close."""
    return tuple(
        gap
        for gap in analysis.blocking_gaps
        if gap.kind in ACTIONABLE_GAP_KINDS
    )


def _blocking_keys(analysis: ResidualAnalysis) -> frozenset[tuple[str, str]]:
    """Return the ``(topic, kind)`` identity of every blocking gap.

    Identity deliberately ignores ``task_id`` and iteration: the planner may
    rename or re-rank a task for the same topic between passes, and that must
    not be mistaken for progress.
    """
    return frozenset(
        (gap.topic_id, gap.kind.value) for gap in analysis.blocking_gaps
    )


def _made_progress(previous: ResidualAnalysis, current: ResidualAnalysis) -> bool:
    """Return whether ``current`` closed or changed any blocking gap.

    A pass made progress when its blocking gap set differs from the previous
    pass's in any way:

    * gaps that disappeared were closed — progress;
    * gaps whose kind changed moved up the ladder — progress;
    * gaps that appeared are a regression, but still a *change* that another
      pass might act on.

    Only an identical blocking gap set means the last pass learned nothing,
    which is the signal that iterating again would repeat the same work.
    This is the loop's structural guarantee that every iteration reduces
    uncertainty or terminates, and it holds even when memory keeps making the
    revised planner input differ across passes.
    """
    return _blocking_keys(previous) != _blocking_keys(current)


def _stable_text(value: object) -> str:
    """Return a deterministic, address-free string for ``value``.

    Memory errors embed exception text that may include object reprs, whose
    memory addresses would otherwise make the recorded history differ between
    two otherwise identical runs.
    """
    return _ADDRESS_PATTERN.sub("0xADDR", str(value).strip())


__all__ = [
    "ACTIONABLE_GAP_KINDS",
    "STOP_EXPLANATIONS",
    "ResearchLoop",
]
