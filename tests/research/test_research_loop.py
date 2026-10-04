"""Tests for the deterministic research loop — Phase 8 Sprint 7.

Covers the decision matrix, the budget, the convergence guarantees (changed
planner input, no-progress gap sets, and structural ``UNRESOLVED`` gaps),
history/metrics serialization, memory integration, replay determinism, and
input rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from predictron_engine.research import (
    IterationPolicy,
    PlannerInput,
    ResearchLoop,
    ResearchMemory,
    ResearchOrchestrator,
)
from predictron_engine.research.exceptions import InvalidIterationInputError
from predictron_engine.research.execution_report import ResearchExecutionResult
from predictron_engine.research.execution_state import ExecutionState
from predictron_engine.research.iteration_models import (
    GapKind,
    IterationRecord,
    LoopDecision,
    LoopMetrics,
    ResearchIterationHistory,
    ResidualAnalysis,
    ResidualGap,
    TerminationReason,
)
from predictron_engine.research.research_loop import (
    ACTIONABLE_GAP_KINDS,
    STOP_EXPLANATIONS,
)

# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------

COMPANY = "Acme"


def _gap(
    topic: str,
    kind: GapKind,
    *,
    iteration: int = 1,
    task_id: str = "",
    evidence_count: int = 0,
    confidence: float = 0.0,
    conflicts: int = 0,
) -> ResidualGap:
    return ResidualGap(
        topic_id=topic,
        kind=kind,
        reason=f"{kind.value}:{topic}",
        iteration=iteration,
        task_id=task_id,
        evidence_count=evidence_count,
        confidence=confidence,
        conflicts=conflicts,
    )


def _analysis(
    *,
    iteration: int = 1,
    gaps: tuple[ResidualGap, ...] = (),
    minimum_confidence: float = 0.0,
    minimum_coverage: float = 1.0,
    mean_confidence: float = 0.0,
    coverage_ratio: float = 0.0,
) -> ResidualAnalysis:
    return ResidualAnalysis(
        analysis_id=f"analysis-{iteration}",
        iteration=iteration,
        gaps=gaps,
        minimum_confidence=minimum_confidence,
        minimum_coverage=minimum_coverage,
        mean_confidence=mean_confidence,
        coverage_ratio=coverage_ratio,
    )


@dataclass
class _ResultStub:
    """Stand-in for a pipeline result: ``_decide`` reads only these two."""

    state: ExecutionState = ExecutionState.COMPLETED
    errors: tuple[object, ...] = ()


def _iterative(max_iterations: int) -> ResearchLoop:
    return ResearchLoop(policy=IterationPolicy(max_iterations=max_iterations))


def _decide(
    loop: ResearchLoop,
    *,
    iteration: int,
    analysis: ResidualAnalysis,
    state: ExecutionState = ExecutionState.COMPLETED,
    errors: tuple[object, ...] = (),
) -> LoopDecision:
    return loop._decide(
        iteration=iteration,
        analysis=analysis,
        result=_ResultStub(state=state, errors=errors),  # type: ignore[arg-type]
    )


class _GapEngineStub:
    """Gap engine returning scripted analyses and scripted input revisions."""

    def __init__(
        self,
        analyses: list[ResidualAnalysis],
        *,
        change_input: bool = True,
    ) -> None:
        self._analyses = analyses
        self._change_input = change_input
        self.calls = 0

    def analyze(self, *, iteration, **kwargs) -> ResidualAnalysis:  # noqa: ANN003
        index = min(self.calls, len(self._analyses) - 1)
        self.calls += 1
        template = self._analyses[index]
        return replace(template, analysis_id=f"analysis-{iteration}", iteration=iteration)

    def revise_planner_input(self, planner_input, analysis, *, context=None):  # noqa: ANN001
        if not self._change_input:
            return planner_input
        return replace(planner_input, description=planner_input.description + ".")


@pytest.fixture
def orchestrator() -> ResearchOrchestrator:
    return ResearchOrchestrator()


@pytest.fixture
def baseline(orchestrator: ResearchOrchestrator) -> ResearchExecutionResult:
    return orchestrator.execute(PlannerInput(company_name=COMPANY))


# ----------------------------------------------------------------------
# Construction and input validation
# ----------------------------------------------------------------------


def test_default_policy_is_single_pass():
    assert IterationPolicy().max_iterations == 1
    assert not IterationPolicy().is_iterative


def test_default_policy_single_pass_matches_sprint_five():
    history = ResearchLoop().run_for_company(COMPANY)
    assert history.count == 1
    # The placeholder collectors leave only structural (UNRESOLVED) gaps,
    # so the single pass converges rather than reporting a spent budget.
    assert history.termination is TerminationReason.CONVERGED


def test_invalid_policy_rejected():
    with pytest.raises(InvalidIterationInputError):
        ResearchLoop(policy="five")  # type: ignore[arg-type]


def test_invalid_planner_input_rejected():
    with pytest.raises(InvalidIterationInputError):
        ResearchLoop().run("Acme")  # type: ignore[arg-type]


def test_orchestrator_without_execute_rejected():
    with pytest.raises(InvalidIterationInputError):
        ResearchLoop(orchestrator=object()).run(PlannerInput(company_name=COMPANY))


def test_orchestrator_returning_wrong_type_rejected():
    class _Bad:
        def execute(self, planner_input):  # noqa: ANN001, ANN201
            return "not a result"

    with pytest.raises(InvalidIterationInputError):
        ResearchLoop(orchestrator=_Bad()).run(PlannerInput(company_name=COMPANY))


def test_gap_engine_defaults_to_policy_thresholds():
    loop = ResearchLoop(policy=IterationPolicy(minimum_confidence=0.4))
    assert loop.policy.minimum_confidence == 0.4
    # The engine is built from the same policy so thresholds cannot drift.
    assert loop.policy is loop._gap_engine.policy


# ----------------------------------------------------------------------
# The decision matrix
# ----------------------------------------------------------------------


def test_decide_execution_failed_on_failed_state():
    decision = _decide(
        ResearchLoop(),
        iteration=1,
        analysis=_analysis(),
        state=ExecutionState.FAILED,
    )
    assert decision.reason is TerminationReason.EXECUTION_FAILED


def test_decide_execution_failed_on_errors():
    decision = _decide(
        ResearchLoop(),
        iteration=1,
        analysis=_analysis(),
        errors=(object(),),
    )
    assert decision.reason is TerminationReason.EXECUTION_FAILED


def test_decide_complete_stops_before_budget():
    # Budget already spent, yet completion still wins.
    loop = ResearchLoop(policy=IterationPolicy(max_iterations=1))
    analysis = _analysis(minimum_coverage=0.0, mean_confidence=0.9)
    decision = _decide(loop, iteration=1, analysis=analysis)
    assert decision.reason is TerminationReason.RESEARCH_COMPLETE


def test_decide_complete_but_completion_ignored_falls_to_partial():
    loop = ResearchLoop(
        policy=IterationPolicy(max_iterations=1, stop_when_complete=False)
    )
    analysis = _analysis(minimum_coverage=0.0, mean_confidence=0.9)
    decision = _decide(loop, iteration=1, analysis=analysis)
    assert decision.reason is TerminationReason.PARTIAL_COMPLETION_ALLOWED


def test_decide_partial_completion_allowed():
    loop = ResearchLoop(
        policy=IterationPolicy(max_iterations=5, allow_partial_completion=True)
    )
    # Thresholds met, but a blocking gap remains.
    analysis = _analysis(
        gaps=(_gap("team", GapKind.MISSING),),
        minimum_coverage=0.0,
        mean_confidence=0.9,
        coverage_ratio=1.0,
    )
    decision = _decide(loop, iteration=1, analysis=analysis)
    assert decision.reason is TerminationReason.PARTIAL_COMPLETION_ALLOWED


def test_decide_partial_not_allowed_with_no_actionable_gaps_converges():
    loop = ResearchLoop(
        policy=IterationPolicy(
            max_iterations=5,
            stop_when_complete=False,
            allow_partial_completion=False,
        )
    )
    analysis = _analysis(minimum_coverage=0.0, mean_confidence=0.9)
    decision = _decide(loop, iteration=1, analysis=analysis)
    assert decision.reason is TerminationReason.CONVERGED


def test_decide_only_stale_gaps_converges():
    analysis = _analysis(gaps=(_gap("team", GapKind.STALE),))
    decision = _decide(_iterative(5), iteration=1, analysis=analysis)
    assert decision.reason is TerminationReason.CONVERGED


def test_decide_only_unresolved_gaps_converges():
    # A missing collector cannot be fixed by another pass, so a run whose only
    # blocking gaps are UNRESOLVED converges instead of burning the budget.
    analysis = _analysis(
        gaps=(
            _gap("business_model", GapKind.UNRESOLVED),
            _gap("team", GapKind.UNRESOLVED),
        )
    )
    decision = _decide(_iterative(5), iteration=1, analysis=analysis)
    assert decision.reason is TerminationReason.CONVERGED


def test_decide_actionable_gap_continues():
    analysis = _analysis(gaps=(_gap("team", GapKind.MISSING),))
    decision = _decide(_iterative(5), iteration=1, analysis=analysis)
    assert decision.proceed
    assert decision.reason is TerminationReason.ITERATION_REQUIRED


def test_decide_actionable_gap_out_of_budget():
    analysis = _analysis(gaps=(_gap("team", GapKind.MISSING),))
    decision = _decide(_iterative(1), iteration=1, analysis=analysis)
    assert decision.reason is TerminationReason.MAX_ITERATIONS_REACHED


def test_decide_unresolved_plus_actionable_continues():
    analysis = _analysis(
        gaps=(
            _gap("business_model", GapKind.UNRESOLVED),
            _gap("team", GapKind.MISSING),
        )
    )
    decision = _decide(_iterative(5), iteration=1, analysis=analysis)
    assert decision.proceed


def test_actionable_gap_kinds_exclude_structural_and_stale():
    assert GapKind.UNRESOLVED not in ACTIONABLE_GAP_KINDS
    assert GapKind.STALE not in ACTIONABLE_GAP_KINDS
    assert GapKind.MISSING in ACTIONABLE_GAP_KINDS
    assert GapKind.LOW_CONFIDENCE in ACTIONABLE_GAP_KINDS


def test_stop_explanations_cover_every_stopping_reason():
    stopping = set(TerminationReason) - {TerminationReason.ITERATION_REQUIRED}
    assert set(STOP_EXPLANATIONS) == stopping
    assert TerminationReason.ITERATION_REQUIRED not in STOP_EXPLANATIONS


# ----------------------------------------------------------------------
# Run-level behaviour
# ----------------------------------------------------------------------


def test_real_pipeline_converges_without_repeating_work(orchestrator):
    loop = ResearchLoop(orchestrator=orchestrator, policy=IterationPolicy(max_iterations=6))
    history = loop.run_for_company(COMPANY, description="Fintech")

    assert history.termination is TerminationReason.CONVERGED
    assert history.count == 1
    assert history.repeated_topics == ()
    gaps = history.analysis.blocking_gaps
    assert gaps and all(gap.kind is GapKind.UNRESOLVED for gap in gaps)


def test_run_exhausts_budget_when_gaps_keep_changing(orchestrator):
    # Each pass closes nothing but raises a *new* gap, so the loop keeps
    # finding something actionable until the budget runs out.
    engine = _GapEngineStub(
        [
            _analysis(gaps=(_gap("t0", GapKind.MISSING),)),
            _analysis(
                gaps=(
                    _gap("t0", GapKind.MISSING),
                    _gap("t1", GapKind.MISSING),
                )
            ),
            _analysis(
                gaps=(
                    _gap("t0", GapKind.MISSING),
                    _gap("t1", GapKind.MISSING),
                    _gap("t2", GapKind.MISSING),
                )
            ),
        ],
        change_input=True,
    )
    loop = ResearchLoop(
        orchestrator=orchestrator,
        policy=IterationPolicy(max_iterations=3),
        gap_engine=engine,
    )
    history = loop.run(PlannerInput(company_name=COMPANY))

    assert history.termination is TerminationReason.MAX_ITERATIONS_REACHED
    assert history.count == 3
    assert history.exhausted_budget


def test_run_converges_when_revised_input_unchanged(orchestrator):
    engine = _GapEngineStub(
        [_analysis(gaps=(_gap("team", GapKind.MISSING),))],
        change_input=False,
    )
    loop = ResearchLoop(
        orchestrator=orchestrator,
        policy=IterationPolicy(max_iterations=5),
        gap_engine=engine,
    )
    history = loop.run(PlannerInput(company_name=COMPANY))

    assert history.termination is TerminationReason.CONVERGED
    assert history.count == 1


def test_run_converges_when_gaps_do_not_change(orchestrator):
    # The revised input differs every pass (memory can do this), but the
    # blocking gap set never changes, so the loop converges after the second
    # pass instead of spending the whole budget.
    engine = _GapEngineStub(
        [
            _analysis(
                gaps=(
                    _gap("team", GapKind.UNRESOLVED, task_id="t1"),
                    _gap("team", GapKind.MISSING, task_id="t2"),
                )
            )
        ],
        change_input=True,
    )
    loop = ResearchLoop(
        orchestrator=orchestrator,
        policy=IterationPolicy(max_iterations=5),
        gap_engine=engine,
    )
    history = loop.run(PlannerInput(company_name=COMPANY))

    assert history.termination is TerminationReason.CONVERGED
    assert history.count == 2


def test_run_returns_history_even_when_memory_write_fails(orchestrator):
    class _FailingMemory(ResearchMemory):
        def add_report(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            raise RuntimeError(f"boom at {object()!r}")

    def _run():
        loop = ResearchLoop(
            orchestrator=orchestrator,
            policy=IterationPolicy(max_iterations=1),
            memory=_FailingMemory(),
        )
        return loop.run_for_company(COMPANY).iterations[0].memory_update

    first = _run()
    second = _run()

    assert first.applied is False
    assert first.error
    # The recorded error is address-normalized, so two failing runs agree.
    assert first.error == second.error


def test_memory_is_written_and_reused(orchestrator):
    memory = ResearchMemory()
    loop = ResearchLoop(
        orchestrator=orchestrator,
        policy=IterationPolicy(max_iterations=2),
        memory=memory,
    )
    history = loop.run_for_company(COMPANY)

    assert history.iterations[0].memory_update.applied
    assert history.iterations[0].memory_update.reports_written == 1
    assert history.memory_context is not None
    assert history.termination is TerminationReason.CONVERGED


def test_metrics_are_exposed(orchestrator):
    loop = ResearchLoop(orchestrator=orchestrator, policy=IterationPolicy(max_iterations=5))
    history = loop.run_for_company(COMPANY)
    metrics = loop.metrics(history)

    assert isinstance(metrics, LoopMetrics)
    assert metrics.iterations == history.count
    assert metrics.termination is history.termination


# ----------------------------------------------------------------------
# Records, history, and determinism
# ----------------------------------------------------------------------


def test_first_record_captures_the_pass(orchestrator):
    loop = ResearchLoop(orchestrator=orchestrator, policy=IterationPolicy(max_iterations=1))
    record = loop.run_for_company(COMPANY, description="Fintech").first

    assert isinstance(record, IterationRecord)
    assert record.iteration == 1
    assert record.planned_tasks
    assert record.evidence_count > 0
    assert record.confidence > 0.0
    assert 0.0 <= record.coverage_ratio <= 1.0
    assert record.gap_count == len(record.residual_gaps)


def test_run_is_deterministic(orchestrator):
    def _once():
        loop = ResearchLoop(
            orchestrator=ResearchOrchestrator(),
            policy=IterationPolicy(max_iterations=4),
        )
        return loop.run_for_company(COMPANY, description="Fintech").to_dict()

    assert _once() == _once()


def test_history_serialization_round_trip(orchestrator):
    loop = ResearchLoop(orchestrator=orchestrator, policy=IterationPolicy(max_iterations=4))
    history = loop.run_for_company(COMPANY, description="Fintech")

    restored = ResearchIterationHistory.from_dict(history.to_dict())
    assert restored.to_dict() == history.to_dict()
    assert restored.termination is history.termination
    assert restored.count == history.count


def test_run_for_company_matches_run(orchestrator):
    loop = ResearchLoop(orchestrator=orchestrator, policy=IterationPolicy(max_iterations=4))
    direct = loop.run(PlannerInput(company_name=COMPANY, description="Fintech"))
    convenience = loop.run_for_company(COMPANY, description="Fintech")

    assert direct.to_dict() == convenience.to_dict()
