"""Tests for the iterative research models — Phase 8 Sprint 7.

Focuses on the deterministic behaviour the loop depends on: enum ordering and
blocking rules, validation, the derived properties each record/analysis
exposes, history and metrics derivation, and JSON round-trips for every model.
"""

from __future__ import annotations

import json

import pytest

from predictron_engine.research.exceptions import InvalidIterationModelError
from predictron_engine.research.execution_state import ExecutionState
from predictron_engine.research.iteration_models import (
    EMPTY_PROGRESS_RATIO,
    ITERATION_SCHEMA_VERSION,
    FollowUpTask,
    GapKind,
    IterationPolicy,
    IterationRecord,
    IterationStatus,
    LoopDecision,
    LoopMetrics,
    MemoryContext,
    MemoryUpdate,
    ResearchIterationHistory,
    ResidualAnalysis,
    ResidualGap,
    TerminationReason,
    TopicInsight,
    TopicKnowledge,
    analysis_digest,
)
from predictron_engine.research.models import PlannerInput

# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------


def _gap(topic: str, kind: GapKind, **kwargs) -> ResidualGap:
    return ResidualGap(topic_id=topic, kind=kind, **kwargs)


def _analysis(iteration: int = 1, **kwargs) -> ResidualAnalysis:
    return ResidualAnalysis(analysis_id=f"a{iteration}", iteration=iteration, **kwargs)


def _record(iteration: int = 1, **kwargs) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        planner_input=PlannerInput(company_name="Acme"),
        **kwargs,
    )


def _history(**kwargs) -> ResearchIterationHistory:
    return ResearchIterationHistory(**kwargs)


# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------


def test_gap_kind_order_is_the_urgency_order():
    assert [k.value for k in GapKind] == [
        "missing",
        "unresolved",
        "failed_validation",
        "conflicting",
        "low_confidence",
        "insufficient",
        "stale",
    ]
    assert GapKind.MISSING.order_value() < GapKind.STALE.order_value()


def test_only_stale_is_non_blocking():
    assert not GapKind.STALE.is_blocking
    assert all(k.is_blocking for k in GapKind if k is not GapKind.STALE)


def test_topic_knowledge_satisfied_is_known_only():
    assert TopicKnowledge.KNOWN.is_satisfied
    assert not any(
        k.is_satisfied for k in TopicKnowledge if k is not TopicKnowledge.KNOWN
    )


def test_iteration_status_from_state():
    assert IterationStatus.for_state(ExecutionState.COMPLETED) is IterationStatus.SUCCEEDED
    assert IterationStatus.for_state(ExecutionState.FAILED) is IterationStatus.FAILED


def test_termination_reason_has_continue_and_stop_values():
    assert TerminationReason.ITERATION_REQUIRED.value == "iteration_required"
    assert TerminationReason.RESEARCH_COMPLETE.value == "research_complete"


# ----------------------------------------------------------------------
# Policy and decision
# ----------------------------------------------------------------------


def test_policy_defaults_are_single_pass_and_iterative_opt_in():
    policy = IterationPolicy()
    assert policy.max_iterations == 1
    assert not policy.is_iterative
    assert policy.is_deterministic


def test_policy_allows_iteration_up_to_budget():
    policy = IterationPolicy(max_iterations=3)
    assert policy.allows_iteration(1)
    assert policy.allows_iteration(2)
    assert not policy.allows_iteration(3)


def test_policy_rejects_negative_budget():
    with pytest.raises(InvalidIterationModelError):
        IterationPolicy(max_iterations=-1)


def test_policy_rejects_out_of_range_confidence():
    with pytest.raises(InvalidIterationModelError):
        IterationPolicy(minimum_confidence=1.5)


def test_loop_decision_stop_and_continue():
    stopped = LoopDecision.stop(TerminationReason.CONVERGED, "done")
    assert stopped.should_stop and stopped.is_terminal
    assert stopped.reason is TerminationReason.CONVERGED

    cont = LoopDecision.continue_loop("more to do")
    assert not cont.should_stop and not cont.is_terminal
    assert cont.reason is TerminationReason.ITERATION_REQUIRED


# ----------------------------------------------------------------------
# Gaps and analysis
# ----------------------------------------------------------------------


def test_gap_order_key_sorts_by_kind_then_topic():
    gaps = [
        _gap("b", GapKind.STALE),
        _gap("a", GapKind.MISSING),
        _gap("z", GapKind.MISSING),
    ]
    ordered = sorted(gaps, key=lambda g: g.order_key)
    assert [(g.kind, g.topic_id) for g in ordered] == [
        (GapKind.MISSING, "a"),
        (GapKind.MISSING, "z"),
        (GapKind.STALE, "b"),
    ]


def test_gap_rejects_blank_topic_and_bad_confidence():
    with pytest.raises(InvalidIterationModelError):
        _gap("   ", GapKind.MISSING)
    with pytest.raises(InvalidIterationModelError):
        _gap("team", GapKind.MISSING, confidence=2.0)


def test_analysis_splits_blocking_and_stale_gaps():
    analysis = _analysis(
        gaps=(
            _gap("a", GapKind.MISSING),
            _gap("b", GapKind.STALE),
            _gap("c", GapKind.LOW_CONFIDENCE),
        )
    )
    assert {g.topic_id for g in analysis.blocking_gaps} == {"a", "c"}
    assert {g.topic_id for g in analysis.stale_gaps} == {"b"}
    assert analysis.has_blocking_gaps
    assert analysis.has_actionable_gaps


def test_analysis_unresolved_is_blocking_but_not_actionable():
    analysis = _analysis(gaps=(_gap("a", GapKind.UNRESOLVED),))
    assert analysis.has_blocking_gaps
    assert not analysis.has_actionable_gaps
    assert analysis.actionable_gaps == ()
    assert _gap("a", GapKind.UNRESOLVED).is_actionable is False
    assert _gap("a", GapKind.MISSING).is_actionable is True
    assert _gap("a", GapKind.STALE).is_actionable is False


def test_analysis_gap_counts_cover_every_kind_in_enum_order():
    analysis = _analysis(
        gaps=(_gap("a", GapKind.MISSING), _gap("b", GapKind.MISSING), _gap("c", GapKind.STALE))
    )
    counts = analysis.gap_counts()
    assert list(counts) == [kind.value for kind in GapKind]
    assert counts["missing"] == 2
    assert counts["stale"] == 1
    assert counts["low_confidence"] == 0


def test_analysis_has_and_gaps_of():
    analysis = _analysis(gaps=(_gap("a", GapKind.MISSING), _gap("b", GapKind.CONFLICTING)))
    assert analysis.has(GapKind.MISSING)
    assert not analysis.has(GapKind.STALE)
    assert [g.topic_id for g in analysis.gaps_of(GapKind.CONFLICTING)] == ["b"]


def test_analysis_thresholds_and_completeness():
    incomplete = _analysis(
        gaps=(_gap("a", GapKind.MISSING),),
        mean_confidence=0.9,
        minimum_confidence=0.0,
        coverage_ratio=1.0,
        minimum_coverage=1.0,
    )
    assert incomplete.thresholds_met
    assert not incomplete.is_complete

    complete = _analysis(
        mean_confidence=0.9,
        minimum_confidence=0.0,
        coverage_ratio=1.0,
        minimum_coverage=1.0,
    )
    assert complete.is_complete
    assert complete.is_empty


def test_analysis_rejects_negative_iteration():
    with pytest.raises(InvalidIterationModelError):
        ResidualAnalysis(analysis_id="a", iteration=-1)


# ----------------------------------------------------------------------
# Memory context
# ----------------------------------------------------------------------


def _insight(topic: str, knowledge: TopicKnowledge, **kwargs) -> TopicInsight:
    return TopicInsight(topic_id=topic, knowledge=knowledge, **kwargs)


def test_topic_insight_staleness_and_research_need():
    stale = _insight(
        "news", TopicKnowledge.KNOWN, age_days=10, freshness_requirement_days=5
    )
    fresh = _insight(
        "team", TopicKnowledge.KNOWN, age_days=1, freshness_requirement_days=5
    )
    assert stale.is_stale
    assert not fresh.is_stale
    assert not stale.needs_research

    missing = _insight("pricing", TopicKnowledge.MISSING)
    assert missing.needs_research


def test_memory_context_groups_topics():
    context = MemoryContext(
        insights=(
            _insight("a", TopicKnowledge.KNOWN),
            _insight("b", TopicKnowledge.PARTIAL),
            _insight("c", TopicKnowledge.MISSING),
            _insight("d", TopicKnowledge.CONFLICTING),
            _insight("e", TopicKnowledge.KNOWN, age_days=9, freshness_requirement_days=1),
        ),
        memory_key="acme",
        company_name="Acme",
    )
    assert context.topic_ids == ("a", "b", "c", "d", "e")
    assert context.known_topics == ("a", "e")
    assert context.partial_topics == ("b",)
    assert context.missing_topics == ("c",)
    assert context.conflicting_topics == ("d",)
    assert context.stale_topics == ("e",)
    assert context.satisfied_topics == ("a", "e")
    assert context.reusable_topics == ("a", "e", "b")
    assert context.topics_reused == 3
    assert context.insight_for("b").knowledge is TopicKnowledge.PARTIAL
    assert context.insight_for("nope") is None


def test_memory_context_is_empty():
    assert MemoryContext().is_empty()
    assert not MemoryContext(insights=(_insight("a", TopicKnowledge.KNOWN),)).is_empty()


# ----------------------------------------------------------------------
# Updates, follow-ups, records
# ----------------------------------------------------------------------


def test_memory_update_changed_requires_a_real_change():
    assert not MemoryUpdate().changed
    # A write that applied but stored nothing new did not change memory.
    assert not MemoryUpdate(applied=True).changed
    assert MemoryUpdate(evidence_stored=1).changed
    assert MemoryUpdate(entities_created=1).changed
    assert MemoryUpdate(entities_updated=1).changed
    assert MemoryUpdate(relationships_created=1).changed
    assert MemoryUpdate(history_entries=1).changed


def test_follow_up_task_blocking_and_order():
    follow = FollowUpTask(task_id="f1", topic_id="team", kind=GapKind.MISSING)
    assert follow.is_blocking
    stale = FollowUpTask(task_id="f2", topic_id="news", kind=GapKind.STALE)
    assert not stale.is_blocking
    assert follow.order_key[0] < stale.order_key[0]


def test_iteration_record_derived_values():
    record = _record(
        planned_tasks=("t1", "t2"),
        planned_task_topics=("a", "b"),
        completed_tasks=("t1",),
        unresolved_tasks=("t2",),
        new_evidence_ids=("e1", "e2"),
        topics_completed=("a",),
        topics_incomplete=("b",),
        residual_gaps=(_gap("b", GapKind.MISSING),),
        state=ExecutionState.COMPLETED,
    )
    assert record.status is IterationStatus.SUCCEEDED
    assert record.is_successful
    assert record.gap_count == 1
    assert record.new_evidence_count == 2
    assert record.topic_count == 2
    assert record.planned_topic_count == 2
    assert not record.coverage_complete
    assert record.step_names() == ()


def test_iteration_record_failed_state():
    record = _record(state=ExecutionState.FAILED)
    assert record.status is IterationStatus.FAILED
    assert not record.is_successful


# ----------------------------------------------------------------------
# History and metrics
# ----------------------------------------------------------------------


def test_history_accessors_and_progressions():
    first = _record(
        1,
        planned_task_topics=("a", "b"),
        topics_completed=("a",),
        topics_incomplete=("b",),
        confidence=0.2,
        coverage_ratio=0.5,
        evidence_count=2,
        residual_gaps=(_gap("b", GapKind.MISSING),),
    )
    second = _record(
        2,
        planned_task_topics=("b",),
        topics_completed=("b",),
        confidence=0.8,
        coverage_ratio=1.0,
        evidence_count=3,
        residual_gaps=(_gap("b", GapKind.MISSING),),
    )
    history = _history(iterations=(first, second))

    assert history.count == 2
    assert history.executed_iterations == 2
    assert history.first is first
    assert history.last is second
    assert history.record_for(2) is second
    assert history.record_for(9) is None
    assert history.confidence_progression == (0.2, 0.8)
    assert history.coverage_progression == (0.5, 1.0)
    assert history.evidence_progression == (2, 3)
    assert history.confidence_improved
    assert history.topics_completed == ("a", "b")
    assert history.repeated_topics == ("b",)
    assert history.is_partial


def test_history_budget_and_convergence_flags():
    records = (_record(1), _record(2))
    spent = _history(
        iterations=records,
        termination=TerminationReason.MAX_ITERATIONS_REACHED,
        policy=IterationPolicy(max_iterations=2),
    )
    assert spent.exhausted_budget

    converged = _history(
        iterations=(_record(1),),
        termination=TerminationReason.CONVERGED,
        policy=IterationPolicy(max_iterations=5),
    )
    assert converged.converged
    assert not converged.is_complete

    complete = _history(
        iterations=(_record(1),),
        termination=TerminationReason.RESEARCH_COMPLETE,
        policy=IterationPolicy(max_iterations=5),
    )
    assert complete.is_complete
    assert complete.converged


def test_empty_history_defaults():
    history = _history()
    assert history.is_empty
    assert history.count == 0
    assert history.first is None
    assert history.last is None
    assert history.current_termination is TerminationReason.ITERATION_REQUIRED


def test_metrics_derive_from_history():
    record = _record(
        planned_tasks=("t1", "t2"),
        completed_tasks=("t1",),
        unresolved_tasks=("t2",),
        planned_task_topics=("a", "b"),
        topics_completed=("a",),
        confidence=0.5,
        coverage_ratio=0.5,
        evidence_count=4,
        new_evidence_ids=("e1", "e2", "e3"),
        memory_update=MemoryUpdate(applied=True),
        residual_gaps=(_gap("b", GapKind.MISSING),),
    )
    history = _history(
        iterations=(record,),
        termination=TerminationReason.MAX_ITERATIONS_REACHED,
        policy=IterationPolicy(max_iterations=1),
    )
    metrics = LoopMetrics.from_history(history)

    assert metrics.iterations == 1
    assert metrics.tasks_planned == 2
    assert metrics.tasks_completed == 1
    assert metrics.tasks_unresolved == 1
    assert metrics.evidence_collected == 4
    assert metrics.new_evidence == 3
    assert metrics.redundant_evidence == 1
    assert metrics.residual_gaps_total == 1
    assert metrics.residual_gaps_final == 1
    assert metrics.topics_planned == 2
    assert metrics.topics_completed == 1
    assert metrics.memory_writes == 1
    assert metrics.planning_efficiency == pytest.approx(0.5)
    assert metrics.budget_used == pytest.approx(1.0)
    assert metrics.is_single_iteration
    assert not metrics.is_converged
    assert metrics.wasted_evidence_ratio == pytest.approx(0.25)


def test_metrics_zero_denominators_are_safe():
    metrics = LoopMetrics.from_history(_history())
    assert metrics.planning_efficiency == EMPTY_PROGRESS_RATIO
    assert metrics.wasted_evidence_ratio == EMPTY_PROGRESS_RATIO


def test_metrics_from_history_rejects_wrong_type():
    with pytest.raises(InvalidIterationModelError):
        LoopMetrics.from_history("history")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------


def _samples() -> list[object]:
    gap = _gap("team", GapKind.MISSING, reason="none", iteration=1, task_id="t1")
    analysis = _analysis(
        gaps=(gap,),
        planned_topics=("team",),
        mean_confidence=0.3,
        coverage_ratio=0.5,
    )
    record = _record(
        planned_tasks=("t1",),
        planned_task_topics=("team",),
        residual_gaps=(gap,),
        memory_update=MemoryUpdate(applied=True, new_evidence_ids=("e1",)),
    )
    return [
        IterationPolicy(max_iterations=3),
        LoopDecision.stop(TerminationReason.CONVERGED, "done"),
        TopicInsight("team", TopicKnowledge.PARTIAL, evidence_count=1, mean_confidence=0.4),
        MemoryContext(
            insights=(TopicInsight("team", TopicKnowledge.PARTIAL),),
            memory_key="acme",
            company_name="Acme",
        ),
        gap,
        analysis,
        FollowUpTask(task_id="f1", topic_id="team", kind=GapKind.MISSING),
        MemoryUpdate(applied=True, reports_written=1, new_evidence_ids=("e1",)),
        record,
        _history(
            iterations=(record,),
            termination=TerminationReason.CONVERGED,
            policy=IterationPolicy(max_iterations=3),
            analysis=analysis,
        ),
        LoopMetrics.from_history(_history(iterations=(record,))),
    ]


@pytest.mark.parametrize("sample", _samples(), ids=lambda s: type(s).__name__)
def test_every_model_round_trips_through_json(sample):
    payload = sample.to_dict()
    restored = type(sample).from_dict(json.loads(json.dumps(payload)))
    assert restored.to_dict() == payload


def test_history_dict_carries_the_schema_version():
    assert _history().to_dict()["schema_version"] == ITERATION_SCHEMA_VERSION


def test_analysis_digest_is_deterministic_and_key_order_insensitive():
    first = analysis_digest({"a": 1, "b": [1, 2]})
    second = analysis_digest({"b": [1, 2], "a": 1})
    assert first == second
    assert first.startswith("analysis_")
    assert first != analysis_digest({"a": 1, "b": [1, 3]})
