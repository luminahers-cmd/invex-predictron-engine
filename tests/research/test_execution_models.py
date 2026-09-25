"""Tests for the execution value objects — Phase 8 Sprint 5.

Covers the execution policy, step records, warnings, errors, timing,
the dispatch projection, the derived statistic/metric/coverage/confidence
blocks, and the immutable ``ResearchExecution`` request.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.dispatcher import Dispatcher
from predictron_engine.research.exceptions import (
    InvalidExecutionInputError,
    InvalidExecutionModelError,
)
from predictron_engine.research.execution_models import (
    EMPTY_COVERAGE_RATIO,
    EMPTY_MEAN_CONFIDENCE,
    EXECUTION_SCHEMA_VERSION,
    RATIO_PRECISION,
    ConfidenceDistribution,
    DispatchOutcome,
    ExecutionCoverage,
    ExecutionError,
    ExecutionMetrics,
    ExecutionPolicy,
    ExecutionStatistics,
    ExecutionStep,
    ExecutionStepStatus,
    ExecutionTiming,
    ExecutionWarning,
    ResearchExecution,
    StepTiming,
)
from predictron_engine.research.execution_state import (
    ACTIVE_STATES,
    ExecutionState,
)
from predictron_engine.research.models import PlannerInput
from predictron_engine.research.planner import ResearchPlanner
from predictron_engine.research.validation_models import EvidenceConfidence


@pytest.fixture
def planner_input() -> PlannerInput:
    return PlannerInput(company_name="Acme")


def _step(**overrides: object) -> ExecutionStep:
    values: dict[str, object] = {
        "name": "planning",
        "state": ExecutionState.PLANNING,
        "status": ExecutionStepStatus.COMPLETED,
        "order": 0,
        "artifact_id": "plan_abc",
        "detail": "3 task(s)",
    }
    values.update(overrides)
    return ExecutionStep(**values)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Module constants
# ----------------------------------------------------------------------


def test_constants_are_exported() -> None:
    assert EXECUTION_SCHEMA_VERSION
    assert RATIO_PRECISION == 6
    assert EMPTY_COVERAGE_RATIO == 0.0
    assert EMPTY_MEAN_CONFIDENCE == 0.0


# ----------------------------------------------------------------------
# ExecutionPolicy
# ----------------------------------------------------------------------


def test_policy_defaults_allow_partial_and_stay_deterministic() -> None:
    policy = ExecutionPolicy()

    assert policy.run_validation
    assert not policy.fail_fast
    assert not policy.measure_duration
    assert policy.allow_partial
    assert policy.is_deterministic


def test_policy_fail_fast_disallows_partial() -> None:
    assert not ExecutionPolicy(fail_fast=True).allow_partial


def test_policy_measuring_duration_is_not_deterministic() -> None:
    assert not ExecutionPolicy(measure_duration=True).is_deterministic


def test_policy_round_trips_through_dict() -> None:
    policy = ExecutionPolicy(
        run_validation=False, fail_fast=True, measure_duration=True
    )

    assert ExecutionPolicy.from_dict(policy.to_dict()) == policy


def test_policy_dict_has_exactly_three_keys() -> None:
    assert set(ExecutionPolicy().to_dict()) == {
        "run_validation",
        "fail_fast",
        "measure_duration",
    }


@pytest.mark.parametrize(
    "name", ("run_validation", "fail_fast", "measure_duration")
)
def test_policy_rejects_non_bool(name: str) -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionPolicy(**{name: 1})  # type: ignore[arg-type]


def test_policy_rejects_non_bool_from_dict() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionPolicy.from_dict(
            {
                "run_validation": "yes",
                "fail_fast": False,
                "measure_duration": False,
            }
        )


# ----------------------------------------------------------------------
# ExecutionStepStatus
# ----------------------------------------------------------------------


def test_step_status_has_three_members() -> None:
    assert [status.value for status in ExecutionStepStatus] == [
        "completed",
        "skipped",
        "failed",
    ]


# ----------------------------------------------------------------------
# ExecutionStep
# ----------------------------------------------------------------------


def test_step_round_trips_through_dict() -> None:
    step = _step(warning_count=2, error_count=0)

    assert ExecutionStep.from_dict(step.to_dict()) == step


def test_step_status_predicates() -> None:
    assert _step().is_completed
    assert not _step().is_failed
    assert _step(status=ExecutionStepStatus.SKIPPED, detail="off").is_skipped
    assert _step(
        status=ExecutionStepStatus.FAILED, detail="boom", error_count=1
    ).is_failed


def test_step_strips_whitespace() -> None:
    step = _step(name="  planning  ", artifact_id="  plan_abc  ")

    assert step.name == "planning"
    assert step.artifact_id == "plan_abc"


def test_step_rejects_blank_name() -> None:
    with pytest.raises(InvalidExecutionModelError):
        _step(name="   ")


def test_step_rejects_non_state() -> None:
    with pytest.raises(InvalidExecutionModelError):
        _step(state="planning")


def test_step_rejects_non_status() -> None:
    with pytest.raises(InvalidExecutionModelError):
        _step(status="completed")


def test_step_rejects_negative_order() -> None:
    with pytest.raises(InvalidExecutionModelError):
        _step(order=-1)


def test_step_rejects_bool_order() -> None:
    with pytest.raises(InvalidExecutionModelError):
        _step(order=True)


def test_failed_step_must_record_an_error() -> None:
    with pytest.raises(InvalidExecutionModelError):
        _step(status=ExecutionStepStatus.FAILED, detail="boom", error_count=0)


def test_completed_step_needs_an_artifact_or_detail() -> None:
    with pytest.raises(InvalidExecutionModelError):
        _step(artifact_id="", detail="")


def test_skipped_step_needs_neither_artifact_nor_detail() -> None:
    step = _step(status=ExecutionStepStatus.SKIPPED, artifact_id="", detail="")

    assert step.is_skipped


def test_step_rejects_bad_order_from_dict() -> None:
    payload = _step().to_dict()
    payload["order"] = "0"

    with pytest.raises(InvalidExecutionModelError):
        ExecutionStep.from_dict(payload)


def test_step_rejects_bad_status_from_dict() -> None:
    payload = _step().to_dict()
    payload["status"] = "nope"

    with pytest.raises(InvalidExecutionModelError):
        ExecutionStep.from_dict(payload)


# ----------------------------------------------------------------------
# ExecutionWarning and ExecutionError
# ----------------------------------------------------------------------


def test_warning_round_trips_through_dict() -> None:
    warning = ExecutionWarning(
        code="unresolved_tasks", message="2 unresolved", step="dispatch"
    )

    assert ExecutionWarning.from_dict(warning.to_dict()) == warning


def test_error_round_trips_through_dict() -> None:
    error = ExecutionError(
        code="planning_failed",
        message="research planning failed: boom",
        step="planning",
        exception_type="ValueError",
    )

    assert ExecutionError.from_dict(error.to_dict()) == error


def test_warning_defaults_are_empty_strings() -> None:
    warning = ExecutionWarning(code="c", message="m")

    assert warning.step == ""
    assert warning.detail == ""


def test_error_defaults_are_empty_strings() -> None:
    error = ExecutionError(code="c", message="m")

    assert error.step == ""
    assert error.exception_type == ""
    assert error.detail == ""


@pytest.mark.parametrize(
    "cls", (ExecutionWarning, ExecutionError)
)
def test_warning_and_error_reject_blank_code(cls: type) -> None:
    with pytest.raises(InvalidExecutionModelError):
        cls(code="  ", message="m")


@pytest.mark.parametrize(
    "cls", (ExecutionWarning, ExecutionError)
)
def test_warning_and_error_reject_blank_message(cls: type) -> None:
    with pytest.raises(InvalidExecutionModelError):
        cls(code="c", message="  ")


def test_warning_rejects_missing_code_from_dict() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionWarning.from_dict({"message": "m"})


def test_error_rejects_non_dict_exception_type_from_dict() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionError.from_dict(
            {"code": "c", "message": "m", "exception_type": 1}
        )


# ----------------------------------------------------------------------
# StepTiming and ExecutionTiming
# ----------------------------------------------------------------------


def test_step_timing_round_trips_through_dict() -> None:
    timing = StepTiming(step="planning", duration_ms=1.5)

    assert StepTiming.from_dict(timing.to_dict()) == timing


def test_step_timing_rejects_negative_duration() -> None:
    with pytest.raises(InvalidExecutionModelError):
        StepTiming(step="planning", duration_ms=-1.0)


def test_step_timing_rejects_blank_step() -> None:
    with pytest.raises(InvalidExecutionModelError):
        StepTiming(step=" ", duration_ms=1.0)


def test_unmeasured_timing_is_all_defaults() -> None:
    timing = ExecutionTiming.unmeasured()

    assert not timing.measured
    assert timing.started_at == ""
    assert timing.finished_at == ""
    assert timing.total_duration_ms == 0.0
    assert timing.step_durations == ()


def test_timing_round_trips_through_dict() -> None:
    timing = ExecutionTiming(
        measured=True,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        total_duration_ms=1000.0,
        step_durations=(StepTiming(step="planning", duration_ms=2.5),),
    )

    assert ExecutionTiming.from_dict(timing.to_dict()) == timing


def test_duration_for_known_step() -> None:
    timing = ExecutionTiming(
        measured=True, step_durations=(StepTiming(step="planning", duration_ms=3.0),)
    )

    assert timing.duration_for("planning") == 3.0


def test_duration_for_unknown_step_is_zero() -> None:
    assert ExecutionTiming().duration_for("planning") == 0.0


def test_timing_rejects_non_step_timing_entries() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionTiming(measured=True, step_durations=("planning",))  # type: ignore[arg-type]


def test_timing_rejects_non_bool_measured() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionTiming(measured=1)  # type: ignore[arg-type]


def test_timing_rejects_bad_entry_from_dict() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionTiming.from_dict(
            {
                "measured": True,
                "started_at": "",
                "finished_at": "",
                "total_duration_ms": 1.0,
                "step_durations": ["planning"],
            }
        )


# ----------------------------------------------------------------------
# DispatchOutcome
# ----------------------------------------------------------------------


def test_dispatch_outcome_defaults_are_empty() -> None:
    outcome = DispatchOutcome()

    assert outcome.task_ids == ()
    assert outcome.collector_ids == ()
    assert outcome.unresolved == ()
    assert outcome.is_complete
    assert outcome.dispatched_count == 0
    assert outcome.unresolved_count == 0


def test_dispatch_outcome_counts_and_lookup() -> None:
    outcome = DispatchOutcome(
        task_ids=("t1", "t2"),
        collector_ids=("a", "b"),
        unresolved=("t3",),
    )

    assert outcome.dispatched_count == 2
    assert outcome.unresolved_count == 1
    assert not outcome.is_complete
    assert outcome.collector_for("t2") == "b"
    assert outcome.collector_for("nope") is None


def test_dispatch_outcome_distinct_collectors_are_sorted() -> None:
    outcome = DispatchOutcome(
        task_ids=("t1", "t2", "t3"),
        collector_ids=("z", "a", "z"),
    )

    assert outcome.distinct_collectors == ("a", "z")


def test_dispatch_outcome_round_trips_through_dict() -> None:
    outcome = DispatchOutcome(
        task_ids=("t1",), collector_ids=("a",), unresolved=("t2",)
    )

    assert DispatchOutcome.from_dict(outcome.to_dict()) == outcome


def test_dispatch_outcome_requires_aligned_id_lists() -> None:
    with pytest.raises(InvalidExecutionModelError):
        DispatchOutcome(task_ids=("t1", "t2"), collector_ids=("a",))


def test_dispatch_outcome_rejects_blank_ids() -> None:
    with pytest.raises(InvalidExecutionModelError):
        DispatchOutcome(task_ids=("  ",), collector_ids=("a",))


def test_dispatch_outcome_normalizes_lists_to_tuples() -> None:
    outcome = DispatchOutcome(
        task_ids=["t1"], collector_ids=["a"], unresolved=[]
    )

    assert outcome.task_ids == ("t1",)
    assert outcome.unresolved == ()


def test_dispatch_outcome_rejects_non_list_from_dict() -> None:
    with pytest.raises(InvalidExecutionModelError):
        DispatchOutcome.from_dict(
            {"task_ids": "t1", "collector_ids": ["a"], "unresolved": []}
        )


def test_dispatch_outcome_projects_a_dispatch_plan(
    planner: ResearchPlanner,
) -> None:
    tasks = planner.create_plan(PlannerInput(company_name="Acme")).tasks[:1]
    outcome = DispatchOutcome.from_dispatch(Dispatcher().dispatch(tasks))

    assert outcome.dispatched_count == 1
    assert outcome.is_complete


# ----------------------------------------------------------------------
# ExecutionStatistics
# ----------------------------------------------------------------------


def test_statistics_default_to_zero() -> None:
    statistics = ExecutionStatistics()

    assert statistics.topics_planned == 0
    assert statistics.evidence_collected == 0
    assert not statistics.is_complete


def test_statistics_topics_researched_aliases_planned() -> None:
    assert ExecutionStatistics(topics_planned=4).topics_researched == 4


def test_statistics_is_complete_when_all_tasks_succeeded() -> None:
    assert ExecutionStatistics(tasks_planned=2, tasks_executed=2).is_complete


def test_statistics_is_incomplete_when_a_task_failed() -> None:
    assert not ExecutionStatistics(
        tasks_planned=2, tasks_executed=1, tasks_failed=1
    ).is_complete


def test_statistics_is_incomplete_when_a_task_is_unresolved() -> None:
    assert not ExecutionStatistics(
        tasks_planned=2, tasks_executed=1, tasks_unresolved=1
    ).is_complete


def test_statistics_is_incomplete_with_no_tasks() -> None:
    assert not ExecutionStatistics().is_complete


def test_statistics_round_trips_through_dict() -> None:
    statistics = ExecutionStatistics(
        topics_planned=2,
        topics_covered=1,
        tasks_planned=3,
        tasks_dispatched=3,
        tasks_executed=2,
        tasks_failed=1,
        tasks_unresolved=1,
        sources_discovered=9,
        distinct_sources=4,
        collectors_executed=2,
        distinct_collectors=2,
        evidence_collected=6,
        evidence_validated=6,
    )

    assert ExecutionStatistics.from_dict(statistics.to_dict()) == statistics


def test_statistics_rejects_negative_counts() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionStatistics(topics_planned=-1)


def test_statistics_rejects_bool_counts() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionStatistics(topics_planned=True)


def test_statistics_rejects_bad_count_from_dict() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionStatistics.from_dict({"topics_planned": "2"})


# ----------------------------------------------------------------------
# ExecutionMetrics
# ----------------------------------------------------------------------


def test_metrics_default_to_zero() -> None:
    metrics = ExecutionMetrics()

    assert metrics.steps_total == 0
    assert metrics.coverage_ratio == EMPTY_COVERAGE_RATIO
    assert metrics.is_clean


def test_metrics_is_not_clean_with_a_warning() -> None:
    assert not ExecutionMetrics(warnings=1).is_clean


def test_metrics_is_not_clean_with_an_error() -> None:
    assert not ExecutionMetrics(errors=1).is_clean


def test_metrics_round_trips_through_dict() -> None:
    metrics = ExecutionMetrics(
        steps_total=5,
        steps_completed=4,
        steps_skipped=1,
        warnings=2,
        validation_issues=7,
        coverage_ratio=0.5,
    )

    assert ExecutionMetrics.from_dict(metrics.to_dict()) == metrics


def test_metrics_rounds_the_coverage_ratio() -> None:
    assert ExecutionMetrics(coverage_ratio=1 / 3).coverage_ratio == round(
        1 / 3, RATIO_PRECISION
    )


@pytest.mark.parametrize("value", (-0.1, 1.1))
def test_metrics_rejects_out_of_range_ratio(value: float) -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionMetrics(coverage_ratio=value)


def test_metrics_rejects_non_number_ratio() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionMetrics(coverage_ratio="1.0")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# ExecutionCoverage
# ----------------------------------------------------------------------


def test_coverage_defaults_are_empty() -> None:
    coverage = ExecutionCoverage()

    assert coverage.planned_count == 0
    assert coverage.covered_count == 0
    assert coverage.ratio == EMPTY_COVERAGE_RATIO
    assert not coverage.is_complete


def test_coverage_computes_ratio() -> None:
    coverage = ExecutionCoverage(
        planned_topics=("a", "b", "c", "d"),
        covered_topics=("a", "b"),
    )

    assert coverage.covered_count == 2
    assert coverage.ratio == 0.5
    assert coverage.uncovered_topics == ("c", "d")
    assert not coverage.is_complete


def test_coverage_is_complete_when_every_topic_is_covered() -> None:
    coverage = ExecutionCoverage(planned_topics=("a", "b"), covered_topics=("b", "a"))

    assert coverage.is_complete
    assert coverage.uncovered_topics == ()


def test_coverage_ignores_covered_topics_outside_the_plan() -> None:
    coverage = ExecutionCoverage(planned_topics=("a",), covered_topics=("z",))

    assert coverage.covered_count == 0


def test_coverage_preserves_planned_order() -> None:
    coverage = ExecutionCoverage(
        planned_topics=("c", "a", "b"), covered_topics=("a",)
    )

    assert coverage.uncovered_topics == ("c", "b")


def test_coverage_deduplicates_and_strips() -> None:
    coverage = ExecutionCoverage(
        planned_topics=(" a ", "a", "b"), covered_topics=("a", "a")
    )

    assert coverage.planned_topics == ("a", "b")
    assert coverage.covered_topics == ("a",)


def test_coverage_rejects_blank_topic() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionCoverage(planned_topics=("  ",))


def test_coverage_round_trips_through_dict() -> None:
    coverage = ExecutionCoverage(
        planned_topics=("a", "b"), covered_topics=("a",)
    )

    assert ExecutionCoverage.from_dict(coverage.to_dict()) == coverage


# ----------------------------------------------------------------------
# ConfidenceDistribution
# ----------------------------------------------------------------------


def test_empty_distribution_is_empty() -> None:
    distribution = ConfidenceDistribution.empty()

    assert distribution.total == 0
    assert distribution.is_empty
    assert distribution.mean_score == EMPTY_MEAN_CONFIDENCE


def test_distribution_counts_total() -> None:
    assert ConfidenceDistribution(high=2, medium=1, low=0, unknown=3).total == 6


@pytest.mark.parametrize(
    ("band", "expected"),
    (
        (EvidenceConfidence.HIGH, 2),
        (EvidenceConfidence.MEDIUM, 1),
        (EvidenceConfidence.LOW, 0),
        (EvidenceConfidence.UNKNOWN, 3),
    ),
)
def test_distribution_count_for_band(band: EvidenceConfidence, expected: int) -> None:
    distribution = ConfidenceDistribution(high=2, medium=1, low=0, unknown=3)

    assert distribution.count_for(band) == expected


def test_distribution_is_keyed_by_band_value() -> None:
    distribution = ConfidenceDistribution(high=1, medium=2)

    assert distribution.distribution() == {
        "high": 1,
        "medium": 2,
        "low": 0,
        "unknown": 0,
    }


def test_top_band_picks_the_strongest_band() -> None:
    assert ConfidenceDistribution(high=1, medium=5).top_band() is (
        EvidenceConfidence.MEDIUM
    )


def test_top_band_breaks_ties_towards_the_stronger_band() -> None:
    assert ConfidenceDistribution(high=2, medium=2).top_band() is (
        EvidenceConfidence.HIGH
    )


def test_top_band_of_an_empty_distribution_is_unknown() -> None:
    assert ConfidenceDistribution().top_band() is EvidenceConfidence.UNKNOWN


def test_distribution_rounds_the_mean_score() -> None:
    distribution = ConfidenceDistribution(mean_score=1 / 3)

    assert distribution.mean_score == round(1 / 3, RATIO_PRECISION)


def test_distribution_rejects_out_of_range_mean() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ConfidenceDistribution(mean_score=1.5)


def test_distribution_rejects_negative_counts() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ConfidenceDistribution(high=-1)


def test_distribution_round_trips_through_dict() -> None:
    distribution = ConfidenceDistribution(
        high=1, medium=2, low=3, unknown=4, mean_score=0.5
    )

    assert ConfidenceDistribution.from_dict(distribution.to_dict()) == distribution


def test_distribution_from_no_summaries_is_empty() -> None:
    distribution = ConfidenceDistribution.from_validations()

    assert distribution.is_empty
    assert distribution.mean_score == EMPTY_MEAN_CONFIDENCE


# ----------------------------------------------------------------------
# ResearchExecution
# ----------------------------------------------------------------------


def test_execution_create_derives_a_stable_id(planner_input: PlannerInput) -> None:
    first = ResearchExecution.create(planner_input)
    second = ResearchExecution.create(planner_input)

    assert first.execution_id == second.execution_id
    assert first.execution_id.startswith("exec_")


def test_execution_id_changes_with_the_company(planner_input: PlannerInput) -> None:
    other = ResearchExecution.create(PlannerInput(company_name="Globex"))

    assert other.execution_id != ResearchExecution.create(planner_input).execution_id


def test_execution_id_changes_with_the_policy(planner_input: PlannerInput) -> None:
    default = ResearchExecution.create(planner_input)

    assert (
        ResearchExecution.create(
            planner_input, policy=ExecutionPolicy(measure_duration=True)
        ).execution_id
        != default.execution_id
    )


def test_execution_id_is_independent_of_wall_clock(
    planner_input: PlannerInput,
) -> None:
    first = ResearchExecution.create(planner_input)
    second = ResearchExecution.create(planner_input)

    assert first.execution_id == second.execution_id


def test_execution_defaults_to_every_active_state(
    planner_input: PlannerInput,
) -> None:
    execution = ResearchExecution.create(planner_input)

    assert execution.requested_states == ACTIVE_STATES
    assert execution.visits(ExecutionState.VALIDATION)


def test_execution_omits_validation_when_disabled(
    planner_input: PlannerInput,
) -> None:
    execution = ResearchExecution.create(
        planner_input, policy=ExecutionPolicy(run_validation=False)
    )

    assert execution.requested_states == (
        ExecutionState.PLANNING,
        ExecutionState.DISCOVERY,
        ExecutionState.COLLECTION,
    )
    assert not execution.visits(ExecutionState.VALIDATION)


def test_execution_exposes_the_company_name(planner_input: PlannerInput) -> None:
    assert ResearchExecution.create(planner_input).company_name == "Acme"


def test_execution_round_trips_through_dict(planner_input: PlannerInput) -> None:
    execution = ResearchExecution.create(
        planner_input, policy=ExecutionPolicy(run_validation=False)
    )

    assert ResearchExecution.from_dict(execution.to_dict()) == execution


def test_execution_create_rejects_non_planner_input() -> None:
    with pytest.raises(InvalidExecutionInputError):
        ResearchExecution.create("Acme")  # type: ignore[arg-type]


def test_execution_rejects_non_planner_input_field() -> None:
    with pytest.raises(InvalidExecutionInputError):
        ResearchExecution(
            schema_version=EXECUTION_SCHEMA_VERSION,
            execution_id="exec_1",
            planner_input="Acme",  # type: ignore[arg-type]
        )


def test_execution_rejects_non_policy() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ResearchExecution(
            schema_version=EXECUTION_SCHEMA_VERSION,
            execution_id="exec_1",
            planner_input=PlannerInput(company_name="Acme"),
            policy={},  # type: ignore[arg-type]
        )


def test_execution_rejects_blank_execution_id() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ResearchExecution(
            schema_version=EXECUTION_SCHEMA_VERSION,
            execution_id="  ",
            planner_input=PlannerInput(company_name="Acme"),
        )


def test_execution_rejects_empty_requested_states() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ResearchExecution(
            schema_version=EXECUTION_SCHEMA_VERSION,
            execution_id="exec_1",
            planner_input=PlannerInput(company_name="Acme"),
            requested_states=(),
        )


def test_execution_requires_the_planning_state() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ResearchExecution(
            schema_version=EXECUTION_SCHEMA_VERSION,
            execution_id="exec_1",
            planner_input=PlannerInput(company_name="Acme"),
            requested_states=(ExecutionState.DISCOVERY,),
        )


def test_execution_rejects_out_of_order_states() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ResearchExecution(
            schema_version=EXECUTION_SCHEMA_VERSION,
            execution_id="exec_1",
            planner_input=PlannerInput(company_name="Acme"),
            requested_states=(
                ExecutionState.PLANNING,
                ExecutionState.VALIDATION,
                ExecutionState.DISCOVERY,
            ),
        )


def test_execution_rejects_duplicate_states() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ResearchExecution(
            schema_version=EXECUTION_SCHEMA_VERSION,
            execution_id="exec_1",
            planner_input=PlannerInput(company_name="Acme"),
            requested_states=(ExecutionState.PLANNING, ExecutionState.PLANNING),
        )


def test_execution_rejects_terminal_states_in_the_request() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ResearchExecution(
            schema_version=EXECUTION_SCHEMA_VERSION,
            execution_id="exec_1",
            planner_input=PlannerInput(company_name="Acme"),
            requested_states=(ExecutionState.PLANNING, ExecutionState.COMPLETED),
        )


def test_execution_rejects_unknown_state_from_dict(
    planner_input: PlannerInput,
) -> None:
    payload = ResearchExecution.create(planner_input).to_dict()
    payload["requested_states"] = ["planning", "nope"]

    with pytest.raises(InvalidExecutionModelError):
        ResearchExecution.from_dict(payload)


def test_execution_is_frozen(planner_input: PlannerInput) -> None:
    execution = ResearchExecution.create(planner_input)

    with pytest.raises(Exception):
        execution.execution_id = "exec_other"  # type: ignore[misc]
