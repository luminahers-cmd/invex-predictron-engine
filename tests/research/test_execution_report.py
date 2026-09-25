"""Tests for the execution report layer — Phase 8 Sprint 5.

Covers ``ValidationOutcome`` projection, the bundled ``ExecutionSummary``,
the content-addressed ``ResearchReport``, the pure
``build_research_report`` derivation, and the
``ResearchExecutionResult`` value object.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.dispatcher import Dispatcher
from predictron_engine.research.exceptions import InvalidExecutionModelError
from predictron_engine.research.execution import EvidenceCollectionEngine
from predictron_engine.research.execution_models import (
    ConfidenceDistribution,
    DispatchOutcome,
    ExecutionCoverage,
    ExecutionError,
    ExecutionPolicy,
    ExecutionStatistics,
    ExecutionStep,
    ExecutionStepStatus,
    ExecutionTiming,
    ExecutionWarning,
    ResearchExecution,
)
from predictron_engine.research.execution_report import (
    NOT_VALIDATED_STATUS,
    REPORT_SCHEMA_VERSION,
    ExecutionSummary,
    ResearchExecutionResult,
    ResearchReport,
    ValidationOutcome,
    build_research_report,
)
from predictron_engine.research.execution_state import (
    ExecutionState,
    ExecutionStateTransition,
)
from predictron_engine.research.models import PlannerInput
from predictron_engine.research.planner import ResearchPlanner
from predictron_engine.research.source_discovery import SourceDiscovery
from predictron_engine.research.validation import EvidenceValidationEngine


@pytest.fixture
def artifacts(planner: ResearchPlanner) -> dict[str, object]:
    """Every real pipeline artifact for a small, fixed research run."""
    plan = planner.create_plan(PlannerInput(company_name="Acme"))
    discovery = SourceDiscovery().discover(plan)
    dispatch = DispatchOutcome.from_dispatch(Dispatcher().dispatch(plan.tasks))
    collection = EvidenceCollectionEngine().execute(plan)
    validation = EvidenceValidationEngine().validate(collection.collection)
    return {
        "plan": plan,
        "discovery": discovery,
        "dispatch": dispatch,
        "collection": collection,
        "validation": validation,
    }


def _report(
    artifacts: dict[str, object],
    *,
    state: ExecutionState = ExecutionState.COMPLETED,
    steps: tuple[ExecutionStep, ...] = (),
    warnings: tuple[ExecutionWarning, ...] = (),
    errors: tuple[ExecutionError, ...] = (),
    timing: ExecutionTiming | None = None,
    execution: ResearchExecution | None = None,
) -> ResearchReport:
    return build_research_report(
        execution=execution or ResearchExecution.create(
            PlannerInput(company_name="Acme")
        ),
        state=state,
        steps=steps,
        plan=artifacts["plan"],  # type: ignore[arg-type]
        discovery=artifacts["discovery"],  # type: ignore[arg-type]
        dispatch=artifacts["dispatch"],  # type: ignore[arg-type]
        collection=artifacts["collection"],  # type: ignore[arg-type]
        validation=artifacts["validation"],  # type: ignore[arg-type]
        warnings=warnings,
        errors=errors,
        timing=timing,
    )


# ----------------------------------------------------------------------
# ValidationOutcome
# ----------------------------------------------------------------------


def test_not_validated_status_constant() -> None:
    assert NOT_VALIDATED_STATUS == "skipped"


def test_default_outcome_is_not_validated() -> None:
    outcome = ValidationOutcome()

    assert not outcome.performed
    assert outcome.status == NOT_VALIDATED_STATUS
    assert outcome.validation_id == ""
    assert outcome.total == 0
    assert not outcome.has_failures
    assert not outcome.is_clean


def test_outcome_from_none_is_not_validated() -> None:
    outcome = ValidationOutcome.from_summary(None)

    assert not outcome.performed
    assert outcome.status == NOT_VALIDATED_STATUS


def test_outcome_projects_a_validation_summary(artifacts: dict[str, object]) -> None:
    validation = artifacts["validation"]
    outcome = ValidationOutcome.from_summary(validation)  # type: ignore[arg-type]

    assert outcome.performed
    assert outcome.status == validation.status.value  # type: ignore[union-attr]
    assert outcome.validation_id == validation.validation_id  # type: ignore[union-attr]
    assert outcome.passed == validation.counts.passed  # type: ignore[union-attr]
    assert outcome.warned == validation.counts.warned  # type: ignore[union-attr]
    assert outcome.failed == validation.counts.failed  # type: ignore[union-attr]
    assert outcome.issues == validation.counts.issues  # type: ignore[union-attr]
    assert outcome.total == validation.counts.total  # type: ignore[union-attr]


def test_outcome_is_clean_only_when_validation_ran_cleanly() -> None:
    assert ValidationOutcome(performed=True).is_clean
    assert not ValidationOutcome(performed=True, issues=1).is_clean
    assert not ValidationOutcome(performed=False).is_clean


def test_outcome_reports_failures(artifacts: dict[str, object]) -> None:
    outcome = ValidationOutcome(performed=True, failed=2)

    assert outcome.has_failures


def test_outcome_rounds_the_mean_confidence() -> None:
    outcome = ValidationOutcome(performed=True, mean_confidence=1 / 3)

    assert outcome.mean_confidence == round(1 / 3, 6)


def test_outcome_rejects_out_of_range_mean_confidence() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ValidationOutcome(mean_confidence=1.5)


def test_outcome_rejects_non_int_counts() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ValidationOutcome(passed="1")  # type: ignore[arg-type]


def test_outcome_rejects_negative_counts() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ValidationOutcome(issues=-1)


def test_outcome_rejects_blank_status() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ValidationOutcome(status="  ")


def test_outcome_rejects_non_bool_performed() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ValidationOutcome(performed=1)  # type: ignore[arg-type]


def test_outcome_round_trips_through_dict(artifacts: dict[str, object]) -> None:
    outcome = ValidationOutcome.from_summary(artifacts["validation"])  # type: ignore[arg-type]

    assert ValidationOutcome.from_dict(outcome.to_dict()) == outcome


def test_outcome_rejects_non_string_validation_id() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ValidationOutcome(validation_id=1)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# ExecutionSummary
# ----------------------------------------------------------------------


def test_summary_defaults_are_empty() -> None:
    summary = ExecutionSummary()

    assert summary.is_clean
    assert summary.statistics == ExecutionStatistics()
    assert summary.coverage == ExecutionCoverage()
    assert summary.confidence == ConfidenceDistribution()
    assert summary.validation == ValidationOutcome()


@pytest.mark.parametrize(
    "field_name",
    ("statistics", "metrics", "coverage", "confidence", "validation"),
)
def test_summary_rejects_wrong_block_types(field_name: str) -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionSummary(**{field_name: {}})  # type: ignore[arg-type]


def test_summary_round_trips_through_dict(artifacts: dict[str, object]) -> None:
    summary = _report(artifacts).summary

    assert ExecutionSummary.from_dict(summary.to_dict()) == summary


# ----------------------------------------------------------------------
# ResearchReport identity
# ----------------------------------------------------------------------


def test_report_schema_version_is_exported() -> None:
    assert REPORT_SCHEMA_VERSION


def test_report_id_is_stable(artifacts: dict[str, object]) -> None:
    assert _report(artifacts).report_id == _report(artifacts).report_id


def test_report_id_is_prefixed(artifacts: dict[str, object]) -> None:
    assert _report(artifacts).report_id.startswith("report_")


def test_report_id_changes_with_the_state(artifacts: dict[str, object]) -> None:
    completed = _report(artifacts).report_id
    failed = _report(artifacts, state=ExecutionState.FAILED).report_id

    assert completed != failed


def test_report_id_changes_with_a_warning(artifacts: dict[str, object]) -> None:
    plain = _report(artifacts).report_id
    warned = _report(
        artifacts, warnings=(ExecutionWarning(code="c", message="m"),)
    ).report_id

    assert plain != warned


def test_report_id_changes_with_an_error(artifacts: dict[str, object]) -> None:
    plain = _report(artifacts).report_id
    errored = _report(
        artifacts,
        state=ExecutionState.FAILED,
        errors=(ExecutionError(code="e", message="m"),),
    ).report_id

    assert plain != errored


def test_report_id_ignores_wall_clock_timing(artifacts: dict[str, object]) -> None:
    unmeasured = _report(artifacts).report_id
    measured = _report(
        artifacts,
        timing=ExecutionTiming(
            measured=True,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:09+00:00",
            total_duration_ms=9000.0,
        ),
    ).report_id

    assert unmeasured == measured


# ----------------------------------------------------------------------
# ResearchReport views
# ----------------------------------------------------------------------


def test_report_is_successful_on_a_clean_completion(
    artifacts: dict[str, object],
) -> None:
    report = _report(artifacts)

    assert report.is_successful
    assert report.state is ExecutionState.COMPLETED
    assert not report.has_errors()
    assert not report.has_warnings()


def test_report_is_not_successful_when_failed(artifacts: dict[str, object]) -> None:
    report = _report(artifacts, state=ExecutionState.FAILED)

    assert not report.is_successful


def test_report_is_not_successful_with_a_recorded_error(
    artifacts: dict[str, object],
) -> None:
    report = _report(artifacts, errors=(ExecutionError(code="e", message="m"),))

    assert not report.is_successful
    assert report.has_errors()


def test_report_exposes_the_digest_blocks(artifacts: dict[str, object]) -> None:
    report = _report(artifacts)

    assert report.statistics == report.summary.statistics
    assert report.metrics == report.summary.metrics
    assert report.coverage == report.summary.coverage
    assert report.confidence == report.summary.confidence
    assert report.validation == report.summary.validation


def test_report_is_deterministic_without_measurement(
    artifacts: dict[str, object],
) -> None:
    report = _report(artifacts)

    assert report.is_deterministic
    assert report.duration_ms == 0.0


def test_report_is_not_deterministic_with_measurement(
    artifacts: dict[str, object],
) -> None:
    report = _report(artifacts, timing=ExecutionTiming(measured=True, total_duration_ms=5.0))

    assert not report.is_deterministic
    assert report.duration_ms == 5.0


def test_report_lists_distinct_codes_in_raise_order(
    artifacts: dict[str, object],
) -> None:
    report = _report(
        artifacts,
        warnings=(
            ExecutionWarning(code="a", message="m"),
            ExecutionWarning(code="b", message="m"),
            ExecutionWarning(code="a", message="m"),
        ),
        errors=(ExecutionError(code="x", message="m"),),
    )

    assert report.warning_codes == ("a", "b")
    assert report.error_codes == ("x",)


def test_report_finds_a_step_by_name(artifacts: dict[str, object]) -> None:
    step = ExecutionStep(
        name="planning",
        state=ExecutionState.PLANNING,
        status=ExecutionStepStatus.COMPLETED,
        order=0,
        detail="1 task(s)",
    )
    report = _report(artifacts, steps=(step,))

    assert report.step("planning") is step
    assert report.step("nope") is None


def test_report_topic_and_evidence_counts(artifacts: dict[str, object]) -> None:
    report = _report(artifacts)
    collection = artifacts["collection"]

    assert report.topic_count == report.statistics.topics_planned
    assert report.evidence_count == collection.evidence_count  # type: ignore[union-attr]


# ----------------------------------------------------------------------
# Derived statistics
# ----------------------------------------------------------------------


def test_statistics_come_from_the_pipeline_artifacts(
    artifacts: dict[str, object],
) -> None:
    statistics = _report(artifacts).statistics
    collection = artifacts["collection"]
    dispatch = artifacts["dispatch"]
    validation = artifacts["validation"]

    assert statistics.tasks_planned == len(collection.executed_tasks) + len(
        collection.unresolved_tasks
    )  # type: ignore[union-attr]
    assert statistics.tasks_dispatched == dispatch.dispatched_count  # type: ignore[union-attr]
    assert statistics.tasks_executed == len(collection.executed_tasks)  # type: ignore[union-attr]
    assert statistics.tasks_failed == len(collection.failed_tasks)  # type: ignore[union-attr]
    assert statistics.evidence_validated == validation.counts.total  # type: ignore[union-attr]


def test_coverage_derives_planned_topics_from_the_plan(
    artifacts: dict[str, object],
) -> None:
    coverage = _report(artifacts).coverage

    assert coverage.planned_count > 0
    assert 0 < coverage.covered_count <= coverage.planned_count
    assert coverage.ratio == round(
        coverage.covered_count / coverage.planned_count, 6
    )


def test_confidence_is_derived_from_the_validation_summary(
    artifacts: dict[str, object],
) -> None:
    validation = artifacts["validation"]
    confidence = _report(artifacts).confidence

    assert confidence.total == validation.counts.total  # type: ignore[union-attr]
    assert confidence.mean_score > 0.0


def test_metrics_count_steps_warnings_and_errors(artifacts: dict[str, object]) -> None:
    step = ExecutionStep(
        name="planning",
        state=ExecutionState.PLANNING,
        status=ExecutionStepStatus.COMPLETED,
        order=0,
        detail="1 task(s)",
    )
    metrics = _report(
        artifacts,
        steps=(step,),
        warnings=(ExecutionWarning(code="c", message="m"),),
        errors=(ExecutionError(code="e", message="m"),),
    ).metrics

    assert metrics.steps_total == 1
    assert metrics.steps_completed == 1
    assert metrics.steps_skipped == 0
    assert metrics.steps_failed == 0
    assert metrics.warnings == 1
    assert metrics.errors == 1
    assert not metrics.is_clean


# ----------------------------------------------------------------------
# Partial reports
# ----------------------------------------------------------------------


def test_report_survives_with_no_artifacts() -> None:
    report = build_research_report(
        execution=ResearchExecution.create(PlannerInput(company_name="Acme")),
        state=ExecutionState.FAILED,
        steps=(),
        plan=None,
        discovery=None,
        dispatch=None,
        collection=None,
        validation=None,
        warnings=(),
        errors=(ExecutionError(code="planning_failed", message="m"),),
    )

    assert report.plan_id == ""
    assert report.statistics == ExecutionStatistics()
    assert report.coverage.ratio == 0.0
    assert not report.validation.performed
    assert report.confidence.is_empty


def test_report_rejects_a_non_execution() -> None:
    with pytest.raises(InvalidExecutionModelError):
        build_research_report(
            execution={},  # type: ignore[arg-type]
            state=ExecutionState.COMPLETED,
            steps=(),
            plan=None,
            discovery=None,
            dispatch=None,
            collection=None,
            validation=None,
            warnings=(),
            errors=(),
        )


def test_report_rejects_duplicate_step_names(artifacts: dict[str, object]) -> None:
    step = ExecutionStep(
        name="planning",
        state=ExecutionState.PLANNING,
        status=ExecutionStepStatus.COMPLETED,
        order=0,
        detail="1 task(s)",
    )

    with pytest.raises(InvalidExecutionModelError):
        _report(artifacts, steps=(step, step))


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------


def test_report_round_trips_through_dict(artifacts: dict[str, object]) -> None:
    step = ExecutionStep(
        name="planning",
        state=ExecutionState.PLANNING,
        status=ExecutionStepStatus.COMPLETED,
        order=0,
        detail="1 task(s)",
    )
    report = _report(
        artifacts,
        steps=(step,),
        warnings=(ExecutionWarning(code="c", message="m"),),
    )

    assert ResearchReport.from_dict(report.to_dict()) == report


def test_report_dict_is_json_ready(artifacts: dict[str, object]) -> None:
    payload = _report(artifacts).to_dict()

    assert set(payload) == {
        "schema_version",
        "report_id",
        "execution_id",
        "company_name",
        "plan_id",
        "state",
        "summary",
        "steps",
        "warnings",
        "errors",
        "timing",
        "is_successful",
    }


def test_report_rejects_unknown_state_from_dict(artifacts: dict[str, object]) -> None:
    payload = _report(artifacts).to_dict()
    payload["state"] = "nope"

    with pytest.raises(InvalidExecutionModelError):
        ResearchReport.from_dict(payload)


# ----------------------------------------------------------------------
# ResearchExecutionResult
# ----------------------------------------------------------------------
def _result(
    artifacts: dict[str, object],
    *,
    state: ExecutionState = ExecutionState.COMPLETED,
    steps: tuple[ExecutionStep, ...] = (),
    transitions: tuple[ExecutionStateTransition, ...] = (),
    execution: ResearchExecution | None = None,
) -> ResearchExecutionResult:
    resolved_execution = execution or ResearchExecution.create(
        PlannerInput(company_name="Acme")
    )
    report = build_research_report(
        execution=resolved_execution,
        state=state,
        steps=steps,
        plan=artifacts["plan"],  # type: ignore[arg-type]
        discovery=artifacts["discovery"],  # type: ignore[arg-type]
        dispatch=artifacts["dispatch"],  # type: ignore[arg-type]
        collection=artifacts["collection"],  # type: ignore[arg-type]
        validation=artifacts["validation"],  # type: ignore[arg-type]
        warnings=(),
        errors=(),
    )
    return ResearchExecutionResult(
        execution=resolved_execution,
        state=state,
        steps=steps,
        transitions=transitions,
        report=report,
        plan=artifacts["plan"],  # type: ignore[arg-type]
        discovery=artifacts["discovery"],  # type: ignore[arg-type]
        dispatch=artifacts["dispatch"],  # type: ignore[arg-type]
        collection=artifacts["collection"],  # type: ignore[arg-type]
        validation=artifacts["validation"],  # type: ignore[arg-type]
    )


def _path() -> tuple[ExecutionStateTransition, ...]:
    """The full legal happy path, from ``NOT_STARTED`` to ``COMPLETED``."""
    return tuple(
        ExecutionStateTransition(sequence=index, source=current, target=target)
        for index, (current, target) in enumerate(
            zip(
                (ExecutionState.NOT_STARTED,
                 ExecutionState.PLANNING,
                 ExecutionState.DISCOVERY,
                 ExecutionState.COLLECTION,
                 ExecutionState.VALIDATION),
                (ExecutionState.PLANNING,
                 ExecutionState.DISCOVERY,
                 ExecutionState.COLLECTION,
                 ExecutionState.VALIDATION,
                 ExecutionState.COMPLETED),
                strict=True,
            )
        )
    )


def test_result_delegates_identifiers(artifacts: dict[str, object]) -> None:
    result = _result(artifacts, transitions=_path())

    assert result.execution_id == result.execution.execution_id
    assert result.report_id == result.report.report_id
    assert result.company_name == "Acme"
    assert result.policy == ExecutionPolicy()


def test_result_delegates_digest_blocks(artifacts: dict[str, object]) -> None:
    result = _result(artifacts, transitions=_path())

    assert result.statistics == result.report.statistics
    assert result.metrics == result.report.metrics
    assert result.coverage == result.report.coverage
    assert result.confidence == result.report.confidence
    assert result.validation_outcome == result.report.validation
    assert result.warnings == ()
    assert result.errors == ()


def test_result_reports_the_visited_state_path(artifacts: dict[str, object]) -> None:
    result = _result(artifacts, transitions=_path())

    assert result.state_path() == (
        ExecutionState.NOT_STARTED,
        ExecutionState.PLANNING,
        ExecutionState.DISCOVERY,
        ExecutionState.COLLECTION,
        ExecutionState.VALIDATION,
        ExecutionState.COMPLETED,
    )


def test_result_succeeded_on_a_clean_completion(artifacts: dict[str, object]) -> None:
    result = _result(artifacts, transitions=_path())

    assert result.succeeded()
    assert not result.has_errors()
    assert not result.has_warnings()
    assert result.is_deterministic


def test_result_is_partial_when_work_is_missing(
    artifacts: dict[str, object],
) -> None:
    result = _result(artifacts, transitions=_path())

    assert result.is_partial
    assert result.succeeded()


def test_result_is_never_partial_when_failed(artifacts: dict[str, object]) -> None:
    failed_path = (
        ExecutionStateTransition(
            sequence=0,
            source=ExecutionState.NOT_STARTED,
            target=ExecutionState.FAILED,
        ),
    )
    result = _result(artifacts, state=ExecutionState.FAILED, transitions=failed_path)

    assert not result.is_partial
    assert not result.succeeded()


def test_result_exposes_evidence_in_deterministic_order(
    artifacts: dict[str, object],
) -> None:
    result = _result(artifacts, transitions=_path())
    evidence = result.evidence()

    assert evidence
    assert evidence == result.evidence()
    assert len(evidence) == result.report.evidence_count


def test_result_evidence_is_empty_without_a_collection() -> None:
    execution = ResearchExecution.create(PlannerInput(company_name="Acme"))
    report = build_research_report(
        execution=execution,
        state=ExecutionState.FAILED,
        steps=(),
        plan=None,
        discovery=None,
        dispatch=None,
        collection=None,
        validation=None,
        warnings=(),
        errors=(),
    )
    result = ResearchExecutionResult(
        execution=execution,
        state=ExecutionState.FAILED,
        steps=(),
        transitions=(
            ExecutionStateTransition(
                sequence=0,
                source=ExecutionState.NOT_STARTED,
                target=ExecutionState.FAILED,
            ),
        ),
        report=report,
    )

    assert result.evidence() == ()


def test_result_step_lookup_delegates_to_the_report(
    artifacts: dict[str, object],
) -> None:
    step = ExecutionStep(
        name="planning",
        state=ExecutionState.PLANNING,
        status=ExecutionStepStatus.COMPLETED,
        order=0,
        detail="1 task(s)",
    )
    result = _result(artifacts, steps=(step,), transitions=_path())

    assert result.step("planning") is step
    assert result.step("nope") is None


def test_result_requires_a_final_transition_into_its_state(
    artifacts: dict[str, object],
) -> None:
    with pytest.raises(InvalidExecutionModelError):
        _result(artifacts, transitions=_path()[:1])


def test_result_rejects_empty_transitions(artifacts: dict[str, object]) -> None:
    with pytest.raises(InvalidExecutionModelError):
        _result(artifacts, transitions=())


def test_result_rejects_out_of_order_transitions(
    artifacts: dict[str, object],
) -> None:
    with pytest.raises(InvalidExecutionModelError):
        _result(artifacts, transitions=tuple(reversed(_path())))


def test_result_rejects_non_execution(artifacts: dict[str, object]) -> None:
    result = _result(artifacts, transitions=_path())

    with pytest.raises(InvalidExecutionModelError):
        ResearchExecutionResult(
            execution={},  # type: ignore[arg-type]
            state=result.state,
            steps=result.steps,
            transitions=result.transitions,
            report=result.report,
        )


def test_result_rejects_non_timing(artifacts: dict[str, object]) -> None:
    result = _result(artifacts, transitions=_path())

    with pytest.raises(InvalidExecutionModelError):
        ResearchExecutionResult(
            execution=result.execution,
            state=result.state,
            steps=result.steps,
            transitions=result.transitions,
            report=result.report,
            timing={},  # type: ignore[arg-type]
        )


def test_result_rejects_non_step_entries(artifacts: dict[str, object]) -> None:
    with pytest.raises(InvalidExecutionModelError):
        _result(artifacts, steps=("planning",), transitions=_path())


def test_result_rejects_non_transition_entries(
    artifacts: dict[str, object],
) -> None:
    with pytest.raises(InvalidExecutionModelError):
        _result(artifacts, transitions=("planning",))


def test_result_round_trips_through_dict(artifacts: dict[str, object]) -> None:
    result = _result(artifacts, transitions=_path())

    assert ResearchExecutionResult.from_dict(result.to_dict()) == result


def test_result_dict_is_json_ready(artifacts: dict[str, object]) -> None:
    payload = _result(artifacts, transitions=_path()).to_dict()

    assert set(payload) == {
        "execution",
        "state",
        "steps",
        "transitions",
        "plan",
        "discovery",
        "dispatch",
        "collection",
        "validation",
        "report",
        "timing",
    }
