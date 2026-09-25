"""Tests for the Research Orchestrator — Phase 8 Sprint 5.

Covers component wiring and reconciliation, the full happy-path pipeline,
step and transition recording, every warning code, every error code,
fail-fast, policy-driven short circuits, determinism, and serialization.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.dispatcher import Dispatcher
from predictron_engine.research.exceptions import InvalidExecutionInputError
from predictron_engine.research.execution import EvidenceCollectionEngine
from predictron_engine.research.execution_models import (
    ExecutionPolicy,
    ResearchExecution,
)
from predictron_engine.research.execution_report import ResearchExecutionResult
from predictron_engine.research.execution_state import (
    ACTIVE_STATES,
    PIPELINE_SEQUENCE,
    ExecutionState,
)
from predictron_engine.research.models import (
    COLLECTION_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    CollectionResult,
    CollectionStatus,
    EvidenceCollection,
    PlannerInput,
    ResearchPlan,
    SourceDiscoveryPlan,
)
from predictron_engine.research.orchestrator import (
    ERROR_COLLECTION_FAILED,
    ERROR_DISCOVERY_FAILED,
    ERROR_DISPATCH_FAILED,
    ERROR_PLANNING_FAILED,
    ERROR_VALIDATION_FAILED,
    PIPELINE_STEPS,
    STEP_COLLECTION,
    STEP_DISCOVERY,
    STEP_DISPATCH,
    STEP_PLANNING,
    STEP_VALIDATION,
    WARNING_COLLECTOR_FAILURES,
    WARNING_EMPTY_PLAN,
    WARNING_NO_EVIDENCE,
    WARNING_NO_SOURCES,
    WARNING_UNRESOLVED_TASKS,
    WARNING_VALIDATION_CONFLICTS,
    WARNING_VALIDATION_FAILED,
    WARNING_VALIDATION_SKIPPED,
    WARNING_VALIDATION_WARNED,
    ResearchOrchestrator,
)
from predictron_engine.research.planner import ResearchPlanner
from predictron_engine.research.source_discovery import SourceDiscovery
from predictron_engine.research.validation import EvidenceValidationEngine
from predictron_engine.research.validation_models import collection_fingerprint


class _ExplodingDispatcher(Dispatcher):
    """A real dispatcher whose registry fails, so wiring stays honest."""

    def dispatch(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")


class _Exploding:
    """A stand-in component whose every call raises."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or RuntimeError("boom")
        self.calls = 0

    def _raise(self) -> None:
        self.calls += 1
        raise self._error

    def create_plan(self, *args: object, **kwargs: object) -> ResearchPlan:
        self._raise()
        raise AssertionError("unreachable")

    def discover(self, *args: object, **kwargs: object) -> SourceDiscoveryPlan:
        self._raise()
        raise AssertionError("unreachable")

    def dispatch(self, *args: object, **kwargs: object) -> object:
        self._raise()
        raise AssertionError("unreachable")

    def execute(self, *args: object, **kwargs: object) -> CollectionResult:
        self._raise()
        raise AssertionError("unreachable")

    def validate(self, *args: object, **kwargs: object) -> object:
        self._raise()
        raise AssertionError("unreachable")


def _orchestrator(
    *,
    planner: object | None = None,
    source_discovery: object | None = None,
    dispatcher: object | None = None,
    collection_engine: object | None = None,
    validation_engine: object | None = None,
    policy: ExecutionPolicy | None = None,
) -> ResearchOrchestrator:
    return ResearchOrchestrator(
        planner=planner,  # type: ignore[arg-type]
        source_discovery=source_discovery,  # type: ignore[arg-type]
        dispatcher=dispatcher,  # type: ignore[arg-type]
        collection_engine=collection_engine,  # type: ignore[arg-type]
        validation_engine=validation_engine,  # type: ignore[arg-type]
        policy=policy,
    )


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------


def test_pipeline_steps_are_declared_in_execution_order() -> None:
    assert PIPELINE_STEPS == (
        STEP_PLANNING,
        STEP_DISCOVERY,
        STEP_DISPATCH,
        STEP_COLLECTION,
        STEP_VALIDATION,
    )


def test_error_codes_are_distinct_stable_strings() -> None:
    codes = (
        ERROR_PLANNING_FAILED,
        ERROR_DISCOVERY_FAILED,
        ERROR_DISPATCH_FAILED,
        ERROR_COLLECTION_FAILED,
        ERROR_VALIDATION_FAILED,
    )

    assert len(set(codes)) == 5
    assert all(code.islower() and " " not in code for code in codes)


def test_warning_codes_are_distinct_stable_strings() -> None:
    codes = (
        WARNING_EMPTY_PLAN,
        WARNING_NO_SOURCES,
        WARNING_UNRESOLVED_TASKS,
        WARNING_COLLECTOR_FAILURES,
        WARNING_NO_EVIDENCE,
        WARNING_VALIDATION_SKIPPED,
        WARNING_VALIDATION_WARNED,
        WARNING_VALIDATION_FAILED,
        WARNING_VALIDATION_CONFLICTS,
    )

    assert len(set(codes)) == 9
    assert all(code.islower() and " " not in code for code in codes)


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------


def test_defaults_are_real_components() -> None:
    orchestrator = ResearchOrchestrator()

    assert isinstance(orchestrator.planner, ResearchPlanner)
    assert isinstance(orchestrator.source_discovery, SourceDiscovery)
    assert isinstance(orchestrator.dispatcher, Dispatcher)
    assert isinstance(orchestrator.collection_engine, EvidenceCollectionEngine)
    assert isinstance(orchestrator.validation_engine, EvidenceValidationEngine)
    assert orchestrator.policy == ExecutionPolicy()


def test_injected_components_are_used_verbatim() -> None:
    planner = ResearchPlanner()
    discovery = SourceDiscovery()
    validation = EvidenceValidationEngine()
    orchestrator = _orchestrator(
        planner=planner, source_discovery=discovery, validation_engine=validation
    )

    assert orchestrator.planner is planner
    assert orchestrator.source_discovery is discovery
    assert orchestrator.validation_engine is validation


def test_dispatcher_and_collection_engine_share_one_registry() -> None:
    orchestrator = ResearchOrchestrator()

    assert orchestrator.collection_engine.dispatcher is orchestrator.dispatcher


def test_supplied_dispatcher_drives_the_default_collection_engine() -> None:
    dispatcher = Dispatcher()
    orchestrator = _orchestrator(dispatcher=dispatcher)

    assert orchestrator.dispatcher is dispatcher
    assert orchestrator.collection_engine.dispatcher is dispatcher


def test_supplied_collection_engine_keeps_precedence() -> None:
    engine = EvidenceCollectionEngine()
    orchestrator = _orchestrator(collection_engine=engine)

    assert orchestrator.collection_engine is engine
    assert orchestrator.dispatcher is engine.dispatcher


def test_supplied_engine_wins_over_a_supplied_dispatcher() -> None:
    engine = EvidenceCollectionEngine()
    other = Dispatcher()
    orchestrator = _orchestrator(
        dispatcher=other, collection_engine=engine
    )

    assert orchestrator.collection_engine is engine
    assert orchestrator.dispatcher is engine.dispatcher


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_happy_path_reaches_completed(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert isinstance(result, ResearchExecutionResult)
    assert result.state is ExecutionState.COMPLETED
    assert result.succeeded()
    assert not result.has_errors()


def test_happy_path_visits_every_active_state(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.state_path() == PIPELINE_SEQUENCE


def test_happy_path_records_every_step(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert [step.name for step in result.steps] == list(PIPELINE_STEPS)
    assert [step.order for step in result.steps] == list(range(5))
    assert all(step.is_completed for step in result.steps)
    assert all(step.artifact_id for step in result.steps)


def test_steps_record_the_state_they_ran_in(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert [step.state for step in result.steps] == [
        ExecutionState.PLANNING,
        ExecutionState.DISCOVERY,
        ExecutionState.COLLECTION,
        ExecutionState.COLLECTION,
        ExecutionState.VALIDATION,
    ]


def test_dispatch_and_collection_share_the_collection_state(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.step(STEP_DISPATCH).state is ExecutionState.COLLECTION  # type: ignore[union-attr]
    assert result.step(STEP_COLLECTION).state is ExecutionState.COLLECTION  # type: ignore[union-attr]


def test_happy_path_produces_every_artifact(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert isinstance(result.plan, ResearchPlan)
    assert isinstance(result.discovery, SourceDiscoveryPlan)
    assert result.dispatch is not None
    assert isinstance(result.collection, CollectionResult)
    assert result.validation is not None


def test_step_artifacts_reference_the_real_identifiers(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.step(STEP_PLANNING).artifact_id == result.plan.plan_id  # type: ignore[union-attr]
    assert result.step(STEP_DISCOVERY).artifact_id.startswith("discovery_")  # type: ignore[union-attr]
    assert result.step(STEP_DISPATCH).artifact_id.startswith("dispatch_")  # type: ignore[union-attr]
    assert result.step(STEP_COLLECTION).artifact_id == result.collection.collection_id  # type: ignore[union-attr]
    assert result.step(STEP_VALIDATION).artifact_id == result.validation.validation_id  # type: ignore[union-attr]


def test_reported_dispatch_matches_the_executed_dispatch(
    company_input: PlannerInput,
) -> None:
    orchestrator = ResearchOrchestrator()
    result = orchestrator.execute(company_input)
    replayed = orchestrator.dispatcher.dispatch(result.plan.tasks)

    assert result.dispatch.task_ids == replayed.task_ids  # type: ignore[union-attr]
    assert result.dispatch.collector_ids == replayed.collector_ids  # type: ignore[union-attr]
    assert result.dispatch.unresolved == replayed.unresolved  # type: ignore[union-attr]


def test_statistics_are_derived_from_the_artifacts(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator().execute(company_input)
    statistics = result.statistics

    assert statistics.topics_planned == len(result.plan.tasks)  # type: ignore[union-attr]
    assert statistics.tasks_planned == len(result.plan.tasks)  # type: ignore[union-attr]
    assert statistics.tasks_executed == len(result.collection.executed_tasks)  # type: ignore[union-attr]
    assert statistics.evidence_collected == result.collection.evidence_count  # type: ignore[union-attr]
    assert statistics.evidence_validated == result.validation.counts.total  # type: ignore[union-attr]


def test_coverage_reflects_uncovered_topics(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.coverage.planned_count == result.statistics.topics_planned
    assert result.coverage.uncovered_topics
    assert result.is_partial


def test_happy_path_collects_evidence(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.evidence()
    assert result.confidence.total == result.statistics.evidence_validated


def test_validation_outcome_is_performed(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.validation_outcome.performed
    assert result.validation_outcome.validation_id == result.validation.validation_id  # type: ignore[union-attr]


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


def test_two_orchestrators_produce_identical_results(
    company_input: PlannerInput,
) -> None:
    first = ResearchOrchestrator().execute(company_input)
    second = ResearchOrchestrator().execute(company_input)

    assert first.report_id == second.report_id
    assert first.execution_id == second.execution_id
    assert first.report.to_dict() == second.report.to_dict()


def test_repeated_execution_is_stable(company_input: PlannerInput) -> None:
    orchestrator = ResearchOrchestrator()

    first = orchestrator.execute(company_input)
    second = orchestrator.execute(company_input)

    assert first.report_id == second.report_id
    assert first.to_dict() == second.to_dict()


def test_default_policy_never_reads_the_wall_clock(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.is_deterministic
    assert not result.timing.measured
    assert result.timing.step_durations == ()
    assert result.timing.started_at == ""
    assert result.duration_ms == 0.0


def test_measurement_does_not_change_the_report_id(
    company_input: PlannerInput,
) -> None:
    policy = ExecutionPolicy(measure_duration=True)
    first = ResearchOrchestrator(policy=policy).execute(company_input)
    second = ResearchOrchestrator(policy=policy).execute(company_input)

    assert first.timing.measured
    assert first.timing.started_at
    assert first.duration_ms > 0.0
    assert first.report_id == second.report_id


def test_timing_values_never_reach_the_report_identity(
    company_input: PlannerInput,
) -> None:
    policy = ExecutionPolicy(measure_duration=True)
    first = ResearchOrchestrator(policy=policy).execute(company_input)
    second = ResearchOrchestrator(policy=policy).execute(company_input)

    assert first.execution_id == second.execution_id
    assert first.report_id == second.report_id
    assert first.report.to_dict()["timing"] != second.report.to_dict()["timing"]


def test_measurement_records_a_duration_per_step(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator(
        policy=ExecutionPolicy(measure_duration=True)
    ).execute(company_input)

    recorded = [entry.step for entry in result.timing.step_durations]
    assert recorded == list(PIPELINE_STEPS)
    assert result.timing.duration_for(STEP_PLANNING) >= 0.0
    assert result.timing.duration_for("nope") == 0.0


def test_different_companies_produce_different_report_ids() -> None:
    first = ResearchOrchestrator().execute_for_company("Acme")
    second = ResearchOrchestrator().execute_for_company("Globex")

    assert first.report_id != second.report_id
    assert first.execution_id != second.execution_id


def test_different_policies_produce_different_execution_ids(
    company_input: PlannerInput,
) -> None:
    default = ResearchOrchestrator().execute(company_input)
    no_validation = ResearchOrchestrator(
        policy=ExecutionPolicy(run_validation=False)
    ).execute(company_input)

    assert default.execution_id != no_validation.execution_id


# ----------------------------------------------------------------------
# Policy: validation disabled
# ----------------------------------------------------------------------


def test_validation_can_be_disabled(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator(
        policy=ExecutionPolicy(run_validation=False)
    ).execute(company_input)

    assert result.state is ExecutionState.COMPLETED
    assert result.succeeded()
    assert result.validation is None


def test_disabling_validation_skips_the_step_and_the_state(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator(
        policy=ExecutionPolicy(run_validation=False)
    ).execute(company_input)

    assert [step.name for step in result.steps] == [
        STEP_PLANNING,
        STEP_DISCOVERY,
        STEP_DISPATCH,
        STEP_COLLECTION,
    ]
    assert ExecutionState.VALIDATION not in result.state_path()


def test_disabling_validation_short_circuits_from_collection(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator(
        policy=ExecutionPolicy(run_validation=False)
    ).execute(company_input)

    assert result.state_path() == (
        ExecutionState.NOT_STARTED,
        ExecutionState.PLANNING,
        ExecutionState.DISCOVERY,
        ExecutionState.COLLECTION,
        ExecutionState.COMPLETED,
    )


def test_disabling_validation_omits_the_requested_state(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator(
        policy=ExecutionPolicy(run_validation=False)
    ).execute(company_input)

    assert result.execution.requested_states == tuple(
        state for state in ACTIVE_STATES if state is not ExecutionState.VALIDATION
    )
    assert not result.execution.visits(ExecutionState.VALIDATION)


def test_disabling_validation_reports_a_not_validated_outcome(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator(
        policy=ExecutionPolicy(run_validation=False)
    ).execute(company_input)

    assert not result.validation_outcome.performed
    assert result.validation_outcome.status == "skipped"
    assert result.confidence.is_empty


def test_disabling_validation_still_collects_evidence(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator(
        policy=ExecutionPolicy(run_validation=False)
    ).execute(company_input)

    assert result.evidence()
    assert result.statistics.evidence_collected > 0
    assert result.statistics.evidence_validated == 0


# ----------------------------------------------------------------------
# Stage failures
# ----------------------------------------------------------------------


def test_planning_failure_fails_the_run(company_input: PlannerInput) -> None:
    result = _orchestrator(planner=_Exploding()).execute(company_input)

    assert result.state is ExecutionState.FAILED
    assert not result.succeeded()
    assert result.error_codes == (ERROR_PLANNING_FAILED,)
    assert result.step(STEP_PLANNING).is_failed  # type: ignore[union-attr]


def test_planning_failure_stops_the_pipeline(company_input: PlannerInput) -> None:
    result = _orchestrator(planner=_Exploding()).execute(company_input)

    assert [step.name for step in result.steps] == [STEP_PLANNING]
    assert result.plan is None
    assert result.discovery is None
    assert result.collection is None


def test_planning_failure_records_the_exception_type(
    company_input: PlannerInput,
) -> None:
    result = _orchestrator(planner=_Exploding()).execute(company_input)

    assert result.errors[0].exception_type == "RuntimeError"
    assert result.errors[0].step == STEP_PLANNING


def test_planning_failure_records_a_stable_message(
    company_input: PlannerInput,
) -> None:
    first = _orchestrator(planner=_Exploding()).execute(company_input)
    second = _orchestrator(planner=_Exploding()).execute(company_input)

    assert first.errors[0].message == second.errors[0].message
    assert "0x" not in first.errors[0].message


def test_planning_failure_sanitizes_memory_addresses(
    company_input: PlannerInput,
) -> None:
    error = ValueError("bad <object at 0x7ffee1234567>")
    result = _orchestrator(planner=_Exploding(error)).execute(company_input)

    assert "0x7ffee1234567" not in result.errors[0].message
    assert "0xADDR" in result.errors[0].message


def test_planning_failure_produces_an_empty_report(
    company_input: PlannerInput,
) -> None:
    result = _orchestrator(planner=_Exploding()).execute(company_input)

    assert result.report.plan_id == ""
    assert result.report.statistics.tasks_planned == 0
    assert result.report.state is ExecutionState.FAILED
    assert not result.report.is_successful


def test_discovery_failure_continues_to_collection(
    company_input: PlannerInput,
) -> None:
    result = _orchestrator(
        source_discovery=_Exploding()
    ).execute(company_input)

    assert result.state is ExecutionState.FAILED
    assert result.error_codes == (ERROR_DISCOVERY_FAILED,)
    assert result.discovery is None
    assert result.collection is not None


def test_discovery_failure_keeps_the_plan(company_input: PlannerInput) -> None:
    result = _orchestrator(
        source_discovery=_Exploding()
    ).execute(company_input)

    assert isinstance(result.plan, ResearchPlan)
    assert result.report.plan_id == result.plan.plan_id


def test_dispatch_failure_fails_both_stages_of_the_shared_registry(
    company_input: PlannerInput,
) -> None:
    result = _orchestrator(
        collection_engine=EvidenceCollectionEngine(
            dispatcher=_ExplodingDispatcher()
        )
    ).execute(company_input)

    assert result.error_codes == (
        ERROR_DISPATCH_FAILED,
        ERROR_COLLECTION_FAILED,
    )
    assert result.state is ExecutionState.FAILED
    assert result.dispatch is None
    assert result.collection is None


def test_a_supplied_engine_overrides_a_supplied_dispatcher(
    company_input: PlannerInput,
) -> None:
    result = _orchestrator(
        dispatcher=_Exploding(), collection_engine=EvidenceCollectionEngine()
    ).execute(company_input)

    assert result.error_codes == ()
    assert result.succeeded()
    assert result.dispatch is not None


def test_collection_failure_is_recorded(company_input: PlannerInput) -> None:
    result = _orchestrator(
        collection_engine=_Exploding()
    ).execute(company_input)

    assert result.state is ExecutionState.FAILED
    assert result.error_codes == (ERROR_COLLECTION_FAILED,)
    assert result.step(STEP_COLLECTION).is_failed  # type: ignore[union-attr]


def test_validation_failure_is_recorded(company_input: PlannerInput) -> None:
    result = _orchestrator(
        validation_engine=_Exploding()
    ).execute(company_input)

    assert result.state is ExecutionState.FAILED
    assert result.error_codes == (ERROR_VALIDATION_FAILED,)
    assert result.validation is None


def test_validation_failure_keeps_the_collection(
    company_input: PlannerInput,
) -> None:
    result = _orchestrator(
        validation_engine=_Exploding()
    ).execute(company_input)

    assert result.collection is not None
    assert result.evidence()


def test_failed_runs_settle_in_the_failed_state(
    company_input: PlannerInput,
) -> None:
    result = _orchestrator(planner=_Exploding()).execute(company_input)

    assert result.state_path()[-1] is ExecutionState.FAILED
    assert result.transitions[-1].is_terminal


def test_failed_runs_never_report_partial_completion(
    company_input: PlannerInput,
) -> None:
    result = _orchestrator(planner=_Exploding()).execute(company_input)

    assert not result.is_partial


# ----------------------------------------------------------------------
# Fail-fast
# ----------------------------------------------------------------------


def test_fail_fast_reraises_the_stage_exception(
    company_input: PlannerInput,
) -> None:
    error = RuntimeError("boom")

    with pytest.raises(RuntimeError) as excinfo:
        _orchestrator(
            planner=_Exploding(error),
            policy=ExecutionPolicy(fail_fast=True),
        ).execute(company_input)

    assert excinfo.value is error


def test_fail_fast_is_opt_in(company_input: PlannerInput) -> None:
    result = _orchestrator(planner=_Exploding()).execute(company_input)

    assert result.state is ExecutionState.FAILED


def test_fail_fast_policy_is_allow_partial_false() -> None:
    assert not ExecutionPolicy(fail_fast=True).allow_partial


# ----------------------------------------------------------------------
# Warnings
# ----------------------------------------------------------------------


def test_unresolved_tasks_are_warned(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert WARNING_UNRESOLVED_TASKS in result.report.warning_codes
    warning = next(
        entry for entry in result.warnings if entry.code == WARNING_UNRESOLVED_TASKS
    )
    assert warning.step == STEP_DISPATCH
    assert warning.detail


def test_collector_failures_are_warned(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert WARNING_COLLECTOR_FAILURES in result.report.warning_codes
    warning = next(
        entry
        for entry in result.warnings
        if entry.code == WARNING_COLLECTOR_FAILURES
    )
    assert warning.step == STEP_COLLECTION


def test_warnings_are_counted_on_their_step(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.step(STEP_DISPATCH).warning_count == 1  # type: ignore[union-attr]
    assert result.step(STEP_PLANNING).warning_count == 0  # type: ignore[union-attr]


def test_warning_count_matches_the_recorded_warnings(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator().execute(company_input)
    total = sum(step.warning_count for step in result.steps)

    assert total == len(result.warnings)


def test_validation_warnings_are_content_level(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.state is ExecutionState.COMPLETED
    assert result.succeeded()
    assert WARNING_VALIDATION_WARNED in result.report.warning_codes


def test_empty_plan_warns_and_still_completes(
    company_input: PlannerInput,
) -> None:
    class _EmptyPlanner:
        def create_plan(self, planner_input: PlannerInput) -> ResearchPlan:
            return ResearchPlanner().create_plan(
                PlannerInput(company_name="Acme")
            ).__class__.from_dict(
                {**ResearchPlanner().create_plan(planner_input).to_dict(), "tasks": []}
            )

    result = _orchestrator(planner=_EmptyPlanner()).execute(company_input)

    assert WARNING_EMPTY_PLAN in result.report.warning_codes
    assert result.state is ExecutionState.COMPLETED


def test_no_sources_warns() -> None:
    class _NoDiscovery:
        def discover(self, plan: ResearchPlan) -> SourceDiscoveryPlan:
            return SourceDiscoveryPlan(
                schema_version=SOURCE_SCHEMA_VERSION,
                plan_id=plan.plan_id,
                company_name=plan.company_name,
                source_tasks=(),
                recommendations=(),
            )

    result = _orchestrator(source_discovery=_NoDiscovery()).execute_for_company(
        "Acme"
    )

    assert WARNING_NO_SOURCES in result.report.warning_codes
    assert result.discovery is not None
    assert result.discovery.recommendations == ()


def test_no_evidence_warns() -> None:
    class _EmptyCollection(EvidenceCollectionEngine):
        def execute(self, plan: ResearchPlan) -> CollectionResult:
            return CollectionResult(
                schema_version=COLLECTION_SCHEMA_VERSION,
                plan_id=plan.plan_id,
                company_name=plan.company_name,
                collection_id="collection_empty",
                status=CollectionStatus.EMPTY,
                collection=EvidenceCollection(evidence=(), topic_order=()),
                executed_tasks=(),
                failed_tasks=plan.tasks and tuple(
                    task.task_id for task in plan.tasks
                ),
                unresolved_tasks=(),
            )

    result = _orchestrator(
        collection_engine=_EmptyCollection()
    ).execute_for_company("Acme")

    assert result.collection is not None
    assert result.collection.evidence_count == 0
    assert WARNING_NO_EVIDENCE in result.report.warning_codes
    assert WARNING_COLLECTOR_FAILURES in result.report.warning_codes


def test_warnings_never_change_the_lifecycle_state(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.has_warnings()
    assert result.state is ExecutionState.COMPLETED


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


def test_execute_rejects_a_non_planner_input() -> None:
    with pytest.raises(InvalidExecutionInputError):
        ResearchOrchestrator().execute("Acme")  # type: ignore[arg-type]


def test_execute_rejects_a_non_planner_input_under_fail_fast() -> None:
    with pytest.raises(InvalidExecutionInputError):
        _orchestrator(
            policy=ExecutionPolicy(fail_fast=True)
        ).execute(None)  # type: ignore[arg-type]


def test_execute_for_company_builds_planner_input() -> None:
    result = ResearchOrchestrator().execute_for_company(
        "Acme",
        website="https://acme.test",
        description="Widgets.",
        prediction_horizon_days=180,
    )

    assert result.company_name == "Acme"
    assert result.execution.planner_input.website == "https://acme.test"
    assert result.execution.planner_input.prediction_horizon_days == 180


def test_execute_for_company_matches_execute() -> None:
    orchestrator = ResearchOrchestrator()

    assert (
        orchestrator.execute_for_company("Acme").report_id
        == orchestrator.execute(PlannerInput(company_name="Acme")).report_id
    )


def test_execute_for_company_rejects_a_blank_name() -> None:
    with pytest.raises(Exception):
        ResearchOrchestrator().execute_for_company("   ")


# ----------------------------------------------------------------------
# Result identity and serialization
# ----------------------------------------------------------------------


def test_result_report_matches_the_result_state(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.report.state is result.state
    assert result.report.report_id == result.report_id
    assert result.report.execution_id == result.execution_id


def test_result_stores_the_stage_artifacts_by_reference(
    company_input: PlannerInput,
) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.collection.plan_id == result.plan.plan_id  # type: ignore[union-attr]
    assert result.collection.company_name == result.plan.company_name  # type: ignore[union-attr]
    assert result.validation.collection_fingerprint == collection_fingerprint(  # type: ignore[union-attr]
        [
            item.evidence_id
            for item in result.collection.collection.items()  # type: ignore[union-attr]
        ]
    )
    assert result.validation.counts.total == result.collection.evidence_count  # type: ignore[union-attr]


def test_result_round_trips_through_dict(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert ResearchExecutionResult.from_dict(result.to_dict()) == result


def test_execution_request_is_stable(company_input: PlannerInput) -> None:
    result = ResearchOrchestrator().execute(company_input)

    assert result.execution == ResearchExecution.create(company_input)
    assert result.execution.schema_version
