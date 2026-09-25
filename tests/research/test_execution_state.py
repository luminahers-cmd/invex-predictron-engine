"""Tests for the execution state machine — Phase 8 Sprint 5.

Covers the ``ExecutionState`` enum, the canonical pipeline ordering, the
closed set of legal transitions, the validation gate, terminal-state
derivation, and the recorded ``ExecutionStateTransition`` value object.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.exceptions import (
    InvalidExecutionModelError,
    InvalidExecutionStateTransitionError,
)
from predictron_engine.research.execution_state import (
    ACTIVE_STATES,
    ALLOWED_TRANSITIONS,
    EXECUTION_STATE_SCHEMA_VERSION,
    PIPELINE_SEQUENCE,
    TERMINAL_STATES,
    ExecutionState,
    ExecutionStateTransition,
    active_state_sequence,
    can_transition,
    is_active_state,
    is_terminal_state,
    resolve_final_state,
    state_sequence,
    successor_state,
    transition_state,
)

# ----------------------------------------------------------------------
# Enum shape
# ----------------------------------------------------------------------


def test_schema_version_is_exported() -> None:
    assert EXECUTION_STATE_SCHEMA_VERSION


def test_seven_states_are_declared() -> None:
    assert len(tuple(ExecutionState)) == 7


def test_declaration_order_is_pipeline_order() -> None:
    assert tuple(ExecutionState) == (
        ExecutionState.NOT_STARTED,
        ExecutionState.PLANNING,
        ExecutionState.DISCOVERY,
        ExecutionState.COLLECTION,
        ExecutionState.VALIDATION,
        ExecutionState.COMPLETED,
        ExecutionState.FAILED,
    )


def test_state_values_are_stable_strings() -> None:
    assert ExecutionState.NOT_STARTED.value == "not_started"
    assert ExecutionState.PLANNING.value == "planning"
    assert ExecutionState.DISCOVERY.value == "discovery"
    assert ExecutionState.COLLECTION.value == "collection"
    assert ExecutionState.VALIDATION.value == "validation"
    assert ExecutionState.COMPLETED.value == "completed"
    assert ExecutionState.FAILED.value == "failed"


def test_state_is_a_string_enum() -> None:
    assert ExecutionState.COMPLETED == "completed"


@pytest.mark.parametrize("state", tuple(ExecutionState))
def test_every_state_serializes(state: ExecutionState) -> None:
    payload = state.to_dict()

    assert payload["value"] == state.value
    assert isinstance(payload["is_terminal"], bool)
    assert isinstance(payload["is_active"], bool)


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "state", (ExecutionState.COMPLETED, ExecutionState.FAILED)
)
def test_terminal_states_report_terminal(state: ExecutionState) -> None:
    assert state.is_terminal
    assert is_terminal_state(state)
    assert not state.is_active
    assert not is_active_state(state)


@pytest.mark.parametrize(
    "state",
    (
        ExecutionState.NOT_STARTED,
        ExecutionState.PLANNING,
        ExecutionState.DISCOVERY,
        ExecutionState.COLLECTION,
        ExecutionState.VALIDATION,
    ),
)
def test_non_terminal_states(state: ExecutionState) -> None:
    assert not state.is_terminal
    assert not is_terminal_state(state)


@pytest.mark.parametrize("state", ACTIVE_STATES)
def test_active_states_report_active(state: ExecutionState) -> None:
    assert state.is_active
    assert is_active_state(state)


def test_not_started_is_neither_active_nor_terminal() -> None:
    state = ExecutionState.NOT_STARTED

    assert not state.is_active
    assert not state.is_terminal


def test_only_failed_is_a_failure() -> None:
    failures = {state for state in ExecutionState if state.is_failure}

    assert failures == {ExecutionState.FAILED}


def test_active_states_tuple_contents() -> None:
    assert ACTIVE_STATES == (
        ExecutionState.PLANNING,
        ExecutionState.DISCOVERY,
        ExecutionState.COLLECTION,
        ExecutionState.VALIDATION,
    )


def test_terminal_states_tuple_contents() -> None:
    assert TERMINAL_STATES == (ExecutionState.COMPLETED, ExecutionState.FAILED)


def test_pipeline_sequence_contents() -> None:
    assert PIPELINE_SEQUENCE == (
        ExecutionState.NOT_STARTED,
        ExecutionState.PLANNING,
        ExecutionState.DISCOVERY,
        ExecutionState.COLLECTION,
        ExecutionState.VALIDATION,
        ExecutionState.COMPLETED,
    )


def test_state_sequence_returns_the_canonical_sequence() -> None:
    assert state_sequence() == PIPELINE_SEQUENCE


def test_active_state_sequence_returns_active_states() -> None:
    assert active_state_sequence() == ACTIVE_STATES


def test_state_sequence_is_not_shared_mutable_state() -> None:
    assert state_sequence() is state_sequence()


# ----------------------------------------------------------------------
# Successors
# ----------------------------------------------------------------------


def test_pipeline_states_have_single_happy_successor() -> None:
    assert successor_state(ExecutionState.NOT_STARTED) is ExecutionState.PLANNING
    assert successor_state(ExecutionState.PLANNING) is ExecutionState.DISCOVERY
    assert successor_state(ExecutionState.DISCOVERY) is ExecutionState.COLLECTION
    assert successor_state(ExecutionState.VALIDATION) is ExecutionState.COMPLETED


def test_failed_has_no_happy_successor() -> None:
    with pytest.raises(InvalidExecutionStateTransitionError):
        successor_state(ExecutionState.FAILED)


def test_not_started_is_on_the_happy_path() -> None:
    assert ExecutionState.NOT_STARTED in PIPELINE_SEQUENCE


@pytest.mark.parametrize("state", TERMINAL_STATES)
def test_terminal_states_have_no_successor(state: ExecutionState) -> None:
    with pytest.raises(InvalidExecutionStateTransitionError):
        successor_state(state)


def test_successors_are_reported_in_canonical_order() -> None:
    successors = ExecutionState.COLLECTION.successors()

    assert successors == (
        ExecutionState.VALIDATION,
        ExecutionState.COMPLETED,
        ExecutionState.FAILED,
    )


def test_terminal_states_have_no_successors() -> None:
    assert ExecutionState.COMPLETED.successors() == ()
    assert ExecutionState.FAILED.successors() == ()


# ----------------------------------------------------------------------
# Transition table
# ----------------------------------------------------------------------


def test_allowed_transitions_covers_every_state() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(ExecutionState)


def test_allowed_transitions_is_read_only() -> None:
    with pytest.raises(TypeError):
        ALLOWED_TRANSITIONS[ExecutionState.PLANNING] = frozenset()  # type: ignore[index]


def test_full_validation_run_is_legal() -> None:
    path = PIPELINE_SEQUENCE

    for current, target in zip(path[:-1], path[1:], strict=True):
        assert can_transition(current, target)
        assert transition_state(current, target) is target


def test_collection_may_short_circuit_to_completed() -> None:
    assert can_transition(ExecutionState.COLLECTION, ExecutionState.COMPLETED)


@pytest.mark.parametrize(
    "state",
    (
        ExecutionState.NOT_STARTED,
        ExecutionState.PLANNING,
        ExecutionState.DISCOVERY,
        ExecutionState.COLLECTION,
        ExecutionState.VALIDATION,
    ),
)
def test_every_active_state_may_fail(state: ExecutionState) -> None:
    assert can_transition(state, ExecutionState.FAILED)


@pytest.mark.parametrize(
    ("current", "target"),
    (
        (ExecutionState.NOT_STARTED, ExecutionState.DISCOVERY),
        (ExecutionState.NOT_STARTED, ExecutionState.COMPLETED),
        (ExecutionState.PLANNING, ExecutionState.COLLECTION),
        (ExecutionState.PLANNING, ExecutionState.COMPLETED),
        (ExecutionState.DISCOVERY, ExecutionState.VALIDATION),
        (ExecutionState.COLLECTION, ExecutionState.PLANNING),
        (ExecutionState.VALIDATION, ExecutionState.COLLECTION),
        (ExecutionState.VALIDATION, ExecutionState.DISCOVERY),
    ),
)
def test_illegal_transitions_are_rejected(
    current: ExecutionState, target: ExecutionState
) -> None:
    assert not can_transition(current, target)

    with pytest.raises(InvalidExecutionStateTransitionError):
        transition_state(current, target)


@pytest.mark.parametrize("state", TERMINAL_STATES)
def test_terminal_states_accept_no_transition(state: ExecutionState) -> None:
    for target in ExecutionState:
        assert not can_transition(state, target)


def test_transition_error_lists_allowed_targets() -> None:
    with pytest.raises(InvalidExecutionStateTransitionError) as excinfo:
        transition_state(ExecutionState.PLANNING, ExecutionState.VALIDATION)

    assert "discovery" in str(excinfo.value)
    assert "failed" in str(excinfo.value)


def test_transition_error_for_terminal_source_mentions_no_transitions() -> None:
    with pytest.raises(InvalidExecutionStateTransitionError) as excinfo:
        transition_state(ExecutionState.COMPLETED, ExecutionState.FAILED)

    assert "no transitions" in str(excinfo.value)


# ----------------------------------------------------------------------
# Final state derivation
# ----------------------------------------------------------------------


def test_final_state_is_completed_without_errors() -> None:
    assert resolve_final_state(has_errors=False) is ExecutionState.COMPLETED


def test_final_state_is_failed_with_errors() -> None:
    assert resolve_final_state(has_errors=True) is ExecutionState.FAILED


def test_final_state_derivation_is_pure() -> None:
    assert resolve_final_state(has_errors=False) is resolve_final_state(
        has_errors=False
    )


# ----------------------------------------------------------------------
# from_value
# ----------------------------------------------------------------------


@pytest.mark.parametrize("state", tuple(ExecutionState))
def test_from_value_round_trips(state: ExecutionState) -> None:
    assert ExecutionState.from_value(state.value) is state


def test_from_value_accepts_a_state_instance() -> None:
    assert (
        ExecutionState.from_value(ExecutionState.FAILED) is ExecutionState.FAILED
    )


@pytest.mark.parametrize("value", (None, 1, 1.5, True, object()))
def test_from_value_rejects_non_strings(value: object) -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionState.from_value(value)


def test_from_value_rejects_unknown_string() -> None:
    with pytest.raises(InvalidExecutionModelError) as excinfo:
        ExecutionState.from_value("nope")

    assert "nope" in str(excinfo.value)


# ----------------------------------------------------------------------
# ExecutionStateTransition
# ----------------------------------------------------------------------


def test_transition_records_sequence_and_states() -> None:
    transition = ExecutionStateTransition(
        sequence=0,
        source=ExecutionState.NOT_STARTED,
        target=ExecutionState.PLANNING,
    )

    assert transition.sequence == 0
    assert transition.source is ExecutionState.NOT_STARTED
    assert transition.target is ExecutionState.PLANNING


def test_transition_into_terminal_is_terminal() -> None:
    transition = ExecutionStateTransition(
        sequence=4,
        source=ExecutionState.VALIDATION,
        target=ExecutionState.COMPLETED,
    )

    assert transition.is_terminal


def test_transition_into_active_is_not_terminal() -> None:
    transition = ExecutionStateTransition(
        sequence=0,
        source=ExecutionState.NOT_STARTED,
        target=ExecutionState.PLANNING,
    )

    assert not transition.is_terminal


def test_transition_rejects_illegal_hop() -> None:
    with pytest.raises(InvalidExecutionStateTransitionError):
        ExecutionStateTransition(
            sequence=0,
            source=ExecutionState.PLANNING,
            target=ExecutionState.VALIDATION,
        )


def test_transition_round_trips_through_dict() -> None:
    transition = ExecutionStateTransition(
        sequence=2,
        source=ExecutionState.DISCOVERY,
        target=ExecutionState.COLLECTION,
    )

    assert ExecutionStateTransition.from_dict(transition.to_dict()) == transition


def test_transition_rejects_negative_sequence() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionStateTransition(
            sequence=-1,
            source=ExecutionState.NOT_STARTED,
            target=ExecutionState.PLANNING,
        )


@pytest.mark.parametrize("sequence", (None, "0", 1.5, True))
def test_transition_rejects_non_int_sequence(sequence: object) -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionStateTransition(
            sequence=sequence,  # type: ignore[arg-type]
            source=ExecutionState.NOT_STARTED,
            target=ExecutionState.PLANNING,
        )


def test_transition_rejects_non_state_source() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionStateTransition(
            sequence=0,
            source="planning",  # type: ignore[arg-type]
            target=ExecutionState.DISCOVERY,
        )


def test_transition_rejects_non_int_sequence_from_dict() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionStateTransition.from_dict(
            {
                "sequence": "0",
                "source": "not_started",
                "target": "planning",
            }
        )


def test_transition_from_dict_rejects_unknown_state() -> None:
    with pytest.raises(InvalidExecutionModelError):
        ExecutionStateTransition.from_dict(
            {
                "sequence": 0,
                "source": "not_started",
                "target": "nope",
            }
        )


def test_transition_is_frozen() -> None:
    transition = ExecutionStateTransition(
        sequence=0,
        source=ExecutionState.NOT_STARTED,
        target=ExecutionState.PLANNING,
    )

    with pytest.raises(Exception):
        transition.sequence = 1  # type: ignore[misc]
