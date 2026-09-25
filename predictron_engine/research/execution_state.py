"""Deterministic execution state machine — Phase 8 Sprint 5.

This module owns the *lifecycle vocabulary* of an orchestrated research
run.  It contains exactly two things:

* :class:`ExecutionState` — the immutable state enum
  (``NOT_STARTED``, ``PLANNING``, ``DISCOVERY``, ``COLLECTION``,
  ``VALIDATION``, ``COMPLETED``, ``FAILED``),
* :data:`ALLOWED_TRANSITIONS` — the closed set of legal transitions,
  plus pure helpers to query and walk it.

The state machine is **total and deterministic**: every query is a pure
function of its arguments, the transition table is a read-only mapping,
and no wall-clock, randomness, I/O, or environment state participates in
any decision.  A given sequence of state visits always produces the same
:class:`ExecutionStateTransition` records.

The orchestrator never invents a transition — it asks
:func:`transition_state` to validate each hop, so an illegal hop is a
hard programming error (it raises
:class:`~predictron_engine.research.exceptions.InvalidExecutionStateTransitionError`)
rather than a silently corrupted run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from predictron_engine.research.exceptions import (
    InvalidExecutionModelError,
    InvalidExecutionStateTransitionError,
)

#: Serialization schema version for execution state artifacts.
EXECUTION_STATE_SCHEMA_VERSION = "1.0.0"


class ExecutionState(str, Enum):
    """The deterministic lifecycle state of one research execution.

    * ``NOT_STARTED`` — the execution has been accepted but no stage ran.
    * ``PLANNING`` — the Research Planner is producing a research plan.
    * ``DISCOVERY`` — Source Discovery is ranking candidate sources.
    * ``COLLECTION`` — dispatch and the Evidence Collection Engine run.
    * ``VALIDATION`` — the Evidence Validation Engine is running.
    * ``COMPLETED`` — every required stage produced a result.
    * ``FAILED`` — at least one required stage produced an error.

    The declaration order is the canonical pipeline order, so
    ``tuple(ExecutionState)`` is itself a deterministic ordering and the
    enum never has to be re-sorted.
    """

    NOT_STARTED = "not_started"
    PLANNING = "planning"
    DISCOVERY = "discovery"
    COLLECTION = "collection"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Return whether no further transition is possible."""
        return self in TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        """Return whether the state represents work in progress."""
        return self in ACTIVE_STATES

    @property
    def is_failure(self) -> bool:
        """Return whether the state is the terminal failure state."""
        return self is ExecutionState.FAILED

    def successors(self) -> tuple[ExecutionState, ...]:
        """Return the legal successor states, in canonical order."""
        return tuple(
            state
            for state in tuple(ExecutionState)
            if state in ALLOWED_TRANSITIONS[self]
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "value": self.value,
            "is_terminal": self.is_terminal,
            "is_active": self.is_active,
        }

    @classmethod
    def from_value(cls, value: object) -> ExecutionState:
        """Restore a state from its serialized ``value`` string."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise InvalidExecutionModelError(
                "expected string for 'value' when restoring ExecutionState"
            )
        try:
            return cls(value)
        except ValueError as exc:
            raise InvalidExecutionModelError(
                f"unknown ExecutionState value: {value!r}"
            ) from exc


#: The canonical happy-path state sequence, in order.
PIPELINE_SEQUENCE: tuple[ExecutionState, ...] = (
    ExecutionState.NOT_STARTED,
    ExecutionState.PLANNING,
    ExecutionState.DISCOVERY,
    ExecutionState.COLLECTION,
    ExecutionState.VALIDATION,
    ExecutionState.COMPLETED,
)

#: States that represent work in progress, in pipeline order.
ACTIVE_STATES: tuple[ExecutionState, ...] = (
    ExecutionState.PLANNING,
    ExecutionState.DISCOVERY,
    ExecutionState.COLLECTION,
    ExecutionState.VALIDATION,
)

#: States from which no further transition is possible.
TERMINAL_STATES: tuple[ExecutionState, ...] = (
    ExecutionState.COMPLETED,
    ExecutionState.FAILED,
)

#: The closed set of legal transitions.
#:
#: The pipeline may only advance one stage at a time, and any active state
#: may exit to :attr:`ExecutionState.FAILED`.  ``COLLECTION`` additionally
#: admits a direct hop to ``COMPLETED``, which is how a run configured
#: without validation short-circuits the ``VALIDATION`` state.  A
#: ``FAILED`` hop may then never be taken from ``COLLECTION`` in that
#: configuration, because nothing remains to fail.  ``COMPLETED`` and
#: ``FAILED`` are terminal: a run is never resurrected.
ALLOWED_TRANSITIONS: Mapping[ExecutionState, frozenset[ExecutionState]] = (
    MappingProxyType(
        {
            ExecutionState.NOT_STARTED: frozenset(
                {ExecutionState.PLANNING, ExecutionState.FAILED}
            ),
            ExecutionState.PLANNING: frozenset(
                {ExecutionState.DISCOVERY, ExecutionState.FAILED}
            ),
            ExecutionState.DISCOVERY: frozenset(
                {ExecutionState.COLLECTION, ExecutionState.FAILED}
            ),
            ExecutionState.COLLECTION: frozenset(
                {
                    ExecutionState.VALIDATION,
                    ExecutionState.COMPLETED,
                    ExecutionState.FAILED,
                }
            ),
            ExecutionState.VALIDATION: frozenset(
                {ExecutionState.COMPLETED, ExecutionState.FAILED}
            ),
            ExecutionState.COMPLETED: frozenset(),
            ExecutionState.FAILED: frozenset(),
        }
    )
)


def is_terminal_state(state: ExecutionState) -> bool:
    """Return whether ``state`` admits no further transition."""
    return state in TERMINAL_STATES


def is_active_state(state: ExecutionState) -> bool:
    """Return whether ``state`` represents work in progress."""
    return state in ACTIVE_STATES


def can_transition(current: ExecutionState, target: ExecutionState) -> bool:
    """Return whether ``current`` -> ``target`` is a legal transition."""
    return target in ALLOWED_TRANSITIONS[current]


def successor_state(current: ExecutionState) -> ExecutionState:
    """Return the next happy-path state after ``current``.

    Raises
    ------
    InvalidExecutionStateTransitionError:
        When ``current`` is terminal or absent from the happy path, so it
        has no successor.
    """
    if current not in PIPELINE_SEQUENCE:
        raise InvalidExecutionStateTransitionError(
            f"{current.value!r} is not on the happy path and has no successor"
        )
    index = PIPELINE_SEQUENCE.index(current)
    if index + 1 >= len(PIPELINE_SEQUENCE):
        raise InvalidExecutionStateTransitionError(
            f"{current.value!r} is a terminal state and has no successor"
        )
    return PIPELINE_SEQUENCE[index + 1]


def transition_state(
    current: ExecutionState,
    target: ExecutionState,
) -> ExecutionState:
    """Return ``target`` after validating the ``current`` -> ``target`` hop.

    This is the single gate every transition passes through.  A legal hop
    returns ``target`` unchanged so callers can write
    ``state = transition_state(state, target)``.

    Raises
    ------
    InvalidExecutionStateTransitionError:
        When the hop is not part of the declared state machine.
    """
    if not can_transition(current, target):
        allowed = ", ".join(
            state.value for state in current.successors()
        ) or "no transitions"
        raise InvalidExecutionStateTransitionError(
            f"illegal execution transition {current.value!r} -> "
            f"{target.value!r} (allowed: {allowed})"
        )
    return target


def resolve_final_state(*, has_errors: bool) -> ExecutionState:
    """Derive the terminal state of a run from whether errors were recorded.

    The derivation is pure: a run that recorded at least one stage error
    is ``FAILED``; every other run is ``COMPLETED`` — including runs whose
    *content* is degraded (empty plan, unresolved tasks, failing evidence).
    Degradation is reported through warnings and statistics, never by
    rewriting the lifecycle state, so the mapping stays unambiguous.
    """
    return ExecutionState.FAILED if has_errors else ExecutionState.COMPLETED


def state_sequence() -> tuple[ExecutionState, ...]:
    """Return the canonical happy-path sequence of states."""
    return PIPELINE_SEQUENCE


def active_state_sequence() -> tuple[ExecutionState, ...]:
    """Return the active states, in pipeline order."""
    return ACTIVE_STATES


@dataclass(frozen=True)
class ExecutionStateTransition:
    """One recorded, validated hop through the execution state machine.

    Parameters
    ----------
    sequence:
        Zero-based position of the hop within the run.  Transitions are
        recorded in ascending ``sequence`` order.
    source:
        The state the run was in before the hop.
    target:
        The state the run entered.  Always a legal successor of ``source``.
    """

    sequence: int
    source: ExecutionState
    target: ExecutionState

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise InvalidExecutionModelError(
                "sequence must be an int for ExecutionStateTransition"
            )
        if self.sequence < 0:
            raise InvalidExecutionModelError(
                "sequence must be non-negative for ExecutionStateTransition"
            )
        if not isinstance(self.source, ExecutionState):
            raise InvalidExecutionModelError(
                "source must be an ExecutionState"
            )
        if not isinstance(self.target, ExecutionState):
            raise InvalidExecutionModelError(
                "target must be an ExecutionState"
            )
        if not can_transition(self.source, self.target):
            raise InvalidExecutionStateTransitionError(
                f"illegal execution transition {self.source.value!r} -> "
                f"{self.target.value!r}"
            )

    @property
    def is_terminal(self) -> bool:
        """Return whether this hop entered a terminal state."""
        return self.target.is_terminal

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "sequence": self.sequence,
            "source": self.source.value,
            "target": self.target.value,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
    ) -> ExecutionStateTransition:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        sequence = data.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise InvalidExecutionModelError("expected int for 'sequence'")
        return cls(
            sequence=sequence,
            source=ExecutionState.from_value(data.get("source")),
            target=ExecutionState.from_value(data.get("target")),
        )


__all__ = [
    "ACTIVE_STATES",
    "ALLOWED_TRANSITIONS",
    "EXECUTION_STATE_SCHEMA_VERSION",
    "PIPELINE_SEQUENCE",
    "TERMINAL_STATES",
    "ExecutionState",
    "ExecutionStateTransition",
    "active_state_sequence",
    "can_transition",
    "is_active_state",
    "is_terminal_state",
    "resolve_final_state",
    "state_sequence",
    "successor_state",
    "transition_state",
]
