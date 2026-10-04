"""Immutable value objects for the Research Intelligence Loop — Phase 8 Sprint 7.

Sprint 7 turns the straight-line Sprint 5 pipeline into a closed loop::

    Plan -> Discover -> Collect -> Validate
         -> Update Memory
         -> Identify Remaining Knowledge Gaps
         -> Generate Revised PlannerInput
         -> Re-plan -> ... until completion or budget reached

This module owns the *vocabulary* of that loop and nothing else.  It holds
the policy that bounds it, the gaps it reasons about, the memory context it
reads back, the per-iteration record it preserves, and the metrics it
reports:

* :class:`GapKind` — why a residual knowledge gap still exists.
* :class:`TopicKnowledge` — how much memory knows about one topic.
* :class:`TopicInsight` — the memory-derived reading of one topic.
* :class:`MemoryContext` — the planner's memory-derived view of the company.
* :class:`ResidualGap` — one unresolved gap, with the evidence behind it.
* :class:`ResidualAnalysis` — every gap found after one iteration.
* :class:`FollowUpTask` — the deterministic follow-up work for one topic.
* :class:`IterationPolicy` — the configurable execution limits.
* :class:`LoopDecision` / :class:`TerminationReason` — deterministic stopping.
* :class:`IterationStatus` / :class:`IterationRecord` — one preserved cycle.
* :class:`ResearchIterationHistory` — every cycle, in order.
* :class:`LoopMetrics` — the deterministic metrics for the whole loop.
* :class:`MemoryUpdate` — the receipt for one memory write.

Design constraints honoured by every model here:

* **Immutable** — frozen dataclasses only; no setters, no mutable
  collections (tuples throughout).
* **Serializable** — a JSON-ready :meth:`to_dict` paired with a strict
  :meth:`from_dict` for every single model, so a whole loop round-trips
  losslessly.
* **Deterministic** — every derived value is a pure function of the stored
  fields.  No wall-clock, no randomness, and no dictionary iteration order
  leaking into output.
* **Free of clocks** — staleness is expressed as a caller-supplied integer
  ``age_days`` compared against a caller-supplied freshness requirement.
  Nothing here reads a date, so "the same inputs always produce the same
  output" survives contact with real deployments.

Like the Sprint 5 execution models, the private ``_require_*``
deserialization helpers below intentionally mirror the ones in
:mod:`predictron_engine.research.models` and
:mod:`predictron_engine.research.execution_models`; keeping them local is the
established convention in this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum

from predictron_engine.research.exceptions import InvalidIterationModelError
from predictron_engine.research.execution_models import (
    EMPTY_COVERAGE_RATIO,
    EMPTY_MEAN_CONFIDENCE,
    RATIO_PRECISION,
    ExecutionError,
    ExecutionStep,
    ExecutionWarning,
)
from predictron_engine.research.execution_state import ExecutionState
from predictron_engine.research.models import PlannerInput
from predictron_engine.research.validation_models import content_digest

#: Serialization schema version for every loop value object.
ITERATION_SCHEMA_VERSION = "1.0.0"

#: Default iteration budget: one pass, which is the Sprint 5 behaviour.
DEFAULT_MAX_ITERATIONS = 1

#: Default required mean confidence before research counts as complete.
DEFAULT_MINIMUM_CONFIDENCE = 0.0

#: Default required topic coverage before research counts as complete.
DEFAULT_MINIMUM_COVERAGE = 1.0

#: Confidence at or above which evidence stops being a weak spot.
#:
#: Deliberately identical to the ``MEDIUM`` boundary of
#: :meth:`~predictron_engine.research.validation_models.EvidenceConfidence.from_score`,
#: so "low confidence" means exactly "would band below medium".
LOW_CONFIDENCE_THRESHOLD = 0.55

#: Evidence items a topic needs before it stops being under-covered.
MINIMUM_EVIDENCE_PER_TOPIC = 1

#: Ratio reported when nothing was planned and nothing could be measured.
EMPTY_PROGRESS_RATIO = 0.0


# ----------------------------------------------------------------------
# Gap vocabulary
# ----------------------------------------------------------------------


class GapKind(str, Enum):
    """Why a residual knowledge gap still exists after an iteration.

    The declaration order is the urgency order, most urgent first, so
    ``tuple(GapKind)`` is itself a deterministic ranking and gap lists can
    be sorted without a separate comparison table.

    * ``MISSING`` — no evidence at all was collected for the topic.
    * ``UNRESOLVED`` — the task could not be dispatched to a collector.
    * ``FAILED_VALIDATION`` — collected evidence failed validation.
    * ``CONFLICTING`` — the topic's evidence contradicts itself.
    * ``LOW_CONFIDENCE`` — the topic's evidence validated but scored weakly.
    * ``INSUFFICIENT`` — the topic has fewer evidence items than required.
    * ``STALE`` — what memory holds is older than the topic's freshness
      requirement.  The only non-blocking kind: stale knowledge is a reason
      to refresh, never a reason to declare research incomplete.
    """

    MISSING = "missing"
    UNRESOLVED = "unresolved"
    FAILED_VALIDATION = "failed_validation"
    CONFLICTING = "conflicting"
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT = "insufficient"
    STALE = "stale"

    def order_value(self) -> int:
        """Return a deterministic rank; lower means more urgent."""
        return tuple(GapKind).index(self)

    @property
    def is_blocking(self) -> bool:
        """Return whether this kind prevents research from being complete."""
        return self is not GapKind.STALE

    @property
    def is_actionable(self) -> bool:
        """Return whether another iteration could plausibly close this kind.

        ``UNRESOLVED`` is blocking but not actionable: dispatch found no
        collector for the task, and the collector registry does not change
        between passes, so re-planning the same topic reproduces the identical
        gap.  ``STALE`` is not actionable because it is not blocking.  Every
        other kind is a finding the next pass can act on.
        """
        return self.is_blocking and self is not GapKind.UNRESOLVED


class TopicKnowledge(str, Enum):
    """How much is already known about one research topic.

    * ``KNOWN`` — complete, trusted evidence exists.
    * ``PARTIAL`` — some evidence exists but it is thin or weak.
    * ``STALE`` — evidence exists but is older than its freshness window.
    * ``CONFLICTING`` — evidence exists and contradicts itself.
    * ``MISSING`` — nothing at all is known.
    """

    KNOWN = "known"
    PARTIAL = "partial"
    STALE = "stale"
    CONFLICTING = "conflicting"
    MISSING = "missing"

    @property
    def is_satisfied(self) -> bool:
        """Return whether this state needs no further research."""
        return self is TopicKnowledge.KNOWN


class TerminationReason(str, Enum):
    """Why the research loop stopped iterating.

    * ``ITERATION_REQUIRED`` — the loop decided another iteration is
      warranted.  This is the only reason that accompanies a decision to
      *continue*; it is deliberately distinct from every stopping reason.
    * ``RESEARCH_COMPLETE`` — every required topic is complete and both
      thresholds are met.
    * ``PARTIAL_COMPLETION_ALLOWED`` — thresholds are met but blocking gaps
      remain, and the policy permits finishing in that state.
    * ``CONVERGED`` — residual gaps remain but none of them can be acted on,
      so another iteration would repeat the same work.
    * ``MAX_ITERATIONS_REACHED`` — the iteration budget was spent.
    * ``EXECUTION_FAILED`` — the pass recorded a fatal stage error.
    """

    ITERATION_REQUIRED = "iteration_required"
    RESEARCH_COMPLETE = "research_complete"
    PARTIAL_COMPLETION_ALLOWED = "partial_completion_allowed"
    CONVERGED = "converged"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    EXECUTION_FAILED = "execution_failed"

    @property
    def is_success(self) -> bool:
        """Return whether this reason means the goal was reached."""
        return self in (
            TerminationReason.RESEARCH_COMPLETE,
            TerminationReason.PARTIAL_COMPLETION_ALLOWED,
        )

    @property
    def is_stopping(self) -> bool:
        """Return whether this reason ends the loop."""
        return self is not TerminationReason.ITERATION_REQUIRED


class IterationStatus(str, Enum):
    """Deterministic outcome of one pipeline pass inside the loop."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @classmethod
    def for_state(cls, state: ExecutionState) -> IterationStatus:
        """Return the status implied by a pass's terminal ``state``."""
        return cls.FAILED if state is ExecutionState.FAILED else cls.SUCCEEDED


# ----------------------------------------------------------------------
# Iteration policy
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class IterationPolicy:
    """Immutable, fully serialized bounds on the research loop.

    Parameters
    ----------
    max_iterations:
        Hard ceiling on how many planning passes may run.  The default of
        ``1`` is exactly the Sprint 5 single-pass behaviour, which is what
        keeps opting into the loop an explicit decision.
    stop_when_complete:
        Whether the loop may stop early as soon as the completion
        thresholds are met.  When ``False`` the loop still stops on
        ``MAX_ITERATIONS_REACHED``, on a failed pass, or when nothing is
        left to act on — it simply refuses to stop *early*.
    minimum_confidence:
        Mean validated confidence the research must reach, in ``[0, 1]``.
    minimum_coverage:
        Fraction of planned topics that must produce evidence, in
        ``[0, 1]``.
    allow_partial_completion:
        Whether the loop may finish while blocking gaps remain, provided the
        thresholds are met.  When ``False`` the loop keeps iterating until
        the thresholds and the gaps agree or the budget runs out.
    minimum_evidence_per_topic:
        Evidence items a topic needs before it stops being under-covered.
    confidence_threshold:
        Confidence at or above which a topic's evidence stops being a weak
        spot.
    use_research_memory:
        Whether an injected
        :class:`~predictron_engine.research.memory.ResearchMemory` is written
        after every pass and read back into the next planner input.

    Determinism
    -----------
    The same policy and the same planner input always produce the same
    execution sequence: every field is a plain value, every comparison is a
    pure function of stored fields, and nothing here consults a clock.
    """

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    stop_when_complete: bool = True
    minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE
    minimum_coverage: float = DEFAULT_MINIMUM_COVERAGE
    allow_partial_completion: bool = True
    minimum_evidence_per_topic: int = MINIMUM_EVIDENCE_PER_TOPIC
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    use_research_memory: bool = True

    def __post_init__(self) -> None:
        _require_non_negative_int(self.max_iterations, "max_iterations")
        if self.max_iterations < 1:
            raise InvalidIterationModelError(
                "max_iterations must be at least 1 for IterationPolicy"
            )
        for field_name in (
            "stop_when_complete",
            "allow_partial_completion",
            "use_research_memory",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise InvalidIterationModelError(
                    f"{field_name} must be a bool for IterationPolicy"
                )
        _require_non_negative_int(
            self.minimum_evidence_per_topic, "minimum_evidence_per_topic"
        )
        for field_name in (
            "minimum_confidence",
            "minimum_coverage",
            "confidence_threshold",
        ):
            _require_unit_interval(getattr(self, field_name), field_name)

    @property
    def is_iterative(self) -> bool:
        """Return whether the policy allows more than one pass."""
        return self.max_iterations > 1

    def allows_iteration(self, iteration: int) -> bool:
        """Return whether another pass may follow ``iteration``."""
        _require_non_negative_int(iteration, "iteration")
        return iteration < self.max_iterations

    @property
    def is_deterministic(self) -> bool:
        """Return whether this policy guarantees deterministic output."""
        return True

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "stop_when_complete": self.stop_when_complete,
            "minimum_confidence": self.minimum_confidence,
            "minimum_coverage": self.minimum_coverage,
            "allow_partial_completion": self.allow_partial_completion,
            "minimum_evidence_per_topic": self.minimum_evidence_per_topic,
            "confidence_threshold": self.confidence_threshold,
            "use_research_memory": self.use_research_memory,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> IterationPolicy:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            max_iterations=_require_int(data, "max_iterations"),
            stop_when_complete=_require_bool(data, "stop_when_complete"),
            minimum_confidence=_require_float(data, "minimum_confidence"),
            minimum_coverage=_require_float(data, "minimum_coverage"),
            allow_partial_completion=_require_bool(
                data, "allow_partial_completion"
            ),
            minimum_evidence_per_topic=_require_int(
                data, "minimum_evidence_per_topic"
            ),
            confidence_threshold=_require_float(data, "confidence_threshold"),
            use_research_memory=_require_bool(data, "use_research_memory"),
        )


@dataclass(frozen=True)
class LoopDecision:
    """The deterministic answer to "should another iteration run?".

    ``proceed`` carries the whole decision: ``True`` means run another pass
    and is always paired with
    :attr:`TerminationReason.ITERATION_REQUIRED`, while ``False`` means stop
    and always names the stopping reason.  The two halves can never
    disagree, and :meth:`__post_init__` enforces exactly that.
    """

    proceed: bool
    reason: TerminationReason
    explanation: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.proceed, bool):
            raise InvalidIterationModelError(
                "proceed must be a bool for LoopDecision"
            )
        if not isinstance(self.reason, TerminationReason):
            raise InvalidIterationModelError(
                "reason must be a TerminationReason for LoopDecision"
            )
        if self.proceed and self.reason.is_stopping:
            raise InvalidIterationModelError(
                "a proceeding decision must not name a stopping reason"
            )
        if not self.proceed and not self.reason.is_stopping:
            raise InvalidIterationModelError(
                "a stopping decision must name a stopping reason"
            )
        object.__setattr__(self, "explanation", self.explanation.strip())

    @classmethod
    def continue_loop(cls, explanation: str = "") -> LoopDecision:
        """Return the decision to run one more iteration."""
        return cls(
            proceed=True,
            reason=TerminationReason.ITERATION_REQUIRED,
            explanation=explanation,
        )

    @classmethod
    def stop(cls, reason: TerminationReason, explanation: str = "") -> LoopDecision:
        """Return the decision to stop iterating."""
        if not isinstance(reason, TerminationReason):
            raise InvalidIterationModelError(
                "reason must be a TerminationReason for LoopDecision"
            )
        if not reason.is_stopping:
            raise InvalidIterationModelError(
                "stop() requires a stopping reason for LoopDecision"
            )
        return cls(proceed=False, reason=reason, explanation=explanation)

    @property
    def should_stop(self) -> bool:
        """Return whether the loop must stop iterating."""
        return not self.proceed

    @property
    def is_terminal(self) -> bool:
        """Return whether this decision ends the loop."""
        return self.should_stop

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "proceed": self.proceed,
            "reason": self.reason.value,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> LoopDecision:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            proceed=_require_bool(data, "proceed"),
            reason=TerminationReason(_require_str(data, "reason")),
            explanation=_require_str(data, "explanation", default=""),
        )


# ----------------------------------------------------------------------
# Memory-derived planner context
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TopicInsight:
    """The memory-derived reading of exactly one research topic.

    ``age_days`` and ``freshness_requirement_days`` are both caller-supplied
    integers, which is what keeps staleness deterministic: the engine never
    reads a date, it compares two numbers the caller already had.
    """

    topic_id: str
    knowledge: TopicKnowledge
    evidence_count: int = 0
    mean_confidence: float = 0.0
    conflicts: int = 0
    age_days: int = 0
    freshness_requirement_days: int = 0
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.topic_id, str) or not self.topic_id.strip():
            raise InvalidIterationModelError(
                "topic_id must be a non-empty string for TopicInsight"
            )
        if not isinstance(self.knowledge, TopicKnowledge):
            raise InvalidIterationModelError(
                "knowledge must be a TopicKnowledge for TopicInsight"
            )
        for field_name in ("evidence_count", "conflicts", "age_days"):
            _require_non_negative_int(getattr(self, field_name), field_name)
        _require_non_negative_int(
            self.freshness_requirement_days, "freshness_requirement_days"
        )
        _require_unit_interval(self.mean_confidence, "mean_confidence")
        object.__setattr__(self, "topic_id", self.topic_id.strip())
        object.__setattr__(self, "source", self.source.strip())

    @property
    def is_stale(self) -> bool:
        """Return whether this insight is past its freshness requirement."""
        return (
            self.freshness_requirement_days > 0
            and self.age_days > self.freshness_requirement_days
        )

    @property
    def is_conflicting(self) -> bool:
        """Return whether memory holds contradictory knowledge."""
        return self.knowledge is TopicKnowledge.CONFLICTING

    @property
    def needs_research(self) -> bool:
        """Return whether this topic still warrants further research."""
        return not self.knowledge.is_satisfied

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "topic_id": self.topic_id,
            "knowledge": self.knowledge.value,
            "evidence_count": self.evidence_count,
            "mean_confidence": self.mean_confidence,
            "conflicts": self.conflicts,
            "age_days": self.age_days,
            "freshness_requirement_days": self.freshness_requirement_days,
            "is_stale": self.is_stale,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TopicInsight:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            topic_id=_require_str(data, "topic_id"),
            knowledge=TopicKnowledge(_require_str(data, "knowledge")),
            evidence_count=_require_int(data, "evidence_count"),
            mean_confidence=_require_float(data, "mean_confidence"),
            conflicts=_require_int(data, "conflicts"),
            age_days=_require_int(data, "age_days"),
            freshness_requirement_days=_require_int(
                data, "freshness_requirement_days"
            ),
            source=_require_str(data, "source", default=""),
        )


@dataclass(frozen=True)
class MemoryContext:
    """Everything memory already knows, as planner input context.

    This is the read side of the closed loop.  The planner already honours
    ``known_topics`` and ``partial_topics``; this value is what derives them
    from
    :class:`~predictron_engine.research.memory.ResearchMemory` instead of
    from a human, which is how memory reduces unnecessary future research
    without changing a single planner rule.

    The derived properties are mutually exclusive by construction: a topic
    appears in exactly one of :attr:`known_topics`,
    :attr:`partial_topics`, :attr:`stale_topics`,
    :attr:`conflicting_topics`, or :attr:`missing_topics`.
    """

    insights: tuple[TopicInsight, ...] = ()
    memory_key: str = ""
    company_name: str = ""

    def __post_init__(self) -> None:
        entries = tuple(self.insights)
        for entry in entries:
            if not isinstance(entry, TopicInsight):
                raise InvalidIterationModelError(
                    "insights entries must be TopicInsight instances for "
                    "MemoryContext"
                )
        seen: set[str] = set()
        for entry in entries:
            if entry.topic_id in seen:
                raise InvalidIterationModelError(
                    f"duplicate topic insight for: {entry.topic_id!r}"
                )
            seen.add(entry.topic_id)
        object.__setattr__(self, "insights", entries)
        object.__setattr__(self, "memory_key", self.memory_key.strip())
        object.__setattr__(self, "company_name", self.company_name.strip())

    # -- Derived views -------------------------------------------------

    @property
    def topic_ids(self) -> tuple[str, ...]:
        """Return the topic identifiers this context describes."""
        return tuple(insight.topic_id for insight in self.insights)

    def _topics_with(self, knowledge: TopicKnowledge) -> tuple[str, ...]:
        return tuple(
            insight.topic_id
            for insight in self.insights
            if insight.knowledge is knowledge
        )

    @property
    def known_topics(self) -> tuple[str, ...]:
        """Return topics memory holds complete, trusted evidence for."""
        return self._topics_with(TopicKnowledge.KNOWN)

    @property
    def partial_topics(self) -> tuple[str, ...]:
        """Return topics memory holds only thin or weak evidence for."""
        return self._topics_with(TopicKnowledge.PARTIAL)

    @property
    def stale_topics(self) -> tuple[str, ...]:
        """Return topics whose evidence is past its freshness window."""
        return tuple(
            insight.topic_id for insight in self.insights if insight.is_stale
        )

    @property
    def conflicting_topics(self) -> tuple[str, ...]:
        """Return topics memory holds contradictory evidence for."""
        return self._topics_with(TopicKnowledge.CONFLICTING)

    @property
    def missing_topics(self) -> tuple[str, ...]:
        """Return topics memory holds nothing at all for."""
        return self._topics_with(TopicKnowledge.MISSING)

    @property
    def satisfied_topics(self) -> tuple[str, ...]:
        """Return topics that need no further research."""
        return tuple(
            insight.topic_id
            for insight in self.insights
            if insight.knowledge.is_satisfied
        )

    @property
    def reusable_topics(self) -> tuple[str, ...]:
        """Return topics where memory saves work: known or partial."""
        return self.known_topics + self.partial_topics

    @property
    def topics_reused(self) -> int:
        """Return how many topics memory lets the loop skip."""
        return len(self.reusable_topics)

    def is_empty(self) -> bool:
        """Return whether this context describes nothing at all."""
        return not self.insights

    def insight_for(self, topic_id: str) -> TopicInsight | None:
        """Return the insight for ``topic_id`` or ``None``."""
        for insight in self.insights:
            if insight.topic_id == topic_id:
                return insight
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "memory_key": self.memory_key,
            "company_name": self.company_name,
            "insights": [insight.to_dict() for insight in self.insights],
            "known_topics": list(self.known_topics),
            "partial_topics": list(self.partial_topics),
            "stale_topics": list(self.stale_topics),
            "conflicting_topics": list(self.conflicting_topics),
            "missing_topics": list(self.missing_topics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MemoryContext:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            insights=tuple(
                TopicInsight.from_dict(item)
                for item in _require_list_of_dicts(data, "insights")
            ),
            memory_key=_require_str(data, "memory_key", default=""),
            company_name=_require_str(data, "company_name", default=""),
        )


# ----------------------------------------------------------------------
# Residual gaps
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualGap:
    """One knowledge gap that survived an iteration.

    The value is a *finding*, not an instruction: it records what is still
    wrong, how much evidence backs the finding, and which iteration found
    it.  Turning findings into work is
    :class:`~predictron_engine.research.residual.ResidualGapEngine`'s job.
    """

    topic_id: str
    kind: GapKind
    reason: str = ""
    iteration: int = 0
    task_id: str = ""
    evidence_count: int = 0
    confidence: float = 0.0
    conflicts: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.topic_id, str) or not self.topic_id.strip():
            raise InvalidIterationModelError(
                "topic_id must be a non-empty string for ResidualGap"
            )
        if not isinstance(self.kind, GapKind):
            raise InvalidIterationModelError(
                "kind must be a GapKind for ResidualGap"
            )
        for field_name in ("iteration", "evidence_count", "conflicts"):
            _require_non_negative_int(getattr(self, field_name), field_name)
        _require_unit_interval(self.confidence, "confidence")
        object.__setattr__(self, "topic_id", self.topic_id.strip())
        object.__setattr__(self, "reason", self.reason.strip())
        object.__setattr__(self, "task_id", self.task_id.strip())

    @property
    def is_blocking(self) -> bool:
        """Return whether this gap prevents research from being complete."""
        return self.kind.is_blocking

    @property
    def is_actionable(self) -> bool:
        """Return whether another iteration could plausibly close this gap."""
        return self.kind.is_actionable

    @property
    def order_key(self) -> tuple[int, str, str]:
        """Return the deterministic sort key for a gap list."""
        return (self.kind.order_value(), self.topic_id, self.reason)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "topic_id": self.topic_id,
            "kind": self.kind.value,
            "reason": self.reason,
            "iteration": self.iteration,
            "task_id": self.task_id,
            "evidence_count": self.evidence_count,
            "confidence": self.confidence,
            "conflicts": self.conflicts,
            "is_blocking": self.is_blocking,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResidualGap:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            topic_id=_require_str(data, "topic_id"),
            kind=GapKind(_require_str(data, "kind")),
            reason=_require_str(data, "reason", default=""),
            iteration=_require_int(data, "iteration"),
            task_id=_require_str(data, "task_id", default=""),
            evidence_count=_require_int(data, "evidence_count"),
            confidence=_require_float(data, "confidence"),
            conflicts=_require_int(data, "conflicts"),
        )


@dataclass(frozen=True)
class ResidualAnalysis:
    """Every residual knowledge gap found after one iteration.

    The analysis is self-contained: it carries the thresholds it was judged
    against, so :meth:`is_complete` and :meth:`thresholds_met` answer the
    completion question without the caller re-deriving anything.  Its
    ``analysis_id`` is a content hash over the deterministic payload, which
    makes "the same iteration always yields the same analysis" observable
    rather than merely asserted.
    """

    analysis_id: str
    iteration: int
    gaps: tuple[ResidualGap, ...] = ()
    planned_topics: tuple[str, ...] = ()
    completed_topics: tuple[str, ...] = ()
    completed_tasks: tuple[str, ...] = ()
    unresolved_tasks: tuple[str, ...] = ()
    failed_tasks: tuple[str, ...] = ()
    evidence_count: int = 0
    mean_confidence: float = EMPTY_MEAN_CONFIDENCE
    coverage_ratio: float = EMPTY_COVERAGE_RATIO
    conflicts: int = 0
    failed_validations: int = 0
    minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE
    minimum_coverage: float = DEFAULT_MINIMUM_COVERAGE
    errors: tuple[ExecutionError, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.analysis_id, str) or not self.analysis_id.strip():
            raise InvalidIterationModelError(
                "analysis_id must be a non-empty string for ResidualAnalysis"
            )
        gaps = tuple(self.gaps)
        for gap in gaps:
            if not isinstance(gap, ResidualGap):
                raise InvalidIterationModelError(
                    "gaps entries must be ResidualGap instances for "
                    "ResidualAnalysis"
                )
        object.__setattr__(self, "gaps", gaps)
        _require_non_negative_int(self.iteration, "iteration")
        for field_name in (
            "evidence_count",
            "conflicts",
            "failed_validations",
        ):
            _require_non_negative_int(getattr(self, field_name), field_name)
        _require_unit_interval(self.mean_confidence, "mean_confidence")
        _require_unit_interval(self.coverage_ratio, "coverage_ratio")
        object.__setattr__(
            self, "minimum_confidence", _round(self.minimum_confidence)
        )
        object.__setattr__(
            self, "minimum_coverage", _round(self.minimum_coverage)
        )
        for field_name in (
            "planned_topics",
            "completed_topics",
            "completed_tasks",
            "unresolved_tasks",
            "failed_tasks",
        ):
            object.__setattr__(
                self, field_name, _clean_ids(getattr(self, field_name))
            )
        errors = tuple(self.errors)
        for entry in errors:
            if not isinstance(entry, ExecutionError):
                raise InvalidIterationModelError(
                    "errors entries must be ExecutionError instances for "
                    "ResidualAnalysis"
                )
        object.__setattr__(self, "errors", errors)

    # -- Derived views -------------------------------------------------

    @property
    def gap_count(self) -> int:
        """Return how many residual gaps were found."""
        return len(self.gaps)

    @property
    def topics_with_gaps(self) -> tuple[str, ...]:
        """Return the distinct topics that still have a gap, sorted."""
        return tuple(sorted({gap.topic_id for gap in self.gaps}))

    @property
    def blocking_gaps(self) -> tuple[ResidualGap, ...]:
        """Return the gaps that prevent completion, in sorted order."""
        return tuple(gap for gap in self.gaps if gap.is_blocking)

    @property
    def stale_gaps(self) -> tuple[ResidualGap, ...]:
        """Return the non-blocking staleness gaps, in sorted order."""
        return tuple(gap for gap in self.gaps if not gap.is_blocking)

    @property
    def has_blocking_gaps(self) -> bool:
        """Return whether any gap prevents completion."""
        return any(gap.is_blocking for gap in self.gaps)

    @property
    def actionable_gaps(self) -> tuple[ResidualGap, ...]:
        """Return the blocking gaps another iteration could plausibly close."""
        return tuple(gap for gap in self.gaps if gap.is_actionable)

    @property
    def has_actionable_gaps(self) -> bool:
        """Return whether another iteration could possibly close a gap.

        This is deliberately narrower than :attr:`has_blocking_gaps`: a gap
        whose kind is not actionable (``UNRESOLVED``) blocks completion but
        cannot be closed by another pass, so it is not counted here.
        """
        return any(gap.is_actionable for gap in self.gaps)

    @property
    def is_empty(self) -> bool:
        """Return whether no residual gap remains at all."""
        return not self.gaps

    @property
    def has_errors(self) -> bool:
        """Return whether the pass recorded a fatal stage error."""
        return bool(self.errors)

    @property
    def confidence_met(self) -> bool:
        """Return whether the mean confidence reached its minimum."""
        return self.mean_confidence >= self.minimum_confidence

    @property
    def coverage_met(self) -> bool:
        """Return whether topic coverage reached its minimum."""
        return self.coverage_ratio >= self.minimum_coverage

    @property
    def thresholds_met(self) -> bool:
        """Return whether both completion thresholds were reached."""
        return self.confidence_met and self.coverage_met

    @property
    def is_complete(self) -> bool:
        """Return whether research counts as complete."""
        return self.thresholds_met and not self.has_blocking_gaps

    def gaps_of(self, kind: GapKind) -> tuple[ResidualGap, ...]:
        """Return every gap of one ``kind``, in sorted order."""
        return tuple(gap for gap in self.gaps if gap.kind is kind)

    def has(self, kind: GapKind) -> bool:
        """Return whether at least one gap of ``kind`` was found."""
        return any(gap.kind is kind for gap in self.gaps)

    def gap_counts(self) -> dict[str, int]:
        """Return gap counts keyed by kind value, in enum order."""
        counts: dict[str, int] = {}
        for kind in GapKind:
            counts[kind.value] = len(self.gaps_of(kind))
        return counts

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "iteration": self.iteration,
            "gaps": [gap.to_dict() for gap in self.gaps],
            "planned_topics": list(self.planned_topics),
            "completed_topics": list(self.completed_topics),
            "completed_tasks": list(self.completed_tasks),
            "unresolved_tasks": list(self.unresolved_tasks),
            "failed_tasks": list(self.failed_tasks),
            "evidence_count": self.evidence_count,
            "mean_confidence": self.mean_confidence,
            "coverage_ratio": self.coverage_ratio,
            "conflicts": self.conflicts,
            "failed_validations": self.failed_validations,
            "minimum_confidence": self.minimum_confidence,
            "minimum_coverage": self.minimum_coverage,
            "thresholds_met": self.thresholds_met,
            "is_complete": self.is_complete,
            "errors": [entry.to_dict() for entry in self.errors],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResidualAnalysis:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            analysis_id=_require_str(data, "analysis_id"),
            iteration=_require_int(data, "iteration"),
            gaps=tuple(
                ResidualGap.from_dict(item)
                for item in _require_list_of_dicts(data, "gaps")
            ),
            planned_topics=tuple(_require_str_list(data, "planned_topics")),
            completed_topics=tuple(
                _require_str_list(data, "completed_topics")
            ),
            completed_tasks=tuple(_require_str_list(data, "completed_tasks")),
            unresolved_tasks=tuple(
                _require_str_list(data, "unresolved_tasks")
            ),
            failed_tasks=tuple(_require_str_list(data, "failed_tasks")),
            evidence_count=_require_int(data, "evidence_count"),
            mean_confidence=_require_float(data, "mean_confidence"),
            coverage_ratio=_require_float(data, "coverage_ratio"),
            conflicts=_require_int(data, "conflicts"),
            failed_validations=_require_int(data, "failed_validations"),
            minimum_confidence=_require_float(data, "minimum_confidence"),
            minimum_coverage=_require_float(data, "minimum_coverage"),
            errors=tuple(
                ExecutionError.from_dict(item)
                for item in _require_list_of_dicts(data, "errors")
            ),
        )


@dataclass(frozen=True)
class FollowUpTask:
    """One deterministic unit of follow-up work.

    Follow-up tasks are *requests*, not executable
    :class:`~predictron_engine.research.models.ResearchTask` values: priority
    scoring, dependency ordering, and source categories all belong to the
    Research Planner.  The engine therefore produces one request per topic
    and lets the existing planner turn the revised
    :class:`~predictron_engine.research.models.PlannerInput` into real tasks,
    which is what guarantees a follow-up plan never restates the priorities
    the planner already computed.
    """

    task_id: str
    topic_id: str
    kind: GapKind
    reason: str = ""
    title: str = ""
    iteration: int = 0
    evidence_count: int = 0
    confidence: float = 0.0
    conflicts: int = 0

    def __post_init__(self) -> None:
        for field_name in ("task_id", "topic_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidIterationModelError(
                    f"{field_name} must be a non-empty string for FollowUpTask"
                )
        if not isinstance(self.kind, GapKind):
            raise InvalidIterationModelError(
                "kind must be a GapKind for FollowUpTask"
            )
        _require_non_negative_int(self.iteration, "iteration")
        _require_non_negative_int(self.evidence_count, "evidence_count")
        _require_non_negative_int(self.conflicts, "conflicts")
        _require_unit_interval(self.confidence, "confidence")
        object.__setattr__(self, "task_id", self.task_id.strip())
        object.__setattr__(self, "topic_id", self.topic_id.strip())
        object.__setattr__(self, "reason", self.reason.strip())
        object.__setattr__(self, "title", self.title.strip())

    @property
    def is_blocking(self) -> bool:
        """Return whether this follow-up must be planned next."""
        return self.kind.is_blocking

    @property
    def order_key(self) -> tuple[int, str]:
        """Return the deterministic sort key for a follow-up list."""
        return (self.kind.order_value(), self.topic_id)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "task_id": self.task_id,
            "topic_id": self.topic_id,
            "kind": self.kind.value,
            "reason": self.reason,
            "title": self.title,
            "iteration": self.iteration,
            "evidence_count": self.evidence_count,
            "confidence": self.confidence,
            "conflicts": self.conflicts,
            "is_blocking": self.is_blocking,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> FollowUpTask:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            task_id=_require_str(data, "task_id"),
            topic_id=_require_str(data, "topic_id"),
            kind=GapKind(_require_str(data, "kind")),
            reason=_require_str(data, "reason", default=""),
            title=_require_str(data, "title", default=""),
            iteration=_require_int(data, "iteration"),
            evidence_count=_require_int(data, "evidence_count"),
            confidence=_require_float(data, "confidence"),
            conflicts=_require_int(data, "conflicts"),
        )


# ----------------------------------------------------------------------
# Memory receipts
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryUpdate:
    """What one iteration's write to research memory achieved.

    ``applied`` is ``False`` when the memory write could not be performed.
    The loop records that rather than raising, because a failed memory write
    degrades the *next* iteration rather than invalidating the current one —
    the same partial-completion contract the Sprint 5 orchestrator follows
    for stage failures.
    """

    applied: bool = False
    reports_written: int = 0
    reports_replayed: int = 0
    evidence_stored: int = 0
    duplicates: int = 0
    entities_created: int = 0
    entities_updated: int = 0
    relationships_created: int = 0
    history_entries: int = 0
    new_evidence_ids: tuple[str, ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.applied, bool):
            raise InvalidIterationModelError(
                "applied must be a bool for MemoryUpdate"
            )
        for entry in fields(self):
            if entry.name in ("applied", "new_evidence_ids", "error"):
                continue
            _require_non_negative_int(getattr(self, entry.name), entry.name)
        object.__setattr__(
            self, "new_evidence_ids", _clean_ids(self.new_evidence_ids)
        )
        object.__setattr__(self, "error", self.error.strip())

    @property
    def changed(self) -> bool:
        """Return whether the write altered what memory believes."""
        return bool(
            self.entities_created
            or self.entities_updated
            or self.relationships_created
            or self.history_entries
            or self.evidence_stored
        )

    @property
    def new_evidence_count(self) -> int:
        """Return how many genuinely new evidence items were stored."""
        return len(self.new_evidence_ids)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "applied": self.applied,
            "reports_written": self.reports_written,
            "reports_replayed": self.reports_replayed,
            "evidence_stored": self.evidence_stored,
            "duplicates": self.duplicates,
            "entities_created": self.entities_created,
            "entities_updated": self.entities_updated,
            "relationships_created": self.relationships_created,
            "history_entries": self.history_entries,
            "new_evidence_ids": list(self.new_evidence_ids),
            "new_evidence_count": self.new_evidence_count,
            "changed": self.changed,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MemoryUpdate:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            applied=_require_bool(data, "applied"),
            reports_written=_require_int(data, "reports_written"),
            reports_replayed=_require_int(data, "reports_replayed"),
            evidence_stored=_require_int(data, "evidence_stored"),
            duplicates=_require_int(data, "duplicates"),
            entities_created=_require_int(data, "entities_created"),
            entities_updated=_require_int(data, "entities_updated"),
            relationships_created=_require_int(data, "relationships_created"),
            history_entries=_require_int(data, "history_entries"),
            new_evidence_ids=tuple(_require_str_list(data, "new_evidence_ids")),
            error=_require_str(data, "error", default=""),
        )


# ----------------------------------------------------------------------
# Iteration history
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class IterationRecord:
    """Everything one pass of the loop achieved, preserved verbatim.

    This is the unit of execution history: the plan that ran, the tasks that
    completed, the tasks that did not, the validation findings, the memory
    write, and the residual gaps that either closed or opened the next
    iteration.  A record is fully serializable, so a loop's whole history
    survives a round trip through JSON.
    """

    iteration: int
    planner_input: PlannerInput
    plan_id: str = ""
    planned_tasks: tuple[str, ...] = ()
    planned_task_topics: tuple[str, ...] = ()
    completed_tasks: tuple[str, ...] = ()
    unresolved_tasks: tuple[str, ...] = ()
    new_evidence_ids: tuple[str, ...] = ()
    evidence_count: int = 0
    topics_completed: tuple[str, ...] = ()
    topics_incomplete: tuple[str, ...] = ()
    confidence: float = EMPTY_MEAN_CONFIDENCE
    coverage_ratio: float = EMPTY_COVERAGE_RATIO
    residual_gaps: tuple[ResidualGap, ...] = ()
    memory_update: MemoryUpdate = field(default_factory=MemoryUpdate)
    state: ExecutionState = ExecutionState.COMPLETED
    steps: tuple[ExecutionStep, ...] = ()
    warnings: tuple[ExecutionWarning, ...] = ()
    errors: tuple[ExecutionError, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.planner_input, PlannerInput):
            raise InvalidIterationModelError(
                "planner_input must be a PlannerInput for IterationRecord"
            )
        _require_non_negative_int(self.iteration, "iteration")
        _require_non_negative_int(self.evidence_count, "evidence_count")
        _require_unit_interval(self.confidence, "confidence")
        _require_unit_interval(self.coverage_ratio, "coverage_ratio")
        if not isinstance(self.state, ExecutionState):
            raise InvalidIterationModelError(
                "state must be an ExecutionState for IterationRecord"
            )
        if not isinstance(self.memory_update, MemoryUpdate):
            raise InvalidIterationModelError(
                "memory_update must be a MemoryUpdate for IterationRecord"
            )
        gaps = tuple(self.residual_gaps)
        for gap in gaps:
            if not isinstance(gap, ResidualGap):
                raise InvalidIterationModelError(
                    "residual_gaps entries must be ResidualGap instances"
                )
        object.__setattr__(self, "residual_gaps", gaps)
        steps = tuple(self.steps)
        for step in steps:
            if not isinstance(step, ExecutionStep):
                raise InvalidIterationModelError(
                    "steps entries must be ExecutionStep instances"
                )
        object.__setattr__(self, "steps", steps)
        warnings = tuple(self.warnings)
        for warning in warnings:
            if not isinstance(warning, ExecutionWarning):
                raise InvalidIterationModelError(
                    "warnings entries must be ExecutionWarning instances"
                )
        object.__setattr__(self, "warnings", warnings)
        errors = tuple(self.errors)
        for error in errors:
            if not isinstance(error, ExecutionError):
                raise InvalidIterationModelError(
                    "errors entries must be ExecutionError instances"
                )
        object.__setattr__(self, "errors", errors)
        for field_name in (
            "planned_tasks",
            "planned_task_topics",
            "completed_tasks",
            "unresolved_tasks",
            "new_evidence_ids",
            "topics_completed",
            "topics_incomplete",
        ):
            object.__setattr__(
                self, field_name, _clean_ids(getattr(self, field_name))
            )
        object.__setattr__(self, "plan_id", self.plan_id.strip())

    @property
    def status(self) -> IterationStatus:
        """Return the deterministic outcome of this pass."""
        return IterationStatus.for_state(self.state)

    @property
    def is_successful(self) -> bool:
        """Return whether the pass completed without a fatal error."""
        return self.status is IterationStatus.SUCCEEDED

    @property
    def gap_count(self) -> int:
        """Return how many residual gaps this pass left behind."""
        return len(self.residual_gaps)

    @property
    def new_evidence_count(self) -> int:
        """Return how many genuinely new evidence items this pass found."""
        return len(self.new_evidence_ids)

    @property
    def topic_count(self) -> int:
        """Return how many topics this pass researched."""
        return len(self.topics_completed) + len(self.topics_incomplete)

    @property
    def planned_topic_count(self) -> int:
        """Return how many distinct topics this pass planned research for."""
        return len(self.planned_task_topics)

    @property
    def coverage_complete(self) -> bool:
        """Return whether every topic this pass planned produced evidence."""
        return not self.topics_incomplete

    def step_names(self) -> tuple[str, ...]:
        """Return the pipeline steps this pass recorded, in order."""
        return tuple(step.name for step in self.steps)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "iteration": self.iteration,
            "status": self.status.value,
            "planner_input": self.planner_input.to_dict(),
            "plan_id": self.plan_id,
            "planned_tasks": list(self.planned_tasks),
            "planned_task_topics": list(self.planned_task_topics),
            "completed_tasks": list(self.completed_tasks),
            "unresolved_tasks": list(self.unresolved_tasks),
            "new_evidence_ids": list(self.new_evidence_ids),
            "evidence_count": self.evidence_count,
            "topics_completed": list(self.topics_completed),
            "topics_incomplete": list(self.topics_incomplete),
            "confidence": self.confidence,
            "coverage_ratio": self.coverage_ratio,
            "residual_gaps": [gap.to_dict() for gap in self.residual_gaps],
            "memory_update": self.memory_update.to_dict(),
            "state": self.state.value,
            "steps": [step.to_dict() for step in self.steps],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "errors": [error.to_dict() for error in self.errors],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> IterationRecord:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            iteration=_require_int(data, "iteration"),
            planner_input=PlannerInput.from_dict(
                _require_dict(data, "planner_input")
            ),
            plan_id=_require_str(data, "plan_id", default=""),
            planned_tasks=tuple(_require_str_list(data, "planned_tasks")),
            planned_task_topics=tuple(
                _require_str_list(data, "planned_task_topics")
            ),
            completed_tasks=tuple(_require_str_list(data, "completed_tasks")),
            unresolved_tasks=tuple(_require_str_list(data, "unresolved_tasks")),
            new_evidence_ids=tuple(
                _require_str_list(data, "new_evidence_ids")
            ),
            evidence_count=_require_int(data, "evidence_count"),
            topics_completed=tuple(_require_str_list(data, "topics_completed")),
            topics_incomplete=tuple(
                _require_str_list(data, "topics_incomplete")
            ),
            confidence=_require_float(data, "confidence"),
            coverage_ratio=_require_float(data, "coverage_ratio"),
            residual_gaps=tuple(
                ResidualGap.from_dict(item)
                for item in _require_list_of_dicts(data, "residual_gaps")
            ),
            memory_update=MemoryUpdate.from_dict(
                _require_dict(data, "memory_update")
            ),
            state=ExecutionState.from_value(
                _require_str(data, "state", default=ExecutionState.COMPLETED.value)
            ),
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
        )


@dataclass(frozen=True)
class ResearchIterationHistory:
    """The complete, ordered history of one research loop.

    The history is the loop's memory of itself: which iterations ran, what
    each of them planned and found, and why the loop stopped.  It is a
    frozen, fully serializable value, so two identical loops produce
    identical history documents.
    """

    iterations: tuple[IterationRecord, ...] = ()
    termination: TerminationReason = TerminationReason.MAX_ITERATIONS_REACHED
    policy: IterationPolicy = field(default_factory=IterationPolicy)
    analysis: ResidualAnalysis | None = None
    memory_context: MemoryContext | None = None

    def __post_init__(self) -> None:
        records = tuple(self.iterations)
        for record in records:
            if not isinstance(record, IterationRecord):
                raise InvalidIterationModelError(
                    "iterations entries must be IterationRecord instances"
                )
        object.__setattr__(self, "iterations", records)
        for index, record in enumerate(records, start=1):
            if record.iteration != index:
                raise InvalidIterationModelError(
                    "iteration numbers must be consecutive and 1-based for "
                    "ResearchIterationHistory"
                )
        if not isinstance(self.termination, TerminationReason):
            raise InvalidIterationModelError(
                "termination must be a TerminationReason for "
                "ResearchIterationHistory"
            )
        if not isinstance(self.policy, IterationPolicy):
            raise InvalidIterationModelError(
                "policy must be an IterationPolicy for "
                "ResearchIterationHistory"
            )
        if self.analysis is not None and not isinstance(
            self.analysis, ResidualAnalysis
        ):
            raise InvalidIterationModelError(
                "analysis must be a ResidualAnalysis or None for "
                "ResearchIterationHistory"
            )
        if self.memory_context is not None and not isinstance(
            self.memory_context, MemoryContext
        ):
            raise InvalidIterationModelError(
                "memory_context must be a MemoryContext or None for "
                "ResearchIterationHistory"
            )

    # -- Derived views -------------------------------------------------

    def __len__(self) -> int:
        """Return how many iterations ran."""
        return len(self.iterations)

    @property
    def count(self) -> int:
        """Return how many iterations ran."""
        return len(self.iterations)

    @property
    def is_empty(self) -> bool:
        """Return whether no iteration ever ran."""
        return not self.iterations

    @property
    def first(self) -> IterationRecord | None:
        """Return the first iteration's record, or ``None``."""
        return self.iterations[0] if self.iterations else None

    @property
    def last(self) -> IterationRecord | None:
        """Return the last iteration's record, or ``None``."""
        return self.iterations[-1] if self.iterations else None

    @property
    def executed_iterations(self) -> int:
        """Return how many iterations actually ran."""
        return len(self.iterations)

    @property
    def exhausted_budget(self) -> bool:
        """Return whether the loop stopped because it ran out of budget."""
        return (
            self.termination is TerminationReason.MAX_ITERATIONS_REACHED
            and self.count >= self.policy.max_iterations
        )

    @property
    def converged(self) -> bool:
        """Return whether the loop stopped because nothing was left to do."""
        return self.termination in (
            TerminationReason.RESEARCH_COMPLETE,
            TerminationReason.CONVERGED,
        )

    @property
    def is_complete(self) -> bool:
        """Return whether research finished without residual blocking gaps."""
        return self.termination is TerminationReason.RESEARCH_COMPLETE

    @property
    def is_partial(self) -> bool:
        """Return whether the loop stopped while gaps remained."""
        final = self.last
        return final is not None and bool(final.gap_count)

    @property
    def current_termination(self) -> TerminationReason:
        """Return the reason for the most recent decision, if any.

        The history's ``termination`` is the *final* reason the loop
        stopped; this is the reason recorded alongside the last iteration,
        which is ``ITERATION_REQUIRED`` when that iteration ran and more
        remained to be done.
        """
        if not self.iterations:
            return TerminationReason.ITERATION_REQUIRED
        return (
            self.termination
            if self.exhausted_budget or self.is_complete
            else TerminationReason.ITERATION_REQUIRED
        )

    @property
    def confidence_progression(self) -> tuple[float, ...]:
        """Return the mean confidence achieved by each iteration."""
        return tuple(record.confidence for record in self.iterations)

    @property
    def coverage_progression(self) -> tuple[float, ...]:
        """Return the topic coverage achieved by each iteration."""
        return tuple(record.coverage_ratio for record in self.iterations)

    @property
    def evidence_progression(self) -> tuple[int, ...]:
        """Return the evidence collected by each iteration."""
        return tuple(record.evidence_count for record in self.iterations)

    @property
    def confidence_improved(self) -> bool:
        """Return whether confidence rose strictly over the whole loop."""
        progression = self.confidence_progression
        return len(progression) > 1 and progression[-1] > progression[0]

    @property
    def topics_completed(self) -> tuple[str, ...]:
        """Return every topic any iteration completed, in first-seen order."""
        completed: list[str] = []
        for record in self.iterations:
            for topic_id in record.topics_completed:
                if topic_id not in completed:
                    completed.append(topic_id)
        return tuple(completed)

    @property
    def repeated_topics(self) -> tuple[str, ...]:
        """Return topics more than one iteration researched, sorted.

        A non-empty result is the signature of wasted work: the loop should
        converge on an empty value.
        """
        counts: dict[str, int] = {}
        for record in self.iterations:
            for topic_id in record.planned_task_topics:
                counts[topic_id] = counts.get(topic_id, 0) + 1
        return tuple(
            sorted(topic_id for topic_id, seen in counts.items() if seen > 1)
        )

    def record_for(self, iteration: int) -> IterationRecord | None:
        """Return the record for ``iteration`` or ``None``."""
        for record in self.iterations:
            if record.iteration == iteration:
                return record
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "schema_version": ITERATION_SCHEMA_VERSION,
            "iterations": [record.to_dict() for record in self.iterations],
            "termination": self.termination.value,
            "policy": self.policy.to_dict(),
            "analysis": (
                self.analysis.to_dict() if self.analysis is not None else None
            ),
            "memory_context": (
                self.memory_context.to_dict()
                if self.memory_context is not None
                else None
            ),
            "iteration_count": self.count,
            "confidence_progression": list(self.confidence_progression),
            "coverage_progression": list(self.coverage_progression),
            "topics_completed": list(self.topics_completed),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResearchIterationHistory:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        analysis_data = _optional_dict(data, "analysis")
        context_data = _optional_dict(data, "memory_context")
        policy_data = _optional_dict(data, "policy")
        return cls(
            iterations=tuple(
                IterationRecord.from_dict(item)
                for item in _require_list_of_dicts(data, "iterations")
            ),
            termination=TerminationReason(
                _require_str(data, "termination")
            ),
            policy=(
                IterationPolicy.from_dict(policy_data)
                if policy_data is not None
                else IterationPolicy()
            ),
            analysis=(
                ResidualAnalysis.from_dict(analysis_data)
                if analysis_data is not None
                else None
            ),
            memory_context=(
                MemoryContext.from_dict(context_data)
                if context_data is not None
                else None
            ),
        )


# ----------------------------------------------------------------------
# Loop metrics
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LoopMetrics:
    """Deterministic metrics for a whole research loop.

    Every field is a pure function of the
    :class:`ResearchIterationHistory` the metrics were derived from, so the
    metrics are reproducible and comparable across runs.  Ratios are
    rounded to
    :data:`~predictron_engine.research.execution_models.RATIO_PRECISION`
    decimals for exactly the reason the rest of the package rounds them:
    floating-point noise must never make two identical loops compare
    unequal.
    """

    iterations: int = 0
    tasks_planned: int = 0
    tasks_completed: int = 0
    tasks_unresolved: int = 0
    evidence_collected: int = 0
    new_evidence: int = 0
    redundant_evidence: int = 0
    residual_gaps_total: int = 0
    residual_gaps_final: int = 0
    topics_planned: int = 0
    topics_completed: int = 0
    confidence_progression: tuple[float, ...] = ()
    coverage_progression: tuple[float, ...] = ()
    confidence_initial: float = EMPTY_MEAN_CONFIDENCE
    confidence_final: float = EMPTY_MEAN_CONFIDENCE
    coverage_final: float = EMPTY_COVERAGE_RATIO
    planning_efficiency: float = EMPTY_PROGRESS_RATIO
    budget_used: float = EMPTY_PROGRESS_RATIO
    memory_topics_reused: int = 0
    memory_writes: int = 0
    repeated_topics: tuple[str, ...] = ()
    termination: TerminationReason = TerminationReason.MAX_ITERATIONS_REACHED

    def __post_init__(self) -> None:
        for field_name in (
            "iterations",
            "tasks_planned",
            "tasks_completed",
            "tasks_unresolved",
            "evidence_collected",
            "new_evidence",
            "redundant_evidence",
            "residual_gaps_total",
            "residual_gaps_final",
            "topics_planned",
            "topics_completed",
            "memory_topics_reused",
            "memory_writes",
        ):
            _require_non_negative_int(getattr(self, field_name), field_name)
        for field_name in (
            "confidence_initial",
            "confidence_final",
            "coverage_final",
            "planning_efficiency",
            "budget_used",
        ):
            value = _require_unit_interval(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, _round(value))
        object.__setattr__(
            self,
            "confidence_progression",
            tuple(_round(value) for value in self.confidence_progression),
        )
        object.__setattr__(
            self,
            "coverage_progression",
            tuple(_round(value) for value in self.coverage_progression),
        )
        object.__setattr__(
            self, "repeated_topics", _clean_ids(self.repeated_topics)
        )
        if not isinstance(self.termination, TerminationReason):
            raise InvalidIterationModelError(
                "termination must be a TerminationReason for LoopMetrics"
            )

    @property
    def is_single_iteration(self) -> bool:
        """Return whether the loop ran exactly one pass."""
        return self.iterations == 1

    @property
    def is_converged(self) -> bool:
        """Return whether the loop finished with no blocking gaps left."""
        return self.residual_gaps_final == 0

    @property
    def confidence_gain(self) -> float:
        """Return how much confidence the loop added, in ``[0, 1]``."""
        return _round(max(0.0, self.confidence_final - self.confidence_initial))

    @property
    def wasted_evidence_ratio(self) -> float:
        """Return redundant evidence over everything collected."""
        if not self.evidence_collected:
            return EMPTY_PROGRESS_RATIO
        return _round(self.redundant_evidence / self.evidence_collected)

    @property
    def repeated_work(self) -> bool:
        """Return whether the loop researched a topic more than once."""
        return bool(self.repeated_topics)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "iterations": self.iterations,
            "tasks_planned": self.tasks_planned,
            "tasks_completed": self.tasks_completed,
            "tasks_unresolved": self.tasks_unresolved,
            "evidence_collected": self.evidence_collected,
            "new_evidence": self.new_evidence,
            "redundant_evidence": self.redundant_evidence,
            "residual_gaps_total": self.residual_gaps_total,
            "residual_gaps_final": self.residual_gaps_final,
            "topics_planned": self.topics_planned,
            "topics_completed": self.topics_completed,
            "confidence_progression": list(self.confidence_progression),
            "coverage_progression": list(self.coverage_progression),
            "confidence_initial": self.confidence_initial,
            "confidence_final": self.confidence_final,
            "confidence_gain": self.confidence_gain,
            "coverage_final": self.coverage_final,
            "planning_efficiency": self.planning_efficiency,
            "budget_used": self.budget_used,
            "memory_topics_reused": self.memory_topics_reused,
            "memory_writes": self.memory_writes,
            "repeated_topics": list(self.repeated_topics),
            "termination": self.termination.value,
            "is_converged": self.is_converged,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> LoopMetrics:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            iterations=_require_int(data, "iterations"),
            tasks_planned=_require_int(data, "tasks_planned"),
            tasks_completed=_require_int(data, "tasks_completed"),
            tasks_unresolved=_require_int(data, "tasks_unresolved"),
            evidence_collected=_require_int(data, "evidence_collected"),
            new_evidence=_require_int(data, "new_evidence"),
            redundant_evidence=_require_int(data, "redundant_evidence"),
            residual_gaps_total=_require_int(data, "residual_gaps_total"),
            residual_gaps_final=_require_int(data, "residual_gaps_final"),
            topics_planned=_require_int(data, "topics_planned"),
            topics_completed=_require_int(data, "topics_completed"),
            confidence_progression=tuple(
                _require_float_list(data, "confidence_progression")
            ),
            coverage_progression=tuple(
                _require_float_list(data, "coverage_progression")
            ),
            confidence_initial=_require_float(data, "confidence_initial"),
            confidence_final=_require_float(data, "confidence_final"),
            coverage_final=_require_float(data, "coverage_final"),
            planning_efficiency=_require_float(data, "planning_efficiency"),
            budget_used=_require_float(data, "budget_used"),
            memory_topics_reused=_require_int(data, "memory_topics_reused"),
            memory_writes=_require_int(data, "memory_writes"),
            repeated_topics=tuple(_require_str_list(data, "repeated_topics")),
            termination=TerminationReason(_require_str(data, "termination")),
        )

    @classmethod
    def from_history(
        cls, history: ResearchIterationHistory
    ) -> LoopMetrics:
        """Derive the metrics from ``history``, purely.

        ``planning_efficiency`` is completed tasks over planned tasks, and
        ``budget_used`` is iterations spent over the budget available: both
        are ``0.0`` when their denominator is zero, so an empty loop
        reports zeros instead of dividing by it.
        """
        if not isinstance(history, ResearchIterationHistory):
            raise InvalidIterationModelError(
                "from_history expects a ResearchIterationHistory instance"
            )
        records = history.iterations
        planned = sum(len(record.planned_tasks) for record in records)
        completed = sum(len(record.completed_tasks) for record in records)
        unresolved = sum(len(record.unresolved_tasks) for record in records)
        collected = sum(record.evidence_count for record in records)
        new_evidence = sum(record.new_evidence_count for record in records)
        planned_topics: list[str] = []
        for record in records:
            for topic_id in record.planned_task_topics:
                if topic_id not in planned_topics:
                    planned_topics.append(topic_id)
        final = records[-1] if records else None
        return cls(
            iterations=len(records),
            tasks_planned=planned,
            tasks_completed=completed,
            tasks_unresolved=unresolved,
            evidence_collected=collected,
            new_evidence=new_evidence,
            redundant_evidence=max(0, collected - new_evidence),
            residual_gaps_total=sum(record.gap_count for record in records),
            residual_gaps_final=final.gap_count if final is not None else 0,
            topics_planned=len(planned_topics),
            topics_completed=len(history.topics_completed),
            confidence_progression=history.confidence_progression,
            coverage_progression=history.coverage_progression,
            confidence_initial=(
                history.confidence_progression[0]
                if history.confidence_progression
                else EMPTY_MEAN_CONFIDENCE
            ),
            confidence_final=(
                history.confidence_progression[-1]
                if history.confidence_progression
                else EMPTY_MEAN_CONFIDENCE
            ),
            coverage_final=(
                history.coverage_progression[-1]
                if history.coverage_progression
                else EMPTY_COVERAGE_RATIO
            ),
            planning_efficiency=(
                completed / planned if planned else EMPTY_PROGRESS_RATIO
            ),
            budget_used=(
                len(records) / history.policy.max_iterations
                if history.policy.max_iterations
                else EMPTY_PROGRESS_RATIO
            ),
            memory_topics_reused=(
                history.memory_context.topics_reused
                if history.memory_context is not None
                else 0
            ),
            memory_writes=sum(
                1 for record in records if record.memory_update.applied
            ),
            repeated_topics=history.repeated_topics,
            termination=history.termination,
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
        raise InvalidIterationModelError(f"expected string for '{key}'")
    return value


def _require_int(data: Mapping[str, object], key: str) -> int:
    """Read and validate an int field from a serialized dictionary."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIterationModelError(f"expected int for '{key}'")
    return value


def _require_float(data: Mapping[str, object], key: str) -> float:
    """Read and validate a number field from a serialized dictionary."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidIterationModelError(f"expected number for '{key}'")
    return float(value)


def _require_bool(data: Mapping[str, object], key: str) -> bool:
    """Read and validate a bool field from a serialized dictionary."""
    value = data.get(key)
    if not isinstance(value, bool):
        raise InvalidIterationModelError(f"expected bool for '{key}'")
    return value


def _require_str_list(data: Mapping[str, object], key: str) -> list[str]:
    """Read and validate a string-list field."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise InvalidIterationModelError(f"expected list for '{key}'")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise InvalidIterationModelError(
                f"expected list of strings for '{key}'"
            )
        result.append(item)
    return result


def _require_float_list(data: Mapping[str, object], key: str) -> list[float]:
    """Read and validate a list-of-numbers field."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise InvalidIterationModelError(f"expected list for '{key}'")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise InvalidIterationModelError(
                f"expected list of numbers for '{key}'"
            )
        result.append(float(item))
    return result


def _require_dict(data: Mapping[str, object], key: str) -> dict[str, object]:
    """Read and validate a nested dictionary field."""
    value = data.get(key)
    if not isinstance(value, dict):
        raise InvalidIterationModelError(f"expected dict for '{key}'")
    return value


def _optional_dict(
    data: Mapping[str, object], key: str
) -> dict[str, object] | None:
    """Read a nested dictionary field that may be absent or ``None``."""
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvalidIterationModelError(f"expected dict or null for '{key}'")
    return value


def _require_list_of_dicts(
    data: Mapping[str, object], key: str
) -> list[dict[str, object]]:
    """Read and validate a list-of-dicts field."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise InvalidIterationModelError(f"expected list for '{key}'")
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise InvalidIterationModelError(
                f"expected list of dicts for '{key}'"
            )
        result.append(item)
    return result


def _require_non_negative_int(value: object, field_name: str) -> None:
    """Validate that ``value`` is a non-negative ``int``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIterationModelError(
            f"{field_name} must be an int for the loop models"
        )
    if value < 0:
        raise InvalidIterationModelError(
            f"{field_name} must be non-negative for the loop models"
        )


def _require_unit_interval(value: object, field_name: str) -> float:
    """Validate that ``value`` is a number within the closed unit interval."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidIterationModelError(
            f"{field_name} must be a number for the loop models"
        )
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise InvalidIterationModelError(
            f"{field_name} must be within [0, 1] for the loop models"
        )
    return number


def _clean_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return ``values`` de-duplicated, order preserved, whitespace stripped."""
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise InvalidIterationModelError(
                "identifiers must be non-empty strings for the loop models"
            )
        stripped = value.strip()
        if stripped not in cleaned:
            cleaned.append(stripped)
    return tuple(cleaned)


def _round(value: float) -> float:
    """Round a derived ratio so float noise cannot break determinism."""
    return round(float(value), RATIO_PRECISION)


def analysis_digest(payload: Mapping[str, object]) -> str:
    """Return the stable content hash identifying a residual analysis."""
    return f"analysis_{content_digest(payload)}"


__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MINIMUM_CONFIDENCE",
    "DEFAULT_MINIMUM_COVERAGE",
    "EMPTY_PROGRESS_RATIO",
    "ITERATION_SCHEMA_VERSION",
    "LOW_CONFIDENCE_THRESHOLD",
    "MINIMUM_EVIDENCE_PER_TOPIC",
    "FollowUpTask",
    "GapKind",
    "IterationPolicy",
    "IterationRecord",
    "IterationStatus",
    "LoopDecision",
    "LoopMetrics",
    "MemoryContext",
    "MemoryUpdate",
    "ResearchIterationHistory",
    "ResidualAnalysis",
    "ResidualGap",
    "TerminationReason",
    "TopicInsight",
    "TopicKnowledge",
    "analysis_digest",
]
