"""Residual gap analysis — Phase 8 Sprint 7.

:class:`ResidualGapEngine` is the pure reasoning core of the research loop.
It answers one question, deterministically:

    after this iteration, what is still missing, and what should the next
    iteration ask for?

It reads the artifacts an iteration produced and writes nothing to any of
them:

* the :class:`~predictron_engine.research.validation_models.ValidationSummary`
  — the **only** source of residual gaps,
* the :class:`~predictron_engine.research.models.ResearchPlan` the iteration
  ran, for the topic set and the task-to-topic mapping,
* the :class:`~predictron_engine.research.execution_models.DispatchOutcome`
  and :class:`~predictron_engine.research.models.CollectionResult`, so a task
  that never reached a collector is distinguishable from one that collected
  nothing, and
* a :class:`~predictron_engine.research.iteration_models.MemoryContext`
  derived from :class:`~predictron_engine.research.memory.ResearchMemory`,
  which supplies freshness and prior knowledge.

It emits two values and mutates nothing:

* a :class:`~predictron_engine.research.iteration_models.ResidualAnalysis`
  — every gap that survived, together with the thresholds it was judged
  against, and
* :class:`~predictron_engine.research.iteration_models.FollowUpTask`
  values — one deterministic request per gap.

It contains **no orchestration**: it does not run the pipeline, write to
memory, iterate, or decide when to stop.  Those belong to
:mod:`predictron_engine.research.research_loop`.

Validation is the only source of gaps
-------------------------------------
A gap can only be raised for a topic the iteration actually examined — one
the plan researched, or one the validation summary covered.  Memory never
invents a gap for a topic nothing looked at.  It contributes exactly two
things: the ``STALE`` classification for a topic that *does* hold evidence
(a freshness overlay, never a blocking gap), and the prior knowledge that
narrows the next
:class:`~predictron_engine.research.models.PlannerInput`.

The gap ladder
--------------
Each topic walks one ladder and reports the **first** condition it
satisfies, so a topic carries at most one blocking gap and a gap count never
inflates by counting one weakness three times:

===  ====================  =========================================
  1  ``UNRESOLVED``        the task never reached a collector
  2  ``FAILED_VALIDATION`` every validated item failed
  3  ``MISSING``           no usable evidence at all exists
  4  ``CONFLICTING``       the evidence contradicts itself
  5  ``LOW_CONFIDENCE``    usable evidence scored below the threshold
  6  ``INSUFFICIENT``      fewer usable items than the policy requires
===  ====================  =========================================

``STALE`` is the one additive gap: it rides alongside any of the six,
because refreshable knowledge can be complete *and* out of date.  It is
never blocking, which is what stops a loop spending its whole budget
chasing freshness nobody asked for.

Determinism
-----------
The engine holds no clock, no randomness, and no mutable state.  Every value
it returns is a pure function of its arguments, gap lists are emitted in
:attr:`~predictron_engine.research.iteration_models.GapKind.order_value`
order, and ``analysis_id`` is a content hash over the deterministic payload —
so the same iteration always yields the same analysis, and a replay is
observable rather than merely asserted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from predictron_engine.research.exceptions import InvalidIterationInputError
from predictron_engine.research.execution_models import (
    EMPTY_MEAN_CONFIDENCE,
    RATIO_PRECISION,
    ConfidenceDistribution,
    DispatchOutcome,
    ExecutionCoverage,
    ExecutionError,
)
from predictron_engine.research.iteration_models import (
    FollowUpTask,
    GapKind,
    IterationPolicy,
    MemoryContext,
    ResidualAnalysis,
    ResidualGap,
    TopicInsight,
    TopicKnowledge,
    analysis_digest,
)
from predictron_engine.research.memory import ResearchMemory
from predictron_engine.research.models import (
    CollectionResult,
    PlannerInput,
    ResearchPlan,
)
from predictron_engine.research.topics import TOPIC_REGISTRY
from predictron_engine.research.validation_models import (
    ValidationSummary,
    content_digest,
)

#: Where a topic insight learned its facts, recorded on every insight so a
#: reader can tell a memory-derived reading from a validation-derived one.
SOURCE_VALIDATION = "validation"
SOURCE_MEMORY = "memory"
SOURCE_BOTH = "memory+validation"

#: Prefix of a derived follow-up request identifier.
FOLLOW_UP_PREFIX = "followup"

#: Prefix of a derived residual gap identity.
GAP_PREFIX = "gap"


class ResidualGapEngine:
    """Deterministic, pure analysis of what a research iteration left open.

    Parameters
    ----------
    policy:
        The
        :class:`~predictron_engine.research.iteration_models.IterationPolicy`
        whose thresholds the analysis is judged against.  Defaults to the
        permissive single-pass policy, so a caller who only wants a gap list
        configures nothing.
    topics:
        Topic registry used to resolve a topic's freshness requirement.  The
        default is the built-in taxonomy; a caller running a custom taxonomy
        passes the matching mapping so freshness and staleness stay correct.

    Notes
    -----
    The engine is stateless: it can be reused across iterations, companies,
    and runs without accumulating anything, and it never mutates the
    artifacts it reads.
    """

    def __init__(
        self,
        *,
        policy: IterationPolicy | None = None,
        topics: Mapping[str, object] | None = None,
    ) -> None:
        self._policy = _require_policy(policy)
        self._topics: Mapping[str, object] = TOPIC_REGISTRY if topics is None else topics

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def policy(self) -> IterationPolicy:
        """Return the thresholds this engine judges against."""
        return self._policy

    @property
    def topics(self) -> Mapping[str, object]:
        """Return the registry used to resolve freshness requirements."""
        return self._topics

    # ------------------------------------------------------------------
    # Memory-derived context
    # ------------------------------------------------------------------

    def build_memory_context(
        self,
        *,
        company_name: str = "",
        topics: Sequence[str] | None = None,
        validation: ValidationSummary | None = None,
        plan: ResearchPlan | None = None,
        memory: ResearchMemory | None = None,
        ages: Mapping[str, int] | None = None,
    ) -> MemoryContext:
        """Derive what memory already knows, per topic.

        The returned context is the *read* side of the closed loop: it turns
        research memory into the ``known_topics`` / ``partial_topics`` the
        planner already understands, so memory reduces future work without a
        single planner rule changing.

        Parameters
        ----------
        company_name:
            The company the context describes; recorded for traceability.
        topics:
            Explicit topic list to describe.  When omitted the union of the
            plan's researched topics and the validation summary's covered
            topics is used.
        validation:
            The iteration's validation summary, used for per-topic evidence
            counts, confidence, and conflicts.
        plan:
            The iteration's plan, used for freshness requirements and to
            widen the topic set when validation did not run.
        memory:
            The research memory to read.  When omitted, insights are derived
            from ``validation`` alone.
        ages:
            Caller-supplied ``topic_id -> age in days`` map.  Staleness is a
            comparison of two integers precisely so the engine never has to
            read a date: a caller that does not care about freshness passes
            nothing and never sees a stale gap.

        Returns
        -------
        MemoryContext
            One insight per described topic, in taxonomy order, so the
            context is byte-identical across runs.
        """
        ages_by_topic = dict(ages) if ages else {}
        view = _ValidationView.of(validation)
        insights = tuple(
            self._insight(
                topic_id,
                view=view,
                plan=plan,
                memory=memory,
                age_days=ages_by_topic.get(topic_id, 0),
            )
            for topic_id in self._topic_sequence(topics, validation, plan)
        )
        return MemoryContext(
            insights=insights,
            memory_key=memory.memory_key() if memory is not None else "",
            company_name=company_name.strip(),
        )

    # ------------------------------------------------------------------
    # Residual analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        *,
        iteration: int,
        validation: ValidationSummary | None = None,
        plan: ResearchPlan | None = None,
        collection: CollectionResult | None = None,
        dispatch: DispatchOutcome | None = None,
        errors: tuple[ExecutionError, ...] = (),
        context: MemoryContext | None = None,
        policy: IterationPolicy | None = None,
    ) -> ResidualAnalysis:
        """Return every knowledge gap that survived ``iteration``.

        Parameters
        ----------
        iteration:
            The 1-based iteration number, recorded on every gap so a reader
            can see which pass found what.
        validation:
            The iteration's validation summary.  This is the only source of
            gap findings; without it the analysis reports missing evidence
            for every planned topic, which is exactly the truth.
        plan:
            The plan the iteration ran.  Supplies the topic set and the
            task-to-topic mapping.
        collection:
            The collection result, used to separate a task that produced
            nothing from one that was never executed.
        dispatch:
            The dispatch outcome, used to detect a task that could not be
            routed to a collector at all.
        errors:
            Stage errors the iteration recorded, carried through so the loop
            can decide to stop without re-reading the execution result.
        context:
            Memory-derived context supplying freshness for the ``STALE``
            classification.
        policy:
            Overrides the engine policy for this analysis only.

        Returns
        -------
        ResidualAnalysis
            The gaps, sorted by urgency then topic then reason, together
            with the thresholds they were judged against.
        """
        effective = _require_policy(policy, fallback=self._policy)
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise InvalidIterationInputError(
                "iteration must be an int for a residual analysis"
            )
        if iteration < 1:
            raise InvalidIterationInputError(
                "iteration must be 1 or greater for a residual analysis"
            )
        if context is not None and not isinstance(context, MemoryContext):
            raise InvalidIterationInputError(
                "context must be a MemoryContext instance or None"
            )
        view = _ValidationView.of(validation)
        topic_order = self._topic_sequence(None, validation, plan)
        tasks_by_topic = _tasks_by_topic(plan)
        unresolved_by_topic = _unresolved_topics(tasks_by_topic, dispatch)
        collected = _collected_counts(collection)
        gaps = self._gaps(
            topic_order=topic_order,
            iteration=iteration,
            tasks_by_topic=tasks_by_topic,
            unresolved_by_topic=unresolved_by_topic,
            view=view,
            collected=collected,
            context=context,
            policy=effective,
        )
        completed_topics = tuple(
            topic_id for topic_id in topic_order if view.usable_count(topic_id) > 0
        )
        unresolved_tasks = _unresolved_ids(dispatch)
        completed_tasks, failed_tasks = _task_outcomes(
            plan, collection, unresolved_tasks
        )
        coverage = ExecutionCoverage(
            planned_topics=plan.topics_researched if plan is not None else (),
            covered_topics=validation.covered_topics if validation is not None else (),
        )
        payload = _analysis_payload(
            iteration=iteration,
            gaps=gaps,
            topic_order=topic_order,
            completed_topics=completed_topics,
            completed_tasks=completed_tasks,
            unresolved_tasks=unresolved_tasks,
            failed_tasks=failed_tasks,
            view=view,
            coverage=coverage,
            errors=errors,
            policy=effective,
        )
        return ResidualAnalysis(
            analysis_id=analysis_digest(payload),
            iteration=iteration,
            gaps=gaps,
            planned_topics=topic_order,
            completed_topics=completed_topics,
            completed_tasks=completed_tasks,
            unresolved_tasks=unresolved_tasks,
            failed_tasks=failed_tasks,
            evidence_count=view.evidence_total,
            mean_confidence=view.mean_confidence,
            coverage_ratio=coverage.ratio,
            conflicts=view.conflicts_total,
            failed_validations=view.failed_total,
            minimum_confidence=effective.minimum_confidence,
            minimum_coverage=effective.minimum_coverage,
            errors=tuple(errors),
        )

    # ------------------------------------------------------------------
    # Follow-up work
    # ------------------------------------------------------------------

    def follow_up_tasks(self, analysis: ResidualAnalysis) -> tuple[FollowUpTask, ...]:
        """Return one deterministic follow-up request per residual gap.

        A topic with no gap produces no request, which is how "already
        satisfied work is never recreated" is enforced here: the engine only
        emits work for a topic it has just found a gap in.

        The requests are *not* executable
        :class:`~predictron_engine.research.models.ResearchTask` values.
        Priority scoring, dependency ordering, and source categories belong
        to the Research Planner, so the engine states what is wrong and lets
        :meth:`revise_planner_input` hand the planner the context it needs to
        build the real tasks.  Reimplementing that scoring here would let the
        two layers disagree about what matters.
        """
        if not isinstance(analysis, ResidualAnalysis):
            raise InvalidIterationInputError(
                "follow_up_tasks expects a ResidualAnalysis instance"
            )
        return tuple(
            FollowUpTask(
                task_id=follow_up_id(gap),
                topic_id=gap.topic_id,
                kind=gap.kind,
                reason=gap.reason,
                title=f"Resolve {gap.kind.value} gap for topic '{gap.topic_id}'",
                iteration=analysis.iteration,
                evidence_count=gap.evidence_count,
                confidence=gap.confidence,
                conflicts=gap.conflicts,
            )
            for gap in analysis.gaps
        )

    def actionable_tasks(self, analysis: ResidualAnalysis) -> tuple[FollowUpTask, ...]:
        """Return only the follow-up requests that warrant another iteration.

        Staleness is excluded: refreshing complete-but-old knowledge is
        legitimate work, but it never makes research *more complete*, so
        spending iteration budget on it would be a loop that cannot finish.
        """
        return tuple(task for task in self.follow_up_tasks(analysis) if task.is_blocking)

    # ------------------------------------------------------------------
    # Planner feedback
    # ------------------------------------------------------------------

    def revise_planner_input(
        self,
        planner_input: PlannerInput,
        analysis: ResidualAnalysis,
        *,
        context: MemoryContext | None = None,
        refresh_stale: bool = False,
    ) -> PlannerInput:
        """Return the planner input the next iteration should run with.

        This is the loop's only feedback channel, and it deliberately uses
        nothing the planner does not already understand:

        * a topic with no blocking gap becomes ``known`` — the planner then
          emits no task for it, so resolved work is never repeated;
        * a topic with a blocking gap becomes ``partial`` — the planner
          keeps researching it, now scoring the gap against what is already
          known rather than against nothing;
        * topics memory already knows are carried in the same way, so memory
          reduces work the loop would otherwise ask for again.

        ``known`` and ``partial`` are disjoint by construction: a topic can
        only be in one, and the planner rejects an input carrying it in both.

        Parameters
        ----------
        planner_input:
            The input the completed iteration ran with.
        analysis:
            The residual analysis of that iteration.
        context:
            Memory-derived context, when the loop has one.
        refresh_stale:
            Whether non-blocking ``STALE`` topics are carried as partial.
            Off by default: a refresh does not make research more complete,
            so by default it does not consume the next iteration.
        """
        if not isinstance(planner_input, PlannerInput):
            raise InvalidIterationInputError(
                "revise_planner_input expects a PlannerInput instance"
            )
        if not isinstance(analysis, ResidualAnalysis):
            raise InvalidIterationInputError(
                "revise_planner_input expects a ResidualAnalysis instance"
            )
        if context is not None and not isinstance(context, MemoryContext):
            raise InvalidIterationInputError(
                "context must be a MemoryContext instance or None"
            )
        gaps = analysis.gaps if refresh_stale else analysis.blocking_gaps
        open_topics = {gap.topic_id for gap in gaps}
        resolved = [
            topic_id
            for topic_id in analysis.planned_topics
            if topic_id not in open_topics
        ]
        if context is not None:
            resolved.extend(context.known_topics)
            open_topics.update(context.partial_topics)
        known = _merge_topics(planner_input.known_topics, resolved)
        partial = _merge_topics(planner_input.partial_topics, sorted(open_topics))
        # A blocking gap outranks an inherited "known" claim: validation is the
        # only authority on completeness, so a topic freshly found deficient is
        # reopened rather than silently dropped. Without this a carried-in
        # partial topic that has since been resolved would land in both sets.
        known = tuple(
            topic_id for topic_id in known if topic_id not in open_topics
        )
        partial = tuple(
            topic_id for topic_id in partial if topic_id not in known
        )
        return PlannerInput(
            company_name=planner_input.company_name,
            website=planner_input.website,
            description=planner_input.description,
            known_topics=known,
            partial_topics=partial,
            prediction_horizon_days=planner_input.prediction_horizon_days,
        )

    # ------------------------------------------------------------------
    # Gap classification
    # ------------------------------------------------------------------

    @staticmethod
    def _gaps(
        *,
        topic_order: tuple[str, ...],
        iteration: int,
        tasks_by_topic: Mapping[str, str],
        unresolved_by_topic: Mapping[str, str],
        view: _ValidationView,
        collected: Mapping[str, int],
        context: MemoryContext | None,
        policy: IterationPolicy,
    ) -> tuple[ResidualGap, ...]:
        """Walk every topic down the ladder and collect what it reports."""
        gaps: list[ResidualGap] = []
        for topic_id in topic_order:
            findings = _TopicFindings(
                usable=view.usable_count(topic_id),
                failed=view.failed_count(topic_id),
                confidence=view.confidence_for(topic_id),
                conflicts=view.conflicts_for(topic_id),
                unresolved=topic_id in unresolved_by_topic,
                collected=collected.get(topic_id, 0),
            )
            task_id = tasks_by_topic.get(topic_id, "")
            blocking = _blocking_gap(
                topic_id=topic_id,
                iteration=iteration,
                task_id=task_id,
                unresolved_task_id=unresolved_by_topic.get(topic_id, ""),
                findings=findings,
                policy=policy,
            )
            if blocking is not None:
                gaps.append(blocking)
            stale = _stale_gap(
                topic_id=topic_id,
                iteration=iteration,
                context=context,
                findings=findings,
            )
            if stale is not None:
                gaps.append(stale)
        return tuple(sorted(gaps, key=lambda gap: gap.order_key))

    # ------------------------------------------------------------------
    # Per-topic derivation
    # ------------------------------------------------------------------

    def _insight(
        self,
        topic_id: str,
        *,
        view: _ValidationView,
        plan: ResearchPlan | None,
        memory: ResearchMemory | None,
        age_days: int,
    ) -> TopicInsight:
        """Derive what is known about one topic: memory first, then validation."""
        _require_non_negative_int(age_days, "age_days")
        requirement = self._freshness_requirement(topic_id, plan)
        usable = view.usable_count(topic_id)
        failed = view.failed_count(topic_id)
        stored = _memory_evidence_count(memory, topic_id)
        return TopicInsight(
            topic_id=topic_id,
            knowledge=_knowledge_state(
                usable=usable,
                failed=failed,
                stored=stored,
                confidence=view.confidence_for(topic_id),
                conflicts=view.conflicts_for(topic_id),
                age_days=age_days,
                freshness_requirement_days=requirement,
                policy=self._policy,
            ),
            evidence_count=max(stored, usable + failed),
            mean_confidence=view.confidence_for(topic_id),
            conflicts=view.conflicts_for(topic_id),
            age_days=age_days,
            freshness_requirement_days=requirement,
            source=_insight_source(usable + failed, stored),
        )

    def _freshness_requirement(
        self,
        topic_id: str,
        plan: ResearchPlan | None,
    ) -> int:
        """Return how many days the topic's knowledge stays fresh.

        The plan's own task is authoritative when it exists, because the
        planner already resolved the taxonomy once; the registry is the
        fallback for topics the plan did not research.  An unresolvable topic
        reports ``0``, which means "no freshness requirement" and therefore
        never produces a stale gap.
        """
        if plan is not None:
            task = plan.task_by_topic(topic_id)
            if task is not None:
                return task.freshness_requirement_days
        topic = self._topics.get(topic_id)
        requirement = getattr(topic, "freshness_requirement_days", 0)
        if isinstance(requirement, bool) or not isinstance(requirement, int):
            return 0
        return max(0, requirement)

    def _topic_sequence(
        self,
        topics: Sequence[str] | None,
        validation: ValidationSummary | None,
        plan: ResearchPlan | None,
    ) -> tuple[str, ...]:
        """Return the topics to reason about, in a stable deterministic order.

        Order is taxonomy position, then any topic outside the taxonomy,
        alphabetically — so the sequence never depends on which artifact
        happened to mention a topic first.
        """
        candidates: list[str] = []
        for source in (
            topics if topics is not None else (),
            plan.topics_researched if plan is not None else (),
            validation.covered_topics if validation is not None else (),
        ):
            for topic_id in source:
                cleaned = topic_id.strip()
                if cleaned and cleaned not in candidates:
                    candidates.append(cleaned)
        position = {
            topic_id: index for index, topic_id in enumerate(self._topics)
        }
        outside = len(position)
        return tuple(
            sorted(
                candidates,
                key=lambda topic_id: (
                    position.get(topic_id, outside),
                    topic_id if topic_id not in position else "",
                ),
            )
        )


# ----------------------------------------------------------------------
# The ladder
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _TopicFindings:
    """What validation and collection say about one topic.

    A tiny internal carrier rather than a public model: it exists only for
    the duration of one ladder walk, and every field is a plain count or flag
    derived from an iteration's artifacts.
    """

    usable: int
    failed: int
    confidence: float
    conflicts: int
    unresolved: bool
    collected: int


def _blocking_gap(
    *,
    topic_id: str,
    iteration: int,
    task_id: str,
    unresolved_task_id: str,
    findings: _TopicFindings,
    policy: IterationPolicy,
) -> ResidualGap | None:
    """Return the first blocking condition the topic satisfies, or ``None``."""
    if findings.unresolved:
        return ResidualGap(
            topic_id=topic_id,
            kind=GapKind.UNRESOLVED,
            reason="no collector is registered for the topic's research task",
            iteration=iteration,
            task_id=unresolved_task_id or task_id,
            evidence_count=findings.usable,
            confidence=findings.confidence,
            conflicts=findings.conflicts,
        )
    if findings.usable == 0 and findings.failed > 0:
        return ResidualGap(
            topic_id=topic_id,
            kind=GapKind.FAILED_VALIDATION,
            reason=f"all {findings.failed} evidence item(s) failed validation",
            iteration=iteration,
            task_id=task_id,
            evidence_count=findings.failed,
            confidence=findings.confidence,
            conflicts=findings.conflicts,
        )
    if findings.usable == 0:
        return ResidualGap(
            topic_id=topic_id,
            kind=GapKind.MISSING,
            reason=(
                "no evidence was collected for the topic"
                if findings.collected == 0
                else "no evidence for the topic passed validation"
            ),
            iteration=iteration,
            task_id=task_id,
            evidence_count=0,
            confidence=EMPTY_MEAN_CONFIDENCE,
            conflicts=findings.conflicts,
        )
    if findings.conflicts > 0:
        return ResidualGap(
            topic_id=topic_id,
            kind=GapKind.CONFLICTING,
            reason=(
                f"{findings.conflicts} evidence conflict(s) contradict "
                "knowledge for the topic"
            ),
            iteration=iteration,
            task_id=task_id,
            evidence_count=findings.usable,
            confidence=findings.confidence,
            conflicts=findings.conflicts,
        )
    if findings.confidence < policy.confidence_threshold:
        return ResidualGap(
            topic_id=topic_id,
            kind=GapKind.LOW_CONFIDENCE,
            reason=(
                f"mean confidence {_ratio(findings.confidence):.2f} is below "
                f"the {_ratio(policy.confidence_threshold):.2f} threshold"
            ),
            iteration=iteration,
            task_id=task_id,
            evidence_count=findings.usable,
            confidence=findings.confidence,
            conflicts=0,
        )
    if findings.usable < policy.minimum_evidence_per_topic:
        return ResidualGap(
            topic_id=topic_id,
            kind=GapKind.INSUFFICIENT,
            reason=(
                f"{findings.usable} usable evidence item(s) is fewer than the "
                f"{policy.minimum_evidence_per_topic} required"
            ),
            iteration=iteration,
            task_id=task_id,
            evidence_count=findings.usable,
            confidence=findings.confidence,
            conflicts=0,
        )
    return None


def _stale_gap(
    *,
    topic_id: str,
    iteration: int,
    context: MemoryContext | None,
    findings: _TopicFindings,
) -> ResidualGap | None:
    """Return the topic's staleness gap, when memory says it is stale.

    A stale topic with no usable evidence is already reported by the blocking
    ladder, and reporting it twice would inflate the gap count for one
    underlying problem, so staleness is only added for a topic that actually
    holds something.
    """
    if context is None or findings.usable == 0:
        return None
    insight = context.insight_for(topic_id)
    if insight is None or not insight.is_stale:
        return None
    return ResidualGap(
        topic_id=topic_id,
        kind=GapKind.STALE,
        reason=(
            f"evidence is {insight.age_days} day(s) old and the topic "
            f"requires {insight.freshness_requirement_days}"
        ),
        iteration=iteration,
        task_id="",
        evidence_count=findings.usable,
        confidence=findings.confidence,
        conflicts=findings.conflicts,
    )


# ----------------------------------------------------------------------
# Validation view
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _ValidationView:
    """Per-topic projections of one validation summary, computed once.

    Every question the ladder asks about a topic — how much usable evidence
    it has, how much failed, how confident it is, how many conflicts it
    carries — is answered from here, so the projections are built once per
    analysis instead of once per topic.
    """

    usable_counts: Mapping[str, int] = field(default_factory=dict)
    failed_counts: Mapping[str, int] = field(default_factory=dict)
    confidence_by_topic: Mapping[str, float] = field(default_factory=dict)
    conflicts_by_topic: Mapping[str, int] = field(default_factory=dict)
    mean_confidence: float = EMPTY_MEAN_CONFIDENCE
    evidence_total: int = 0
    conflicts_total: int = 0
    failed_total: int = 0

    @classmethod
    def of(cls, validation: ValidationSummary | None) -> _ValidationView:
        """Build the view for ``validation``, or the empty view.

        Always go through this constructor: the dataclass signature takes the
        projections directly, so passing a ``ValidationSummary`` positionally
        would silently produce a view whose every topic reports nothing.
        """
        if validation is None:
            return cls()
        usable: dict[str, int] = {}
        failed: dict[str, int] = {}
        confidence: dict[str, float] = {}
        for breakdown in validation.by_topic:
            usable[breakdown.topic_id] = breakdown.passed + breakdown.warned
            failed[breakdown.topic_id] = breakdown.failed
            confidence[breakdown.topic_id] = _ratio(breakdown.mean_confidence)
        return cls(
            usable_counts=usable,
            failed_counts=failed,
            confidence_by_topic=confidence,
            conflicts_by_topic=_conflict_counts(validation),
            mean_confidence=ConfidenceDistribution.from_validations(
                (validation,)
            ).mean_score,
            evidence_total=validation.counts.total,
            conflicts_total=validation.counts.conflicts,
            failed_total=validation.counts.failed,
        )

    def usable_count(self, topic_id: str) -> int:
        """Return how many of the topic's evidence items are usable."""
        return self.usable_counts.get(topic_id, 0)

    def failed_count(self, topic_id: str) -> int:
        """Return how many of the topic's evidence items failed."""
        return self.failed_counts.get(topic_id, 0)

    def confidence_for(self, topic_id: str) -> float:
        """Return the topic's rounded mean confidence."""
        return self.confidence_by_topic.get(topic_id, EMPTY_MEAN_CONFIDENCE)

    def conflicts_for(self, topic_id: str) -> int:
        """Return how many conflicts the topic's evidence carries."""
        return self.conflicts_by_topic.get(topic_id, 0)

    def findings_for(
        self,
        topic_id: str,
        *,
        unresolved: bool,
        collected: int,
    ) -> _TopicFindings:
        """Return everything the ladder needs about one topic."""
        return _TopicFindings(
            usable=self.usable_count(topic_id),
            failed=self.failed_count(topic_id),
            confidence=self.confidence_for(topic_id),
            conflicts=self.conflicts_for(topic_id),
            unresolved=unresolved,
            collected=collected,
        )


def _conflict_counts(validation: ValidationSummary) -> dict[str, int]:
    """Return the number of conflicts per topic, deterministically attributed.

    A conflict names the evidence it involves and every evidence item knows
    its topic, so a conflict is attributed to the topic of the items it
    concerns.  A conflict spanning two topics counts once, against the
    lexicographically first topic, which keeps the per-topic total equal to
    the number of conflicts instead of double counting it.
    """
    if not validation.conflicts:
        return {}
    topic_by_evidence = {
        entry.evidence_id: entry.topic_id for entry in validation.validations
    }
    counts: dict[str, int] = {}
    for conflict in validation.conflicts:
        topics = sorted(
            {
                topic_by_evidence[evidence_id]
                for evidence_id in (conflict.evidence_a, conflict.evidence_b)
                if evidence_id and evidence_id in topic_by_evidence
            }
        )
        if not topics:
            continue
        counts[topics[0]] = counts.get(topics[0], 0) + 1
    return counts


# ----------------------------------------------------------------------
# Plan, collection, and memory projections
# ----------------------------------------------------------------------


def _tasks_by_topic(plan: ResearchPlan | None) -> dict[str, str]:
    """Return the planned task identifier for each planned topic."""
    if plan is None:
        return {}
    return {task.topic_id: task.task_id for task in plan.tasks}


def _unresolved_ids(dispatch: DispatchOutcome | None) -> tuple[str, ...]:
    """Return the task identifiers that never reached a collector."""
    if dispatch is None:
        return ()
    return tuple(dispatch.unresolved)


def _unresolved_topics(
    tasks_by_topic: Mapping[str, str],
    dispatch: DispatchOutcome | None,
) -> dict[str, str]:
    """Return the unresolved task identifier for each unresolved topic."""
    if dispatch is None:
        return {}
    topic_by_task = {task_id: topic for topic, task_id in tasks_by_topic.items()}
    return {
        topic_by_task[task_id]: task_id
        for task_id in dispatch.unresolved
        if task_id in topic_by_task
    }


def _collected_counts(collection: CollectionResult | None) -> dict[str, int]:
    """Return how many evidence items the collection holds per topic."""
    if collection is None:
        return {}
    return {
        topic_id: collection.collection.evidence_count(topic_id)
        for topic_id in collection.collection.topics_covered()
    }


def _task_outcomes(
    plan: ResearchPlan | None,
    collection: CollectionResult | None,
    unresolved_tasks: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the planned tasks that completed and those that never ran.

    A task *failed* when the collector ran and produced nothing; a task was
    *never executed* when it was not in the collection's executed set.  The
    distinction matters: only the second kind is worth retrying with a
    different collector.
    """
    if plan is None:
        return (), ()
    failed = collection.failed_tasks if collection is not None else ()
    executed = set(collection.executed_tasks) if collection is not None else set()
    unresolved = set(unresolved_tasks)
    completed = tuple(
        task.task_id
        for task in plan.tasks
        if task.task_id not in unresolved and task.task_id not in failed
    )
    never_ran = tuple(
        task_id for task_id in failed if task_id not in executed
    )
    return completed, never_ran


def _memory_evidence_count(memory: ResearchMemory | None, topic_id: str) -> int:
    """Return how many evidence items memory holds for ``topic_id``."""
    if memory is None:
        return 0
    return len(memory.retrieve_by_topic(topic_id).evidence)


def _knowledge_state(
    *,
    usable: int,
    failed: int,
    stored: int,
    confidence: float,
    conflicts: int,
    age_days: int,
    freshness_requirement_days: int,
    policy: IterationPolicy,
) -> TopicKnowledge:
    """Classify what is known about a topic from memory and validation.

    The order is the ladder's, plus the two states only memory can assert:
    ``MISSING`` for a topic nothing has ever observed, and ``STALE`` for
    knowledge past its freshness window.  A topic memory already holds counts
    towards what is known even when this iteration re-examined it and found
    it thin, because memory reducing future work is the whole reason the loop
    consults it.
    """
    if usable + failed + stored == 0:
        return TopicKnowledge.MISSING
    if usable > 0 and conflicts > 0:
        return TopicKnowledge.CONFLICTING
    if (
        freshness_requirement_days > 0
        and age_days > freshness_requirement_days
        and (usable > 0 or stored > 0)
    ):
        return TopicKnowledge.STALE
    if (
        usable >= max(1, policy.minimum_evidence_per_topic)
        and confidence >= policy.confidence_threshold
        and conflicts == 0
    ):
        return TopicKnowledge.KNOWN
    return TopicKnowledge.PARTIAL


def _insight_source(validated: int, stored: int) -> str:
    """Return where an insight's facts came from."""
    if validated and stored:
        return SOURCE_BOTH
    if stored:
        return SOURCE_MEMORY
    return SOURCE_VALIDATION


# ----------------------------------------------------------------------
# Identity and payload
# ----------------------------------------------------------------------


def gap_id(gap: ResidualGap) -> str:
    """Return the stable, clock-free identity of one residual gap.

    Identity is ``(topic, kind, task)`` and deliberately **not** the
    iteration number: the same unresolved weakness discovered twice must
    reduce to one unit of work, otherwise a loop would keep rediscovering it
    forever.  ``reason`` is excluded so a wording change never orphans a
    tracked gap.
    """
    if not isinstance(gap, ResidualGap):
        raise InvalidIterationInputError("gap_id expects a ResidualGap instance")
    payload = {
        "topic_id": gap.topic_id,
        "kind": gap.kind.value,
        "task_id": gap.task_id,
    }
    return f"{GAP_PREFIX}_{content_digest(payload)}"


def follow_up_id(gap: ResidualGap) -> str:
    """Return the stable identifier of the follow-up request for ``gap``."""
    if not isinstance(gap, ResidualGap):
        raise InvalidIterationInputError("follow_up_id expects a ResidualGap instance")
    return f"{FOLLOW_UP_PREFIX}_{gap.kind.value}_{gap.topic_id}"


def _analysis_payload(
    *,
    iteration: int,
    gaps: tuple[ResidualGap, ...],
    topic_order: tuple[str, ...],
    completed_topics: tuple[str, ...],
    completed_tasks: tuple[str, ...],
    unresolved_tasks: tuple[str, ...],
    failed_tasks: tuple[str, ...],
    view: _ValidationView,
    coverage: ExecutionCoverage,
    errors: tuple[ExecutionError, ...],
    policy: IterationPolicy,
) -> dict[str, object]:
    """Return the canonical payload hashed into ``analysis_id``.

    The measured values are included on purpose: two iterations that examined
    the same topics and found the same gaps but reached different confidence
    really are different analyses, and the identifier must say so.
    """
    return {
        "iteration": iteration,
        "gaps": [gap.to_dict() for gap in gaps],
        "planned_topics": list(topic_order),
        "completed_topics": list(completed_topics),
        "completed_tasks": list(completed_tasks),
        "unresolved_tasks": list(unresolved_tasks),
        "failed_tasks": list(failed_tasks),
        "evidence_count": view.evidence_total,
        "mean_confidence": view.mean_confidence,
        "coverage_ratio": coverage.ratio,
        "conflicts": view.conflicts_total,
        "failed_validations": view.failed_total,
        "minimum_confidence": policy.minimum_confidence,
        "minimum_coverage": policy.minimum_coverage,
        "confidence_threshold": policy.confidence_threshold,
        "minimum_evidence_per_topic": policy.minimum_evidence_per_topic,
        "errors": [error.to_dict() for error in errors],
    }


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


def _require_policy(
    policy: IterationPolicy | None,
    *,
    fallback: IterationPolicy | None = None,
) -> IterationPolicy:
    """Return the policy to use, rejecting anything that is not one."""
    if policy is None:
        return fallback if fallback is not None else IterationPolicy()
    if not isinstance(policy, IterationPolicy):
        raise InvalidIterationInputError("policy must be an IterationPolicy instance")
    return policy


def _require_non_negative_int(value: object, field_name: str) -> None:
    """Validate that ``value`` is a non-negative ``int``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIterationInputError(
            f"{field_name} must be an int for the residual gap engine"
        )
    if value < 0:
        raise InvalidIterationInputError(
            f"{field_name} must be non-negative for the residual gap engine"
        )


def _merge_topics(
    existing: Iterable[str],
    additions: Iterable[str],
) -> tuple[str, ...]:
    """Return the de-duplicated union of two topic sequences, order preserved."""
    merged: list[str] = []
    for topic_id in (*tuple(existing), *tuple(additions)):
        cleaned = topic_id.strip()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
    return tuple(merged)


def _ratio(value: float) -> float:
    """Round a derived ratio so float noise cannot break determinism."""
    return round(float(value), RATIO_PRECISION)


__all__ = [
    "FOLLOW_UP_PREFIX",
    "GAP_PREFIX",
    "SOURCE_BOTH",
    "SOURCE_MEMORY",
    "SOURCE_VALIDATION",
    "ResidualGapEngine",
    "follow_up_id",
    "gap_id",
]
