"""Tests for the Residual Gap Engine — Phase 8 Sprint 7.

Covers the gap ladder (every rung), the ``STALE`` overlay, deterministic
identity and ordering, the planner-feedback contract that stops satisfied
work from being recreated, memory-derived context, and input rejection.
"""

from __future__ import annotations

import json

import pytest

from predictron_engine.research.exceptions import InvalidIterationInputError
from predictron_engine.research.execution_models import DispatchOutcome
from predictron_engine.research.iteration_models import (
    GapKind,
    IterationPolicy,
    MemoryContext,
    ResidualAnalysis,
    ResidualGap,
    TopicInsight,
    TopicKnowledge,
)
from predictron_engine.research.models import (
    CollectionResult,
    CollectionStatus,
    Evidence,
    EvidenceCollection,
    EvidenceReference,
    PlannerInput,
    ResearchPlan,
    ResearchPriority,
    ResearchTask,
)
from predictron_engine.research.residual import (
    ResidualGapEngine,
    follow_up_id,
    gap_id,
)
from predictron_engine.research.validation_models import (
    VALIDATION_SCHEMA_VERSION,
    EvidenceConfidence,
    EvidenceConflict,
    EvidenceValidation,
    TopicValidation,
    ValidationCounts,
    ValidationMetadata,
    ValidationStatus,
    ValidationSummary,
    collection_fingerprint,
    compute_fingerprint,
)

PLAN_SCHEMA_VERSION = "1.0.0"
COLLECTION_SCHEMA_VERSION = "1.0.0"


def _reference() -> EvidenceReference:
    return EvidenceReference(
        source_category="web_search",
        source_identifier="placeholder",
        url="https://example.com",
        description="A ref.",
    )


def _evidence(topic_id: str = "product", **overrides: object) -> Evidence:
    values: dict[str, object] = {
        "task_id": f"research_{topic_id}",
        "topic_id": topic_id,
        "collector_id": f"{topic_id}_collector",
        "category": "product_claims",
        "claim": "Acme automates invoicing for small businesses.",
        "confidence": 0.8,
        "reference": _reference(),
    }
    values.update(overrides)
    return Evidence(**values)


def _task(topic_id: str, **overrides: object) -> ResearchTask:
    values: dict[str, object] = {
        "task_id": f"research_{topic_id}",
        "topic_id": topic_id,
        "title": f"Research {topic_id}",
        "description": "Gather evidence.",
        "priority": ResearchPriority.MEDIUM,
        "priority_score": 0.5,
        "importance": 0.5,
        "prediction_impact": 0.5,
        "freshness_requirement_days": 90,
        "dependency_ids": (),
        "source_categories": ("web_search",),
        "priority_rank": 1,
        "execution_order": 1,
    }
    values.update(overrides)
    return ResearchTask(**values)


def _plan(*topic_ids: str, **overrides: object) -> ResearchPlan:
    task_overrides = overrides.pop("task_overrides", {})
    values: dict[str, object] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": "plan_test",
        "company_name": "Acme",
        "input": PlannerInput(company_name="Acme"),
        "gaps": (),
        "tasks": tuple(
            _task(topic_id, **task_overrides) for topic_id in topic_ids
        ),
        "topics_researched": topic_ids,
        "available_source_categories": ("web_search",),
    }
    values.update(overrides)
    return ResearchPlan(**values)


def _validation(
    topic_id: str = "product",
    *,
    status: ValidationStatus = ValidationStatus.PASSED,
    score: float = 0.9,
    claim: str = "Acme automates invoicing for small businesses.",
) -> EvidenceValidation:
    evidence = _evidence(topic_id, claim=claim)
    return EvidenceValidation(
        evidence_id=evidence.evidence_id,
        task_id=evidence.task_id,
        topic_id=topic_id,
        fingerprint=compute_fingerprint(evidence),
        status=status,
        confidence_score=score,
        confidence=EvidenceConfidence.from_score(score),
        validator_results=(),
        issues=(),
        conflicts=(),
        duplicate_count=1,
    )


def _summary(
    validations: tuple[EvidenceValidation, ...],
    *,
    conflicts: tuple[EvidenceConflict, ...] = (),
    by_topic: tuple[TopicValidation, ...] | None = None,
) -> ValidationSummary:
    topics = tuple(dict.fromkeys(item.topic_id for item in validations))
    passed = sum(1 for v in validations if v.status == ValidationStatus.PASSED)
    warned = sum(1 for v in validations if v.status == ValidationStatus.WARNED)
    failed = sum(1 for v in validations if v.status == ValidationStatus.FAILED)
    fingerprint = collection_fingerprint([v.evidence_id for v in validations])
    breakdown = by_topic if by_topic is not None else tuple(
        TopicValidation(
            topic_id=topic_id,
            evidence_count=sum(1 for v in validations if v.topic_id == topic_id),
            passed=sum(
                1
                for v in validations
                if v.topic_id == topic_id and v.status == ValidationStatus.PASSED
            ),
            warned=sum(
                1
                for v in validations
                if v.topic_id == topic_id and v.status == ValidationStatus.WARNED
            ),
            failed=sum(
                1
                for v in validations
                if v.topic_id == topic_id and v.status == ValidationStatus.FAILED
            ),
            mean_confidence=round(
                sum(
                    v.confidence_score
                    for v in validations
                    if v.topic_id == topic_id
                )
                / max(1, sum(1 for v in validations if v.topic_id == topic_id)),
                6,
            ),
        )
        for topic_id in topics
    )
    return ValidationSummary(
        schema_version=VALIDATION_SCHEMA_VERSION,
        validation_id="validation_test",
        collection_fingerprint=fingerprint,
        status=ValidationStatus.PASSED,
        metadata=ValidationMetadata(
            schema_version=VALIDATION_SCHEMA_VERSION,
            engine_version="1.0.0",
            collection_fingerprint=fingerprint,
        ),
        validations=validations,
        conflicts=conflicts,
        counts=ValidationCounts(
            total=len(validations),
            passed=passed,
            warned=warned,
            failed=failed,
            skipped=0,
            issues=0,
            conflicts=len(conflicts),
            duplicates=0,
            topics_covered=len(topics),
        ),
        covered_topics=topics,
        by_topic=breakdown,
    )


def _collection(*topic_ids: str) -> CollectionResult:
    evidence = tuple(_evidence(topic_id) for topic_id in topic_ids)
    return CollectionResult(
        schema_version=COLLECTION_SCHEMA_VERSION,
        plan_id="plan_test",
        company_name="Acme",
        collection_id="collection_test",
        status=CollectionStatus.PARTIAL,
        collection=EvidenceCollection(evidence=evidence, topic_order=topic_ids),
        executed_tasks=tuple(f"research_{topic_id}" for topic_id in topic_ids),
        failed_tasks=(),
        unresolved_tasks=(),
    )


def _context(insights: tuple[TopicInsight, ...]) -> MemoryContext:
    return MemoryContext(insights=insights, memory_key="mem", company_name="Acme")


# ----------------------------------------------------------------------
# The gap ladder
# ----------------------------------------------------------------------


class TestGapLadder:
    """Each rung of the ladder is reported for exactly the topics that hit it."""

    def test_missing_when_no_evidence_at_all(self) -> None:
        analysis = ResidualGapEngine().analyze(iteration=1, plan=_plan("product"))

        assert [gap.kind for gap in analysis.gaps] == [GapKind.MISSING]
        assert analysis.gaps[0].topic_id == "product"
        assert analysis.gaps[0].task_id == "research_product"

    def test_missing_when_collected_but_nothing_passed(self) -> None:
        """Evidence that exists but failed validation is missing knowledge."""
        analysis = ResidualGapEngine().analyze(
            iteration=1,
            plan=_plan("product"),
            collection=_collection("product"),
        )

        assert analysis.gaps[0].kind == GapKind.MISSING

    def test_failed_validation_when_every_item_failed(self) -> None:
        summary = _summary(
            (
                _validation("product", status=ValidationStatus.FAILED, score=0.1),
                _validation("product", status=ValidationStatus.FAILED, score=0.2),
            )
        )
        analysis = ResidualGapEngine().analyze(
            iteration=1, plan=_plan("product"), validation=summary
        )

        assert [gap.kind for gap in analysis.gaps] == [GapKind.FAILED_VALIDATION]
        assert analysis.gaps[0].evidence_count == 2

    def test_unresolved_wins_over_every_other_rung(self) -> None:
        """A task that never reached a collector outranks its other problems."""
        summary = _summary(
            (_validation("product", status=ValidationStatus.FAILED, score=0.1),)
        )
        analysis = ResidualGapEngine().analyze(
            iteration=1,
            plan=_plan("product"),
            validation=summary,
            dispatch=DispatchOutcome(unresolved=("research_product",)),
        )

        assert [gap.kind for gap in analysis.gaps] == [GapKind.UNRESOLVED]

    def test_conflicting_when_evidence_contradicts(self) -> None:
        first = _validation("product", claim="Acme serves clinics.")
        second = _validation("product", claim="Acme serves retailers.")
        conflict = EvidenceConflict(
            conflict_type="contradiction",
            evidence_a=first.evidence_id,
            evidence_b=second.evidence_id,
            description="Contradiction.",
        )
        summary = _summary((first, second), conflicts=(conflict,))
        analysis = ResidualGapEngine().analyze(
            iteration=1, plan=_plan("product"), validation=summary
        )

        assert [gap.kind for gap in analysis.gaps] == [GapKind.CONFLICTING]
        assert analysis.gaps[0].conflicts == 1

    def test_low_confidence_when_usable_but_weak(self) -> None:
        summary = _summary((_validation("product", score=0.2),))
        analysis = ResidualGapEngine().analyze(
            iteration=1, plan=_plan("product"), validation=summary
        )

        assert [gap.kind for gap in analysis.gaps] == [GapKind.LOW_CONFIDENCE]
        assert "below" in analysis.gaps[0].reason

    def test_insufficient_when_confident_but_thin(self) -> None:
        """Enough confidence, not enough evidence for the policy's minimum."""
        policy = IterationPolicy(
            minimum_evidence_per_topic=3,
            confidence_threshold=0.5,
        )
        summary = _summary((_validation("product", score=0.95),))
        analysis = ResidualGapEngine(policy=policy).analyze(
            iteration=1, plan=_plan("product"), validation=summary
        )

        assert [gap.kind for gap in analysis.gaps] == [GapKind.INSUFFICIENT]

    def test_no_gap_when_the_topic_is_fully_satisfied(self) -> None:
        summary = _summary((_validation("product", score=0.95),))
        analysis = ResidualGapEngine().analyze(
            iteration=1, plan=_plan("product"), validation=summary
        )

        assert analysis.gaps == ()
        assert analysis.is_complete is True
        assert analysis.completed_topics == ("product",)

    def test_one_rung_per_topic_not_one_per_weakness(self) -> None:
        """A topic with several problems still reports a single blocking gap."""
        first = _validation("product", score=0.1, claim="Acme serves clinics.")
        second = _validation("product", score=0.1, claim="Acme serves retailers.")
        conflict = EvidenceConflict(
            conflict_type="contradiction",
            evidence_a=first.evidence_id,
            evidence_b=second.evidence_id,
            description="Contradiction.",
        )
        summary = _summary((first, second), conflicts=(conflict,))
        policy = IterationPolicy(
            minimum_evidence_per_topic=99,
            confidence_threshold=0.99,
        )
        analysis = ResidualGapEngine(policy=policy).analyze(
            iteration=1, plan=_plan("product"), validation=summary
        )

        assert len(analysis.gaps) == 1

    def test_confidence_outranks_insufficient_evidence(self) -> None:
        """The higher rung on the ladder is the one reported."""
        policy = IterationPolicy(
            minimum_evidence_per_topic=99,
            confidence_threshold=0.99,
        )
        summary = _summary((_validation("product", score=0.1),))
        analysis = ResidualGapEngine(policy=policy).analyze(
            iteration=1, plan=_plan("product"), validation=summary
        )

        assert [gap.kind for gap in analysis.gaps] == [GapKind.LOW_CONFIDENCE]


# ----------------------------------------------------------------------
# Stale is additive, never blocking
# ----------------------------------------------------------------------


class TestStaleGap:
    """Freshness rides alongside the ladder without inflating the gap count."""

    def test_stale_adds_to_a_satisfied_topic(self) -> None:
        summary = _summary((_validation("product", score=0.95),))
        context = _context(
            (
                TopicInsight(
                    topic_id="product",
                    knowledge=TopicKnowledge.STALE,
                    evidence_count=1,
                    mean_confidence=0.95,
                    age_days=400,
                    freshness_requirement_days=90,
                    source="validation",
                ),
            )
        )
        analysis = ResidualGapEngine().analyze(
            iteration=1,
            plan=_plan("product"),
            validation=summary,
            context=context,
        )

        assert [gap.kind for gap in analysis.gaps] == [GapKind.STALE]
        assert analysis.blocking_gaps == ()
        assert analysis.is_complete is True

    def test_stale_not_reported_for_a_missing_topic(self) -> None:
        """A topic with nothing to refresh is missing, not stale."""
        context = _context(
            (
                TopicInsight(
                    topic_id="product",
                    knowledge=TopicKnowledge.MISSING,
                    age_days=400,
                    freshness_requirement_days=90,
                ),
            )
        )
        analysis = ResidualGapEngine().analyze(
            iteration=1, plan=_plan("product"), context=context
        )

        assert [gap.kind for gap in analysis.gaps] == [GapKind.MISSING]

    def test_stale_accompanies_a_blocking_gap(self) -> None:
        summary = _summary((_validation("product", score=0.2),))
        context = _context(
            (
                TopicInsight(
                    topic_id="product",
                    knowledge=TopicKnowledge.STALE,
                    age_days=400,
                    freshness_requirement_days=90,
                ),
            )
        )
        analysis = ResidualGapEngine().analyze(
            iteration=1,
            plan=_plan("product"),
            validation=summary,
            context=context,
        )

        kinds = [gap.kind for gap in analysis.gaps]
        assert GapKind.LOW_CONFIDENCE in kinds
        assert GapKind.STALE in kinds
        assert len(analysis.blocking_gaps) == 1


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


class TestDeterminism:
    """The same inputs always produce byte-identical output."""

    def _analyze(self) -> ResidualAnalysis:
        summary = _summary((_validation("product", score=0.3),))
        return ResidualGapEngine().analyze(
            iteration=1,
            plan=_plan("product", "team", "market"),
            validation=summary,
        )

    def test_repeated_analysis_is_identical(self) -> None:
        first, second = self._analyze(), self._analyze()

        assert first.to_dict() == second.to_dict()
        assert first.analysis_id == second.analysis_id

    def test_analysis_id_changes_with_findings(self) -> None:
        summary = _summary((_validation("product", score=0.3),))
        policy = IterationPolicy(minimum_evidence_per_topic=5)
        baseline = ResidualGapEngine().analyze(
            iteration=1, plan=_plan("product"), validation=summary
        )
        stricter = ResidualGapEngine(policy=policy).analyze(
            iteration=1, plan=_plan("product"), validation=summary
        )

        assert baseline.analysis_id != stricter.analysis_id

    def test_gap_order_is_independent_of_input_order(self) -> None:
        summary = _summary(
            (
                _validation("market", score=0.1),
                _validation("product", score=0.1),
                _validation("team", score=0.1),
            )
        )
        forward = ResidualGapEngine().analyze(
            iteration=1,
            plan=_plan("market", "product", "team"),
            validation=summary,
        )
        reverse = ResidualGapEngine().analyze(
            iteration=1,
            plan=_plan("team", "product", "market"),
            validation=summary,
        )

        assert [g.topic_id for g in forward.gaps] == [g.topic_id for g in reverse.gaps]

    def test_gap_identity_excludes_iteration_and_reason(self) -> None:
        """Rediscovering one weakness is still one unit of work."""
        gap = ResidualGap(topic_id="product", kind=GapKind.MISSING)
        same = ResidualGap(
            topic_id="product",
            kind=GapKind.MISSING,
            reason="Different wording.",
            iteration=7,
        )

        assert gap_id(gap) == gap_id(same)

    def test_gap_identity_separates_kinds(self) -> None:
        missing = gap_id(ResidualGap(topic_id="product", kind=GapKind.MISSING))
        unresolved = gap_id(
            ResidualGap(topic_id="product", kind=GapKind.UNRESOLVED)
        )

        assert missing != unresolved

    def test_gap_identity_is_stable_across_calls(self) -> None:
        gap = ResidualGap(topic_id="product", kind=GapKind.INSUFFICIENT)

        assert gap_id(gap) == gap_id(gap)
        assert gap_id(gap).startswith("gap_")

    def test_follow_up_id_is_readable_and_stable(self) -> None:
        gap = ResidualGap(topic_id="product", kind=GapKind.MISSING)

        assert follow_up_id(gap) == "followup_missing_product"
        assert follow_up_id(gap) == follow_up_id(gap)

    def test_analysis_survives_serialization_round_trip(self) -> None:
        analysis = self._analyze()

        restored = ResidualAnalysis.from_dict(analysis.to_dict())

        assert restored.to_dict() == analysis.to_dict()

    def test_analysis_is_json_ready(self) -> None:
        json.dumps(self._analyze().to_dict())


# ----------------------------------------------------------------------
# Planner feedback
# ----------------------------------------------------------------------


class TestPlannerFeedback:
    """Feedback uses only concepts the planner already understands."""

    def _engine_and_analysis(self) -> tuple[ResidualGapEngine, ResidualAnalysis]:
        summary = _summary((_validation("product", score=0.95),))
        engine = ResidualGapEngine()
        return engine, engine.analyze(
            iteration=1,
            plan=_plan("product", "team"),
            validation=summary,
        )

    def test_resolved_topic_becomes_known(self) -> None:
        engine, analysis = self._engine_and_analysis()
        original = PlannerInput(company_name="Acme")

        revised = engine.revise_planner_input(original, analysis)

        assert "product" in revised.known_topics
        assert "team" in revised.partial_topics

    def test_known_and_partial_stay_disjoint(self) -> None:
        engine, analysis = self._engine_and_analysis()
        original = PlannerInput(
            company_name="Acme",
            known_topics=("market",),
            partial_topics=("product",),
        )

        revised = engine.revise_planner_input(original, analysis)

        assert set(revised.known_topics).isdisjoint(revised.partial_topics)

    def test_a_topic_already_known_is_not_reopened(self) -> None:
        """Satisfied work is never recreated, even if the plan still lists it."""
        summary = _summary((_validation("product", score=0.95),))
        engine = ResidualGapEngine()
        analysis = engine.analyze(
            iteration=1, plan=_plan("product"), validation=summary
        )
        original = PlannerInput(company_name="Acme", known_topics=("product",))

        revised = engine.revise_planner_input(original, analysis)

        assert "product" not in revised.partial_topics
        assert "product" in revised.known_topics

    def test_carried_identity_fields_are_preserved(self) -> None:
        engine, analysis = self._engine_and_analysis()
        original = PlannerInput(
            company_name="Acme",
            website="https://acme.test",
            description="An ACME.",
            prediction_horizon_days=90,
        )

        revised = engine.revise_planner_input(original, analysis)

        assert revised.company_name == "Acme"
        assert revised.website == "https://acme.test"
        assert revised.description == "An ACME."
        assert revised.prediction_horizon_days == 90

    def test_stale_topic_does_not_consume_the_next_iteration(self) -> None:
        """A refresh makes nothing more complete, so it does not block."""
        summary = _summary((_validation("product", score=0.95),))
        context = _context(
            (
                TopicInsight(
                    topic_id="product",
                    knowledge=TopicKnowledge.STALE,
                    age_days=400,
                    freshness_requirement_days=90,
                ),
            )
        )
        engine = ResidualGapEngine()
        analysis = engine.analyze(
            iteration=1, plan=_plan("product"), validation=summary, context=context
        )
        original = PlannerInput(company_name="Acme")

        default = engine.revise_planner_input(original, analysis)
        refreshed = engine.revise_planner_input(
            original, analysis, refresh_stale=True
        )

        assert "product" in default.known_topics
        assert "product" in refreshed.partial_topics

    def test_memory_known_topics_narrow_future_work(self) -> None:
        """Memory reduces the next plan instead of being re-researched."""
        summary = _summary((_validation("product", score=0.95),))
        context = _context(
            (
                TopicInsight(
                    topic_id="product",
                    knowledge=TopicKnowledge.KNOWN,
                    evidence_count=4,
                    mean_confidence=0.95,
                ),
                TopicInsight(
                    topic_id="team",
                    knowledge=TopicKnowledge.PARTIAL,
                    evidence_count=1,
                    mean_confidence=0.4,
                ),
            )
        )
        engine = ResidualGapEngine()
        analysis = engine.analyze(
            iteration=1,
            plan=_plan("product", "team", "market"),
            validation=summary,
            context=context,
        )

        revised = engine.revise_planner_input(
            PlannerInput(company_name="Acme"), analysis, context=context
        )

        assert "product" in revised.known_topics
        assert "market" in revised.partial_topics

    def test_revision_is_deterministic(self) -> None:
        engine, analysis = self._engine_and_analysis()
        original = PlannerInput(company_name="Acme")

        first = engine.revise_planner_input(original, analysis)
        second = engine.revise_planner_input(original, analysis)

        assert first.to_dict() == second.to_dict()


# ----------------------------------------------------------------------
# Follow-up requests
# ----------------------------------------------------------------------


class TestFollowUpTasks:
    """One deterministic request per gap, and none for satisfied work."""

    def test_no_gaps_means_no_follow_up_work(self) -> None:
        summary = _summary((_validation("product", score=0.95),))
        engine = ResidualGapEngine()
        analysis = engine.analyze(
            iteration=1, plan=_plan("product"), validation=summary
        )

        assert engine.follow_up_tasks(analysis) == ()
        assert engine.actionable_tasks(analysis) == ()

    def test_one_request_per_gap(self) -> None:
        engine = ResidualGapEngine()
        analysis = engine.analyze(iteration=1, plan=_plan("product", "team"))

        tasks = engine.follow_up_tasks(analysis)

        assert len(tasks) == 2
        assert {task.topic_id for task in tasks} == {"product", "team"}

    def test_stale_is_excluded_from_actionable_work(self) -> None:
        summary = _summary((_validation("product", score=0.95),))
        context = _context(
            (
                TopicInsight(
                    topic_id="product",
                    knowledge=TopicKnowledge.STALE,
                    age_days=400,
                    freshness_requirement_days=90,
                ),
            )
        )
        engine = ResidualGapEngine()
        analysis = engine.analyze(
            iteration=1, plan=_plan("product"), validation=summary, context=context
        )

        assert len(engine.follow_up_tasks(analysis)) == 1
        assert engine.actionable_tasks(analysis) == ()

    def test_request_carries_its_gap_context(self) -> None:
        engine = ResidualGapEngine()
        analysis = engine.analyze(iteration=3, plan=_plan("product"))

        task = engine.follow_up_tasks(analysis)[0]

        assert task.task_id == "followup_missing_product"
        assert task.kind == GapKind.MISSING
        assert task.iteration == 3
        assert task.is_blocking is True
        assert task.reason == analysis.gaps[0].reason

    def test_follow_ups_are_deterministic(self) -> None:
        engine = ResidualGapEngine()
        analysis = engine.analyze(iteration=1, plan=_plan("product", "team"))

        first = engine.follow_up_tasks(analysis)
        second = engine.follow_up_tasks(analysis)

        assert [t.to_dict() for t in first] == [t.to_dict() for t in second]


# ----------------------------------------------------------------------
# Memory context
# ----------------------------------------------------------------------


class TestMemoryContext:
    """Context is derived from artifacts and memory without mutating them."""

    def test_context_describes_planned_topics(self) -> None:
        context = ResidualGapEngine().build_memory_context(
            company_name="Acme",
            plan=_plan("product", "team"),
        )

        assert {insight.topic_id for insight in context.insights} == {
            "product",
            "team",
        }
        assert context.company_name == "Acme"

    def test_unvalidated_topic_is_missing(self) -> None:
        context = ResidualGapEngine().build_memory_context(
            plan=_plan("product"),
        )

        assert context.insights[0].knowledge == TopicKnowledge.MISSING
        assert context.insights[0].evidence_count == 0

    def test_validated_topic_is_known(self) -> None:
        summary = _summary((_validation("product", score=0.95),))
        context = ResidualGapEngine().build_memory_context(
            plan=_plan("product"),
            validation=summary,
        )

        assert context.insights[0].knowledge == TopicKnowledge.KNOWN
        assert context.insights[0].source == "validation"

    def test_ages_are_caller_supplied_not_read_from_a_clock(self) -> None:
        summary = _summary((_validation("product", score=0.95),))
        engine = ResidualGapEngine()

        fresh = engine.build_memory_context(
            plan=_plan("product"), validation=summary, ages={"product": 10}
        )
        aged = engine.build_memory_context(
            plan=_plan("product"), validation=summary, ages={"product": 400}
        )

        assert fresh.insights[0].knowledge == TopicKnowledge.KNOWN
        assert aged.insights[0].knowledge == TopicKnowledge.STALE

    def test_freshness_requirement_comes_from_the_plan_task(self) -> None:
        summary = _summary((_validation("product", score=0.95),))
        context = ResidualGapEngine().build_memory_context(
            plan=_plan("product", task_overrides={"freshness_requirement_days": 7}),
            validation=summary,
        )

        assert context.insights[0].freshness_requirement_days == 7

    def test_known_and_partial_projections_follow_insights(self) -> None:
        summary = _summary((_validation("product", score=0.95),))
        context = ResidualGapEngine().build_memory_context(
            plan=_plan("product"),
            validation=summary,
        )

        assert context.known_topics == ("product",)
        assert context.partial_topics == ()

    def test_context_order_is_stable(self) -> None:
        engine = ResidualGapEngine()

        first = engine.build_memory_context(plan=_plan("team", "product"))
        second = engine.build_memory_context(plan=_plan("product", "team"))

        assert [i.topic_id for i in first.insights] == [
            i.topic_id for i in second.insights
        ]

    def test_context_is_json_serializable(self) -> None:
        context = ResidualGapEngine().build_memory_context(plan=_plan("product"))

        json.dumps(context.to_dict())


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


class TestInputValidation:
    """Bad input is rejected loudly rather than analysed into a wrong answer."""

    @pytest.mark.parametrize("iteration", [0, -1, True, "1", 1.0, None])
    def test_invalid_iteration_rejected(self, iteration: object) -> None:
        with pytest.raises(InvalidIterationInputError):
            ResidualGapEngine().analyze(iteration=iteration)  # type: ignore[arg-type]

    def test_wrong_context_type_rejected(self) -> None:
        with pytest.raises(InvalidIterationInputError):
            ResidualGapEngine().analyze(
                iteration=1,
                plan=_plan("product"),
                context="not a context",  # type: ignore[arg-type]
            )

    def test_wrong_policy_type_rejected(self) -> None:
        with pytest.raises(InvalidIterationInputError):
            ResidualGapEngine(policy="not a policy")  # type: ignore[arg-type]

    def test_wrong_analysis_type_rejected_by_follow_up(self) -> None:
        with pytest.raises(InvalidIterationInputError):
            ResidualGapEngine().follow_up_tasks("not an analysis")  # type: ignore[arg-type]

    def test_negative_age_rejected(self) -> None:
        with pytest.raises(InvalidIterationInputError):
            ResidualGapEngine().build_memory_context(
                plan=_plan("product"),
                ages={"product": -1},
            )

    def test_wrong_gap_type_rejected_by_identity_helpers(self) -> None:
        with pytest.raises(InvalidIterationInputError):
            gap_id("not a gap")  # type: ignore[arg-type]
        with pytest.raises(InvalidIterationInputError):
            follow_up_id("not a gap")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Purity
# ----------------------------------------------------------------------


class TestPurity:
    """The engine holds no state and mutates nothing it reads."""

    def test_one_engine_serves_many_analyses(self) -> None:
        engine = ResidualGapEngine()
        first = engine.analyze(iteration=1, plan=_plan("product"))
        second = engine.analyze(iteration=1, plan=_plan("team"))

        assert first.analysis_id != second.analysis_id
        assert first.gaps[0].topic_id == "product"
        assert second.gaps[0].topic_id == "team"

    def test_analysis_does_not_mutate_its_inputs(self) -> None:
        plan = _plan("product")
        summary = _summary((_validation("product", score=0.3),))
        plan_before = plan.to_dict()
        summary_before = summary.to_dict()

        ResidualGapEngine().analyze(iteration=1, plan=plan, validation=summary)

        assert plan.to_dict() == plan_before
        assert summary.to_dict() == summary_before

    def test_engine_exposes_its_configuration(self) -> None:
        policy = IterationPolicy(minimum_evidence_per_topic=7)
        engine = ResidualGapEngine(policy=policy)

        assert engine.policy is policy
        assert engine.topics
