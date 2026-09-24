"""Rule engine for the Research Planner — Phase 8 Sprint 1.

A rule inspects a :class:`PlannerInput` and decides whether a research
area is under-covered, returning a :class:`KnowledgeGap` when it is.

Extending the engine does not require modifying planner logic: subclass
:class:`ResearchRule`, override ``evaluate``, and register it with
:func:`register_rule`.  Registered rules run *before* the built-in
coverage rules, so a registered rule can override the default behaviour
for a topic.  Rules are evaluated in deterministic registry order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from predictron_engine.research.exceptions import RuleRegistrationError
from predictron_engine.research.models import (
    EvidenceStatus,
    KnowledgeGap,
    PlannerInput,
    ResearchTopic,
)
from predictron_engine.research.topics import RESEARCH_TOPICS, get_topic


def coverage_status(planner_input: PlannerInput, topic_id: str) -> EvidenceStatus:
    """Return the evidence status for a topic according to the input."""
    if topic_id in planner_input.known_topics:
        return EvidenceStatus.COMPLETE
    if topic_id in planner_input.partial_topics:
        return EvidenceStatus.PARTIAL
    return EvidenceStatus.NONE


class ResearchRule(ABC):
    """Base contract every research rule must satisfy."""

    @property
    @abstractmethod
    def topic_id(self) -> str:
        """The identifier of the topic this rule reasons about."""

    @abstractmethod
    def evaluate(self, planner_input: PlannerInput) -> KnowledgeGap | None:
        """Return a knowledge gap for the input, or ``None`` when covered."""


class CoverageRule(ResearchRule):
    """Generic coverage rule bound to one research topic.

    Returns a gap whenever the planner input does not already carry
    complete evidence for the topic.
    """

    def __init__(self, topic: ResearchTopic) -> None:
        self._topic = topic

    @property
    def topic_id(self) -> str:
        return self._topic.topic_id

    def evaluate(self, planner_input: PlannerInput) -> KnowledgeGap | None:
        status = coverage_status(planner_input, self.topic_id)
        if status is EvidenceStatus.COMPLETE:
            return None
        if status is EvidenceStatus.PARTIAL:
            reason = (
                f"Partial {self._topic.name.lower()} evidence supplied; "
                "further research recommended."
            )
        else:
            reason = (
                f"No {self._topic.name.lower()} evidence supplied for planning."
            )
        return KnowledgeGap(
            topic_id=self.topic_id,
            evidence_status=status,
            reason=reason,
        )


class MissingFoundersRule(CoverageRule):
    """Missing founders evidence → create a Founders research task."""

    def __init__(self) -> None:
        super().__init__(get_topic("founders"))


class MissingFundingRule(CoverageRule):
    """Missing funding evidence → create a Funding research task."""

    def __init__(self) -> None:
        super().__init__(get_topic("funding"))


class MissingPricingRule(CoverageRule):
    """Missing pricing evidence → create a Pricing research task."""

    def __init__(self) -> None:
        super().__init__(get_topic("pricing"))


class MissingMarketRule(CoverageRule):
    """Missing market information → create a Market research task."""

    def __init__(self) -> None:
        super().__init__(get_topic("market"))


class MissingTechnologyRule(CoverageRule):
    """Missing technology evidence → create a Technology research task."""

    def __init__(self) -> None:
        super().__init__(get_topic("technology"))


# Built-in rules: one coverage rule per taxonomy topic, in taxonomy order.
DEFAULT_RULES: tuple[ResearchRule, ...] = tuple(
    CoverageRule(topic) for topic in RESEARCH_TOPICS
)

# Extension registry; new rules are appended in registration order.
_EXTENSIONS: list[ResearchRule] = []


def register_rule(rule: ResearchRule) -> None:
    """Register a custom rule for future plans.

    Registered rules run before the built-in coverage rules, so they can
    override the default behaviour for a topic.  A rule may reference a
    topic that is not in the built-in taxonomy; such plans require a
    :class:`~predictron_engine.research.planner.ResearchPlanner`
    constructed with a matching custom ``topics`` registry.

    Raises
    ------
    RuleRegistrationError:
        When the rule is not a :class:`ResearchRule` or a rule is
        already registered for the same topic.
    """
    if not isinstance(rule, ResearchRule):
        raise RuleRegistrationError("rule must be a ResearchRule instance")
    if not rule.topic_id.strip():
        raise RuleRegistrationError("rule topic_id must not be empty")
    for existing in _EXTENSIONS:
        if existing.topic_id == rule.topic_id:
            raise RuleRegistrationError(
                f"rule already registered for topic: {rule.topic_id!r}"
            )
    _EXTENSIONS.append(rule)


def registered_rules() -> tuple[ResearchRule, ...]:
    """Return the currently registered extension rules, in order."""
    return tuple(_EXTENSIONS)


def research_rules() -> tuple[ResearchRule, ...]:
    """Return the full rule set: registered extensions then built-ins.

    Extensions run first so a registered rule overrides the built-in
    coverage rule for its topic.
    """
    return registered_rules() + DEFAULT_RULES


def evaluate_rules(
    planner_input: PlannerInput,
    rules: tuple[ResearchRule, ...] | None = None,
) -> tuple[KnowledgeGap, ...]:
    """Evaluate rules in deterministic order, deduplicating by topic.

    The first rule encountered for a topic decides the outcome: it may
    produce a gap, or it may suppress the topic entirely by returning
    ``None`` (which overrides any later rule for the same topic).  When
    ``rules`` is ``None`` the full registered rule set is used.
    """
    selected = rules if rules is not None else research_rules()
    gaps: list[KnowledgeGap] = []
    decided: set[str] = set()
    for rule in selected:
        if rule.topic_id in decided:
            continue
        decided.add(rule.topic_id)
        gap = rule.evaluate(planner_input)
        if gap is not None:
            gaps.append(gap)
    return tuple(gaps)
