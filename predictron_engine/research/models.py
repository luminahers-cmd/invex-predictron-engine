"""Research Intelligence models — Phase 8 Sprints 1, 2, and 3.

Immutable value objects describing the deterministic planning layer
(Sprint 1), the source discovery layer (Sprint 2), and the evidence
collection layer (Sprint 3):

* :class:`PlannerInput` — what a caller tells the planner.
* :class:`ResearchTopic` — one node in the research taxonomy.
* :class:`KnowledgeGap` — a missing or under-covered research area.
* :class:`ResearchTask` — one concrete piece of research to perform.
* :class:`ResearchPlan` — the complete, executable research plan.
* :class:`ResearchPriority` — deterministic priority band.
* :class:`EvidenceStatus` — how much evidence exists for a topic.
* :class:`SourceCategory` — typed category of a research source.
* :class:`ResearchSource` — metadata describing a candidate source.
* :class:`SourceScore` — component scores for one source against a topic.
* :class:`SourceRank` — one source ranked for one topic.
* :class:`SourceRecommendation` — ranked sources for one research task.
* :class:`SourceDiscoveryPlan` — ranked sources for a whole plan.
* :class:`Evidence` — one deterministic piece of collected evidence.
* :class:`EvidenceReference` — source provenance for an evidence item.
* :class:`EvidenceMetadata` — metadata describing one collector.
* :class:`EvidenceCollection` — aggregate of collected evidence.
* :class:`CollectionStatus` — overall collection outcome.
* :class:`CollectionResult` — the result of executing a research plan.

Models are ``frozen`` dataclasses so they cannot mutate after creation.
Every model exposes ``to_dict`` / ``from_dict`` for lossless
serialization.  Nothing here performs I/O, calls models, or touches the
network — the planning, discovery, and collection layers are pure and
deterministic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

from predictron_engine.research.exceptions import (
    InvalidEvidence,
    InvalidPlannerInputError,
)

PLAN_SCHEMA_VERSION = "1.0.0"
SOURCE_SCHEMA_VERSION = "1.0.0"
COLLECTION_SCHEMA_VERSION = "1.0.0"


class ResearchPriority(str, Enum):
    """Deterministic priority band for a research task."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def order_value(self) -> int:
        """Deterministic rank; higher means more urgent."""
        return len(tuple(ResearchPriority)) - tuple(ResearchPriority).index(
            self
        )


class EvidenceStatus(str, Enum):
    """How much evidence the planner input carries for a topic."""

    NONE = "none"
    PARTIAL = "partial"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PlannerInput:
    """Immutable description of what a caller already knows.

    Parameters
    ----------
    company_name:
        Name of the company being researched.  Required and stripped.
    website:
        Optional website URL.  When supplied the planner treats the
        ``company_website`` source category as available.
    description:
        Optional free-form description supplied by the caller.
    known_topics:
        Topic identifiers for which evidence is already complete;
        those topics are excluded from the plan.
    partial_topics:
        Topic identifiers for which only partial evidence exists;
        those topics still produce tasks with a ``PARTIAL`` gap.
    prediction_horizon_days:
        Length of the prediction window; scales the freshness weight.
    """

    company_name: str
    website: str = ""
    description: str = ""
    known_topics: tuple[str, ...] = ()
    partial_topics: tuple[str, ...] = ()
    prediction_horizon_days: int = 365

    def __post_init__(self) -> None:
        name = self.company_name.strip()
        if not name:
            raise InvalidPlannerInputError("company_name must not be empty")
        website = self.website.strip()
        _validate_website(website)
        if self.prediction_horizon_days < 1:
            raise InvalidPlannerInputError(
                "prediction_horizon_days must be greater than zero"
            )
        known = tuple(dict.fromkeys(self.known_topics))
        partial = tuple(dict.fromkeys(self.partial_topics))
        overlap = sorted(set(known) & set(partial))
        if overlap:
            raise InvalidPlannerInputError(
                f"topics cannot be both known and partial: {overlap}"
            )
        object.__setattr__(self, "company_name", name)
        object.__setattr__(self, "website", website)
        object.__setattr__(self, "known_topics", known)
        object.__setattr__(self, "partial_topics", partial)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "company_name": self.company_name,
            "website": self.website,
            "description": self.description,
            "known_topics": list(self.known_topics),
            "partial_topics": list(self.partial_topics),
            "prediction_horizon_days": self.prediction_horizon_days,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PlannerInput:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            company_name=_require_str(data, "company_name"),
            website=_require_str(data, "website", default=""),
            description=_require_str(data, "description", default=""),
            known_topics=tuple(_require_str_list(data, "known_topics")),
            partial_topics=tuple(_require_str_list(data, "partial_topics")),
            prediction_horizon_days=_require_int(data, "prediction_horizon_days"),
        )


def _validate_website(website: str) -> None:
    """Validate an optional website URL (scheme must be http or https)."""
    if not website:
        return
    parts = urlsplit(website)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise InvalidPlannerInputError(f"invalid website URL: {website!r}")


@dataclass(frozen=True)
class ResearchTopic:
    """One node in the deterministic research taxonomy.

    Parameters
    ----------
    topic_id:
        Stable snake_case identifier, e.g. ``"founders"``.
    name:
        Human-readable name, e.g. ``"Founders"``.
    description:
        What researching this topic answers.
    importance:
        Base importance of the topic to a prediction, in ``[0, 1]``.
    prediction_impact:
        How strongly the topic moves the overall prediction, in ``[0, 1]``.
    freshness_requirement_days:
        Days after which evidence for this topic should be refreshed.
    freshness_sensitivity:
        How quickly evidence decays for this topic, in ``[0, 1]``.
    source_categories:
        Preferred source *categories* (metadata only — never scraping).
    dependencies:
        Topic identifiers that must be researched before this topic.
    """

    topic_id: str
    name: str
    description: str
    importance: float
    prediction_impact: float
    freshness_requirement_days: int
    freshness_sensitivity: float
    source_categories: tuple[str, ...]
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.topic_id.strip():
            raise ValueError("topic_id must not be empty")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.description.strip():
            raise ValueError("description must not be empty")
        if not _in_unit_range(self.importance):
            raise ValueError("importance must be within [0, 1]")
        if not _in_unit_range(self.prediction_impact):
            raise ValueError("prediction_impact must be within [0, 1]")
        if not _in_unit_range(self.freshness_sensitivity):
            raise ValueError("freshness_sensitivity must be within [0, 1]")
        if self.freshness_requirement_days < 1:
            raise ValueError(
                "freshness_requirement_days must be greater than zero"
            )
        if not self.source_categories:
            raise ValueError("source_categories must not be empty")
        if self.topic_id in self.dependencies:
            raise ValueError("a topic cannot depend on itself")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "description": self.description,
            "importance": self.importance,
            "prediction_impact": self.prediction_impact,
            "freshness_requirement_days": self.freshness_requirement_days,
            "freshness_sensitivity": self.freshness_sensitivity,
            "source_categories": list(self.source_categories),
            "dependencies": list(self.dependencies),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResearchTopic:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            topic_id=_require_str(data, "topic_id"),
            name=_require_str(data, "name"),
            description=_require_str(data, "description"),
            importance=_require_float(data, "importance"),
            prediction_impact=_require_float(data, "prediction_impact"),
            freshness_requirement_days=_require_int(
                data, "freshness_requirement_days"
            ),
            freshness_sensitivity=_require_float(data, "freshness_sensitivity"),
            source_categories=tuple(_require_str_list(data, "source_categories")),
            dependencies=tuple(_require_str_list(data, "dependencies")),
        )


@dataclass(frozen=True)
class KnowledgeGap:
    """A missing or under-covered research area detected by the planner."""

    topic_id: str
    evidence_status: EvidenceStatus
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "topic_id": self.topic_id,
            "evidence_status": self.evidence_status.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> KnowledgeGap:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        status_raw = _require_str(data, "evidence_status")
        return cls(
            topic_id=_require_str(data, "topic_id"),
            evidence_status=EvidenceStatus(status_raw),
            reason=_require_str(data, "reason"),
        )


@dataclass(frozen=True)
class ResearchTask:
    """One concrete, ordered unit of research within a plan."""

    task_id: str
    topic_id: str
    title: str
    description: str
    priority: ResearchPriority
    priority_score: float
    importance: float
    prediction_impact: float
    freshness_requirement_days: int
    dependency_ids: tuple[str, ...]
    source_categories: tuple[str, ...]
    priority_rank: int
    execution_order: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "task_id": self.task_id,
            "topic_id": self.topic_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "priority_score": self.priority_score,
            "importance": self.importance,
            "prediction_impact": self.prediction_impact,
            "freshness_requirement_days": self.freshness_requirement_days,
            "dependency_ids": list(self.dependency_ids),
            "source_categories": list(self.source_categories),
            "priority_rank": self.priority_rank,
            "execution_order": self.execution_order,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResearchTask:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            task_id=_require_str(data, "task_id"),
            topic_id=_require_str(data, "topic_id"),
            title=_require_str(data, "title"),
            description=_require_str(data, "description"),
            priority=ResearchPriority(_require_str(data, "priority")),
            priority_score=_require_float(data, "priority_score"),
            importance=_require_float(data, "importance"),
            prediction_impact=_require_float(data, "prediction_impact"),
            freshness_requirement_days=_require_int(
                data, "freshness_requirement_days"
            ),
            dependency_ids=tuple(_require_str_list(data, "dependency_ids")),
            source_categories=tuple(
                _require_str_list(data, "source_categories")
            ),
            priority_rank=_require_int(data, "priority_rank"),
            execution_order=_require_int(data, "execution_order"),
        )


@dataclass(frozen=True)
class ResearchPlan:
    """The complete, executable deterministic research plan.

    Parameters
    ----------
    schema_version:
        Plan serialization schema version.
    plan_id:
        Stable content hash identifying the plan.
    company_name:
        Company under research.
    input:
        The :class:`PlannerInput` that produced this plan.
    gaps:
        Knowledge gaps detected, in deterministic rule order.
    tasks:
        Research tasks in executable (dependency) order.
    topics_researched:
        Topic identifiers covered by the plan, sorted.
    available_source_categories:
        Source categories already available to the planner.
    """

    schema_version: str
    plan_id: str
    company_name: str
    input: PlannerInput
    gaps: tuple[KnowledgeGap, ...]
    tasks: tuple[ResearchTask, ...]
    topics_researched: tuple[str, ...]
    available_source_categories: tuple[str, ...]

    def task_by_topic(self, topic_id: str) -> ResearchTask | None:
        """Return the task for ``topic_id`` or ``None`` when absent."""
        for task in self.tasks:
            if task.topic_id == topic_id:
                return task
        return None

    def priority_ordered_tasks(self) -> tuple[ResearchTask, ...]:
        """Return tasks sorted by priority (score desc, rank asc)."""
        return tuple(
            sorted(
                self.tasks,
                key=lambda t: (-t.priority_score, t.priority_rank),
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a fully JSON-ready dictionary."""
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "company_name": self.company_name,
            "input": self.input.to_dict(),
            "gaps": [gap.to_dict() for gap in self.gaps],
            "tasks": [task.to_dict() for task in self.tasks],
            "topics_researched": list(self.topics_researched),
            "available_source_categories": list(
                self.available_source_categories
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResearchPlan:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            schema_version=_require_str(data, "schema_version"),
            plan_id=_require_str(data, "plan_id"),
            company_name=_require_str(data, "company_name"),
            input=PlannerInput.from_dict(
                _require_dict(data, "input"),
            ),
            gaps=tuple(
                KnowledgeGap.from_dict(item)
                for item in _require_list_of_dicts(data, "gaps")
            ),
            tasks=tuple(
                ResearchTask.from_dict(item)
                for item in _require_list_of_dicts(data, "tasks")
            ),
            topics_researched=tuple(_require_str_list(data, "topics_researched")),
            available_source_categories=tuple(
                _require_str_list(data, "available_source_categories")
            ),
        )


class SourceCategory(str, Enum):
    """Typed category describing one research source (metadata only).

    The category is a controlled label; no scraping, HTTP, or collection
    happens for any category.  Later Phase 8 sprints consume these labels.
    """

    OFFICIAL_WEBSITE = "official_website"
    COMPANY_BLOG = "company_blog"
    ENGINEERING_BLOG = "engineering_blog"
    DOCUMENTATION = "documentation"
    API_DOCUMENTATION = "api_documentation"
    DEVELOPER_DOCS = "developer_docs"
    GITHUB = "github"
    NEWS = "news"
    PRESS_RELEASES = "press_releases"
    CASE_STUDIES = "case_studies"
    G2 = "g2"
    CAPTERRA = "capterra"
    APP_STORE = "app_store"
    GOOGLE_PLAY = "google_play"
    REDDIT = "reddit"
    HACKER_NEWS = "hacker_news"
    LINKEDIN = "linkedin"
    WELLFOUND = "wellfound"
    JOB_LISTINGS = "job_listings"
    CAREERS = "careers"
    CRUNCHBASE = "crunchbase"
    SEC_EDGAR = "sec_edgar"
    YC = "yc"
    PRODUCT_HUNT = "product_hunt"
    PATENTS = "patents"
    RESEARCH_PAPERS = "research_papers"


@dataclass(frozen=True)
class ResearchSource:
    """Immutable metadata describing one candidate research source.

    Sources are metadata only: no endpoint, no client, no credential.
    Each source describes *where* evidence for a research topic would be
    found so downstream collection sprints can consume it directly.

    Parameters
    ----------
    identifier:
        Stable snake_case identifier, e.g. ``"crunchbase"``.
    display_name:
        Human-readable name, e.g. ``"Crunchbase"``.
    source_category:
        Typed :class:`SourceCategory` for the source.
    trust_score:
        How authoritative the source is, in ``[0, 1]``.
    freshness_score:
        How current the source's data tends to be, in ``[0, 1]``.
    coverage_score:
        How broad the source's coverage is, in ``[0, 1]``.
    relative_cost:
        Relative monetary/effort cost to collect from, in ``[0, 1]``.
    supports_structured_data:
        Whether the source exposes structured data (vs. raw prose).
    preferred_topics:
        Topic identifiers the source covers best, in preference order.
    description:
        Optional human description of the source.
    """

    identifier: str
    display_name: str
    source_category: SourceCategory
    trust_score: float
    freshness_score: float
    coverage_score: float
    relative_cost: float
    supports_structured_data: bool
    preferred_topics: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("identifier must not be empty")
        if not self.display_name.strip():
            raise ValueError("display_name must not be empty")
        if not isinstance(self.source_category, SourceCategory):
            raise ValueError("source_category must be a SourceCategory")
        if not _in_unit_range(self.trust_score):
            raise ValueError("trust_score must be within [0, 1]")
        if not _in_unit_range(self.freshness_score):
            raise ValueError("freshness_score must be within [0, 1]")
        if not _in_unit_range(self.coverage_score):
            raise ValueError("coverage_score must be within [0, 1]")
        if not _in_unit_range(self.relative_cost):
            raise ValueError("relative_cost must be within [0, 1]")
        topics = tuple(dict.fromkeys(self.preferred_topics))
        if not topics:
            raise ValueError("preferred_topics must not be empty")
        object.__setattr__(self, "identifier", self.identifier.strip())
        object.__setattr__(self, "display_name", self.display_name.strip())
        object.__setattr__(self, "preferred_topics", topics)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "identifier": self.identifier,
            "display_name": self.display_name,
            "source_category": self.source_category.value,
            "trust_score": self.trust_score,
            "freshness_score": self.freshness_score,
            "coverage_score": self.coverage_score,
            "relative_cost": self.relative_cost,
            "supports_structured_data": self.supports_structured_data,
            "preferred_topics": list(self.preferred_topics),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResearchSource:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            identifier=_require_str(data, "identifier"),
            display_name=_require_str(data, "display_name"),
            source_category=SourceCategory(_require_str(data, "source_category")),
            trust_score=_require_float(data, "trust_score"),
            freshness_score=_require_float(data, "freshness_score"),
            coverage_score=_require_float(data, "coverage_score"),
            relative_cost=_require_float(data, "relative_cost"),
            supports_structured_data=_require_bool(
                data, "supports_structured_data"
            ),
            preferred_topics=tuple(
                _require_str_list(data, "preferred_topics")
            ),
            description=_require_str(data, "description", default=""),
        )


@dataclass(frozen=True)
class SourceScore:
    """Component and total scores for one source against one topic.

    All component scores are normalized values in ``[0, 1]``; ``total``
    is the deterministic weighted combination used for ranking.
    """

    trust: float
    coverage: float
    freshness: float
    cost: float
    structured_data: float
    topic_fit: float
    total: float

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "trust": self.trust,
            "coverage": self.coverage,
            "freshness": self.freshness,
            "cost": self.cost,
            "structured_data": self.structured_data,
            "topic_fit": self.topic_fit,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SourceScore:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            trust=_require_float(data, "trust"),
            coverage=_require_float(data, "coverage"),
            freshness=_require_float(data, "freshness"),
            cost=_require_float(data, "cost"),
            structured_data=_require_float(data, "structured_data"),
            topic_fit=_require_float(data, "topic_fit"),
            total=_require_float(data, "total"),
        )


@dataclass(frozen=True)
class SourceRank:
    """One source ranked for one topic.

    ``rank`` is a 1-based deterministic position; ``source`` is the
    candidate and ``score`` the computed :class:`SourceScore` behind it.
    """

    rank: int
    source: ResearchSource
    score: SourceScore

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be at least 1")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "rank": self.rank,
            "source": self.source.to_dict(),
            "score": self.score.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SourceRank:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            rank=_require_int(data, "rank"),
            source=ResearchSource.from_dict(_require_dict(data, "source")),
            score=SourceScore.from_dict(_require_dict(data, "score")),
        )


@dataclass(frozen=True)
class SourceRecommendation:
    """The ranked candidate sources for one :class:`ResearchTask`.

    ``sources`` holds :class:`SourceRank` entries already ordered by
    the deterministic ranker (rank 1 first).  The task reference is by
    identifier pair so the recommendation stays a lightweight value.
    """

    task_id: str
    topic_id: str
    sources: tuple[SourceRank, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.topic_id.strip():
            raise ValueError("topic_id must not be empty")
        object.__setattr__(self, "task_id", self.task_id.strip())
        object.__setattr__(self, "topic_id", self.topic_id.strip())

    def top_source(self) -> ResearchSource | None:
        """Return the highest-ranked source, or ``None`` when empty."""
        if not self.sources:
            return None
        return self.sources[0].source

    def source_identifiers(self) -> tuple[str, ...]:
        """Return ranked source identifiers."""
        return tuple(entry.source.identifier for entry in self.sources)

    def score_for(self, identifier: str) -> SourceScore | None:
        """Return the score for ``identifier``, or ``None`` when absent."""
        for entry in self.sources:
            if entry.source.identifier == identifier:
                return entry.score
        return None

    def has_recommendations(self) -> bool:
        """Return whether at least one source was recommended."""
        return bool(self.sources)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "task_id": self.task_id,
            "topic_id": self.topic_id,
            "sources": [entry.to_dict() for entry in self.sources],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SourceRecommendation:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            task_id=_require_str(data, "task_id"),
            topic_id=_require_str(data, "topic_id"),
            sources=tuple(
                SourceRank.from_dict(item)
                for item in _require_list_of_dicts(data, "sources")
            ),
        )


@dataclass(frozen=True)
class SourceDiscoveryPlan:
    """Ranked candidate sources for an entire :class:`ResearchPlan`.

    ``source_tasks`` preserves the research plan's tasks verbatim so
    dependency and execution order survive discovery.  Every task has a
    matching :class:`SourceRecommendation` (positionally aligned).
    """

    schema_version: str
    plan_id: str
    company_name: str
    source_tasks: tuple[ResearchTask, ...]
    recommendations: tuple[SourceRecommendation, ...]

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")
        if not self.plan_id.strip():
            raise ValueError("plan_id must not be empty")
        if not self.company_name.strip():
            raise ValueError("company_name must not be empty")
        tasks_by_topic = {task.topic_id: task for task in self.source_tasks}
        for recommendation in self.recommendations:
            task = tasks_by_topic.get(recommendation.topic_id)
            if task is None:
                raise ValueError(
                    "recommendation references an unknown topic "
                    f"for task: {recommendation.topic_id!r}"
                )
            if task.task_id != recommendation.task_id:
                raise ValueError(
                    "recommendation task_id does not match its task"
                )

    @property
    def topics_covered(self) -> tuple[str, ...]:
        """Return the topic identifiers covered, in plan order."""
        return tuple(task.topic_id for task in self.source_tasks)

    def recommendation_for(self, topic_id: str) -> SourceRecommendation | None:
        """Return the recommendation for ``topic_id`` or ``None``."""
        for recommendation in self.recommendations:
            if recommendation.topic_id == topic_id:
                return recommendation
        return None

    def sources_for(self, topic_id: str) -> tuple[ResearchSource, ...]:
        """Return ranked sources for ``topic_id`` (empty when absent)."""
        recommendation = self.recommendation_for(topic_id)
        if recommendation is None:
            return ()
        return tuple(entry.source for entry in recommendation.sources)

    def source_identifiers_for(self, topic_id: str) -> tuple[str, ...]:
        """Return ranked source identifiers for ``topic_id``."""
        recommendation = self.recommendation_for(topic_id)
        if recommendation is None:
            return ()
        return recommendation.source_identifiers()

    def to_dict(self) -> dict[str, object]:
        """Serialize to a fully JSON-ready dictionary."""
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "company_name": self.company_name,
            "source_tasks": [task.to_dict() for task in self.source_tasks],
            "recommendations": [
                recommendation.to_dict()
                for recommendation in self.recommendations
            ],
            "topics_covered": list(self.topics_covered),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SourceDiscoveryPlan:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            schema_version=_require_str(data, "schema_version"),
            plan_id=_require_str(data, "plan_id"),
            company_name=_require_str(data, "company_name"),
            source_tasks=tuple(
                ResearchTask.from_dict(item)
                for item in _require_list_of_dicts(data, "source_tasks")
            ),
            recommendations=tuple(
                SourceRecommendation.from_dict(item)
                for item in _require_list_of_dicts(data, "recommendations")
            ),
        )


def _in_unit_range(value: float) -> bool:
    """Whether a float lies in the closed unit interval."""
    return 0.0 <= value <= 1.0


def _require_str(
    data: Mapping[str, object],
    key: str,
    *,
    default: str | None = None,
) -> str:
    """Read and validate a string field from a serialized dictionary."""
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"expected string for '{key}'")
    return value


def _require_int(data: Mapping[str, object], key: str) -> int:
    """Read and validate an int field from a serialized dictionary."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected int for '{key}'")
    return value


def _require_float(data: Mapping[str, object], key: str) -> float:
    """Read and validate a float field from a serialized dictionary."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected number for '{key}'")
    return float(value)


def _require_bool(data: Mapping[str, object], key: str) -> bool:
    """Read and validate a bool field from a serialized dictionary."""
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"expected bool for '{key}'")
    return value


def _require_str_list(
    data: Mapping[str, object],
    key: str,
) -> list[str]:
    """Read and validate a string-list field."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"expected list for '{key}'")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"expected list of strings for '{key}'")
        result.append(item)
    return result


def _require_dict(
    data: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    """Read and validate a nested dictionary field."""
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"expected dict for '{key}'")
    return value


def _require_list_of_dicts(
    data: Mapping[str, object],
    key: str,
) -> list[dict[str, object]]:
    """Read and validate a list-of-dicts field."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"expected list for '{key}'")
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"expected list of dicts for '{key}'")
        result.append(item)
    return result


# ----------------------------------------------------------------------
# Sprint 3 — Evidence Collection models
# ----------------------------------------------------------------------


class CollectionStatus(str, Enum):
    """Deterministic outcome of a collection run over a research plan.

    * ``SUCCESS`` — every task in the plan produced evidence.
    * ``PARTIAL`` — some tasks produced evidence and others failed.
    * ``FAILED`` — no task produced evidence.
    * ``EMPTY`` — the plan contained no tasks to execute.
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    EMPTY = "empty"


@dataclass(frozen=True)
class EvidenceReference:
    """Source provenance attached to one :class:`Evidence` item.

    The reference is metadata only — placeholders describe *where* the
    evidence would be found.  It never performs or describes a network
    request.
    """

    source_category: str
    source_identifier: str
    url: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        category = self.source_category.strip()
        identifier = self.source_identifier.strip()
        url = self.url.strip()
        description = self.description.strip()
        if not category:
            raise InvalidEvidence("source_category must not be empty")
        if not identifier:
            raise InvalidEvidence("source_identifier must not be empty")
        object.__setattr__(self, "source_category", category)
        object.__setattr__(self, "source_identifier", identifier)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "description", description)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "source_category": self.source_category,
            "source_identifier": self.source_identifier,
            "url": self.url,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvidenceReference:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            source_category=_require_str(data, "source_category"),
            source_identifier=_require_str(data, "source_identifier"),
            url=_require_str(data, "url", default=""),
            description=_require_str(data, "description", default=""),
        )


@dataclass(frozen=True)
class EvidenceMetadata:
    """Deterministic metadata describing one evidence collector."""

    collector_id: str
    display_name: str
    description: str = ""
    supported_topics: tuple[str, ...] = ()
    deterministic: bool = True

    def __post_init__(self) -> None:
        collector_id = self.collector_id.strip()
        display_name = self.display_name.strip()
        description = self.description.strip()
        topics = tuple(dict.fromkeys(self.supported_topics))
        if not collector_id:
            raise InvalidEvidence("collector_id must not be empty")
        if not display_name:
            raise InvalidEvidence("display_name must not be empty")
        if not topics:
            raise InvalidEvidence("supported_topics must not be empty")
        object.__setattr__(self, "collector_id", collector_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "supported_topics", topics)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "collector_id": self.collector_id,
            "display_name": self.display_name,
            "description": self.description,
            "supported_topics": list(self.supported_topics),
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvidenceMetadata:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            collector_id=_require_str(data, "collector_id"),
            display_name=_require_str(data, "display_name"),
            description=_require_str(data, "description", default=""),
            supported_topics=tuple(_require_str_list(data, "supported_topics")),
            deterministic=_require_bool(data, "deterministic"),
        )


@dataclass(frozen=True)
class Evidence:
    """One deterministic placeholder piece of collected evidence.

    ``evidence_id`` is a stable content hash over the item's fields, so
    the same claim always yields the same identifier and ordering is
    fully deterministic.
    """

    task_id: str
    topic_id: str
    collector_id: str
    category: str
    claim: str
    confidence: float
    reference: EvidenceReference
    evidence_id: str = ""

    def __post_init__(self) -> None:
        task_id = self.task_id.strip()
        topic_id = self.topic_id.strip()
        collector_id = self.collector_id.strip()
        category = self.category.strip()
        claim = self.claim.strip()
        if not task_id:
            raise InvalidEvidence("task_id must not be empty")
        if not topic_id:
            raise InvalidEvidence("topic_id must not be empty")
        if not collector_id:
            raise InvalidEvidence("collector_id must not be empty")
        if not category:
            raise InvalidEvidence("category must not be empty")
        if not claim:
            raise InvalidEvidence("claim must not be empty")
        if not _in_unit_range(self.confidence):
            raise InvalidEvidence("confidence must be within [0, 1]")
        if not isinstance(self.reference, EvidenceReference):
            raise InvalidEvidence("reference must be an EvidenceReference")
        evidence_id = _evidence_digest(
            {
                "task_id": task_id,
                "topic_id": topic_id,
                "collector_id": collector_id,
                "category": category,
                "claim": claim,
                "confidence": self.confidence,
                "reference": self.reference.to_dict(),
            }
        )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "topic_id", topic_id)
        object.__setattr__(self, "collector_id", collector_id)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "claim", claim)
        object.__setattr__(self, "evidence_id", evidence_id)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "task_id": self.task_id,
            "topic_id": self.topic_id,
            "collector_id": self.collector_id,
            "category": self.category,
            "claim": self.claim,
            "confidence": self.confidence,
            "reference": self.reference.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Evidence:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            task_id=_require_str(data, "task_id"),
            topic_id=_require_str(data, "topic_id"),
            collector_id=_require_str(data, "collector_id"),
            category=_require_str(data, "category"),
            claim=_require_str(data, "claim"),
            confidence=_require_float(data, "confidence"),
            reference=EvidenceReference.from_dict(
                _require_dict(data, "reference")
            ),
        )


@dataclass(frozen=True)
class EvidenceCollection:
    """An immutable, deterministically ordered aggregation of evidence.

    Items are re-sorted on construction by topic order and then by
    ``evidence_id`` so any insertion order produces an identical
    collection — the same plan always yields the same aggregate.
    """

    evidence: tuple[Evidence, ...] = ()
    topic_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        items = tuple(self.evidence)
        for item in items:
            if not isinstance(item, Evidence):
                raise InvalidEvidence(
                    "collection entries must be Evidence instances"
                )
        order = tuple(dict.fromkeys(self.topic_order))
        positions = {topic_id: position for position, topic_id in enumerate(order)}

        def _position(item: Evidence) -> int:
            return positions.get(item.topic_id, len(order))

        ordered = tuple(
            sorted(items, key=lambda item: (_position(item), item.evidence_id))
        )
        object.__setattr__(self, "evidence", ordered)
        object.__setattr__(self, "topic_order", order)

    def __len__(self) -> int:
        """Return the number of evidence items."""
        return len(self.evidence)

    def items(self) -> tuple[Evidence, ...]:
        """Return the evidence items in deterministic order."""
        return self.evidence

    def for_topic(self, topic_id: str) -> tuple[Evidence, ...]:
        """Return the evidence items for ``topic_id``, ordered."""
        return tuple(item for item in self.evidence if item.topic_id == topic_id)

    def has_topic(self, topic_id: str) -> bool:
        """Return whether any evidence exists for ``topic_id``."""
        return any(item.topic_id == topic_id for item in self.evidence)

    def topics_covered(self) -> tuple[str, ...]:
        """Return topics with evidence, in deterministic topic order."""
        covered = [
            topic_id
            for topic_id in self.topic_order
            if any(item.topic_id == topic_id for item in self.evidence)
        ]
        covered.extend(_topics_not_in_order(self.evidence, self.topic_order))
        return tuple(covered)

    def evidence_count(self, topic_id: str) -> int:
        """Return how many evidence items exist for ``topic_id``."""
        return sum(1 for item in self.evidence if item.topic_id == topic_id)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a fully JSON-ready dictionary."""
        return {
            "evidence": [item.to_dict() for item in self.evidence],
            "topic_order": list(self.topic_order),
            "topics_covered": list(self.topics_covered()),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvidenceCollection:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            evidence=tuple(
                Evidence.from_dict(item)
                for item in _require_list_of_dicts(data, "evidence")
            ),
            topic_order=tuple(_require_str_list(data, "topic_order")),
        )


@dataclass(frozen=True)
class CollectionResult:
    """The deterministic result of executing a research plan.

    ``collection_id`` is a stable content hash so storing or comparing
    results is deterministic.  ``executed_tasks`` preserves the plan's
    execution order; ``failed_tasks`` and ``unresolved_tasks`` list
    task identifiers that did not produce evidence.
    """

    schema_version: str
    plan_id: str
    company_name: str
    collection_id: str
    status: CollectionStatus
    collection: EvidenceCollection
    executed_tasks: tuple[str, ...] = ()
    failed_tasks: tuple[str, ...] = ()
    unresolved_tasks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        schema_version = self.schema_version.strip()
        plan_id = self.plan_id.strip()
        company_name = self.company_name.strip()
        collection_id = self.collection_id.strip()
        if not schema_version:
            raise InvalidEvidence("schema_version must not be empty")
        if not plan_id:
            raise InvalidEvidence("plan_id must not be empty")
        if not company_name:
            raise InvalidEvidence("company_name must not be empty")
        if not collection_id:
            raise InvalidEvidence("collection_id must not be empty")
        if not isinstance(self.status, CollectionStatus):
            raise InvalidEvidence("status must be a CollectionStatus")
        if not isinstance(self.collection, EvidenceCollection):
            raise InvalidEvidence(
                "collection must be an EvidenceCollection"
            )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "company_name", company_name)
        object.__setattr__(self, "collection_id", collection_id)
        object.__setattr__(
            self,
            "executed_tasks",
            tuple(dict.fromkeys(self.executed_tasks)),
        )
        object.__setattr__(
            self, "failed_tasks", tuple(dict.fromkeys(self.failed_tasks))
        )
        object.__setattr__(
            self,
            "unresolved_tasks",
            tuple(dict.fromkeys(self.unresolved_tasks)),
        )

    @property
    def topics_covered(self) -> tuple[str, ...]:
        """Return the topics with evidence, in deterministic order."""
        return self.collection.topics_covered()

    @property
    def evidence_count(self) -> int:
        """Return the total number of collected evidence items."""
        return len(self.collection)

    def has_failures(self) -> bool:
        """Return whether any task failed or was left unresolved."""
        return bool(self.failed_tasks or self.unresolved_tasks)

    def successfully_executed(self) -> bool:
        """Return whether the plan ran without failures or misses."""
        return not self.has_failures()

    def to_dict(self) -> dict[str, object]:
        """Serialize to a fully JSON-ready dictionary."""
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "company_name": self.company_name,
            "collection_id": self.collection_id,
            "status": self.status.value,
            "collection": self.collection.to_dict(),
            "executed_tasks": list(self.executed_tasks),
            "failed_tasks": list(self.failed_tasks),
            "unresolved_tasks": list(self.unresolved_tasks),
            "topics_covered": list(self.topics_covered),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CollectionResult:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            schema_version=_require_str(data, "schema_version"),
            plan_id=_require_str(data, "plan_id"),
            company_name=_require_str(data, "company_name"),
            collection_id=_require_str(data, "collection_id"),
            status=CollectionStatus(_require_str(data, "status")),
            collection=EvidenceCollection.from_dict(
                _require_dict(data, "collection")
            ),
            executed_tasks=tuple(_require_str_list(data, "executed_tasks")),
            failed_tasks=tuple(_require_str_list(data, "failed_tasks")),
            unresolved_tasks=tuple(
                _require_str_list(data, "unresolved_tasks")
            ),
        )


def _topics_not_in_order(
    evidence: tuple[Evidence, ...],
    topic_order: tuple[str, ...],
) -> tuple[str, ...]:
    """Return topics outside ``topic_order``, sorted for determinism.

    Deterministic fallback ordering for evidence whose topic is not part
    of the declared topic order (for example hand-built collections).
    """
    ordered = set(topic_order)
    topics = sorted(
        {item.topic_id for item in evidence if item.topic_id not in ordered}
    )
    return tuple(topics)


def _evidence_digest(payload: Mapping[str, object]) -> str:
    """Return a stable content-hash identifier for an evidence item."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"ev_{digest[:16]}"
