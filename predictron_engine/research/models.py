"""Research Planner models — Phase 8 Sprint 1 (Autonomous Research Intelligence).

Immutable value objects describing the deterministic planning layer:

* :class:`PlannerInput` — what a caller tells the planner.
* :class:`ResearchTopic` — one node in the research taxonomy.
* :class:`KnowledgeGap` — a missing or under-covered research area.
* :class:`ResearchTask` — one concrete piece of research to perform.
* :class:`ResearchPlan` — the complete, executable research plan.
* :class:`ResearchPriority` — deterministic priority band.
* :class:`EvidenceStatus` — how much evidence exists for a topic.

Models are ``frozen`` dataclasses so they cannot mutate after creation.
Every model exposes ``to_dict`` / ``from_dict`` for lossless
serialization.  Nothing here performs I/O, calls models, or touches the
network — the planning layer is pure and deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

from predictron_engine.research.exceptions import InvalidPlannerInputError

PLAN_SCHEMA_VERSION = "1.0.0"


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
