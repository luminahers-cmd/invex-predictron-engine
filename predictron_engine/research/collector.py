"""Evidence collector interface — Phase 8 Sprint 3.

:class:`EvidenceCollector` is the abstract contract every collector in
the Evidence Collection Engine must satisfy.  A collector:

* ``collect(task, ...)`` — executes one research task and returns
  deterministic :class:`~predictron_engine.research.models.Evidence`
  objects.
* ``supports(topic)`` — reports whether it can collect a topic.
* ``metadata()`` — reports deterministic, serializable metadata.

The sprint builds the **production architecture** that later collectors
plug into.  Collectors in this sprint are strictly deterministic
placeholders: no network requests, no browser automation, no search
engines, no LLMs, and no APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from predictron_engine.research.exceptions import UnsupportedTopic
from predictron_engine.research.models import (
    Evidence,
    EvidenceMetadata,
    ResearchTask,
)


class EvidenceCollector(ABC):
    """Abstract contract every evidence collector must satisfy.

    Concrete collectors declare fixed class attributes:

    ``collector_id``
        Stable snake_case identifier, e.g. ``"founder_collector"``.
    ``display_name``
        Human-readable name, e.g. ``"Founders"``.
    ``description``
        Free-form description of what the collector produces.
    ``supported_topics``
        The research topic identifiers the collector can handle.

    ``supports`` and ``metadata`` are implemented generically from those
    attributes; subclasses only implement :meth:`collect`.  Collectors
    never perform I/O — they return deterministic placeholder evidence.
    """

    collector_id: ClassVar[str]
    display_name: ClassVar[str]
    description: ClassVar[str] = ""
    supported_topics: ClassVar[tuple[str, ...]]

    # ------------------------------------------------------------------
    # Required contract
    # ------------------------------------------------------------------

    @abstractmethod
    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]:
        """Collect deterministic placeholder evidence for ``task``.

        Parameters
        ----------
        task:
            The :class:`ResearchTask` being executed.
        company_name:
            The company under research (used in deterministic claims).
        website:
            Optional company website supplied at planning time.

        Returns
        -------
        A non-empty tuple of deterministic :class:`Evidence` objects.

        Raises
        ------
        UnsupportedTopic:
            When ``task.topic_id`` is not in this collector's
            ``supported_topics``.
        """

    # ------------------------------------------------------------------
    # Generic behaviour
    # ------------------------------------------------------------------

    def supports(self, topic_id: str) -> bool:
        """Return whether this collector can collect ``topic_id``."""
        return topic_id in _class_topics(type(self))

    def metadata(self) -> EvidenceMetadata:
        """Return immutable, serializable metadata for this collector.

        The metadata is deterministic — it never depends on execution
        state, wall-clock time, or the input being collected.
        """
        cls = type(self)
        return EvidenceMetadata(
            collector_id=cls.collector_id,
            display_name=cls.display_name,
            description=getattr(cls, "description", ""),
            supported_topics=_class_topics(cls),
            deterministic=True,
        )


def _class_topics(cls: type[EvidenceCollector]) -> tuple[str, ...]:
    """Return ``cls.supported_topics`` validated as a non-empty tuple.

    Raises a descriptive ``TypeError`` when a subclass forgot to declare
    its supported topics so misconfiguration fails loudly and early
    instead of at collection time.
    """
    topics = getattr(cls, "supported_topics", ())
    if not isinstance(topics, tuple) or not topics:
        raise TypeError(f"{cls.__name__} must declare supported_topics")
    for topic in topics:
        if not isinstance(topic, str) or not topic.strip():
            raise TypeError(
                f"{cls.__name__} declares an invalid supported topic"
            )
    return tuple(dict.fromkeys(topics))


def enforce_supported(
    collector: EvidenceCollector,
    topics: Sequence[str] | str,
) -> None:
    """Raise :class:`UnsupportedTopic` unless ``topics`` are covered.

    Used by collectors and the dispatcher so unsupported topics fail
    deterministically instead of silently producing empty results.
    """
    candidates = (topics,) if isinstance(topics, str) else tuple(topics)
    for topic in candidates:
        if not collector.supports(topic):
            raise UnsupportedTopic(
                f"collector {collector.metadata().collector_id!r} does "
                f"not support topic: {topic!r}"
            )
