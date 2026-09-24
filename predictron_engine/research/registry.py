"""Collector registry — Phase 8 Sprint 3.

:class:`CollectorRegistry` is the single place collectors are registered,
unregistered, resolved, and enumerated.  The dispatcher and the
execution engine must never know concrete collector classes: every
lookup flows through this registry.

The registry is deterministic: registration order is preserved, topic
resolution is stable (first registered collector wins), and enumeration
always returns collection order.
"""

from __future__ import annotations

from collections.abc import Sequence

from predictron_engine.research.collector import EvidenceCollector
from predictron_engine.research.collectors import DEFAULT_COLLECTORS
from predictron_engine.research.exceptions import (
    CollectorNotFound,
    DuplicateCollector,
)


class CollectorRegistry:
    """A mutable catalog of :class:`EvidenceCollector` instances.

    Parameters
    ----------
    collectors:
        Optional initial collectors registered in the given order.  When
        ``None`` the registry starts empty.
    """

    def __init__(
        self,
        collectors: Sequence[EvidenceCollector] | None = None,
    ) -> None:
        self._collectors: list[EvidenceCollector] = []
        if collectors is not None:
            for collector in collectors:
                self.register(collector)

    def __len__(self) -> int:
        """Return the number of registered collectors."""
        return len(self._collectors)

    def __contains__(self, collector_id: object) -> bool:
        """Return whether a collector with ``collector_id`` is registered."""
        return isinstance(collector_id, str) and self.has_collector(collector_id)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, collector: EvidenceCollector) -> EvidenceCollector:
        """Register ``collector`` and return it.

        Registration order is preserved and determines resolution order.

        Raises
        ------
        TypeError:
            When ``collector`` is not an :class:`EvidenceCollector`.
        DuplicateCollector:
            When a collector with the same ``collector_id`` is already
            registered.
        """
        if not isinstance(collector, EvidenceCollector):
            raise TypeError(
                "register expects an EvidenceCollector instance"
            )
        collector_id = collector.metadata().collector_id
        if self.has_collector(collector_id):
            raise DuplicateCollector(
                f"collector already registered: {collector_id!r}"
            )
        self._collectors.append(collector)
        return collector

    def unregister(self, collector_id: str) -> EvidenceCollector | None:
        """Remove and return the collector for ``collector_id``.

        Returns ``None`` when no such collector is registered (the
        operation is idempotent).
        """
        for position, collector in enumerate(self._collectors):
            if collector.metadata().collector_id == collector_id:
                del self._collectors[position]
                return collector
        return None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def list_collectors(self) -> tuple[EvidenceCollector, ...]:
        """Return registered collectors in registration order."""
        return tuple(self._collectors)

    def get(self, collector_id: str) -> EvidenceCollector | None:
        """Return the collector for ``collector_id``, or ``None``."""
        for collector in self._collectors:
            if collector.metadata().collector_id == collector_id:
                return collector
        return None

    def has_collector(self, collector_id: str) -> bool:
        """Return whether a collector with ``collector_id`` is registered."""
        return self.get(collector_id) is not None

    def collectors_for(self, topic_id: str) -> tuple[EvidenceCollector, ...]:
        """Return collectors supporting ``topic_id``, in registration order."""
        return tuple(
            collector
            for collector in self._collectors
            if collector.supports(topic_id)
        )

    def resolve(self, topic_id: str) -> EvidenceCollector:
        """Resolve a single collector for ``topic_id``.

        The first registered collector that supports ``topic_id`` wins,
        which makes resolution deterministic regardless of catalog shape.

        Raises
        ------
        CollectorNotFound:
            When no registered collector supports ``topic_id``.
        """
        collectors = self.collectors_for(topic_id)
        if not collectors:
            raise CollectorNotFound(
                f"no registered collector supports topic: {topic_id!r}"
            )
        return collectors[0]

    def topics_covered(self) -> tuple[str, ...]:
        """Return every supported topic, sorted deterministically."""
        covered: set[str] = set()
        for collector in self._collectors:
            covered.update(collector.metadata().supported_topics)
        return tuple(sorted(covered))


# Default registry shared by the execution engine and convenience API.
COLLECTOR_REGISTRY = CollectorRegistry(DEFAULT_COLLECTORS)


def default_collector_registry() -> CollectorRegistry:
    """Return a fresh registry preloaded with the default collectors."""
    return CollectorRegistry(DEFAULT_COLLECTORS)


def register_collector(collector: EvidenceCollector) -> EvidenceCollector:
    """Register ``collector`` on the shared registry."""
    return COLLECTOR_REGISTRY.register(collector)


def unregister_collector(collector_id: str) -> EvidenceCollector | None:
    """Unregister ``collector_id`` from the shared registry."""
    return COLLECTOR_REGISTRY.unregister(collector_id)


def resolve_collector(topic_id: str) -> EvidenceCollector:
    """Resolve a collector for ``topic_id`` on the shared registry."""
    return COLLECTOR_REGISTRY.resolve(topic_id)


def list_collectors() -> tuple[EvidenceCollector, ...]:
    """List the shared registry's collectors in registration order."""
    return COLLECTOR_REGISTRY.list_collectors()


def collectors_for(topic_id: str) -> tuple[EvidenceCollector, ...]:
    """Return shared-registry collectors supporting ``topic_id``."""
    return COLLECTOR_REGISTRY.collectors_for(topic_id)


def has_collector(collector_id: str) -> bool:
    """Return whether ``collector_id`` is registered on the shared registry."""
    return COLLECTOR_REGISTRY.has_collector(collector_id)


def covered_topics() -> tuple[str, ...]:
    """Return the topics the shared registry can collect, sorted."""
    return COLLECTOR_REGISTRY.topics_covered()
