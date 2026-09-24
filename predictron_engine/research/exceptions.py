"""Exception hierarchy for the Research Intelligence subsystem.

Every error raised by the research planning (Sprint 1), source discovery
(Sprint 2), and evidence collection (Sprint 3) layers derives from
:class:`ResearchError` so callers can handle failures uniformly without
coupling to implementation details.
"""

from __future__ import annotations


class ResearchError(Exception):
    """Base class for every error raised by the research planning layer."""


class InvalidPlannerInputError(ResearchError, ValueError):
    """Raised when a :class:`PlannerInput` violates the input contract."""


class UnknownTopicError(ResearchError):
    """Raised when a topic identifier is not part of the research taxonomy."""


class DependencyCycleError(ResearchError):
    """Raised when research task dependencies form a cycle."""


class RuleRegistrationError(ResearchError):
    """Raised when a research rule cannot be registered."""


class UnknownSourceError(ResearchError):
    """Raised when a source identifier is not part of the source registry."""


class SourceRegistryError(ResearchError, ValueError):
    """Raised when a source registry cannot be constructed."""


class InvalidDiscoveryInputError(ResearchError, ValueError):
    """Raised when a :class:`SourceDiscovery` receives invalid input."""


class InvalidCollectionInputError(ResearchError, ValueError):
    """Raised when a collection component receives invalid input.

    Used by the dispatcher and the execution engine when the input is not
    a :class:`ResearchTask` / :class:`ResearchPlan` instance or violates
    the collection contract.
    """


class CollectorNotFound(ResearchError):  # noqa: N818
    """Raised when no registered collector exists for a research topic."""


class DuplicateCollector(ResearchError):  # noqa: N818
    """Raised when a collector is registered twice under the same id."""


class UnsupportedTopic(ResearchError, ValueError):  # noqa: N818
    """Raised when a collector is asked to collect an unsupported topic.

    A collector may reject a topic it does not support at ``collect``
    time even when the dispatcher resolved it; this guards against
    mis-registered collectors.
    """


class CollectorExecutionError(ResearchError):
    """Raised when a collector fails while executing a research task."""


class InvalidEvidence(ResearchError, ValueError):  # noqa: N818
    """Raised when an evidence object violates the model contract.

    Used by the evidence models when fields are malformed and by the
    execution engine when a collector returns objects that are not
    :class:`Evidence` instances.
    """
