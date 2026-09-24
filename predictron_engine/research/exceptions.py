"""Exception hierarchy for the Research Planner subsystem.

Every error raised by the research planning layer derives from
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
