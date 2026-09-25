"""Exception hierarchy for the Research Intelligence subsystem.

Every error raised by the research planning (Sprint 1), source discovery
(Sprint 2), evidence collection (Sprint 3), evidence validation (Sprint 4),
and research execution (Sprint 5) layers derives from
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


class ValidationError(ResearchError):
    """Base class for every error raised by the Evidence Validation Engine.

    Phase 8 Sprint 4.  Anything that goes wrong while validating collected
    evidence raises a subclass of :class:`ValidationError` so callers can
    handle validation failures uniformly without coupling to internals.
    """


class InvalidEvidenceError(ValidationError, ValueError):  # noqa: N818
    """Raised when the validation engine receives invalid evidence.

    Used when an input is not an :class:`Evidence` instance, when a whole
    :class:`EvidenceCollection` cannot be validated, or when evidence
    violates the validation-layer schema contract.
    """


class DuplicateEvidenceError(ValidationError):
    """Raised when an evidence item is registered/added twice.

    Distinct from the (non-fatal) duplicate *detection* reported through
    :class:`~predictron_engine.research.validation_models.EvidenceConflict`:
    this error indicates a hard duplicate that prevents a validation
    operation from proceeding deterministically.
    """


class ValidationRuleError(ValidationError, ValueError):
    """Raised when a validation rule is malformed or misconfigured."""


class UnsupportedValidation(ValidationError):  # noqa: N818
    """Raised when a validator cannot handle the supplied evidence.

    Used by the registry when no validator supports a topic and by
    validators that reject evidence they do not understand.
    """


class FingerprintError(ValidationError, ValueError):
    """Raised when an evidence fingerprint cannot be computed."""


class RegistryError(ValidationError, ValueError):
    """Raised when a validator registry is misused or misconfigured.

    Covers type violations, duplicate registrations, and deserialization
    failures.
    """


class DuplicateValidator(RegistryError):  # noqa: N818
    """Raised when a validator is registered twice under the same id."""


class ResearchExecutionError(ResearchError):
    """Base class for every error raised by the Research Orchestrator.

    Phase 8 Sprint 5.  Anything that goes wrong while orchestrating the
    research pipeline raises a subclass of :class:`ResearchExecutionError`
    so callers can distinguish orchestration faults from the domain
    failures produced by the individual layers.

    Note that a *stage* failure inside a run is not an error: it is
    recorded deterministically on the
    :class:`~predictron_engine.research.execution_models.ExecutionError`
    value object and the run continues (partial completion) unless the
    orchestrator was configured to fail fast.
    """


class InvalidExecutionInputError(ResearchExecutionError, ValueError):  # noqa: N818
    """Raised when the orchestrator receives input that cannot be run.

    Used when ``execute`` is not handed a
    :class:`~predictron_engine.research.models.PlannerInput` instance.
    There is no execution to report in that case, so the call fails fast
    regardless of the configured policy.
    """


class InvalidExecutionModelError(ResearchExecutionError, ValueError):  # noqa: N818
    """Raised when an execution model violates the model contract.

    Used by the execution value objects when required fields are missing,
    malformed, or of the wrong type — both on construction and during
    deserialization.
    """


class InvalidExecutionStateTransitionError(  # noqa: N818
    ResearchExecutionError, ValueError
):
    """Raised when an execution state transition is not permitted.

    The orchestrator only ever walks the declared pipeline
    (``NOT_STARTED`` -> ``PLANNING`` -> ``DISCOVERY`` -> ``COLLECTION``
    -> ``VALIDATION`` -> ``COMPLETED``), plus an exit to ``FAILED`` from
    any active state.  Anything else is an orchestrator bug and raises
    rather than being silently accepted.
    """
