"""Exception hierarchy for the Research Intelligence subsystem.

Every error raised by the research planning (Sprint 1), source discovery
(Sprint 2), evidence collection (Sprint 3), evidence validation (Sprint 4),
research execution (Sprint 5), research memory / knowledge graph
(Sprint 6), and research intelligence loop (Sprint 7) layers derives from
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


class ResearchMemoryError(ResearchError):
    """Base class for every error raised by Research Memory and the graph.

    Phase 8 Sprint 6.  Anything that goes wrong while persisting research
    or building the internal knowledge graph raises a subclass of
    :class:`ResearchMemoryError` so callers can distinguish persistence
    faults from the domain failures produced by the earlier sprints.
    """


class InvalidGraphModelError(ResearchMemoryError, ValueError):  # noqa: N818
    """Raised when a knowledge-graph value object breaks its contract.

    Used on construction and during deserialization when a field is
    missing, malformed, or of the wrong type.
    """


class InvalidRelationshipError(ResearchMemoryError, ValueError):  # noqa: N818
    """Raised when a relationship is not valid for its endpoint types.

    The research knowledge graph is strongly typed: every relationship has
    a fixed set of permitted source and target entity types, and every
    relationship must carry confidence, provenance, a timestamp, and at
    least one supporting evidence identifier.
    """


class InvalidMemoryInputError(ResearchMemoryError, ValueError):  # noqa: N818
    """Raised when :class:`ResearchMemory` receives input it cannot store.

    Also raised when an add operation is asked to overwrite an existing
    record: research memory is append-only and never rewrites history.
    """


class ImmutableRecordError(ResearchMemoryError, ValueError):  # noqa: N818
    """Raised when an append-only invariant would be violated.

    A stored :class:`~predictron_engine.research.execution_report.ResearchReport`,
    :class:`~predictron_engine.research.models.Evidence` item, or history
    entry can never be replaced or edited.  Attempting to do so raises this
    error rather than silently dropping the new value.
    """


class EntityLinkingError(ResearchMemoryError, ValueError):  # noqa: N818
    """Raised when two names cannot be reduced to a canonical entity.

    Entity linking is rule-based; when a name carries no usable characters
    there is no canonical key, so no identifier can be derived and the
    operation fails instead of inventing one.
    """


class DeduplicationError(ResearchMemoryError, ValueError):  # noqa: N818
    """Raised when a duplicate-detection key cannot be computed.

    Raised by :func:`~predictron_engine.research.deduplication.duplicate_key`
    for empty inputs, which have no deterministic content digest.
    """


class ResearchIterationError(ResearchError):
    """Base class for every error raised by the Research Intelligence Loop.

    Phase 8 Sprint 7.  Anything that goes wrong while driving the research
    pipeline iteratively raises a subclass of this error so callers can
    distinguish loop faults from the domain failures produced by the earlier
    sprints.

    Note that a *pass* that fails inside the loop is not an error: it is
    recorded deterministically on the pass's
    :class:`~predictron_engine.research.execution_models.ExecutionError`
    value object and the loop terminates with
    :attr:`~predictron_engine.research.iteration_models.TerminationReason.EXECUTION_FAILED`,
    exactly mirroring how the Sprint 5 orchestrator treats stage failures.
    """


class InvalidIterationInputError(ResearchIterationError, ValueError):  # noqa: N818
    """Raised when the research loop receives input it cannot run.

    Used when the loop is not handed a
    :class:`~predictron_engine.research.models.PlannerInput` instance or a
    run that does not describe a research pass.  There is no loop to report
    in that case, so the call fails immediately.
    """


class InvalidIterationModelError(ResearchIterationError, ValueError):  # noqa: N818
    """Raised when a loop value object violates its model contract.

    Used by the iteration models when required fields are missing, malformed,
    or of the wrong type — both on construction and during deserialization.
    """
