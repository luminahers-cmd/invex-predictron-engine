"""Evidence Validation models — Phase 8 Sprint 4.

Immutable, deterministic value objects produced by the Evidence
Validation Engine:

* :class:`ValidationSeverity` — ordered severity of a detected problem.
* :class:`ValidationStatus` — deterministic pass/fail status.
* :class:`EvidenceConfidence` — deterministic confidence band.
* :class:`ValidationIssue` — one problem found by a validator.
* :class:`ValidationResult` — the outcome of one validator for one item.
* :class:`EvidenceConflict` — a structured conflict involving evidence.
* :class:`EvidenceFingerprint` — deterministic fingerprint of evidence.
* :class:`ValidatorMetadata` — serializable metadata for one validator.
* :class:`ValidationMetadata` — serializable metadata for one run.
* :class:`ValidationCounts` — deterministic aggregate counters.
* :class:`TopicValidation` — per-topic validation breakdown.
* :class:`EvidenceValidation` — the validated view of one evidence item.
* :class:`ValidationSummary` — the complete deterministic result.

Every model is a ``frozen`` dataclass exposing ``to_dict`` /
``from_dict`` for lossless serialization.  Nothing here performs I/O,
calls models, or touches the network — the validation layer is pure and
deterministic with no randomness.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from predictron_engine.research.exceptions import (
    FingerprintError,
    InvalidEvidenceError,
)
from predictron_engine.research.models import Evidence

VALIDATION_SCHEMA_VERSION = "1.0.0"
FINGERPRINT_ALGORITHM = "sha256"
FINGERPRINT_PREFIX = "evf_"


class ValidationSeverity(str, Enum):
    """Ordered severity of a :class:`ValidationIssue`.

    ``CRITICAL`` is the most severe; ``INFO`` the least.  Ranking is
    deterministic and drives status aggregation.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def rank(self) -> int:
        """Return a deterministic rank; higher means more severe."""
        return tuple(ValidationSeverity).index(self)

    def is_failure(self) -> bool:
        """Return whether this severity fails an evidence item."""
        return self in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)

    def is_notice(self) -> bool:
        """Return whether this severity should be surfaced as a warning."""
        return self is ValidationSeverity.WARNING


class ValidationStatus(str, Enum):
    """Deterministic status of one evidence item or a whole run.

    * ``PASSED`` — no errors or warnings surfaced.
    * ``WARNED`` — warnings surfaced but nothing failed.
    * ``FAILED`` — at least one error or critical issue surfaced.
    * ``SKIPPED`` — no evidence was present to validate.
    """

    PASSED = "passed"
    WARNED = "warned"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvidenceConfidence(str, Enum):
    """Deterministic confidence band derived from a score.

    The mapping from a float score in ``[0, 1]`` to a band is a pure
    function and therefore reproducible for identical inputs.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

    @classmethod
    def from_score(cls, score: float) -> EvidenceConfidence:
        """Return the confidence band for a score in ``[0, 1]``."""
        if score >= 0.8:
            return cls.HIGH
        if score >= 0.55:
            return cls.MEDIUM
        if score >= 0.3:
            return cls.LOW
        return cls.UNKNOWN


@dataclass(frozen=True)
class ValidationIssue:
    """One deterministic problem found during validation.

    Parameters
    ----------
    code:
        Stable machine-readable code, e.g. ``"missing_reference"``.
    message:
        Human-readable description of the problem.
    severity:
        The :class:`ValidationSeverity` of this issue.
    validator_id:
        The validator that raised this issue (empty for engine-level).
    evidence_id:
        The evidence the issue relates to (empty for run-level issues).
    field:
        Optional name of the offending field.
    """

    code: str
    message: str
    severity: ValidationSeverity
    validator_id: str = ""
    evidence_id: str = ""
    field: str = ""

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise InvalidEvidenceError("issue code must not be empty")
        if not self.message.strip():
            raise InvalidEvidenceError("issue message must not be empty")
        if not isinstance(self.severity, ValidationSeverity):
            raise InvalidEvidenceError("severity must be a ValidationSeverity")

    def is_failure(self) -> bool:
        """Return whether this issue fails its evidence item."""
        return self.severity.is_failure()

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "validator_id": self.validator_id,
            "evidence_id": self.evidence_id,
            "field": self.field,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ValidationIssue:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            code=_require_str(data, "code"),
            message=_require_str(data, "message"),
            severity=ValidationSeverity(_require_str(data, "severity")),
            validator_id=_require_str(data, "validator_id", default=""),
            evidence_id=_require_str(data, "evidence_id", default=""),
            field=_require_str(data, "field", default=""),
        )


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of one validator for one evidence item.

    ``score`` is a deterministic value in ``[0, 1]``; ``passed`` reports
    whether the validator accepted the evidence; ``issues`` carry the
    specific problems found (empty when the validator passed cleanly).
    """

    validator_id: str
    validator_name: str
    passed: bool
    score: float
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if not self.validator_id.strip():
            raise InvalidEvidenceError("validator_id must not be empty")
        if not self.validator_name.strip():
            raise InvalidEvidenceError("validator_name must not be empty")
        if not _in_unit_range(self.score):
            raise InvalidEvidenceError("score must be within [0, 1]")

    @property
    def status(self) -> ValidationStatus:
        """Return the deterministic status derived from ``issues``."""
        if any(issue.is_failure() for issue in self.issues):
            return ValidationStatus.FAILED
        if any(issue.severity.is_notice() for issue in self.issues):
            return ValidationStatus.WARNED
        return ValidationStatus.PASSED

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "validator_id": self.validator_id,
            "validator_name": self.validator_name,
            "passed": self.passed,
            "score": self.score,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ValidationResult:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            validator_id=_require_str(data, "validator_id"),
            validator_name=_require_str(data, "validator_name"),
            passed=_require_bool(data, "passed"),
            score=_require_float(data, "score"),
            issues=tuple(
                ValidationIssue.from_dict(item)
                for item in _require_list_of_dicts(data, "issues")
            ),
        )


@dataclass(frozen=True)
class EvidenceConflict:
    """A deterministic, structured conflict involving evidence.

    Pairwise conflicts set ``evidence_a`` and ``evidence_b`` to the two
    evidence identifiers involved.  Self-diagnostics (missing references,
    invalid timestamps, incomplete metadata) leave ``evidence_b`` empty.

    Parameters
    ----------
    conflict_type:
        Stable machine-readable type, e.g. ``"conflicting_values"`` or
        ``"duplicate_evidence"``.
    evidence_a:
        First evidence identifier (always the smaller when pairwise).
    evidence_b:
        Optional second evidence identifier (empty for self-diagnostics).
    description:
        Human-readable account of the conflict.
    severity:
        Deterministic :class:`ValidationSeverity` of the conflict.
    field:
        Optional name of the field/attribute in conflict.
    value_a:
        Optional canonical value held by ``evidence_a``.
    value_b:
        Optional canonical value held by ``evidence_b``.
    """

    conflict_type: str
    evidence_a: str
    evidence_b: str = ""
    description: str = ""
    severity: ValidationSeverity = ValidationSeverity.WARNING
    field: str = ""
    value_a: str = ""
    value_b: str = ""

    def __post_init__(self) -> None:
        if not self.conflict_type.strip():
            raise InvalidEvidenceError("conflict_type must not be empty")
        if not self.evidence_a.strip():
            raise InvalidEvidenceError("evidence_a must not be empty")
        if self.evidence_b and self.evidence_a == self.evidence_b:
            raise InvalidEvidenceError(
                "a pairwise conflict needs two distinct evidence ids"
            )
        if not isinstance(self.severity, ValidationSeverity):
            raise InvalidEvidenceError("severity must be a ValidationSeverity")

    @property
    def is_pairwise(self) -> bool:
        """Return whether this conflict involves two evidence items."""
        return bool(self.evidence_b)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "conflict_type": self.conflict_type,
            "evidence_a": self.evidence_a,
            "evidence_b": self.evidence_b,
            "description": self.description,
            "severity": self.severity.value,
            "field": self.field,
            "value_a": self.value_a,
            "value_b": self.value_b,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvidenceConflict:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            conflict_type=_require_str(data, "conflict_type"),
            evidence_a=_require_str(data, "evidence_a"),
            evidence_b=_require_str(data, "evidence_b", default=""),
            description=_require_str(data, "description", default=""),
            severity=ValidationSeverity(_require_str(data, "severity")),
            field=_require_str(data, "field", default=""),
            value_a=_require_str(data, "value_a", default=""),
            value_b=_require_str(data, "value_b", default=""),
        )


@dataclass(frozen=True)
class EvidenceFingerprint:
    """A deterministic fingerprint of one :class:`Evidence` object.

    Identical evidence always produces an identical fingerprint: the
    fingerprint is a content hash over the evidence's canonical JSON so
    future deduplication and knowledge graph ingestion can rely on it.

    Parameters
    ----------
    algorithm:
        The hash algorithm used (currently ``"sha256"``).
    value:
        The full hex digest.
    canonical:
        The canonical JSON payload the digest was computed over.
    prefix:
        Stable short prefix used for identifiers built from the digest.
    """

    algorithm: str
    value: str
    canonical: str = ""
    prefix: str = FINGERPRINT_PREFIX

    def __post_init__(self) -> None:
        if not self.algorithm.strip():
            raise FingerprintError("algorithm must not be empty")
        if not self.value.strip():
            raise FingerprintError("value must not be empty")

    @property
    def short(self) -> str:
        """Return the conventional short identifier for the fingerprint."""
        return f"{self.prefix}{self.value[:16]}"

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "algorithm": self.algorithm,
            "value": self.value,
            "canonical": self.canonical,
            "prefix": self.prefix,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvidenceFingerprint:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            algorithm=_require_str(data, "algorithm"),
            value=_require_str(data, "value"),
            canonical=_require_str(data, "canonical", default=""),
            prefix=_require_str(data, "prefix", default=FINGERPRINT_PREFIX),
        )


@dataclass(frozen=True)
class ValidatorMetadata:
    """Deterministic, serializable metadata describing one validator."""

    validator_id: str
    display_name: str
    description: str = ""
    supported_topics: tuple[str, ...] = ()
    deterministic: bool = True

    def __post_init__(self) -> None:
        validator_id = self.validator_id.strip()
        display_name = self.display_name.strip()
        if not validator_id:
            raise InvalidEvidenceError("validator_id must not be empty")
        if not display_name:
            raise InvalidEvidenceError("display_name must not be empty")
        object.__setattr__(self, "validator_id", validator_id)
        object.__setattr__(self, "display_name", display_name)

    @property
    def is_generic(self) -> bool:
        """Return whether this validator applies to every topic."""
        return not self.supported_topics

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "validator_id": self.validator_id,
            "display_name": self.display_name,
            "description": self.description,
            "supported_topics": list(self.supported_topics),
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ValidatorMetadata:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            validator_id=_require_str(data, "validator_id"),
            display_name=_require_str(data, "display_name"),
            description=_require_str(data, "description", default=""),
            supported_topics=tuple(_require_str_list(data, "supported_topics")),
            deterministic=_require_bool(data, "deterministic"),
        )


@dataclass(frozen=True)
class ValidationMetadata:
    """Deterministic metadata describing one validation run.

    No wall-clock timestamps are stored: the validation layer is pure and
    identical inputs always produce identical metadata.
    """

    schema_version: str
    engine_version: str
    collection_fingerprint: str
    validator_ids: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise InvalidEvidenceError("schema_version must not be empty")
        if not self.engine_version.strip():
            raise InvalidEvidenceError("engine_version must not be empty")
        if not self.collection_fingerprint.strip():
            raise InvalidEvidenceError(
                "collection_fingerprint must not be empty"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "collection_fingerprint": self.collection_fingerprint,
            "validator_ids": list(self.validator_ids),
            "rule_ids": list(self.rule_ids),
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ValidationMetadata:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            schema_version=_require_str(data, "schema_version"),
            engine_version=_require_str(data, "engine_version"),
            collection_fingerprint=_require_str(
                data, "collection_fingerprint"
            ),
            validator_ids=tuple(_require_str_list(data, "validator_ids")),
            rule_ids=tuple(_require_str_list(data, "rule_ids")),
            deterministic=_require_bool(data, "deterministic"),
        )


@dataclass(frozen=True)
class ValidationCounts:
    """Deterministic aggregate counters for a validation run."""

    total: int
    passed: int
    warned: int
    failed: int
    skipped: int
    issues: int
    conflicts: int
    duplicates: int
    topics_covered: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "total": self.total,
            "passed": self.passed,
            "warned": self.warned,
            "failed": self.failed,
            "skipped": self.skipped,
            "issues": self.issues,
            "conflicts": self.conflicts,
            "duplicates": self.duplicates,
            "topics_covered": self.topics_covered,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ValidationCounts:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            total=_require_int(data, "total"),
            passed=_require_int(data, "passed"),
            warned=_require_int(data, "warned"),
            failed=_require_int(data, "failed"),
            skipped=_require_int(data, "skipped"),
            issues=_require_int(data, "issues"),
            conflicts=_require_int(data, "conflicts"),
            duplicates=_require_int(data, "duplicates"),
            topics_covered=_require_int(data, "topics_covered"),
        )


@dataclass(frozen=True)
class TopicValidation:
    """Per-topic validation breakdown for one research topic."""

    topic_id: str
    evidence_count: int
    passed: int
    warned: int
    failed: int
    mean_confidence: float

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "topic_id": self.topic_id,
            "evidence_count": self.evidence_count,
            "passed": self.passed,
            "warned": self.warned,
            "failed": self.failed,
            "mean_confidence": self.mean_confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TopicValidation:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            topic_id=_require_str(data, "topic_id"),
            evidence_count=_require_int(data, "evidence_count"),
            passed=_require_int(data, "passed"),
            warned=_require_int(data, "warned"),
            failed=_require_int(data, "failed"),
            mean_confidence=_require_float(data, "mean_confidence"),
        )


@dataclass(frozen=True)
class EvidenceValidation:
    """The validated view of one evidence item.

    Parameters
    ----------
    evidence_id:
        The validated evidence's stable identifier.
    task_id:
        The research task that produced the evidence.
    topic_id:
        The research topic the evidence belongs to.
    fingerprint:
        The deterministic :class:`EvidenceFingerprint` of the evidence.
    status:
        Deterministic :class:`ValidationStatus` for this item.
    confidence_score:
        Deterministic score in ``[0, 1]``.
    confidence:
        Deterministic :class:`EvidenceConfidence` band.
    validator_results:
        Per-validator :class:`ValidationResult` entries in registry order.
    issues:
        All issues raised for the item, in deterministic order.
    conflicts:
        Conflicting values for the item as :class:`EvidenceConflict`.
    duplicate_count:
        How many copies of this evidence exist in the collection.
    """

    evidence_id: str
    task_id: str
    topic_id: str
    fingerprint: EvidenceFingerprint
    status: ValidationStatus
    confidence_score: float
    confidence: EvidenceConfidence
    validator_results: tuple[ValidationResult, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()
    conflicts: tuple[EvidenceConflict, ...] = ()
    duplicate_count: int = 0

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise InvalidEvidenceError("evidence_id must not be empty")
        if not _in_unit_range(self.confidence_score):
            raise InvalidEvidenceError(
                "confidence_score must be within [0, 1]"
            )
        if not isinstance(self.status, ValidationStatus):
            raise InvalidEvidenceError("status must be a ValidationStatus")
        if not isinstance(self.confidence, EvidenceConfidence):
            raise InvalidEvidenceError(
                "confidence must be an EvidenceConfidence"
            )
        if self.duplicate_count < 0:
            raise InvalidEvidenceError(
                "duplicate_count must be non-negative"
            )

    def passed(self) -> bool:
        """Return whether the item passed validation."""
        return self.status is ValidationStatus.PASSED

    def has_failed(self) -> bool:
        """Return whether the item failed validation."""
        return self.status is ValidationStatus.FAILED

    def has_warnings(self) -> bool:
        """Return whether the item only carried warnings."""
        return self.status is ValidationStatus.WARNED

    def issue_codes(self) -> tuple[str, ...]:
        """Return the distinct issue codes, in first-seen order."""
        codes: list[str] = []
        for issue in self.issues:
            if issue.code not in codes:
                codes.append(issue.code)
        return tuple(codes)

    def result_for(self, validator_id: str) -> ValidationResult | None:
        """Return the validator result for ``validator_id`` or ``None``."""
        for result in self.validator_results:
            if result.validator_id == validator_id:
                return result
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a fully JSON-ready dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "task_id": self.task_id,
            "topic_id": self.topic_id,
            "fingerprint": self.fingerprint.to_dict(),
            "status": self.status.value,
            "confidence_score": self.confidence_score,
            "confidence": self.confidence.value,
            "validator_results": [
                result.to_dict() for result in self.validator_results
            ],
            "issues": [issue.to_dict() for issue in self.issues],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "duplicate_count": self.duplicate_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvidenceValidation:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            evidence_id=_require_str(data, "evidence_id"),
            task_id=_require_str(data, "task_id"),
            topic_id=_require_str(data, "topic_id"),
            fingerprint=EvidenceFingerprint.from_dict(
                _require_dict(data, "fingerprint")
            ),
            status=ValidationStatus(_require_str(data, "status")),
            confidence_score=_require_float(data, "confidence_score"),
            confidence=EvidenceConfidence(_require_str(data, "confidence")),
            validator_results=tuple(
                ValidationResult.from_dict(item)
                for item in _require_list_of_dicts(data, "validator_results")
            ),
            issues=tuple(
                ValidationIssue.from_dict(item)
                for item in _require_list_of_dicts(data, "issues")
            ),
            conflicts=tuple(
                EvidenceConflict.from_dict(item)
                for item in _require_list_of_dicts(data, "conflicts")
            ),
            duplicate_count=_require_int(data, "duplicate_count"),
        )


@dataclass(frozen=True)
class ValidationSummary:
    """The complete deterministic result of a validation run.

    Parameters
    ----------
    schema_version:
        Validation serialization schema version.
    validation_id:
        Stable content hash identifying the run.
    collection_fingerprint:
        Deterministic fingerprint of the validated collection.
    status:
        Overall deterministic :class:`ValidationStatus`.
    metadata:
        Run-level :class:`ValidationMetadata`.
    validations:
        Per-evidence :class:`EvidenceValidation` entries in collection order.
    conflicts:
        Every :class:`EvidenceConflict` detected by the run.
    counts:
        Aggregate :class:`ValidationCounts`.
    covered_topics:
        Topics with evidence, in deterministic topic order.
    by_topic:
        Per-topic :class:`TopicValidation` breakdown in topic order.
    """

    schema_version: str
    validation_id: str
    collection_fingerprint: str
    status: ValidationStatus
    metadata: ValidationMetadata
    validations: tuple[EvidenceValidation, ...] = ()
    conflicts: tuple[EvidenceConflict, ...] = ()
    counts: ValidationCounts = field(
        default_factory=lambda: ValidationCounts(
            total=0,
            passed=0,
            warned=0,
            failed=0,
            skipped=0,
            issues=0,
            conflicts=0,
            duplicates=0,
            topics_covered=0,
        )
    )
    covered_topics: tuple[str, ...] = ()
    by_topic: tuple[TopicValidation, ...] = ()

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise InvalidEvidenceError("schema_version must not be empty")
        if not self.validation_id.strip():
            raise InvalidEvidenceError("validation_id must not be empty")
        if not isinstance(self.status, ValidationStatus):
            raise InvalidEvidenceError("status must be a ValidationStatus")
        if not isinstance(self.metadata, ValidationMetadata):
            raise InvalidEvidenceError(
                "metadata must be a ValidationMetadata"
            )

    def validation_for(self, evidence_id: str) -> EvidenceValidation | None:
        """Return the validation for ``evidence_id`` or ``None``."""
        for validation in self.validations:
            if validation.evidence_id == evidence_id:
                return validation
        return None

    def confidence_for(self, evidence_id: str) -> EvidenceConfidence | None:
        """Return the confidence band for ``evidence_id`` or ``None``."""
        validation = self.validation_for(evidence_id)
        if validation is None:
            return None
        return validation.confidence

    def has_failures(self) -> bool:
        """Return whether any evidence item failed validation."""
        return any(validation.has_failed() for validation in self.validations)

    def issues(self) -> tuple[ValidationIssue, ...]:
        """Return every issue across all validated items, in order."""
        return tuple(
            issue
            for validation in self.validations
            for issue in validation.issues
        )

    def issue_count_by_severity(
        self,
    ) -> dict[str, int]:
        """Return issue counts keyed by deterministic severity value."""
        tally: dict[str, int] = {}
        for issue in self.issues():
            tally[issue.severity.value] = tally.get(issue.severity.value, 0) + 1
        return tally

    def topic_for(self, topic_id: str) -> TopicValidation | None:
        """Return the :class:`TopicValidation` for ``topic_id`` or ``None``."""
        for topic in self.by_topic:
            if topic.topic_id == topic_id:
                return topic
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a fully JSON-ready dictionary."""
        return {
            "schema_version": self.schema_version,
            "validation_id": self.validation_id,
            "collection_fingerprint": self.collection_fingerprint,
            "status": self.status.value,
            "metadata": self.metadata.to_dict(),
            "validations": [
                validation.to_dict() for validation in self.validations
            ],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "counts": self.counts.to_dict(),
            "covered_topics": list(self.covered_topics),
            "by_topic": [topic.to_dict() for topic in self.by_topic],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ValidationSummary:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            schema_version=_require_str(data, "schema_version"),
            validation_id=_require_str(data, "validation_id"),
            collection_fingerprint=_require_str(
                data, "collection_fingerprint"
            ),
            status=ValidationStatus(_require_str(data, "status")),
            metadata=ValidationMetadata.from_dict(
                _require_dict(data, "metadata")
            ),
            validations=tuple(
                EvidenceValidation.from_dict(item)
                for item in _require_list_of_dicts(data, "validations")
            ),
            conflicts=tuple(
                EvidenceConflict.from_dict(item)
                for item in _require_list_of_dicts(data, "conflicts")
            ),
            counts=ValidationCounts.from_dict(
                _require_dict(data, "counts")
            ),
            covered_topics=tuple(_require_str_list(data, "covered_topics")),
            by_topic=tuple(
                TopicValidation.from_dict(item)
                for item in _require_list_of_dicts(data, "by_topic")
            ),
        )


# ----------------------------------------------------------------------
# Fingerprinting
# ----------------------------------------------------------------------


def compute_fingerprint(evidence: Evidence) -> EvidenceFingerprint:
    """Compute the deterministic fingerprint of one ``Evidence`` object.

    The fingerprint hashes the evidence's canonical JSON (sorted keys,
    compact separators) with SHA-256.  Two identical evidence objects
    always produce identical fingerprints, and dictionaries with the same
    content always produce the same canonical payload regardless of
    insertion order.

    Raises
    ------
    FingerprintError:
        When ``evidence`` is not an :class:`Evidence` instance.
    """
    if not isinstance(evidence, Evidence):
        raise FingerprintError(
            "fingerprint expects an Evidence instance"
        )
    canonical = canonical_json(evidence.to_dict())
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return EvidenceFingerprint(
        algorithm=FINGERPRINT_ALGORITHM,
        value=digest,
        canonical=canonical,
        prefix=FINGERPRINT_PREFIX,
    )


def collection_fingerprint(evidence_ids: Sequence[str]) -> str:
    """Return a stable fingerprint for an ordered sequence of evidence ids."""
    canonical = canonical_json({"evidence_ids": list(evidence_ids)})
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"collection_{digest[:16]}"


def content_digest(payload: Mapping[str, object]) -> str:
    """Return a stable SHA-256 hex digest over ``payload``.

    The payload is canonicalized (sorted keys, compact separators) before
    hashing so the digest never depends on dictionary insertion order.
    """
    canonical = canonical_json(payload)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{digest[:16]}"


def canonical_json(payload: object) -> str:
    """Canonicalize ``payload`` into a deterministic JSON string."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
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
