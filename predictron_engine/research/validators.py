"""Deterministic Evidence Validators — Phase 8 Sprint 4.

This module defines the abstract :class:`EvidenceValidator` contract and
the ten default deterministic validators:

* :class:`SchemaValidator` — required fields and schema shape.
* :class:`FreshnessValidator` — maximum evidence age vs. freshness rules.
* :class:`CompletenessValidator` — how complete each evidence item is.
* :class:`ConsistencyValidator` — internal coherence of one item.
* :class:`ConflictValidator` — cross-evidence conflicts from context.
* :class:`ReferenceValidator` — reference integrity and provenance.
* :class:`DuplicateEvidenceValidator` — duplicate copies in a collection.
* :class:`ConfidenceValidator` — stated confidence vs. thresholds.
* :class:`SourceMetadataValidator` — source metadata quality.
* :class:`TopicCoverageValidator` — topic consistency and coverage.

Every validator is stateless apart from its class-declared identity and
observes only the supplied :class:`Evidence` plus the deterministic
:class:`~predictron_engine.research.validation_rules.ValidationContext`
prepared by the engine.  No validator performs I/O, makes API calls, or
uses any model — outputs are fully reproducible.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar
from urllib.parse import urlsplit

from predictron_engine.research.models import Evidence, SourceCategory
from predictron_engine.research.topics import (
    SOURCE_CATEGORIES,
    get_topic,
    has_topic,
)
from predictron_engine.research.validation_models import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidatorMetadata,
)
from predictron_engine.research.validation_rules import (
    ValidationContext,
    is_snake_case,
)

_EVIDENCE_ID_PATTERN = re.compile(r"^ev_[0-9a-f]{16}$")
_KNOWN_SOURCE_CATEGORIES = frozenset(
    set(SOURCE_CATEGORIES) | {item.value for item in SourceCategory}
)


@dataclass(frozen=True)
class EvidenceValidator(ABC):
    """Abstract contract every evidence validator must satisfy.

    Concrete validators declare fixed class attributes (``validator_id``,
    ``display_name``, ``description``) and implement :meth:`validate`.
    Validators do not hold mutable state, so the same input always yields
    the same :class:`ValidationResult`.
    """

    validator_id: ClassVar[str]
    display_name: ClassVar[str]
    description: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Fail loudly when a concrete validator is misconfigured."""
        super().__init_subclass__(**kwargs)
        if not isinstance(getattr(cls, "validator_id", ""), str):
            raise TypeError(f"{cls.__name__} must declare validator_id")
        if not getattr(cls, "validator_id", "").strip():
            raise TypeError(f"{cls.__name__} must declare a validator_id")
        if not getattr(cls, "display_name", "").strip():
            raise TypeError(f"{cls.__name__} must declare a display_name")

    # ------------------------------------------------------------------
    # Required contract
    # ------------------------------------------------------------------

    @abstractmethod
    def validate(
        self,
        evidence: Evidence,
        *,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        """Validate ``evidence`` and return a deterministic result.

        Parameters
        ----------
        evidence:
            The :class:`Evidence` item being validated.
        context:
            Optional deterministic :class:`ValidationContext` carrying
            collection-level signals.  When ``None`` a default context is
            used so validators remain safe to call standalone.
        """

    # ------------------------------------------------------------------
    # Generic behaviour
    # ------------------------------------------------------------------

    def supports(self, topic_id: str) -> bool:
        """Return whether this validator handles ``topic_id``."""
        topics = self.metadata().supported_topics
        if not topics:
            return True
        return topic_id in topics

    def metadata(self) -> ValidatorMetadata:
        """Return immutable, serializable metadata for this validator."""
        cls = type(self)
        return ValidatorMetadata(
            validator_id=cls.validator_id,
            display_name=cls.display_name,
            description=getattr(cls, "description", ""),
            supported_topics=getattr(cls, "supported_topics", ()),
            deterministic=True,
        )


def _context(context: ValidationContext | None) -> ValidationContext:
    """Return ``context`` or a fresh default context."""
    return context if context is not None else ValidationContext()


def _units(
    validator: EvidenceValidator,
    evidence: Evidence,
    passed: int,
    total: int,
    issues: list[ValidationIssue],
) -> ValidationResult:
    """Build a result whose score is the fraction of checks passed."""
    if total <= 0:
        return ValidationResult(
            validator_id=validator.metadata().validator_id,
            validator_name=validator.metadata().display_name,
            passed=True,
            score=1.0,
            issues=tuple(issues),
        )
    score = passed / total
    return ValidationResult(
        validator_id=validator.metadata().validator_id,
        validator_name=validator.metadata().display_name,
        passed=not any(issue.is_failure() for issue in issues),
        score=score,
        issues=tuple(issues),
    )


def _issue(
    validator: EvidenceValidator,
    evidence: Evidence,
    code: str,
    message: str,
    severity: ValidationSeverity,
    *,
    field: str = "",
) -> ValidationIssue:
    """Build a deterministic :class:`ValidationIssue` for one validator."""
    return ValidationIssue(
        code=code,
        message=message,
        severity=severity,
        validator_id=validator.metadata().validator_id,
        evidence_id=evidence.evidence_id,
        field=field,
    )


def _valid_web_url(url: str) -> bool:
    """Return whether ``url`` is a well-formed http(s) URL."""
    if not url.strip():
        return True
    parts = urlsplit(url.strip())
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


# ----------------------------------------------------------------------
# Default validators
# ----------------------------------------------------------------------


class SchemaValidator(EvidenceValidator):
    """Validates the required schema of one evidence item."""

    validator_id = "schema_validator"
    display_name = "Schema"
    description = "Required fields and schema shape of an evidence item."

    def validate(
        self,
        evidence: Evidence,
        *,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        rules = _context(context).rules
        required = rules.get_parameter(
            "required_fields", "required_fields", ()
        )
        fields = tuple(required) if isinstance(required, tuple | list) else ()
        checks: list[tuple[str, bool]] = []
        for field_name in fields:
            present = False
            if field_name == "reference":
                present = evidence.reference is not None
            else:
                value = getattr(evidence, field_name, "")
                present = isinstance(value, str) and bool(value.strip())
            checks.append((field_name, present))
        checks.append(("confidence_in_range", 0.0 <= evidence.confidence <= 1.0))
        checks.append(
            (
                "evidence_id_complete",
                _EVIDENCE_ID_PATTERN.fullmatch(evidence.evidence_id) is not None,
            )
        )
        issues: list[ValidationIssue] = []
        passed = 0
        for field_name, ok in checks:
            if ok:
                passed += 1
            elif field_name == "confidence_in_range":
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "confidence_out_of_range",
                        "Confidence is outside the [0, 1] range.",
                        ValidationSeverity.ERROR,
                        field="confidence",
                    )
                )
            elif field_name == "evidence_id_complete":
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "malformed_evidence_id",
                        "Evidence id does not match the stable id format.",
                        ValidationSeverity.ERROR,
                        field="evidence_id",
                    )
                )
            else:
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "missing_required_field",
                        f"Required field {field_name!r} is empty.",
                        ValidationSeverity.ERROR,
                        field=field_name,
                    )
                )
        return _units(self, evidence, passed, len(checks), issues)


class FreshnessValidator(EvidenceValidator):
    """Validates that evidence is not older than its freshness rules."""

    validator_id = "freshness_validator"
    display_name = "Freshness"
    description = "Evidence within its freshness window and maximum age."

    def validate(
        self,
        evidence: Evidence,
        *,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        ctx = _context(context)
        rules = ctx.rules
        issues: list[ValidationIssue] = []
        enabled = rules.get_bool_parameter(
            "max_evidence_age", "enabled", True
        )
        if not enabled:
            return ValidationResult(
                validator_id=self.metadata().validator_id,
                validator_name=self.metadata().display_name,
                passed=True,
                score=1.0,
                issues=tuple(issues),
            )
        max_age_days = rules.get_int_parameter(
            "max_evidence_age", "max_age_days", 365
        )
        rule_time = rules.get_parameter("max_evidence_age", "reference_time", "")
        reference_time = (
            rule_time if isinstance(rule_time, str) and rule_time.strip()
            else ctx.reference_time
        )
        timestamp = ctx.timestamp_for(evidence.evidence_id)
        if timestamp is None or not timestamp.strip():
            issues.append(
                _issue(
                    self,
                    evidence,
                    "missing_evidence_timestamp",
                    "No evidence timestamp provided; freshness not assessed.",
                    ValidationSeverity.INFO,
                    field="timestamp",
                )
            )
            return ValidationResult(
                validator_id=self.metadata().validator_id,
                validator_name=self.metadata().display_name,
                passed=True,
                score=1.0,
                issues=tuple(issues),
            )
        if reference_time is None or not reference_time.strip():
            issues.append(
                _issue(
                    self,
                    evidence,
                    "missing_reference_time",
                    "No reference time provided; age cannot be computed.",
                    ValidationSeverity.INFO,
                    field="reference_time",
                )
            )
            return ValidationResult(
                validator_id=self.metadata().validator_id,
                validator_name=self.metadata().display_name,
                passed=True,
                score=1.0,
                issues=tuple(issues),
            )
        try:
            collected = datetime.fromisoformat(timestamp.strip().replace("Z", "+00:00"))
            reference = datetime.fromisoformat(
                reference_time.strip().replace("Z", "+00:00")
            )
        except ValueError:
            issues.append(
                _issue(
                    self,
                    evidence,
                    "invalid_timestamp",
                    "Evidence timestamp is not a valid ISO-8601 value.",
                    ValidationSeverity.ERROR,
                    field="timestamp",
                )
            )
            return ValidationResult(
                validator_id=self.metadata().validator_id,
                validator_name=self.metadata().display_name,
                passed=False,
                score=0.0,
                issues=tuple(issues),
            )
        age_days = max(0, (reference - collected).days)
        requirement = _freshness_requirement(evidence.topic_id, default=max_age_days)
        scale = min(requirement, max_age_days)
        if age_days <= scale:
            score = 1.0
        else:
            score = max(0.0, scale / age_days)
            if age_days > max_age_days:
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "exceeds_maximum_age",
                        f"Evidence age ({age_days} days) exceeds the maximum "
                        f"of {max_age_days} days.",
                        ValidationSeverity.ERROR,
                        field="timestamp",
                    )
                )
            else:
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "exceeds_freshness_window",
                        f"Evidence age ({age_days} days) exceeds the "
                        f"{requirement}-day freshness window for this topic.",
                        ValidationSeverity.WARNING,
                        field="timestamp",
                    )
                )
        return ValidationResult(
            validator_id=self.metadata().validator_id,
            validator_name=self.metadata().display_name,
            passed=not any(issue.is_failure() for issue in issues),
            score=score,
            issues=tuple(issues),
        )


def _freshness_requirement(topic_id: str, *, default: int) -> int:
    """Return the taxonomy freshness requirement for ``topic_id``."""
    if not has_topic(topic_id):
        return default
    return get_topic(topic_id).freshness_requirement_days


class CompletenessValidator(EvidenceValidator):
    """Validates how complete one piece of evidence is."""

    validator_id = "completeness_validator"
    display_name = "Completeness"
    description = "Completeness of claims, references, and provenance."

    def validate(
        self,
        evidence: Evidence,
        *,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        ctx = _context(context)
        rules = ctx.rules
        min_claim_chars = rules.get_int_parameter(
            "minimum_completeness", "min_claim_chars", 20
        )
        require_url = rules.get_bool_parameter(
            "minimum_completeness", "require_reference_url", True
        )
        require_description = rules.get_bool_parameter(
            "minimum_completeness", "require_description", False
        )
        min_score = rules.get_float_parameter(
            "minimum_completeness", "min_score", 0.5
        )
        checks: list[tuple[str, bool]] = [
            ("claim_length", len(evidence.claim) >= min_claim_chars),
            ("collector_id", bool(evidence.collector_id.strip())),
            ("category", bool(evidence.category.strip())),
            ("reference_identifier", bool(evidence.reference.source_identifier.strip())),
            ("reference_url", _valid_web_url(evidence.reference.url)),
            ("reference_description", bool(evidence.reference.description.strip())),
            ("confidence", evidence.confidence > 0.0),
        ]
        if not require_url:
            checks = [(name, ok) for name, ok in checks if name != "reference_url"]
        if not require_description:
            checks = [
                (name, ok) for name, ok in checks if name != "reference_description"
            ]
        passed = sum(ok for _, ok in checks)
        score = passed / len(checks) if checks else 1.0
        issues: list[ValidationIssue] = []
        if score < min_score:
            issues.append(
                _issue(
                    self,
                    evidence,
                    "incomplete_evidence",
                    f"Evidence completeness {score:.2f} is below the "
                    f"required minimum {min_score:.2f}.",
                    ValidationSeverity.WARNING,
                )
            )
        return _units(self, evidence, passed, len(checks), issues)


@dataclass(frozen=True)
class ConsistencyValidator(EvidenceValidator):
    """Validates the internal coherence of one evidence item."""

    validator_id: ClassVar[str] = "consistency_validator"
    display_name: ClassVar[str] = "Consistency"
    description: ClassVar[str] = "Internal consistency of one evidence item."

    def validate(
        self,
        evidence: Evidence,
        *,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        checks: list[tuple[str, bool]] = [
            ("reference_url_scheme", _valid_web_url(evidence.reference.url)),
            ("reference_category", bool(evidence.reference.source_category.strip())),
            ("reference_identifier", bool(evidence.reference.source_identifier.strip())),
            (
                "evidence_id_format",
                _EVIDENCE_ID_PATTERN.fullmatch(evidence.evidence_id) is not None,
            ),
            ("claim_usable", bool(evidence.claim.strip())),
        ]
        issues: list[ValidationIssue] = []
        passed = 0
        for name, ok in checks:
            if ok:
                passed += 1
            elif name == "reference_url_scheme":
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "invalid_reference_url",
                        "Reference URL is not a well-formed http(s) URL.",
                        ValidationSeverity.ERROR,
                        field="reference.url",
                    )
                )
            else:
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "inconsistent_field",
                        f"Field {name!r} is inconsistent with the evidence contract.",
                        ValidationSeverity.WARNING,
                        field=name,
                    )
                )
        if not _category_preferred(evidence):
            issues.append(
                _issue(
                    self,
                    evidence,
                    "non_preferred_source_category",
                    "The reference source category is not preferred for "
                    "this research topic.",
                    ValidationSeverity.WARNING,
                    field="reference.source_category",
                )
            )
        return _units(self, evidence, passed, len(checks), issues)


def _category_preferred(evidence: Evidence) -> bool:
    """Return whether the reference category suits the evidence topic."""
    if not has_topic(evidence.topic_id):
        return True
    preferred = get_topic(evidence.topic_id).source_categories
    if not preferred:
        return True
    return evidence.reference.source_category in preferred


class ConflictValidator(EvidenceValidator):
    """Reports cross-evidence conflicts for one evidence item."""

    validator_id = "conflict_validator"
    display_name = "Conflict"
    description = "Cross-evidence contradictions and conflicts."

    def validate(
        self,
        evidence: Evidence,
        *,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        ctx = _context(context)
        conflicts = ctx.conflicts_for(evidence.evidence_id)
        relevant = tuple(
            conflict
            for conflict in conflicts
            if conflict.conflict_type != "duplicate_evidence"
        )
        issues = [
            _issue(
                self,
                evidence,
                "evidence_conflict",
                conflict.description,
                conflict.severity,
                field=conflict.field,
            )
            for conflict in relevant
        ]
        score = 1.0 / (1.0 + len(relevant)) if len(relevant) else 1.0
        return ValidationResult(
            validator_id=self.metadata().validator_id,
            validator_name=self.metadata().display_name,
            passed=not relevant,
            score=score,
            issues=tuple(issues),
        )


@dataclass(frozen=True)
class ReferenceValidator(EvidenceValidator):
    """Validates reference provenance and integrity."""

    validator_id: ClassVar[str] = "reference_validator"
    display_name: ClassVar[str] = "Reference"
    description: ClassVar[str] = "Reference provenance and integrity."

    def validate(
        self,
        evidence: Evidence,
        *,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        rules = _context(context).rules
        require_url = rules.get_bool_parameter(
            "reference_integrity", "require_url", True
        )
        require_description = rules.get_bool_parameter(
            "reference_integrity", "require_description", False
        )
        reference = evidence.reference
        checks: list[tuple[str, bool]] = [
            ("source_category", bool(reference.source_category.strip())),
            ("source_identifier", bool(reference.source_identifier.strip())),
            ("url_well_formed", _valid_web_url(reference.url)),
            ("url_present", bool(reference.url.strip())),
            ("description_present", bool(reference.description.strip())),
        ]
        if not require_url:
            checks = [(name, ok) for name, ok in checks if name != "url_present"]
        if not require_description:
            checks = [
                (name, ok) for name, ok in checks if name != "description_present"
            ]
        issues: list[ValidationIssue] = []
        passed = 0
        for name, ok in checks:
            if ok:
                passed += 1
            elif name in {"source_category", "source_identifier", "url_well_formed"}:
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "reference_integrity",
                        f"Reference field {name!r} is invalid.",
                        ValidationSeverity.ERROR,
                        field=f"reference.{name}",
                    )
                )
            else:
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "missing_reference_field",
                        f"Reference field {name!r} is missing.",
                        ValidationSeverity.WARNING,
                        field=f"reference.{name}",
                    )
                )
        return _units(self, evidence, passed, len(checks), issues)


class DuplicateEvidenceValidator(EvidenceValidator):
    """Reports evidence that appears more than once in a collection."""

    validator_id = "duplicate_validator"
    display_name = "Duplicate Evidence"
    description = "Duplicate copies within a collected collection."

    def validate(
        self,
        evidence: Evidence,
        *,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        ctx = _context(context)
        rules = ctx.rules
        enabled = rules.get_bool_parameter(
            "duplicate_detection", "enabled", True
        )
        treat_as_error = rules.get_bool_parameter(
            "duplicate_detection", "treat_as_error", False
        )
        if not enabled:
            return ValidationResult(
                validator_id=self.metadata().validator_id,
                validator_name=self.metadata().display_name,
                passed=True,
                score=1.0,
                issues=(),
            )
        count = ctx.duplicate_count_for(evidence.evidence_id)
        if count <= 1:
            return ValidationResult(
                validator_id=self.metadata().validator_id,
                validator_name=self.metadata().display_name,
                passed=True,
                score=1.0,
                issues=(),
            )
        severity = (
            ValidationSeverity.ERROR
            if treat_as_error
            else ValidationSeverity.WARNING
        )
        issues = (
            _issue(
                self,
                evidence,
                "duplicate_evidence",
                f"Evidence appears {count} times in the collection.",
                severity,
                field="evidence_id",
            ),
        )
        return ValidationResult(
            validator_id=self.metadata().validator_id,
            validator_name=self.metadata().display_name,
            passed=False,
            score=1.0 / count,
            issues=issues,
        )


class ConfidenceValidator(EvidenceValidator):
    """Validates the stated confidence of one evidence item."""

    validator_id = "confidence_validator"
    display_name = "Confidence"
    description = "Stated confidence of an evidence item."

    def validate(
        self,
        evidence: Evidence,
        *,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        rules = _context(context).rules
        min_stated = rules.get_float_parameter(
            "confidence_threshold", "min_stated_score", 0.0
        )
        issues: list[ValidationIssue] = []
        if evidence.confidence < min_stated:
            issues.append(
                _issue(
                    self,
                    evidence,
                    "stated_confidence_below_threshold",
                    f"Stated confidence {evidence.confidence:.2f} is below "
                    f"the required minimum {min_stated:.2f}.",
                    ValidationSeverity.WARNING,
                    field="confidence",
                )
            )
        if evidence.confidence == 0.0:
            issues.append(
                _issue(
                    self,
                    evidence,
                    "zero_confidence_evidence",
                    "Evidence carries zero stated confidence.",
                    ValidationSeverity.ERROR,
                    field="confidence",
                )
            )
        return ValidationResult(
            validator_id=self.metadata().validator_id,
            validator_name=self.metadata().display_name,
            passed=not any(issue.is_failure() for issue in issues),
            score=evidence.confidence,
            issues=tuple(issues),
        )


class SourceMetadataValidator(EvidenceValidator):
    """Validates the quality of source metadata on an evidence item."""

    validator_id = "source_metadata_validator"
    display_name = "Source Metadata"
    description = "Source metadata quality and controlled vocabulary."

    def validate(
        self,
        evidence: Evidence,
        *,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        rules = _context(context).rules
        require_snake = rules.get_bool_parameter(
            "metadata_quality", "require_snake_case_ids", True
        )
        require_known_category = rules.get_bool_parameter(
            "metadata_quality", "require_known_category", True
        )
        reference = evidence.reference
        checks: list[tuple[str, bool]] = [
            ("collector_id", bool(evidence.collector_id.strip())),
            (
                "collector_id_snake",
                is_snake_case(evidence.collector_id) if require_snake else True,
            ),
            (
                "source_category_known",
                reference.source_category in _KNOWN_SOURCE_CATEGORIES
                if require_known_category
                else True,
            ),
            ("source_identifier", bool(reference.source_identifier.strip())),
        ]
        issues: list[ValidationIssue] = []
        passed = 0
        for name, ok in checks:
            if ok:
                passed += 1
            elif name == "collector_id_snake":
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "non_snake_case_collector_id",
                        "Collector id does not follow the snake_case convention.",
                        ValidationSeverity.WARNING,
                        field="collector_id",
                    )
                )
            elif name == "source_category_known":
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "unknown_source_category",
                        "The source category is outside the controlled vocabulary.",
                        ValidationSeverity.WARNING,
                        field="reference.source_category",
                    )
                )
            else:
                issues.append(
                    _issue(
                        self,
                        evidence,
                        "incomplete_source_metadata",
                        f"Source metadata field {name!r} is missing.",
                        ValidationSeverity.WARNING,
                        field=name,
                    )
                )
        return _units(self, evidence, passed, len(checks), issues)


class TopicCoverageValidator(EvidenceValidator):
    """Validates topic consistency and per-topic coverage."""

    validator_id = "topic_coverage_validator"
    display_name = "Topic Coverage"
    description = "Topic consistency and minimum coverage per topic."

    def validate(
        self,
        evidence: Evidence,
        *,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        ctx = _context(context)
        rules = ctx.rules
        require_known = rules.get_bool_parameter(
            "topic_consistency", "require_known_topic", True
        )
        coverage_enabled = rules.get_bool_parameter(
            "cross_evidence_consistency", "enabled", True
        )
        min_evidence = rules.get_int_parameter(
            "cross_evidence_consistency", "min_evidence_per_topic", 1
        )
        issues: list[ValidationIssue] = []
        if require_known and not has_topic(evidence.topic_id):
            issues.append(
                _issue(
                    self,
                    evidence,
                    "unknown_topic",
                    f"Topic {evidence.topic_id!r} is not part of the research taxonomy.",
                    ValidationSeverity.ERROR,
                    field="topic_id",
                )
            )
        if evidence.topic_id not in ctx.topics_in_collection:
            issues.append(
                _issue(
                    self,
                    evidence,
                    "topic_not_in_collection",
                    "Evidence topic is not covered by the collection.",
                    ValidationSeverity.ERROR,
                    field="topic_id",
                )
            )
        count = ctx.evidence_count_by_topic.get(evidence.topic_id, 0)
        if coverage_enabled and count < min_evidence:
            issues.append(
                _issue(
                    self,
                    evidence,
                    "topic_under_covered",
                    f"Topic {evidence.topic_id!r} has {count} evidence items "
                    f"but at least {min_evidence} are required.",
                    ValidationSeverity.WARNING,
                    field="topic_id",
                )
            )
        score = min(1.0, count / min_evidence) if min_evidence else 1.0
        return ValidationResult(
            validator_id=self.metadata().validator_id,
            validator_name=self.metadata().display_name,
            passed=not any(issue.is_failure() for issue in issues),
            score=score,
            issues=tuple(issues),
        )


# Default validators in deterministic registration order.
DEFAULT_VALIDATORS: tuple[EvidenceValidator, ...] = (
    SchemaValidator(),
    FreshnessValidator(),
    CompletenessValidator(),
    ConsistencyValidator(),
    ConflictValidator(),
    ReferenceValidator(),
    DuplicateEvidenceValidator(),
    ConfidenceValidator(),
    SourceMetadataValidator(),
    TopicCoverageValidator(),
)

# A shared, immutable catalog used by the registry to reconstruct
# validators when deserializing.  Instances are stateless singletons.
VALIDATOR_CATALOG: dict[str, EvidenceValidator] = {
    validator.metadata().validator_id: validator
    for validator in DEFAULT_VALIDATORS
}

__all__ = [
    "CompletenessValidator",
    "ConfidenceValidator",
    "ConflictValidator",
    "ConsistencyValidator",
    "DEFAULT_VALIDATORS",
    "DuplicateEvidenceValidator",
    "EvidenceValidator",
    "FreshnessValidator",
    "ReferenceValidator",
    "SchemaValidator",
    "SourceMetadataValidator",
    "TopicCoverageValidator",
    "VALIDATOR_CATALOG",
]
