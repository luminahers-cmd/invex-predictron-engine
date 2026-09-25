"""Configurable validation rules — Phase 8 Sprint 4.

Rules configure the deterministic validators without modifying engine
logic.  Each rule is an immutable, serializable object exposing:

* a stable ``rule_id`` (``ClassVar``),
* a human-readable ``description`` (``ClassVar``),
* fixed ``parameters`` exposed through :meth:`parameters`,
* lossless ``to_dict`` / ``from_dict`` serialization.

A :class:`ValidationRuleSet` groups rules for one run, enforcing unique
rule identifiers, and validators read their configuration through it.
Section :class:`ValidationContext` carries the deterministic run context
every validator observes (timestamps, duplicate/conflict indexes, topic
coverage) so validators stay stateless and pluggable.

The rule engine is pure: no I/O, no randomness, and identical inputs
always produce identical configurations.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from predictron_engine.research.exceptions import ValidationRuleError
from predictron_engine.research.validation_models import (
    EvidenceConfidence,
    EvidenceConflict,
    _require_bool,
    _require_float,
    _require_str,
)

_SNAKE_CASE = re.compile(r"^[a-z_][a-z0-9_]*$")


class ValidationRule(ABC):
    """Base contract every validation rule must satisfy.

    Concrete rules declare fixed ``rule_id`` and ``description`` class
    attributes and expose their configuration through :meth:`parameters`.
    """

    rule_id: ClassVar[str]
    description: ClassVar[str] = ""

    def __init__(self) -> None:
        # Re-run the guard for subclass instances.
        cls = type(self)
        rule_id = getattr(cls, "rule_id", "")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValidationRuleError(
                f"{cls.__name__} must declare a non-empty rule_id"
            )
        if not isinstance(getattr(cls, "description", ""), str):
            raise ValidationRuleError(
                f"{cls.__name__}.description must be a string"
            )
        parameters = self.parameters()
        if not isinstance(parameters, Mapping):
            raise ValidationRuleError(
                f"{cls.__name__}.parameters() must return a Mapping"
            )

    @abstractmethod
    def parameters(self) -> Mapping[str, object]:
        """Return the rule's fixed, deterministic parameters."""

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        rule_id = getattr(type(self), "rule_id", "")
        return {
            "rule_id": rule_id,
            "description": getattr(type(self), "description", ""),
            "parameters": dict(self.parameters()),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ValidationRule:
        """Deserialize by dispatching on the ``rule_id`` field.

        Raises
        ------
        ValidationRuleError:
            When the payload is malformed or references an unknown rule.
        """
        return rule_from_dict(data)


@dataclass(frozen=True)
class RequiredFieldsRule(ValidationRule):
    """Configures which evidence fields are mandatory."""

    rule_id: ClassVar[str] = "required_fields"
    description: ClassVar[str] = "Required fields on each evidence item."

    required_fields: tuple[str, ...] = (
        "task_id",
        "topic_id",
        "collector_id",
        "category",
        "claim",
        "reference",
    )

    def __post_init__(self) -> None:
        if not self.required_fields:
            raise ValidationRuleError(
                "required_fields must contain at least one field"
            )

    def parameters(self) -> Mapping[str, object]:
        return {"required_fields": list(self.required_fields)}


@dataclass(frozen=True)
class MaxEvidenceAgeRule(ValidationRule):
    """Configures maximum admissible age for collected evidence.

    ``max_age_days`` caps how old evidence may be before it is flagged.
    ``reference_time`` is an optional deterministic ISO-8601 timestamp
    used as the "now" for age computation; when omitted the engine falls
    back to its own reference time (which must then be supplied by the
    caller to keep determinism).
    """

    rule_id: ClassVar[str] = "max_evidence_age"
    description: ClassVar[str] = "Maximum age allowed for evidence."

    max_age_days: int = 365
    reference_time: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_age_days < 0:
            raise ValidationRuleError(
                "max_age_days must be non-negative"
            )
        if not isinstance(self.enabled, bool):
            raise ValidationRuleError("enabled must be a bool")
        if not isinstance(self.reference_time, str):
            raise ValidationRuleError("reference_time must be a string")

    def parameters(self) -> Mapping[str, object]:
        return {
            "max_age_days": self.max_age_days,
            "reference_time": self.reference_time,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class MinimumCompletenessRule(ValidationRule):
    """Configures the minimum completeness a piece of evidence needs."""

    rule_id: ClassVar[str] = "minimum_completeness"
    description: ClassVar[str] = "Minimum completeness threshold for evidence."

    min_score: float = 0.5
    min_claim_chars: int = 20
    require_reference_url: bool = True
    require_description: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValidationRuleError(
                "min_score must be within [0, 1]"
            )
        if self.min_claim_chars < 0:
            raise ValidationRuleError(
                "min_claim_chars must be non-negative"
            )
        if not isinstance(self.require_reference_url, bool):
            raise ValidationRuleError(
                "require_reference_url must be a bool"
            )
        if not isinstance(self.require_description, bool):
            raise ValidationRuleError("require_description must be a bool")

    def parameters(self) -> Mapping[str, object]:
        return {
            "min_score": self.min_score,
            "min_claim_chars": self.min_claim_chars,
            "require_reference_url": self.require_reference_url,
            "require_description": self.require_description,
        }


@dataclass(frozen=True)
class DuplicateDetectionRule(ValidationRule):
    """Configures duplicate detection behaviour."""

    rule_id: ClassVar[str] = "duplicate_detection"
    description: ClassVar[str] = "Duplicate detection for collected evidence."

    enabled: bool = True
    treat_as_error: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValidationRuleError("enabled must be a bool")
        if not isinstance(self.treat_as_error, bool):
            raise ValidationRuleError("treat_as_error must be a bool")

    def parameters(self) -> Mapping[str, object]:
        return {
            "enabled": self.enabled,
            "treat_as_error": self.treat_as_error,
        }


@dataclass(frozen=True)
class TopicConsistencyRule(ValidationRule):
    """Configures topic consistency enforcement."""

    rule_id: ClassVar[str] = "topic_consistency"
    description: ClassVar[str] = "Evidence must reference known topics."

    require_known_topic: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.require_known_topic, bool):
            raise ValidationRuleError(
                "require_known_topic must be a bool"
            )

    def parameters(self) -> Mapping[str, object]:
        return {"require_known_topic": self.require_known_topic}


@dataclass(frozen=True)
class ReferenceIntegrityRule(ValidationRule):
    """Configures reference integrity enforcement."""

    rule_id: ClassVar[str] = "reference_integrity"
    description: ClassVar[str] = "References must be complete and well-formed."

    require_url: bool = True
    require_description: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.require_url, bool):
            raise ValidationRuleError("require_url must be a bool")
        if not isinstance(self.require_description, bool):
            raise ValidationRuleError("require_description must be a bool")

    def parameters(self) -> Mapping[str, object]:
        return {
            "require_url": self.require_url,
            "require_description": self.require_description,
        }


@dataclass(frozen=True)
class CrossEvidenceConsistencyRule(ValidationRule):
    """Configures cross-evidence consistency and topic coverage."""

    rule_id: ClassVar[str] = "cross_evidence_consistency"
    description: ClassVar[str] = "Cross-evidence consistency and coverage."

    min_evidence_per_topic: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.min_evidence_per_topic < 1:
            raise ValidationRuleError(
                "min_evidence_per_topic must be at least 1"
            )
        if not isinstance(self.enabled, bool):
            raise ValidationRuleError("enabled must be a bool")

    def parameters(self) -> Mapping[str, object]:
        return {
            "min_evidence_per_topic": self.min_evidence_per_topic,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class ConfidenceThresholdRule(ValidationRule):
    """Configures minimum confidence thresholds for evidence."""

    rule_id: ClassVar[str] = "confidence_threshold"
    description: ClassVar[str] = "Minimum confidence thresholds for evidence."

    min_band: EvidenceConfidence = EvidenceConfidence.LOW
    min_stated_score: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.min_band, EvidenceConfidence):
            raise ValidationRuleError("min_band must be an EvidenceConfidence")
        if not 0.0 <= self.min_stated_score <= 1.0:
            raise ValidationRuleError(
                "min_stated_score must be within [0, 1]"
            )

    def parameters(self) -> Mapping[str, object]:
        return {
            "min_band": self.min_band.value,
            "min_stated_score": self.min_stated_score,
        }


@dataclass(frozen=True)
class MetadataQualityRule(ValidationRule):
    """Configures source metadata quality enforcement."""

    rule_id: ClassVar[str] = "metadata_quality"
    description: ClassVar[str] = "Source metadata must be well-formed."

    require_snake_case_ids: bool = True
    require_known_category: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.require_snake_case_ids, bool):
            raise ValidationRuleError(
                "require_snake_case_ids must be a bool"
            )
        if not isinstance(self.require_known_category, bool):
            raise ValidationRuleError(
                "require_known_category must be a bool"
            )

    def parameters(self) -> Mapping[str, object]:
        return {
            "require_snake_case_ids": self.require_snake_case_ids,
            "require_known_category": self.require_known_category,
        }


# ----------------------------------------------------------------------
# Rule registry (deterministic deserialization dispatch)
# ----------------------------------------------------------------------

_RULE_TYPES: dict[str, type[ValidationRule]] = {
    cls.rule_id: cls
    for cls in (
        RequiredFieldsRule,
        MaxEvidenceAgeRule,
        MinimumCompletenessRule,
        DuplicateDetectionRule,
        TopicConsistencyRule,
        ReferenceIntegrityRule,
        CrossEvidenceConsistencyRule,
        ConfidenceThresholdRule,
        MetadataQualityRule,
    )
}


def rule_from_dict(data: Mapping[str, object]) -> ValidationRule:
    """Deserialize a rule by its ``rule_id`` using fixed parameters.

    Raises
    ------
    ValidationRuleError:
        When the payload is malformed or references an unknown rule id.
    """
    rule_id = _require_str(data, "rule_id")
    rule_cls = _RULE_TYPES.get(rule_id)
    if rule_cls is None:
        raise ValidationRuleError(f"unknown validation rule: {rule_id!r}")
    parameters = data.get("parameters")
    if not isinstance(parameters, dict):
        raise ValidationRuleError(
            f"rule {rule_id!r} serialization is missing parameters"
        )
    try:
        return _rule_from_parameters(rule_cls, parameters)
    except (TypeError, ValueError) as exc:
        raise ValidationRuleError(
            f"invalid parameters for rule {rule_id!r}"
        ) from exc


def _rule_from_parameters(
    rule_cls: type[ValidationRule],
    parameters: dict[str, object],
) -> ValidationRule:
    """Reconstruct a rule instance from its declared parameters."""
    if rule_cls is RequiredFieldsRule:
        required = parameters.get("required_fields")
        if isinstance(required, list) and all(
            isinstance(item, str) for item in required
        ):
            return RequiredFieldsRule(required_fields=tuple(required))
    if rule_cls is MaxEvidenceAgeRule:
        return MaxEvidenceAgeRule(
            max_age_days=int(_require_float(parameters, "max_age_days")),
            reference_time=_require_str(parameters, "reference_time"),
            enabled=_require_bool(
                parameters, "enabled"
            ),
        )
    if rule_cls is MinimumCompletenessRule:
        return MinimumCompletenessRule(
            min_score=_require_float(parameters, "min_score"),
            min_claim_chars=int(_require_float(parameters, "min_claim_chars")),
            require_reference_url=_require_bool(
                parameters, "require_reference_url"
            ),
            require_description=_require_bool(
                parameters, "require_description"
            ),
        )
    if rule_cls is DuplicateDetectionRule:
        return DuplicateDetectionRule(
            enabled=_require_bool(parameters, "enabled"),
            treat_as_error=_require_bool(parameters, "treat_as_error"),
        )
    if rule_cls is TopicConsistencyRule:
        return TopicConsistencyRule(
            require_known_topic=_require_bool(
                parameters, "require_known_topic"
            )
        )
    if rule_cls is ReferenceIntegrityRule:
        return ReferenceIntegrityRule(
            require_url=_require_bool(parameters, "require_url"),
            require_description=_require_bool(
                parameters, "require_description"
            ),
        )
    if rule_cls is CrossEvidenceConsistencyRule:
        return CrossEvidenceConsistencyRule(
            min_evidence_per_topic=int(
                _require_float(parameters, "min_evidence_per_topic")
            ),
            enabled=_require_bool(parameters, "enabled"),
        )
    if rule_cls is ConfidenceThresholdRule:
        min_band = EvidenceConfidence(
            _require_str(parameters, "min_band")
        )
        return ConfidenceThresholdRule(
            min_band=min_band,
            min_stated_score=_require_float(
                parameters, "min_stated_score"
            ),
        )
    if rule_cls is MetadataQualityRule:
        return MetadataQualityRule(
            require_snake_case_ids=_require_bool(
                parameters, "require_snake_case_ids"
            ),
            require_known_category=_require_bool(
                parameters, "require_known_category"
            ),
        )
    raise ValidationRuleError(f"unsupported rule class: {rule_cls.__name__}")


# ----------------------------------------------------------------------
# Rule set container
# ----------------------------------------------------------------------


def is_snake_case(value: str) -> bool:
    """Return whether ``value`` matches the snake_case convention."""
    return bool(_SNAKE_CASE.fullmatch(value))


@dataclass(frozen=True)
class ValidationRuleSet:
    """An immutable set of validation rules with unique rule ids.

    Rules are ordered deterministically (the order they were supplied).
    Lookup by ``rule_id`` is stable; querying a missing rule returns
    ``None`` so validators can degrade gracefully.
    """

    rules: tuple[ValidationRule, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        ordered: list[ValidationRule] = []
        for rule in self.rules:
            if not isinstance(rule, ValidationRule):
                raise ValidationRuleError(
                    "rules must be ValidationRule instances"
                )
            rule_id = getattr(type(rule), "rule_id", "")
            if rule_id in seen:
                raise ValidationRuleError(
                    f"duplicate validation rule: {rule_id!r}"
                )
            seen.add(rule_id)
            ordered.append(rule)
        object.__setattr__(self, "rules", tuple(ordered))

    def __len__(self) -> int:
        """Return the number of rules in the set."""
        return len(self.rules)

    def __contains__(self, rule_id: object) -> bool:
        """Return whether ``rule_id`` is present."""
        return isinstance(rule_id, str) and self.has_rule(rule_id)

    def get(self, rule_id: str) -> ValidationRule | None:
        """Return the rule for ``rule_id`` or ``None``."""
        for rule in self.rules:
            if getattr(type(rule), "rule_id", "") == rule_id:
                return rule
        return None

    def has_rule(self, rule_id: str) -> bool:
        """Return whether ``rule_id`` is present."""
        return self.get(rule_id) is not None

    def rule_ids(self) -> tuple[str, ...]:
        """Return rule identifiers in deterministic order."""
        return tuple(getattr(type(rule), "rule_id", "") for rule in self.rules)

    def get_parameter(
        self,
        rule_id: str,
        key: str,
        default: object,
    ) -> object:
        """Return a parameter from ``rule_id`` or ``default``.

        Validators use this to read configuration without knowing whether
        a rule is present.  The default value must be provided so lookup
        stays total and deterministic.
        """
        rule = self.get(rule_id)
        if rule is None:
            return default
        return rule.parameters().get(key, default)

    def get_bool_parameter(self, rule_id: str, key: str, default: bool) -> bool:
        """Return a boolean parameter with enforced boolean typing."""
        value = self.get_parameter(rule_id, key, default)
        if not isinstance(value, bool):
            raise ValidationRuleError(
                f"parameter {key!r} of rule {rule_id!r} must be a bool"
            )
        return value

    def get_int_parameter(
        self, rule_id: str, key: str, default: int
    ) -> int:
        """Return an integer parameter with enforced integer typing."""
        value = self.get_parameter(rule_id, key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationRuleError(
                f"parameter {key!r} of rule {rule_id!r} must be an int"
            )
        return value

    def get_float_parameter(
        self, rule_id: str, key: str, default: float
    ) -> float:
        """Return a float parameter with enforced float typing."""
        value = self.get_parameter(rule_id, key, default)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValidationRuleError(
                f"parameter {key!r} of rule {rule_id!r} must be a number"
            )
        return float(value)

    def to_dict(self) -> dict[str, object]:
        """Serialize every rule to a JSON-ready dictionary."""
        return {"rules": [rule.to_dict() for rule in self.rules]}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ValidationRuleSet:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        rules = data.get("rules")
        if not isinstance(rules, list):
            raise ValidationRuleError("expected a 'rules' list")
        return cls(rules=tuple(rule_from_dict(item) for item in rules))


# Default rule configuration used when a caller supplies none.
DEFAULT_RULES: tuple[ValidationRule, ...] = (
    RequiredFieldsRule(),
    MaxEvidenceAgeRule(),
    MinimumCompletenessRule(),
    DuplicateDetectionRule(),
    TopicConsistencyRule(),
    ReferenceIntegrityRule(),
    CrossEvidenceConsistencyRule(),
    ConfidenceThresholdRule(),
    MetadataQualityRule(),
)

DEFAULT_RULE_SET: ValidationRuleSet = ValidationRuleSet(DEFAULT_RULES)


# ----------------------------------------------------------------------
# Deterministic run context
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationContext:
    """Immutable, deterministic context observed by every validator.

    Collection-level signals (duplicate counts, conflicts, topic coverage,
    evidence timestamps) are computed once by the engine and passed to
    stateless validators through this context.  No field depends on wall
    clock time or randomness: identical inputs produce identical contexts.
    """

    rules: ValidationRuleSet = DEFAULT_RULE_SET
    timestamp_by_evidence: Mapping[str, str] = field(default_factory=dict)
    reference_time: str | None = None
    previous_scores: Mapping[str, float] = field(default_factory=dict)
    duplicate_count_by_evidence: Mapping[str, int] = field(
        default_factory=dict
    )
    conflicts_by_evidence: Mapping[str, tuple[EvidenceConflict, ...]] = field(
        default_factory=dict
    )
    topics_in_collection: tuple[str, ...] = ()
    evidence_count_by_topic: Mapping[str, int] = field(default_factory=dict)
    topic_order: tuple[str, ...] = ()

    def timestamp_for(self, evidence_id: str) -> str | None:
        """Return the deterministic timestamp for ``evidence_id``."""
        return self.timestamp_by_evidence.get(evidence_id)

    def duplicate_count_for(self, evidence_id: str) -> int:
        """Return how many copies of ``evidence_id`` exist."""
        return self.duplicate_count_by_evidence.get(evidence_id, 1)

    def conflicts_for(
        self, evidence_id: str
    ) -> tuple[EvidenceConflict, ...]:
        """Return the conflicts involving ``evidence_id``."""
        return self.conflicts_by_evidence.get(evidence_id, ())
