"""Tests for the validator registry — Phase 8 Sprint 4.

Covers registration order and lookup, duplicate and type errors,
unregister idempotency, topic-scoped resolution, serialization, schema
version enforcement, instance isolation, and the shared-registry
convenience API.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.exceptions import (
    DuplicateValidator,
    RegistryError,
    UnsupportedValidation,
)
from predictron_engine.research.validation_models import (
    VALIDATION_SCHEMA_VERSION,
    ValidationResult,
)
from predictron_engine.research.validator_registry import (
    VALIDATOR_REGISTRY,
    ValidatorRegistry,
    default_validator_registry,
    has_validator,
    list_validators,
    register_validator,
    resolve_validator,
    unregister_validator,
)
from predictron_engine.research.validators import (
    DEFAULT_VALIDATORS,
    EvidenceValidator,
    FreshnessValidator,
    SchemaValidator,
)


class _TopicValidator(EvidenceValidator):
    validator_id = "topic_validator"
    display_name = "Topic"
    supported_topics = ("product",)

    def validate(self, evidence, *, context=None) -> ValidationResult:
        return ValidationResult(
            validator_id=self.validator_id,
            validator_name=self.display_name,
            passed=True,
            score=1.0,
        )


class _OtherTopicValidator(EvidenceValidator):
    validator_id = "other_topic_validator"
    display_name = "Other Topic"
    supported_topics = ("product", "market")

    def validate(self, evidence, *, context=None) -> ValidationResult:
        return ValidationResult(
            validator_id=self.validator_id,
            validator_name=self.display_name,
            passed=True,
            score=1.0,
        )


def _ids(validators) -> tuple[str, ...]:
    return tuple(validator.metadata().validator_id for validator in validators)


def test_register_get_list_preserve_order() -> None:
    registry = ValidatorRegistry((SchemaValidator(), FreshnessValidator()))
    assert _ids(registry.list_validators()) == ("schema_validator", "freshness_validator")
    assert registry.get("schema_validator") is not None
    assert registry.get("freshness_validator") is not None
    assert registry.get("missing") is None
    assert registry.validator_ids() == (
        "schema_validator",
        "freshness_validator",
    )
    assert registry.metadata()[0].display_name == "Schema"


def test_duplicate_registration_rejected() -> None:
    registry = ValidatorRegistry((SchemaValidator(),))
    with pytest.raises(DuplicateValidator):
        registry.register(SchemaValidator())


def test_register_rejects_non_validator() -> None:
    registry = ValidatorRegistry()
    with pytest.raises(RegistryError):
        registry.register(object())  # type: ignore[arg-type]


def test_unregister_removes_and_is_idempotent() -> None:
    registry = ValidatorRegistry((SchemaValidator(), FreshnessValidator()))
    removed = registry.unregister("schema_validator")
    assert removed is not None
    assert registry.validator_ids() == ("freshness_validator",)
    assert registry.unregister("schema_validator") is None
    assert registry.unregister("never_registered") is None


def test_length_and_contains() -> None:
    registry = ValidatorRegistry((SchemaValidator(),))
    assert len(registry) == 1
    assert "schema_validator" in registry
    assert "missing" not in registry
    assert 42 not in registry


def test_validators_for_filters_by_topic() -> None:
    topic_validator = _TopicValidator()
    registry = ValidatorRegistry((topic_validator, SchemaValidator()))
    assert _ids(registry.validators_for("product")) == (
        "topic_validator",
        "schema_validator",
    )
    assert _ids(registry.validators_for("founders")) == ("schema_validator",)


def test_resolve_first_registered_wins() -> None:
    first = _TopicValidator()
    second = _OtherTopicValidator()
    registry = ValidatorRegistry((first, second))
    assert registry.resolve("product") is first


def test_resolve_unsupported_topic_raises() -> None:
    with pytest.raises(UnsupportedValidation):
        ValidatorRegistry().resolve("product")


def test_serialization_roundtrip() -> None:
    registry = ValidatorRegistry((SchemaValidator(), FreshnessValidator()))
    decoded = ValidatorRegistry.from_dict(registry.to_dict())  # type: ignore[arg-type]
    assert decoded.validator_ids() == registry.validator_ids()
    assert decoded.metadata()[1].display_name == "Freshness"


def test_from_dict_rejects_unsupported_version() -> None:
    with pytest.raises(RegistryError):
        ValidatorRegistry.from_dict(  # type: ignore[arg-type]
            {"schema_version": "9.9.9", "validators": []}
        )


def test_from_dict_rejects_unknown_validator() -> None:
    with pytest.raises(RegistryError):
        ValidatorRegistry.from_dict(  # type: ignore[arg-type]
            {
                "schema_version": VALIDATION_SCHEMA_VERSION,
                "validators": [{"validator_id": "ghost", "display_name": "Ghost"}],
            }
        )


def test_from_dict_rejects_malformed_payload() -> None:
    with pytest.raises(RegistryError):
        ValidatorRegistry.from_dict({"schema_version": VALIDATION_SCHEMA_VERSION})  # type: ignore[arg-type]


def test_registry_instances_are_isolated() -> None:
    first = ValidatorRegistry()
    second = ValidatorRegistry()
    first.register(_TopicValidator())
    assert len(first) == 1
    assert len(second) == 0
    assert second.validator_ids() == ()


def test_default_registry_contains_defaults() -> None:
    expected = tuple(
        validator.metadata().validator_id for validator in DEFAULT_VALIDATORS
    )
    assert VALIDATOR_REGISTRY.validator_ids() == expected
    for validator_id in expected:
        assert VALIDATOR_REGISTRY.has_validator(validator_id)


def test_default_registry_fresh_instance() -> None:
    fresh = default_validator_registry()
    assert fresh is not VALIDATOR_REGISTRY
    assert fresh.validator_ids() == VALIDATOR_REGISTRY.validator_ids()
    fresh.unregister("schema_validator")
    assert VALIDATOR_REGISTRY.has_validator("schema_validator")


def test_shared_registry_convenience_api() -> None:
    custom = _TopicValidator()
    original = has_validator(custom.validator_id)
    register_validator(custom)
    try:
        assert has_validator(custom.validator_id)
        assert list_validators()[-1] is custom
        assert custom in VALIDATOR_REGISTRY.validators_for("product")
        assert (
            resolve_validator("product").metadata().validator_id
            == "schema_validator"
        )
    finally:
        unregister_validator(custom.validator_id)
    assert has_validator(custom.validator_id) is original
