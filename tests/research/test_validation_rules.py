"""Tests for the configurable validation rules — Phase 8 Sprint 4.

Covers rule identity and parameters, serialization dispatch, parameter
validation, the :class:`ValidationRuleSet` container, typed parameter
lookups, the deterministic :class:`ValidationContext`, and the
snake_case helper.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import pytest

from predictron_engine.research.exceptions import ValidationRuleError
from predictron_engine.research.validation_models import EvidenceConflict
from predictron_engine.research.validation_rules import (
    DEFAULT_RULE_SET,
    DEFAULT_RULES,
    ConfidenceThresholdRule,
    CrossEvidenceConsistencyRule,
    DuplicateDetectionRule,
    MaxEvidenceAgeRule,
    MetadataQualityRule,
    MinimumCompletenessRule,
    ReferenceIntegrityRule,
    RequiredFieldsRule,
    TopicConsistencyRule,
    ValidationContext,
    ValidationRule,
    ValidationRuleSet,
    is_snake_case,
    rule_from_dict,
)

ALL_RULES: tuple[ValidationRule, ...] = (
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


class _WeirdRule(ValidationRule):
    rule_id = "weird_rule"
    description = "A rule with a wrongly typed parameter."

    def parameters(self) -> Mapping[str, object]:
        return {"flag": "not-a-bool"}


def test_default_rules_unique_and_typed() -> None:
    assert len(DEFAULT_RULES) == len({type(rule) for rule in DEFAULT_RULES}) == 9
    for rule in DEFAULT_RULES:
        assert rule.rule_id.strip()
        assert isinstance(rule.parameters(), Mapping)


def test_every_rule_roundtrips() -> None:
    for rule in ALL_RULES:
        rebuilt = rule_from_dict(rule.to_dict())
        assert type(rebuilt) is type(rule)
        assert rebuilt.parameters() == rule.parameters()


def test_rule_parameters_roundtrip_with_custom_values() -> None:
    rule = RequiredFieldsRule(required_fields=("claim", "reference"))
    rebuilt = rule_from_dict(rule.to_dict())
    assert rebuilt.parameters() == {"required_fields": ["claim", "reference"]}


def test_rule_from_dict_unknown_rule() -> None:
    with pytest.raises(ValidationRuleError):
        rule_from_dict({"rule_id": "no_such_rule", "parameters": {}})


def test_rule_from_dict_missing_parameters() -> None:
    with pytest.raises(ValidationRuleError):
        rule_from_dict({"rule_id": "required_fields"})
    with pytest.raises(ValidationRuleError):
        rule_from_dict({"rule_id": "required_fields", "parameters": "nope"})


def test_rule_requires_declared_id() -> None:
    class _NoIdRule(ValidationRule):
        description = "missing id"

        def parameters(self) -> Mapping[str, object]:
            return {}

    with pytest.raises(ValidationRuleError):
        _NoIdRule()


@pytest.mark.parametrize(
    ("factory",),
    [
        (lambda: MaxEvidenceAgeRule(max_age_days=-1),),
        (lambda: MinimumCompletenessRule(min_score=1.5),),
        (lambda: CrossEvidenceConsistencyRule(min_evidence_per_topic=0),),
        (lambda: ConfidenceThresholdRule(min_stated_score=-0.1),),
    ],
)
def test_rule_parameter_validation(factory) -> None:
    with pytest.raises(ValidationRuleError):
        factory()


def test_confidence_threshold_rejects_wrong_band() -> None:
    with pytest.raises(ValidationRuleError):
        ConfidenceThresholdRule(min_band="high")


def test_rule_set_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationRuleError):
        ValidationRuleSet((RequiredFieldsRule(), RequiredFieldsRule()))


def test_rule_set_rejects_non_rule() -> None:
    with pytest.raises(ValidationRuleError):
        ValidationRuleSet((object(),))


def test_rule_set_lookup_and_contains() -> None:
    rules = ValidationRuleSet(DEFAULT_RULES)
    assert "required_fields" in rules
    assert len(rules) == 9
    assert rules.has_rule("confidence_threshold")
    assert not rules.has_rule("missing")
    assert rules.get("missing") is None
    assert rules.get("required_fields") == RequiredFieldsRule()
    assert rules.rule_ids() == tuple(rule.rule_id for rule in DEFAULT_RULES)


def test_rule_set_get_parameter_defaults() -> None:
    rules = ValidationRuleSet()
    assert rules.get_parameter("missing_rule", "key", "fallback") == "fallback"
    assert rules.get_parameter("required_fields", "no_such_key", 42) == 42


def test_rule_set_typed_accessors() -> None:
    rules = ValidationRuleSet(DEFAULT_RULES)
    assert rules.get_bool_parameter("duplicate_detection", "enabled", False)
    assert rules.get_int_parameter("max_evidence_age", "max_age_days", 0) == 365
    assert (
        rules.get_float_parameter("minimum_completeness", "min_score", 0.0)
        == pytest.approx(0.5)
    )


def test_rule_set_typed_accessor_enforces_types() -> None:
    rules = ValidationRuleSet((_WeirdRule(),))
    with pytest.raises(ValidationRuleError):
        rules.get_bool_parameter("weird_rule", "flag", False)
    with pytest.raises(ValidationRuleError):
        rules.get_int_parameter("weird_rule", "flag", 0)


def test_rule_set_roundtrip() -> None:
    rules = ValidationRuleSet((MinimumCompletenessRule(min_score=0.7),))
    rebuilt = ValidationRuleSet.from_dict(rules.to_dict())  # type: ignore[arg-type]
    assert rebuilt.rule_ids() == ("minimum_completeness",)


def test_default_rule_set_matches_defaults() -> None:
    assert DEFAULT_RULE_SET.rule_ids() == tuple(
        rule.rule_id for rule in DEFAULT_RULES
    )
    assert len(DEFAULT_RULE_SET) == 9


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("valid_name_1", True),
        ("name", True),
        ("_hidden", True),
        ("CamelCase", False),
        ("has space", False),
        ("has-dash", False),
        ("9starts_digit", False),
    ],
)
def test_is_snake_case(value: str, expected: bool) -> None:
    assert is_snake_case(value) is expected


def test_context_defaults() -> None:
    context = ValidationContext()
    assert context.timestamp_for("ev_x") is None
    assert context.duplicate_count_for("ev_x") == 1
    assert context.conflicts_for("ev_x") == ()
    assert context.reference_time is None


def test_context_lookups() -> None:
    conflict = EvidenceConflict(
        conflict_type="conflicting_values", evidence_a="ev_a", evidence_b="ev_b"
    )
    context = ValidationContext(
        timestamp_by_evidence={"ev_a": "2026-01-01T00:00:00Z"},
        duplicate_count_by_evidence={"ev_a": 3},
        conflicts_by_evidence={"ev_a": (conflict,)},
    )
    assert context.timestamp_for("ev_a") == "2026-01-01T00:00:00Z"
    assert context.duplicate_count_for("ev_a") == 3
    assert context.conflicts_for("ev_a") == (conflict,)
    assert context.timestamp_for("missing") is None


def test_context_is_frozen() -> None:
    context = ValidationContext()
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.reference_time = "2026-09-24T00:00:00Z"
