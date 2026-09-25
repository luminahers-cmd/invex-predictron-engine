"""Tests for the default evidence validators — Phase 8 Sprint 4.

Each test exercises a single validator against hand-built evidence and a
deterministic :class:`ValidationContext`, asserting exact scores, issue
codes, severities, and statuses.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.models import Evidence, EvidenceReference
from predictron_engine.research.validation_models import (
    EvidenceConflict,
    ValidationSeverity,
    ValidationStatus,
)
from predictron_engine.research.validation_rules import (
    ConfidenceThresholdRule,
    CrossEvidenceConsistencyRule,
    DuplicateDetectionRule,
    MaxEvidenceAgeRule,
    MinimumCompletenessRule,
    RequiredFieldsRule,
    ValidationContext,
    ValidationRuleSet,
)
from predictron_engine.research.validators import (
    DEFAULT_VALIDATORS,
    VALIDATOR_CATALOG,
    CompletenessValidator,
    ConfidenceValidator,
    ConflictValidator,
    ConsistencyValidator,
    DuplicateEvidenceValidator,
    FreshnessValidator,
    ReferenceValidator,
    SchemaValidator,
    SourceMetadataValidator,
    TopicCoverageValidator,
)

_REFERENCE_TIME = "2026-09-24T00:00:00Z"


def _reference(
    *,
    url: str = "https://example.com",
    source_category: str = "web_search",
    source_identifier: str = "placeholder",
    description: str = "A ref.",
) -> EvidenceReference:
    return EvidenceReference(
        source_category=source_category,
        source_identifier=source_identifier,
        url=url,
        description=description,
    )


def _evidence(**overrides: object) -> Evidence:
    values: dict[str, object] = {
        "task_id": "research_product",
        "topic_id": "product",
        "collector_id": "product_collector",
        "category": "product_claims",
        "claim": "Acme automates invoicing for small businesses end to end.",
        "confidence": 0.8,
        "reference": _reference(),
    }
    values.update(overrides)
    return Evidence(**values)


def _timestamp_context(
    evidence: Evidence, timestamp: str, *, reference_time: str = _REFERENCE_TIME
) -> ValidationContext:
    return ValidationContext(
        timestamp_by_evidence={evidence.evidence_id: timestamp},
        reference_time=reference_time,
    )


# ----------------------------------------------------------------------
# SchemaValidator
# ----------------------------------------------------------------------


def test_schema_passes_valid_evidence() -> None:
    result = SchemaValidator().validate(_evidence())
    assert result.passed
    assert result.score == 1.0
    assert result.status is ValidationStatus.PASSED
    assert result.issues == ()


def test_schema_uses_required_field_rules() -> None:
    rules = ValidationRuleSet((RequiredFieldsRule(required_fields=("claim",)),))
    result = SchemaValidator().validate(_evidence(), context=ValidationContext(rules=rules))
    assert result.passed
    assert result.score == 1.0


# ----------------------------------------------------------------------
# FreshnessValidator
# ----------------------------------------------------------------------


def test_freshness_missing_timestamp_is_info() -> None:
    result = FreshnessValidator().validate(_evidence())
    assert result.issues[0].code == "missing_evidence_timestamp"
    assert result.issues[0].severity is ValidationSeverity.INFO
    assert result.passed
    assert result.score == 1.0


def test_freshness_within_window() -> None:
    evidence = _evidence()
    result = FreshnessValidator().validate(
        evidence, context=_timestamp_context(evidence, "2026-09-01T00:00:00Z")
    )
    assert result.passed
    assert result.score == 1.0
    assert result.issues == ()


def test_freshness_past_window_warns() -> None:
    evidence = _evidence()
    result = FreshnessValidator().validate(
        evidence, context=_timestamp_context(evidence, "2026-01-01T00:00:00Z")
    )
    assert result.issues[0].code == "exceeds_freshness_window"
    assert result.issues[0].severity is ValidationSeverity.WARNING
    assert result.passed
    assert 0.0 < result.score < 1.0


def test_freshness_over_maximum_age_fails() -> None:
    evidence = _evidence()
    result = FreshnessValidator().validate(
        evidence, context=_timestamp_context(evidence, "2024-09-01T00:00:00Z")
    )
    assert result.issues[0].code == "exceeds_maximum_age"
    assert result.issues[0].severity is ValidationSeverity.ERROR
    assert not result.passed


def test_freshness_invalid_timestamp_fails() -> None:
    evidence = _evidence()
    result = FreshnessValidator().validate(
        evidence, context=_timestamp_context(evidence, "not-a-date")
    )
    assert result.issues[0].code == "invalid_timestamp"
    assert result.issues[0].severity is ValidationSeverity.ERROR
    assert not result.passed
    assert result.score == 0.0


def test_freshness_disabled_always_passes() -> None:
    evidence = _evidence()
    rules = ValidationRuleSet((MaxEvidenceAgeRule(enabled=False),))
    result = FreshnessValidator().validate(evidence, context=ValidationContext(rules=rules))
    assert result.passed
    assert result.score == 1.0
    assert result.issues == ()


# ----------------------------------------------------------------------
# CompletenessValidator
# ----------------------------------------------------------------------


def test_completeness_valid() -> None:
    result = CompletenessValidator().validate(_evidence())
    assert result.score == 1.0
    assert result.issues == ()
    assert result.status is ValidationStatus.PASSED


def test_completeness_below_threshold_warns() -> None:
    rules = ValidationRuleSet((MinimumCompletenessRule(min_score=0.9),))
    evidence = _evidence(
        claim="Short.",
        confidence=0.1,
        reference=_reference(url="not a url"),
    )
    result = CompletenessValidator().validate(
        evidence, context=ValidationContext(rules=rules)
    )
    assert result.issues[0].code == "incomplete_evidence"
    assert result.issues[0].severity is ValidationSeverity.WARNING
    assert 0.0 < result.score < 1.0


# ----------------------------------------------------------------------
# ConsistencyValidator
# ----------------------------------------------------------------------


def test_consistency_valid() -> None:
    result = ConsistencyValidator().validate(_evidence())
    assert result.score == 1.0
    assert result.issues == ()


def test_consistency_rejects_malformed_url() -> None:
    evidence = _evidence(reference=_reference(url="not a url"))
    result = ConsistencyValidator().validate(evidence)
    assert result.issues[0].code == "invalid_reference_url"
    assert result.issues[0].severity is ValidationSeverity.ERROR
    assert not result.passed


def test_consistency_flags_non_preferred_category() -> None:
    evidence = _evidence(reference=_reference(source_category="linkedin"))
    result = ConsistencyValidator().validate(evidence)
    assert any(
        issue.code == "non_preferred_source_category" for issue in result.issues
    )
    assert any(
        issue.severity is ValidationSeverity.WARNING for issue in result.issues
    )


# ----------------------------------------------------------------------
# ReferenceValidator
# ----------------------------------------------------------------------


def test_reference_valid() -> None:
    result = ReferenceValidator().validate(_evidence())
    assert result.score == 1.0
    assert result.issues == ()


def test_reference_missing_url_warns() -> None:
    evidence = _evidence(reference=_reference(url=""))
    result = ReferenceValidator().validate(evidence)
    assert result.issues[0].code == "missing_reference_field"
    assert result.issues[0].severity is ValidationSeverity.WARNING
    assert result.score == pytest.approx(0.75)


def test_reference_invalid_url_fails() -> None:
    evidence = _evidence(reference=_reference(url="ftp://example.com"))
    result = ReferenceValidator().validate(evidence)
    assert result.issues[0].code == "reference_integrity"
    assert result.issues[0].severity is ValidationSeverity.ERROR
    assert not result.passed


# ----------------------------------------------------------------------
# DuplicateEvidenceValidator
# ----------------------------------------------------------------------


def test_duplicate_single_copy_passes() -> None:
    result = DuplicateEvidenceValidator().validate(_evidence())
    assert result.passed
    assert result.score == 1.0
    assert result.issues == ()


def test_duplicate_warns() -> None:
    evidence = _evidence()
    context = ValidationContext(
        duplicate_count_by_evidence={evidence.evidence_id: 3}
    )
    result = DuplicateEvidenceValidator().validate(evidence, context=context)
    assert not result.passed
    assert result.score == pytest.approx(1 / 3)
    assert result.issues[0].code == "duplicate_evidence"
    assert result.issues[0].severity is ValidationSeverity.WARNING


def test_duplicate_treat_as_error() -> None:
    evidence = _evidence()
    rules = ValidationRuleSet((DuplicateDetectionRule(treat_as_error=True),))
    context = ValidationContext(
        duplicate_count_by_evidence={evidence.evidence_id: 2}, rules=rules
    )
    result = DuplicateEvidenceValidator().validate(evidence, context=context)
    assert result.issues[0].severity is ValidationSeverity.ERROR


# ----------------------------------------------------------------------
# ConflictValidator
# ----------------------------------------------------------------------


def test_conflict_reports_relevant_conflicts() -> None:
    evidence = _evidence()
    conflict = EvidenceConflict(
        conflict_type="conflicting_values",
        evidence_a=evidence.evidence_id,
        evidence_b="ev_other",
        description="Conflicting claims.",
    )
    context = ValidationContext(
        conflicts_by_evidence={evidence.evidence_id: (conflict,)}
    )
    result = ConflictValidator().validate(evidence, context=context)
    assert not result.passed
    assert result.score == 0.5
    assert result.issues[0].code == "evidence_conflict"


def test_conflict_ignores_duplicates() -> None:
    evidence = _evidence()
    conflict = EvidenceConflict(
        conflict_type="duplicate_evidence", evidence_a=evidence.evidence_id
    )
    context = ValidationContext(
        conflicts_by_evidence={evidence.evidence_id: (conflict,)}
    )
    result = ConflictValidator().validate(evidence, context=context)
    assert result.passed
    assert result.score == 1.0
    assert result.issues == ()


# ----------------------------------------------------------------------
# ConfidenceValidator
# ----------------------------------------------------------------------


def test_confidence_passes() -> None:
    result = ConfidenceValidator().validate(_evidence())
    assert result.passed
    assert result.score == pytest.approx(0.8)
    assert result.issues == ()


def test_confidence_zero_is_error() -> None:
    result = ConfidenceValidator().validate(_evidence(confidence=0.0))
    assert result.issues[0].code == "zero_confidence_evidence"
    assert result.issues[0].severity is ValidationSeverity.ERROR
    assert not result.passed
    assert result.score == 0.0


def test_confidence_below_stated_threshold_warns() -> None:
    rules = ValidationRuleSet((ConfidenceThresholdRule(min_stated_score=0.5),))
    result = ConfidenceValidator().validate(
        _evidence(confidence=0.2), context=ValidationContext(rules=rules)
    )
    assert result.issues[0].code == "stated_confidence_below_threshold"
    assert result.issues[0].severity is ValidationSeverity.WARNING
    assert result.passed


# ----------------------------------------------------------------------
# SourceMetadataValidator
# ----------------------------------------------------------------------


def test_source_metadata_valid() -> None:
    result = SourceMetadataValidator().validate(_evidence())
    assert result.score == 1.0
    assert result.issues == ()


def test_source_metadata_flags_issues() -> None:
    evidence = _evidence(
        collector_id="ProductCollector",
        reference=_reference(source_category="unknown_category"),
    )
    result = SourceMetadataValidator().validate(evidence)
    codes = {issue.code for issue in result.issues}
    assert "non_snake_case_collector_id" in codes
    assert "unknown_source_category" in codes


# ----------------------------------------------------------------------
# TopicCoverageValidator
# ----------------------------------------------------------------------


def test_topic_coverage_valid() -> None:
    context = ValidationContext(
        topics_in_collection=("product",),
        evidence_count_by_topic={"product": 1},
    )
    result = TopicCoverageValidator().validate(_evidence(), context=context)
    assert result.passed
    assert result.score == 1.0
    assert result.issues == ()


def test_topic_coverage_unknown_topic_fails() -> None:
    context = ValidationContext(
        topics_in_collection=("product",),
        evidence_count_by_topic={"product": 1},
    )
    result = TopicCoverageValidator().validate(
        _evidence(topic_id="not_a_topic"), context=context
    )
    assert any(issue.code == "unknown_topic" for issue in result.issues)
    assert result.issues[0].severity is ValidationSeverity.ERROR


def test_topic_coverage_under_covered_warns() -> None:
    rules = ValidationRuleSet(
        (CrossEvidenceConsistencyRule(min_evidence_per_topic=2),)
    )
    context = ValidationContext(
        rules=rules,
        topics_in_collection=("product",),
        evidence_count_by_topic={"product": 1},
    )
    result = TopicCoverageValidator().validate(_evidence(), context=context)
    assert any(issue.code == "topic_under_covered" for issue in result.issues)


# ----------------------------------------------------------------------
# Validator contracts and determinism
# ----------------------------------------------------------------------


def test_validator_metadata_and_support() -> None:
    assert len(DEFAULT_VALIDATORS) == 10
    assert len(VALIDATOR_CATALOG) == 10
    for validator in DEFAULT_VALIDATORS:
        metadata = validator.metadata()
        assert metadata.validator_id.strip()
        assert metadata.display_name.strip()
        assert metadata.deterministic
        assert validator.supports("any_topic")


def test_validator_results_are_deterministic() -> None:
    evidence = _evidence()
    first = [validator.validate(evidence) for validator in DEFAULT_VALIDATORS]
    second = [validator.validate(evidence) for validator in DEFAULT_VALIDATORS]
    assert first == second
