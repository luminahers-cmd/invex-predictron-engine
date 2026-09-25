"""Tests for the Evidence Validation Engine — Phase 8 Sprint 4.

Covers the public ``validate`` API, deterministic summaries, confidence
scoring, conflict and duplicate detection, freshness failures, custom
registries and rules, history blending, minimum-confidence bands,
per-topic breakdowns, aggregate counts, and strict determinism.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.exceptions import InvalidEvidenceError
from predictron_engine.research.models import (
    Evidence,
    EvidenceCollection,
    EvidenceReference,
)
from predictron_engine.research.validation import (
    CONFLICT_PENALTY,
    DUPLICATE_PENALTY,
    EvidenceValidationEngine,
    compute_confidence_score,
)
from predictron_engine.research.validation_models import (
    EvidenceConfidence,
    ValidationSeverity,
    ValidationStatus,
    compute_fingerprint,
)
from predictron_engine.research.validation_rules import (
    ConfidenceThresholdRule,
    DuplicateDetectionRule,
    RequiredFieldsRule,
)
from predictron_engine.research.validator_registry import (
    ValidatorRegistry,
    default_validator_registry,
)
from predictron_engine.research.validators import SchemaValidator


def _reference(*, url: str = "https://example.com") -> EvidenceReference:
    return EvidenceReference(
        source_category="web_search",
        source_identifier="placeholder",
        url=url,
        description="A ref.",
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


def _collection(*items: Evidence) -> EvidenceCollection:
    return EvidenceCollection(
        evidence=tuple(items), topic_order=("product", "market", "founders")
    )


def _engine(
    registry: ValidatorRegistry | None = None, **kwargs: object
) -> EvidenceValidationEngine:
    if registry is None:
        registry = default_validator_registry()
    kwargs.setdefault("registry", registry)
    return EvidenceValidationEngine(**kwargs)


def test_validate_rejects_non_collection() -> None:
    with pytest.raises(InvalidEvidenceError):
        _engine().validate(None)  # type: ignore[arg-type]


def test_empty_collection_is_skipped() -> None:
    summary = _engine().validate(EvidenceCollection())
    assert summary.status is ValidationStatus.SKIPPED
    assert summary.counts.total == 0
    assert summary.counts.skipped == 1
    assert summary.validations == ()


def test_valid_collection_passes() -> None:
    summary = _engine().validate(_collection(_evidence()))
    assert summary.status is ValidationStatus.PASSED
    assert summary.counts.total == 1
    assert summary.counts.passed == 1
    assert summary.counts.failed == 0
    validation = summary.validations[0]
    assert validation.status is ValidationStatus.PASSED
    assert len(validation.validator_results) == 10
    assert validation.confidence_score == pytest.approx(0.98)
    assert validation.confidence is EvidenceConfidence.HIGH
    assert validation.duplicate_count == 1
    assert len(summary.metadata.validator_ids) == 10
    assert summary.metadata.deterministic is True
    assert summary.metadata.schema_version


def test_validation_is_deterministic() -> None:
    items = (
        _evidence(),
        _evidence(
            topic_id="market",
            task_id="research_market",
            category="market_conditions",
            claim="The invoicing market is growing quickly.",
        ),
        _evidence(
            topic_id="founders",
            task_id="research_founders",
            category="founder_background",
            claim="The founders previously built a logistics startup.",
        ),
    )
    collection = _collection(*items)
    first = _engine().validate(collection)
    second = _engine().validate(collection)
    third = _engine().validate(collection)
    assert first == second == third
    assert first.validation_id == second.validation_id
    assert first.collection_fingerprint == second.collection_fingerprint


def test_fingerprint_single() -> None:
    evidence = _evidence()
    assert _engine().fingerprint(evidence) == compute_fingerprint(evidence)


def test_missing_timestamp_is_info_only() -> None:
    summary = _engine().validate(_collection(_evidence()))
    validation = summary.validations[0]
    assert validation.status is ValidationStatus.PASSED
    codes = {issue.code for issue in validation.issues}
    assert codes == {"missing_evidence_timestamp"}
    assert {issue.severity for issue in validation.issues} == {
        ValidationSeverity.INFO
    }


def test_conflicting_values_detected() -> None:
    first = _evidence()
    second = _evidence(claim="The platform does not support invoicing at all.")
    summary = _engine().validate(_collection(first, second))
    conflicts = [
        conflict
        for conflict in summary.conflicts
        if conflict.conflict_type == "conflicting_values"
    ]
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert {conflict.evidence_a, conflict.evidence_b} == {
        first.evidence_id,
        second.evidence_id,
    }
    assert conflict.severity is ValidationSeverity.WARNING
    assert summary.counts.conflicts == 1
    for validation in summary.validations:
        assert validation.status is ValidationStatus.WARNED
        assert any(issue.code == "evidence_conflict" for issue in validation.issues)
        assert validation.confidence_score == pytest.approx(
            0.98 * CONFLICT_PENALTY
        )


def test_duplicate_detection() -> None:
    item = _evidence()
    summary = _engine().validate(_collection(item, item))
    assert summary.counts.duplicates == 1
    dup_conflicts = [
        conflict
        for conflict in summary.conflicts
        if conflict.conflict_type == "duplicate_evidence"
    ]
    assert len(dup_conflicts) == 1
    assert dup_conflicts[0].severity is ValidationSeverity.WARNING
    for validation in summary.validations:
        assert validation.duplicate_count == 2
        assert validation.confidence_score == pytest.approx(
            0.98 * DUPLICATE_PENALTY
        )
        assert any(
            issue.code == "duplicate_evidence" for issue in validation.issues
        )
        assert validation.status is ValidationStatus.WARNED


def test_duplicate_treat_as_error_fails_summary() -> None:
    item = _evidence()
    rules = (DuplicateDetectionRule(treat_as_error=True),)
    summary = _engine().validate(_collection(item, item), rules=rules)
    assert summary.status is ValidationStatus.FAILED
    dup = [
        conflict
        for conflict in summary.conflicts
        if conflict.conflict_type == "duplicate_evidence"
    ][0]
    assert dup.severity is ValidationSeverity.ERROR


def test_freshness_failure_fails_summary() -> None:
    evidence = _evidence()
    summary = _engine().validate(
        _collection(evidence),
        evidence_timestamps={evidence.evidence_id: "2024-09-01T00:00:00Z"},
        reference_time="2026-09-24T00:00:00Z",
    )
    validation = summary.validation_for(evidence.evidence_id)
    assert validation is not None
    assert validation.status is ValidationStatus.FAILED
    assert any(
        issue.code == "exceeds_maximum_age" for issue in validation.issues
    )
    assert summary.status is ValidationStatus.FAILED
    assert summary.counts.failed == 1


def test_previous_validations_blend_history() -> None:
    evidence = _evidence()
    base = (
        _engine().validate(_collection(evidence)).validations[0].confidence_score
    )
    with_history = _engine().validate(
        _collection(evidence), previous_validations={evidence.evidence_id: 1.0}
    ).validations[0].confidence_score
    assert with_history == pytest.approx(0.9 * base + 0.1 * 1.0)
    assert with_history > base


def test_custom_registry_controls_validators() -> None:
    registry = ValidatorRegistry((SchemaValidator(),))
    summary = _engine(registry=registry).validate(_collection(_evidence()))
    assert summary.metadata.validator_ids == ("schema_validator",)
    validation = summary.validations[0]
    assert len(validation.validator_results) == 1
    assert validation.result_for("schema_validator") is not None


def test_custom_required_field_rules() -> None:
    rules = (RequiredFieldsRule(required_fields=("claim",)),)
    summary = _engine().validate(_collection(_evidence()), rules=rules)
    schema = summary.validations[0].result_for("schema_validator")
    assert schema is not None
    assert schema.passed
    assert schema.score == 1.0


def test_minimum_band_issue_appended() -> None:
    evidence = _evidence(
        claim="Short.",
        confidence=0.2,
        reference=_reference(url="not a url"),
    )
    rules = (ConfidenceThresholdRule(min_band=EvidenceConfidence.HIGH),)
    summary = _engine().validate(_collection(evidence), rules=rules)
    validation = summary.validations[0]
    assert validation.confidence is EvidenceConfidence.MEDIUM
    assert any(
        issue.code == "confidence_below_minimum" for issue in validation.issues
    )


def test_validation_id_is_stable_and_sensitive() -> None:
    collection = _collection(_evidence())
    first = _engine().validate(collection)
    second = _engine().validate(collection)
    assert first.validation_id == second.validation_id
    assert first.validation_id.startswith("validation_")
    other = _engine().validate(
        _collection(_evidence(claim="A noticeably different claim text."))
    )
    assert first.validation_id != other.validation_id


def test_by_topic_breakdown() -> None:
    product = _evidence()
    product_two = _evidence(
        category="product_pricing",
        claim="The platform now offers transparent per-license pricing.",
    )
    market = _evidence(
        topic_id="market",
        task_id="research_market",
        category="market_conditions",
        claim="The invoicing market is growing quickly.",
    )
    summary = _engine().validate(_collection(product, product_two, market))
    product_topic = summary.topic_for("product")
    market_topic = summary.topic_for("market")
    assert product_topic is not None
    assert market_topic is not None
    assert product_topic.evidence_count == 2
    product_scores = [
        validation.confidence_score
        for validation in summary.validations
        if validation.topic_id == "product"
    ]
    assert product_topic.mean_confidence == pytest.approx(
        sum(product_scores) / 2
    )
    assert market_topic.evidence_count == 1
    assert market_topic.mean_confidence == pytest.approx(
        summary.topic_for("market").mean_confidence  # type: ignore[union-attr]
    )


def test_issue_severity_tally() -> None:
    summary = _engine().validate(_collection(_evidence()))
    assert summary.issue_count_by_severity() == {"info": 1}


def test_counts_match_validation_state() -> None:
    first = _evidence()
    second = _evidence(
        category="market_conditions",
        claim="The TAM here is huge and growing fast.",
    )
    summary = _engine().validate(_collection(first, second))
    assert summary.counts.total == len(summary.validations) == 2
    assert summary.counts.issues == sum(
        len(validation.issues) for validation in summary.validations
    )
    assert summary.counts.topics_covered == len(summary.covered_topics)
    assert (
        summary.counts.passed
        + summary.counts.warned
        + summary.counts.failed
        == summary.counts.total
    )
    assert summary.counts.duplicates == 0
    assert summary.issues() == tuple(
        issue
        for validation in summary.validations
        for issue in validation.issues
    )


def test_compute_confidence_score_math() -> None:
    evidence = _evidence()
    factors = {
        "completeness": 1.0,
        "metadata": 1.0,
        "reference": 1.0,
        "freshness": 1.0,
        "consistency": 1.0,
        "stated": 0.8,
    }
    assert compute_confidence_score(evidence, factors) == pytest.approx(0.98)
    tripled = compute_confidence_score(evidence, factors, duplicate_count=3)
    assert tripled == pytest.approx(0.98 * DUPLICATE_PENALTY**2)
    conflicted = compute_confidence_score(evidence, factors, conflict_count=2)
    assert conflicted == pytest.approx(0.98 * CONFLICT_PENALTY**2)
    blended = compute_confidence_score(evidence, factors, previous_score=1.0)
    assert blended == pytest.approx(0.9 * 0.98 + 0.1)


def test_compute_confidence_score_fallbacks_and_clamping() -> None:
    evidence = _evidence()
    assert compute_confidence_score(evidence, {}) == pytest.approx(
        evidence.confidence
    )
    assert compute_confidence_score(
        evidence, {"no_such_factor": 0.0}
    ) == pytest.approx(evidence.confidence)
    assert compute_confidence_score(evidence, {"completeness": 5.0}) == pytest.approx(
        1.0
    )
    assert compute_confidence_score(
        evidence, {"completeness": 0.5, "metadata": 0.5}
    ) == pytest.approx(0.5)
    assert compute_confidence_score(
        evidence, {"completeness": -1.0}
    ) == pytest.approx(0.0)
