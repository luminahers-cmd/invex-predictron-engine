"""Tests for the validation models — Phase 8 Sprint 4.

Covers the deterministic value objects: severity/status/confidence
enums, validation issues, results, conflicts, fingerprints, metadata,
per-topic breakdowns, the summary aggregation, and lossless JSON
serialization of every model.
"""

from __future__ import annotations

import json

import pytest

from predictron_engine.research.exceptions import (
    FingerprintError,
    InvalidEvidenceError,
)
from predictron_engine.research.models import Evidence, EvidenceReference
from predictron_engine.research.validation_models import (
    FINGERPRINT_ALGORITHM,
    FINGERPRINT_PREFIX,
    VALIDATION_SCHEMA_VERSION,
    EvidenceConfidence,
    EvidenceConflict,
    EvidenceFingerprint,
    EvidenceValidation,
    TopicValidation,
    ValidationCounts,
    ValidationIssue,
    ValidationMetadata,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
    ValidationSummary,
    ValidatorMetadata,
    canonical_json,
    collection_fingerprint,
    compute_fingerprint,
    content_digest,
)


def _reference(
    *, url: str = "https://example.com", description: str = "A ref."
) -> EvidenceReference:
    return EvidenceReference(
        source_category="web_search",
        source_identifier="placeholder",
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


def _issue(severity: ValidationSeverity, code: str = "code") -> ValidationIssue:
    return ValidationIssue(code=code, message="A message.", severity=severity)


def _result(*issues: ValidationIssue) -> ValidationResult:
    return ValidationResult(
        validator_id="schema_validator",
        validator_name="Schema",
        passed=True,
        score=1.0,
        issues=issues,
    )


def _validation(
    evidence: Evidence,
    status: ValidationStatus = ValidationStatus.PASSED,
    score: float = 0.98,
    issues: tuple[ValidationIssue, ...] = (),
) -> EvidenceValidation:
    return EvidenceValidation(
        evidence_id=evidence.evidence_id,
        task_id=evidence.task_id,
        topic_id=evidence.topic_id,
        fingerprint=compute_fingerprint(evidence),
        status=status,
        confidence_score=score,
        confidence=EvidenceConfidence.from_score(score),
        validator_results=(_result(),),
        issues=issues,
        conflicts=(),
        duplicate_count=1,
    )


def _summary(validation: EvidenceValidation) -> ValidationSummary:
    return ValidationSummary(
        schema_version=VALIDATION_SCHEMA_VERSION,
        validation_id="validation_test",
        collection_fingerprint=collection_fingerprint([validation.evidence_id]),
        status=validation.status,
        metadata=ValidationMetadata(
            schema_version=VALIDATION_SCHEMA_VERSION,
            engine_version="1.0.0",
            collection_fingerprint=collection_fingerprint([validation.evidence_id]),
        ),
        validations=(validation,),
        conflicts=(),
        counts=ValidationCounts(
            total=1,
            passed=1,
            warned=0,
            failed=0,
            skipped=0,
            issues=len(validation.issues),
            conflicts=0,
            duplicates=0,
            topics_covered=1,
        ),
        covered_topics=("product",),
        by_topic=(
            TopicValidation(
                topic_id="product",
                evidence_count=1,
                passed=1,
                warned=0,
                failed=0,
                mean_confidence=validation.confidence_score,
            ),
        ),
    )


def test_severity_ranking() -> None:
    ranks = [severity.rank() for severity in ValidationSeverity]
    assert ranks == [0, 1, 2, 3]
    assert list(ValidationSeverity) == [
        ValidationSeverity.INFO,
        ValidationSeverity.WARNING,
        ValidationSeverity.ERROR,
        ValidationSeverity.CRITICAL,
    ]


def test_severity_failure_and_notice() -> None:
    assert ValidationSeverity.ERROR.is_failure()
    assert ValidationSeverity.CRITICAL.is_failure()
    assert not ValidationSeverity.WARNING.is_failure()
    assert not ValidationSeverity.INFO.is_failure()
    assert ValidationSeverity.WARNING.is_notice()
    assert not ValidationSeverity.ERROR.is_notice()


def test_status_values() -> None:
    assert [status.value for status in ValidationStatus] == [
        "passed",
        "warned",
        "failed",
        "skipped",
    ]


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (0.8, EvidenceConfidence.HIGH),
        (0.9, EvidenceConfidence.HIGH),
        (1.0, EvidenceConfidence.HIGH),
        (0.55, EvidenceConfidence.MEDIUM),
        (0.7, EvidenceConfidence.MEDIUM),
        (0.3, EvidenceConfidence.LOW),
        (0.4, EvidenceConfidence.LOW),
        (0.29, EvidenceConfidence.UNKNOWN),
        (0.0, EvidenceConfidence.UNKNOWN),
    ],
)
def test_confidence_band_boundaries(score: float, band: EvidenceConfidence) -> None:
    assert EvidenceConfidence.from_score(score) is band


def test_result_status_derivation() -> None:
    assert _result().status is ValidationStatus.PASSED
    assert _result(_issue(ValidationSeverity.INFO)).status is ValidationStatus.PASSED
    assert _result(_issue(ValidationSeverity.WARNING)).status is ValidationStatus.WARNED
    assert _result(_issue(ValidationSeverity.ERROR)).status is ValidationStatus.FAILED
    assert (
        _result(_issue(ValidationSeverity.CRITICAL)).status is ValidationStatus.FAILED
    )


def test_issue_validates_code_and_message() -> None:
    with pytest.raises(InvalidEvidenceError):
        ValidationIssue(code=" ", message="msg", severity=ValidationSeverity.WARNING)
    with pytest.raises(InvalidEvidenceError):
        ValidationIssue(code="code", message="", severity=ValidationSeverity.WARNING)


def test_issue_failure_flag() -> None:
    assert _issue(ValidationSeverity.ERROR).is_failure()
    assert not _issue(ValidationSeverity.WARNING).is_failure()


def test_result_rejects_out_of_range_score() -> None:
    with pytest.raises(InvalidEvidenceError):
        ValidationResult(validator_id="v", validator_name="V", passed=True, score=1.1)
    with pytest.raises(InvalidEvidenceError):
        ValidationResult(validator_id="v", validator_name="V", passed=False, score=-0.1)


def test_issue_roundtrip() -> None:
    issue = _issue(ValidationSeverity.WARNING, code="missing_reference")
    assert ValidationIssue.from_dict(issue.to_dict()) == issue
    assert json.loads(json.dumps(issue.to_dict())) == issue.to_dict()


def test_result_roundtrip() -> None:
    result = _result(_issue(ValidationSeverity.WARNING))
    assert ValidationResult.from_dict(result.to_dict()) == result


def test_paired_conflict_requires_distinct_evidence() -> None:
    with pytest.raises(InvalidEvidenceError):
        EvidenceConflict(
            conflict_type="conflicting_values", evidence_a="ev_a", evidence_b="ev_a"
        )


def test_conflict_pairwise_flag_and_roundtrip() -> None:
    paired = EvidenceConflict(
        conflict_type="conflicting_values",
        evidence_a="ev_a",
        evidence_b="ev_b",
        description="Conflicting claims.",
    )
    assert paired.is_pairwise
    assert ValidationSeverity.WARNING is paired.severity
    assert EvidenceConflict.from_dict(paired.to_dict()) == paired
    solo = EvidenceConflict(conflict_type="duplicate_evidence", evidence_a="ev_a")
    assert not solo.is_pairwise


def test_fingerprint_is_deterministic_and_well_formed() -> None:
    fp = compute_fingerprint(_evidence())
    assert fp.algorithm == FINGERPRINT_ALGORITHM
    assert len(fp.value) == 64
    assert fp.short.startswith(FINGERPRINT_PREFIX)
    assert fp.canonical
    assert compute_fingerprint(_evidence()) == fp


def test_fingerprint_changes_with_content() -> None:
    original = compute_fingerprint(_evidence())
    changed = compute_fingerprint(_evidence(claim="A completely different claim."))
    assert changed != original


def test_fingerprint_rejects_non_evidence() -> None:
    with pytest.raises(FingerprintError):
        compute_fingerprint({"not": "evidence"})


def test_fingerprint_roundtrip() -> None:
    fp = compute_fingerprint(_evidence())
    assert EvidenceFingerprint.from_dict(fp.to_dict()) == fp


def test_validator_metadata_strips_and_generic() -> None:
    meta = ValidatorMetadata(validator_id="  schema_validator ", display_name=" Schema ")
    assert meta.validator_id == "schema_validator"
    assert meta.display_name == "Schema"
    assert meta.is_generic
    specific = ValidatorMetadata(
        validator_id="v", display_name="V", supported_topics=("product",)
    )
    assert not specific.is_generic
    assert ValidatorMetadata.from_dict(meta.to_dict()) == meta


def test_validation_roundtrip_and_accessors() -> None:
    evidence = _evidence()
    validation = _validation(
        evidence,
        issues=(_issue(ValidationSeverity.INFO, code="info_code"),),
    )
    decoded = EvidenceValidation.from_dict(validation.to_dict())
    assert decoded == validation
    assert validation.passed()
    assert not validation.has_failed()
    assert not validation.has_warnings()
    assert validation.issue_codes() == ("info_code",)
    assert validation.result_for("schema_validator") is not None
    assert validation.result_for("missing") is None


def test_summary_roundtrip() -> None:
    validation = _validation(_evidence())
    summary = _summary(validation)
    rebuilt = ValidationSummary.from_dict(summary.to_dict())
    assert rebuilt == summary
    assert json.loads(json.dumps(summary.to_dict())) == summary.to_dict()


def test_summary_lookup_helpers() -> None:
    validation = _validation(_evidence())
    summary = _summary(validation)
    evidence_id = validation.evidence_id
    assert summary.validation_for(evidence_id) == validation
    assert summary.validation_for("missing") is None
    assert summary.confidence_for(evidence_id) is validation.confidence
    assert summary.confidence_for("missing") is None
    assert not summary.has_failures()
    assert summary.topic_for("product") is not None
    assert summary.topic_for("missing") is None
    assert summary.issues() == validation.issues


def test_summary_issue_severity_tally() -> None:
    validation = _validation(
        _evidence(),
        issues=(
            _issue(ValidationSeverity.WARNING, code="a"),
            _issue(ValidationSeverity.WARNING, code="b"),
            _issue(ValidationSeverity.ERROR, code="c"),
        ),
    )
    summary = _summary(validation)
    assert summary.issue_count_by_severity() == {"warning": 2, "error": 1}


def test_counts_and_topic_roundtrip() -> None:
    counts = ValidationCounts(
        total=3,
        passed=1,
        warned=1,
        failed=1,
        skipped=0,
        issues=4,
        conflicts=2,
        duplicates=1,
        topics_covered=2,
    )
    assert ValidationCounts.from_dict(counts.to_dict()) == counts
    topic = TopicValidation(
        topic_id="product",
        evidence_count=2,
        passed=1,
        warned=1,
        failed=0,
        mean_confidence=0.9,
    )
    assert TopicValidation.from_dict(topic.to_dict()) == topic


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == canonical_json({"a": [2, 3], "b": 1})
    assert canonical_json({"a": 1}) == '{"a":1}'


def test_collection_fingerprint_is_stable_and_sensitive() -> None:
    assert collection_fingerprint(["ev_1", "ev_2"]) == collection_fingerprint(
        ["ev_1", "ev_2"]
    )
    assert collection_fingerprint(["ev_1", "ev_2"]).startswith("collection_")
    assert collection_fingerprint(["ev_1", "ev_2"]) != collection_fingerprint(
        ["ev_2", "ev_1"]
    )


def test_content_digest_ignores_insertion_order() -> None:
    assert content_digest({"x": 1, "y": 2}) == content_digest({"y": 2, "x": 1})
    assert content_digest({"x": 1, "y": 2}) != content_digest({"x": 2, "y": 1})
