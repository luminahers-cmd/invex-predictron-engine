"""Benchmark validator — compares actual engine output against expected outcomes.

Provides structured validation of benchmark case results against the
expected outcome specifications defined in each case. Produces detailed
validation findings that feed into benchmark reports.

Usage:
    from benchmarks.benchmark_validator import BenchmarkValidator
    validator = BenchmarkValidator()
    findings = validator.validate_case(case, result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from benchmarks.benchmark_runner import CaseResult
from benchmarks.startup_cases.cases import ExpectedOutcomes

logger = logging.getLogger(__name__)


class FindingSeverity(Enum):
    """Severity level for a benchmark validation finding."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    INFO = "info"


@dataclass
class ValidationFinding:
    """A single finding from benchmark case validation."""

    field_name: str
    severity: FindingSeverity
    expected: Any
    actual: Any
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_name,
            "severity": self.severity.value,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        }


@dataclass
class CaseValidationResult:
    """Complete validation result for a single benchmark case."""

    case_id: str
    case_label: str
    passed: bool
    findings: list[ValidationFinding] = field(default_factory=list)
    pass_count: int = 0
    warn_count: int = 0
    fail_count: int = 0
    info_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_label": self.case_label,
            "passed": self.passed,
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "info_count": self.info_count,
            "findings": [f.to_dict() for f in self.findings],
        }


class BenchmarkValidator:
    """Validates benchmark case results against expected outcomes.

    Compares actual engine output with the expected outcome specification
    embedded in each benchmark case. Produces detailed validation findings
    indicating passes, warnings, and failures.
    """

    def validate_case(
        self,
        case: dict[str, Any],
        result: CaseResult,
    ) -> CaseValidationResult:
        """Validate a single benchmark case result against its expectations.

        Args:
            case: The benchmark case definition (from BENCHMARK_CASES).
            result: The CaseResult produced by the benchmark runner.

        Returns:
            CaseValidationResult with all findings.
        """
        findings: list[ValidationFinding] = []

        if not result.success:
            findings.append(ValidationFinding(
                field_name="execution",
                severity=FindingSeverity.FAIL,
                expected=True,
                actual=False,
                message=f"Benchmark case failed: {result.error}",
            ))
            return CaseValidationResult(
                case_id=result.case_id,
                case_label=result.case_label,
                passed=False,
                findings=findings,
                fail_count=1,
            )

        outcomes = case.get("expected_outcomes")
        if outcomes is None:
            findings.append(ValidationFinding(
                field_name="expected_outcomes",
                severity=FindingSeverity.INFO,
                expected="ExpectedOutcomes instance",
                actual=None,
                message="No expected outcomes defined for this case",
            ))
            return CaseValidationResult(
                case_id=result.case_id,
                case_label=result.case_label,
                passed=True,
                findings=findings,
                info_count=1,
            )

        assert result.report is not None
        report = result.report

        self._check_feature_match(report, outcomes, findings)
        self._check_score_range(report, outcomes, findings)
        self._check_confidence_range(report, outcomes, findings)
        self._check_observation_count(report, outcomes, findings)
        self._check_recommendation_count(report, outcomes, findings)
        self._check_evidence_count(report, outcomes, findings)
        self._check_score_dimensions(report, outcomes, findings)
        self._check_recommendation_categories(report, outcomes, findings)

        pass_count = sum(1 for f in findings if f.severity == FindingSeverity.PASS)
        warn_count = sum(1 for f in findings if f.severity == FindingSeverity.WARN)
        fail_count = sum(1 for f in findings if f.severity == FindingSeverity.FAIL)
        info_count = sum(1 for f in findings if f.severity == FindingSeverity.INFO)

        return CaseValidationResult(
            case_id=result.case_id,
            case_label=result.case_label,
            passed=fail_count == 0,
            findings=findings,
            pass_count=pass_count,
            warn_count=warn_count,
            fail_count=fail_count,
            info_count=info_count,
        )

    def validate_all(
        self,
        cases: list[dict[str, Any]],
        results: list[CaseResult],
    ) -> list[CaseValidationResult]:
        """Validate all benchmark case results.

        Args:
            cases: List of benchmark case definitions.
            results: List of CaseResult objects from the benchmark runner.

        Returns:
            List of CaseValidationResult objects.
        """
        result_map = {r.case_id: r for r in results}
        validations: list[CaseValidationResult] = []

        for case in cases:
            case_id = case["id"]
            result = result_map.get(case_id)
            if result is None:
                validations.append(CaseValidationResult(
                    case_id=case_id,
                    case_label=case.get("label", ""),
                    passed=False,
                    findings=[ValidationFinding(
                        field_name="execution",
                        severity=FindingSeverity.FAIL,
                        expected="CaseResult",
                        actual=None,
                        message=f"No result found for case {case_id}",
                    )],
                    fail_count=1,
                ))
                continue

            validations.append(self.validate_case(case, result))

        return validations

    def _check_feature_match(
        self,
        report: Any,
        outcomes: ExpectedOutcomes,
        findings: list[ValidationFinding],
    ) -> None:
        """Check extracted features match expectations."""
        features = report.features

        if outcomes.expected_industry is not None:
            actual = features.industry
            passed = actual == outcomes.expected_industry
            findings.append(ValidationFinding(
                field_name="industry",
                severity=FindingSeverity.PASS if passed else FindingSeverity.FAIL,
                expected=outcomes.expected_industry,
                actual=actual,
                message=(
                    "" if passed
                    else f"Industry mismatch: expected "
                    f"{outcomes.expected_industry}, got {actual}"
                ),
            ))

        if outcomes.expected_business_model is not None:
            actual = features.business_model
            passed = actual == outcomes.expected_business_model
            findings.append(ValidationFinding(
                field_name="business_model",
                severity=FindingSeverity.PASS if passed else FindingSeverity.WARN,
                expected=outcomes.expected_business_model,
                actual=actual,
                message=(
                    "" if passed
                    else f"Business model mismatch: expected "
                    f"{outcomes.expected_business_model}, got {actual}"
                ),
            ))

        if outcomes.expected_customer_type is not None:
            actual = features.customer_type
            passed = actual == outcomes.expected_customer_type
            findings.append(ValidationFinding(
                field_name="customer_type",
                severity=FindingSeverity.PASS if passed else FindingSeverity.WARN,
                expected=outcomes.expected_customer_type,
                actual=actual,
                message=(
                    "" if passed
                    else f"Customer type mismatch: expected "
                    f"{outcomes.expected_customer_type}, got {actual}"
                ),
            ))

        if outcomes.expected_has_revenue is not None:
            actual = features.has_revenue
            passed = actual == outcomes.expected_has_revenue
            findings.append(ValidationFinding(
                field_name="has_revenue",
                severity=FindingSeverity.PASS if passed else FindingSeverity.WARN,
                expected=outcomes.expected_has_revenue,
                actual=actual,
                message=(
                    "" if passed
                    else f"Revenue signal mismatch: expected "
                    f"{outcomes.expected_has_revenue}, got {actual}"
                ),
            ))

    def _check_score_range(
        self,
        report: Any,
        outcomes: ExpectedOutcomes,
        findings: list[ValidationFinding],
    ) -> None:
        """Check overall score falls within expected range."""
        score = report.overall_score
        in_range = outcomes.expected_min_score <= score <= outcomes.expected_max_score
        findings.append(ValidationFinding(
            field_name="overall_score",
            severity=FindingSeverity.PASS if in_range else FindingSeverity.FAIL,
            expected=f"[{outcomes.expected_min_score}, {outcomes.expected_max_score}]",
            actual=round(score, 2),
            message=(
                "" if in_range
                else f"Score {score:.1f} outside expected range "
                f"[{outcomes.expected_min_score}, {outcomes.expected_max_score}]"
            ),
        ))

    def _check_confidence_range(
        self,
        report: Any,
        outcomes: ExpectedOutcomes,
        findings: list[ValidationFinding],
    ) -> None:
        """Check overall confidence falls within expected range."""
        conf = report.overall_confidence
        in_range = outcomes.expected_min_confidence <= conf <= outcomes.expected_max_confidence
        findings.append(ValidationFinding(
            field_name="overall_confidence",
            severity=FindingSeverity.PASS if in_range else FindingSeverity.WARN,
            expected=f"[{outcomes.expected_min_confidence}, {outcomes.expected_max_confidence}]",
            actual=round(conf, 4),
            message=(
                "" if in_range
                else f"Confidence {conf:.4f} outside expected range "
                f"[{outcomes.expected_min_confidence}, {outcomes.expected_max_confidence}]"
            ),
        ))

    def _check_observation_count(
        self,
        report: Any,
        outcomes: ExpectedOutcomes,
        findings: list[ValidationFinding],
    ) -> None:
        """Check minimum observation count."""
        count = len(report.observations)
        passed = count >= outcomes.expected_min_observations
        findings.append(ValidationFinding(
            field_name="observation_count",
            severity=FindingSeverity.PASS if passed else FindingSeverity.WARN,
            expected=f">= {outcomes.expected_min_observations}",
            actual=count,
            message=(
                "" if passed
                else f"Only {count} observations, expected >= {outcomes.expected_min_observations}"
            ),
        ))

    def _check_recommendation_count(
        self,
        report: Any,
        outcomes: ExpectedOutcomes,
        findings: list[ValidationFinding],
    ) -> None:
        """Check minimum recommendation count."""
        count = len(report.recommendations)
        passed = count >= outcomes.expected_min_recommendations
        findings.append(ValidationFinding(
            field_name="recommendation_count",
            severity=FindingSeverity.PASS if passed else FindingSeverity.WARN,
            expected=f">= {outcomes.expected_min_recommendations}",
            actual=count,
            message=(
                "" if passed
                else f"Only {count} recommendations, "
                f"expected >= {outcomes.expected_min_recommendations}"
            ),
        ))

    def _check_evidence_count(
        self,
        report: Any,
        outcomes: ExpectedOutcomes,
        findings: list[ValidationFinding],
    ) -> None:
        """Check minimum evidence count."""
        count = len(report.evidence)
        passed = count >= outcomes.expected_min_evidence
        findings.append(ValidationFinding(
            field_name="evidence_count",
            severity=FindingSeverity.PASS if passed else FindingSeverity.WARN,
            expected=f">= {outcomes.expected_min_evidence}",
            actual=count,
            message=(
                "" if passed
                else f"Only {count} evidence items, expected >= {outcomes.expected_min_evidence}"
            ),
        ))

    def _check_score_dimensions(
        self,
        report: Any,
        outcomes: ExpectedOutcomes,
        findings: list[ValidationFinding],
    ) -> None:
        """Check that expected score dimensions are present."""
        if not outcomes.expected_score_dimensions:
            return

        actual_dims = {s.dimension for s in report.scores}
        missing = set(outcomes.expected_score_dimensions) - actual_dims

        findings.append(ValidationFinding(
            field_name="score_dimensions",
            severity=FindingSeverity.PASS if not missing else FindingSeverity.WARN,
            expected=sorted(outcomes.expected_score_dimensions),
            actual=sorted(actual_dims),
            message=(
                "" if not missing
                else f"Missing score dimensions: {sorted(missing)}"
            ),
        ))

    def _check_recommendation_categories(
        self,
        report: Any,
        outcomes: ExpectedOutcomes,
        findings: list[ValidationFinding],
    ) -> None:
        """Check that expected recommendation categories are present."""
        if not outcomes.expected_recommendation_categories:
            return

        actual_cats = {rec.category for rec in report.recommendations}
        missing = set(outcomes.expected_recommendation_categories) - actual_cats

        findings.append(ValidationFinding(
            field_name="recommendation_categories",
            severity=FindingSeverity.PASS if not missing else FindingSeverity.INFO,
            expected=sorted(outcomes.expected_recommendation_categories),
            actual=sorted(actual_cats),
            message=(
                "" if not missing
                else f"Missing recommendation categories: {sorted(missing)}"
            ),
        ))
