"""Tests for feature validation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    FeatureCategory,
    FeatureSnapshot,
    FeatureStatus,
    ValueType,
)
from predictron_engine.feature_store.validation import (
    FeatureValidationReport,
    FeatureValidator,
    Severity,
    ValidationIssue,
)


def make_snap(
    fid: str,
    value=None,
    status: FeatureStatus = FeatureStatus.COMPUTED,
    computed_at: datetime | None = None,
    category: FeatureCategory = FeatureCategory.COMPANY,
    computation_version: str = "1.0.0",
    error_message: str | None = None,
    evidence_references: list | None = None,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        company_id="c1",
        feature_id=fid,
        feature_name=fid,
        category=category,
        value=value,
        value_type=ValueType.FLOAT if isinstance(value, float) else ValueType.NONE,
        status=status,
        computation_version=computation_version,
        computed_at=computed_at or datetime.now(UTC),
        error_message=error_message,
        evidence_references=evidence_references or [],
    )


class TestValidationIssue:
    def test_to_dict(self):
        issue = ValidationIssue(
            feature_id="f1", severity=Severity.ERROR,
            category="computation_error", message="boom",
        )
        data = issue.to_dict()
        assert data["feature_id"] == "f1"
        assert data["severity"] == "error"
        assert data["category"] == "computation_error"


class TestValidationReport:
    def test_empty_report_valid(self):
        report = FeatureValidationReport(company_id="c1")
        assert report.is_valid
        assert report.error_count == 0
        assert report.warning_count == 0
        assert report.info_count == 0

    def test_error_makes_invalid(self):
        report = FeatureValidationReport(company_id="c1")
        report.issues.append(ValidationIssue("f1", Severity.ERROR, "x", "y"))
        assert not report.is_valid
        assert report.error_count == 1

    def test_warning_does_not_invalidate(self):
        report = FeatureValidationReport(company_id="c1")
        report.issues.append(ValidationIssue("f1", Severity.WARNING, "x", "y"))
        assert report.is_valid
        assert report.warning_count == 1

    def test_coverage(self):
        report = FeatureValidationReport(
            company_id="c1", total_features=10, computed_features=5,
        )
        assert report.coverage == 0.5

    def test_coverage_zero(self):
        report = FeatureValidationReport(company_id="c1")
        assert report.coverage == 0.0

    def test_to_dict(self):
        report = FeatureValidationReport(company_id="c1", total_features=5)
        data = report.to_dict()
        assert data["company_id"] == "c1"
        assert "coverage" in data
        assert "is_valid" in data


class TestFeatureValidator:
    def test_valid_feature_set(self, registry, fake_record, engine):
        fs = engine.build_company_features(fake_record)
        validator = FeatureValidator(registry)
        report = validator.validate(fs)
        assert report.is_valid

    def test_missing_feature_warning(self, registry, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        # Remove one feature to trigger missing warning
        _ = fs.features.pop("company_age")
        validator = FeatureValidator(registry)
        report = validator.validate(fs)
        assert any(
            i.category == "missing_feature" and i.severity == Severity.WARNING
            for i in report.issues
        )
        # Missing feature is only a warning, so still valid
        assert report.is_valid

    def test_computation_error_detected(self, registry):
        fs = CompanyFeatureSet(
            company_id="c1",
            features={
                "company_age": make_snap(
                    "company_age", status=FeatureStatus.COMPUTATION_ERROR,
                    error_message="boom",
                ),
            },
        )
        validator = FeatureValidator(registry)
        report = validator.validate(fs)
        assert not report.is_valid
        assert any(
            i.category == "computation_error" for i in report.issues
        )

    def test_missing_dependency_detected(self, registry):
        fs = CompanyFeatureSet(
            company_id="c1",
            features={
                "company_age": make_snap(
                    "company_age", status=FeatureStatus.MISSING_DEPENDENCY,
                ),
            },
        )
        validator = FeatureValidator(registry)
        report = validator.validate(fs)
        assert any(
            i.category == "missing_dependency" for i in report.issues
        )
        assert report.missing_dependency_features == 1

    def test_invalid_value_range(self, registry, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        old = fs.features["company_age"]
        fs.features["company_age"] = make_snap(
            "company_age", value=-100.0,
            computed_at=old.computed_at,
        )
        validator = FeatureValidator(registry)
        report = validator.validate(fs)
        assert any(
            i.category == "value_range" for i in report.issues
        )

    def test_stale_feature_detected(self, registry):
        old_time = datetime.now(UTC) - timedelta(days=700)
        fs = CompanyFeatureSet(
            company_id="c1",
            features={
                "company_age": make_snap("company_age", value=5.0,
                                         computed_at=old_time),
            },
        )
        validator = FeatureValidator(registry)
        report = validator.validate(fs)
        assert any(
            i.category == "stale_feature" for i in report.issues
        )

    def test_fresh_feature_not_stale(self, registry):
        fs = CompanyFeatureSet(
            company_id="c1",
            features={
                "company_age": make_snap("company_age", value=5.0),
            },
        )
        validator = FeatureValidator(registry)
        report = validator.validate(fs)
        assert not any(
            i.category == "stale_feature" for i in report.issues
        )

    def test_version_mismatch_detected(self, registry):
        fs = CompanyFeatureSet(
            company_id="c1",
            features={
                "company_age": make_snap(
                    "company_age", value=5.0,
                    computation_version="0.9.0",
                ),
            },
        )
        validator = FeatureValidator(registry)
        report = validator.validate(fs)
        assert any(
            i.category == "version_mismatch" for i in report.issues
        )

    def test_dependency_integrity(self):
        from predictron_engine.feature_store.models import FeatureDefinition
        from predictron_engine.feature_store.registry import FeatureRegistry
        reg = FeatureRegistry()
        reg.register(
            FeatureDefinition(
                feature_id="a", feature_name="A",
                category=FeatureCategory.COMPANY,
                description="d", value_type=ValueType.FLOAT,
                dependencies=["b"],
            ),
            lambda record, deps, ctx: (1.0, []),
        )
        reg.register(
            FeatureDefinition(
                feature_id="b", feature_name="B",
                category=FeatureCategory.COMPANY,
                description="d", value_type=ValueType.FLOAT,
            ),
            lambda record, deps, ctx: (2.0, []),
        )
        # a is computed but b is missing from set
        fs = CompanyFeatureSet(
            company_id="c1",
            features={
                "a": make_snap("a", value=1.0),
            },
        )
        validator = FeatureValidator(reg)
        report = validator.validate(fs)
        assert any(
            i.category == "dependency_integrity" for i in report.issues
        )

    def test_dependency_bad_status(self, registry, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        # simulate failed dependency
        fs.features["industry"] = make_snap(
            "industry", status=FeatureStatus.COMPUTATION_ERROR,
        )
        validator = FeatureValidator(registry)
        report = validator.validate(fs)
        assert any(
            i.category == "dependency_integrity" for i in report.issues
        )

    def test_missing_evidence_info(self, registry, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        # Mock evidence away
        fs.features["company_age"].evidence_references = []
        validator = FeatureValidator(registry)
        report = validator.validate(fs)
        assert any(
            i.category == "missing_evidence" for i in report.issues
        )

    def test_validate_all(self, registry, engine, fake_record):
        fs1 = engine.build_company_features(fake_record)
        fs2 = engine.build_company_features(fake_record)
        validator = FeatureValidator(registry)
        reports = validator.validate_all([fs1, fs2])
        assert len(reports) == 2

    def test_issues_sorted(self, registry, fake_record, engine):
        fs = engine.build_company_features(fake_record)
        validator = FeatureValidator(registry)
        report = validator.validate(fs)
        severities = [i.severity.value for i in report.issues]
        assert severities == sorted(severities)

    def test_custom_max_age(self, registry):
        old_time = datetime.now(UTC) - timedelta(days=30)
        fs = CompanyFeatureSet(
            company_id="c1",
            features={
                "company_age": make_snap("company_age", value=5.0,
                                         computed_at=old_time),
            },
        )
        validator = FeatureValidator(registry, max_age_days=10)
        report = validator.validate(fs)
        assert any(
            i.category == "stale_feature" for i in report.issues
        )

    def test_custom_computation_version(self, registry):
        fs = CompanyFeatureSet(
            company_id="c1",
            features={
                "company_age": make_snap("company_age", value=5.0),
            },
        )
        validator = FeatureValidator(registry, computation_version="9.9.9")
        report = validator.validate(fs)
        assert any(
            i.category == "version_mismatch" for i in report.issues
        )

    def test_deterministic_report(self, registry, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        validator = FeatureValidator(registry)
        r1 = validator.validate(fs).to_dict()
        r2 = validator.validate(fs).to_dict()
        r1.pop("validated_at")
        r2.pop("validated_at")
        assert r1 == r2
