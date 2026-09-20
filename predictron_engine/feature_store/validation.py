"""Feature validation.

Validates computed feature sets for missing dependencies, invalid values,
staleness, version mismatches, and computation integrity. Generates
deterministic validation reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    FeatureStatus,
)
from predictron_engine.feature_store.registry import FeatureRegistry


class Severity(str, Enum):
    """Validation issue severity."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    feature_id: str
    severity: Severity
    category: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "feature_id": self.feature_id,
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
        }


@dataclass
class FeatureValidationReport:
    """Deterministic validation report for a feature set."""

    company_id: str
    validated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    total_features: int = 0
    computed_features: int = 0
    failed_features: int = 0
    missing_dependency_features: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(
            i.severity == Severity.ERROR for i in self.issues
        )

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.INFO)

    @property
    def coverage(self) -> float:
        if self.total_features == 0:
            return 0.0
        return round(self.computed_features / self.total_features, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            "validated_at": self.validated_at.isoformat(),
            "total_features": self.total_features,
            "computed_features": self.computed_features,
            "failed_features": self.failed_features,
            "missing_dependency_features": self.missing_dependency_features,
            "coverage": self.coverage,
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "issues": [i.to_dict() for i in self.issues],
        }


class FeatureValidator:
    """Validates computed feature sets."""

    def __init__(
        self,
        registry: FeatureRegistry,
        *,
        max_age_days: int = 365,
        computation_version: str = "1.0.0",
    ) -> None:
        self._registry = registry
        self._max_age_days = max_age_days
        self._computation_version = computation_version

    def validate(
        self, feature_set: CompanyFeatureSet
    ) -> FeatureValidationReport:
        """Run all validations and return a report."""
        report = FeatureValidationReport(
            company_id=feature_set.company_id,
            total_features=self._registry.count(),
            computed_features=len(feature_set.computed_features()),
            failed_features=len(feature_set.failed_features()),
            missing_dependency_features=sum(
                1 for f in feature_set.features.values()
                if f.status == FeatureStatus.MISSING_DEPENDENCY
            ),
        )

        self._check_missing_features(feature_set, report)
        self._check_failed_features(feature_set, report)
        self._check_value_ranges(feature_set, report)
        self._check_staleness(feature_set, report)
        self._check_version_mismatches(feature_set, report)
        self._check_dependency_integrity(feature_set, report)
        self._check_evidence_references(feature_set, report)

        report.issues.sort(key=lambda i: (i.severity.value, i.feature_id))
        return report

    def validate_all(
        self, feature_sets: list[CompanyFeatureSet]
    ) -> list[FeatureValidationReport]:
        """Validate multiple feature sets."""
        return [self.validate(fs) for fs in feature_sets]

    def _check_missing_features(
        self, fs: CompanyFeatureSet, report: FeatureValidationReport
    ) -> None:
        registered = set(self._registry.list_ids())
        computed = set(fs.features.keys())
        for fid in sorted(registered - computed):
            report.issues.append(
                ValidationIssue(
                    feature_id=fid,
                    severity=Severity.WARNING,
                    category="missing_feature",
                    message=f"Feature '{fid}' not computed",
                )
            )

    def _check_failed_features(
        self, fs: CompanyFeatureSet, report: FeatureValidationReport
    ) -> None:
        for snap in fs.features.values():
            if snap.status == FeatureStatus.COMPUTATION_ERROR:
                report.issues.append(
                    ValidationIssue(
                        feature_id=snap.feature_id,
                        severity=Severity.ERROR,
                        category="computation_error",
                        message=snap.error_message or "Unknown computation error",
                    )
                )
            elif snap.status == FeatureStatus.MISSING_DEPENDENCY:
                report.issues.append(
                    ValidationIssue(
                        feature_id=snap.feature_id,
                        severity=Severity.WARNING,
                        category="missing_dependency",
                        message=snap.error_message or "Missing dependency",
                    )
                )

    def _check_value_ranges(
        self, fs: CompanyFeatureSet, report: FeatureValidationReport
    ) -> None:
        for snap in fs.features.values():
            if snap.status != FeatureStatus.COMPUTED:
                continue
            defn = self._registry.get(snap.feature_id)
            if defn is None:
                continue
            if snap.value is None:
                continue
            if defn.min_value is not None and isinstance(snap.value, int | float):
                if snap.value < defn.min_value:
                    report.issues.append(
                        ValidationIssue(
                            feature_id=snap.feature_id,
                            severity=Severity.WARNING,
                            category="value_range",
                            message=(
                                f"Value {snap.value} below minimum {defn.min_value}"
                            ),
                        )
                    )
            if defn.max_value is not None and isinstance(snap.value, int | float):
                if snap.value > defn.max_value:
                    report.issues.append(
                        ValidationIssue(
                            feature_id=snap.feature_id,
                            severity=Severity.WARNING,
                            category="value_range",
                            message=(
                                f"Value {snap.value} above maximum {defn.max_value}"
                            ),
                        )
                    )

    def _check_staleness(
        self, fs: CompanyFeatureSet, report: FeatureValidationReport
    ) -> None:
        cutoff = datetime.now(UTC) - timedelta(days=self._max_age_days)
        for snap in fs.features.values():
            if snap.computed_at < cutoff:
                report.issues.append(
                    ValidationIssue(
                        feature_id=snap.feature_id,
                        severity=Severity.INFO,
                        category="stale_feature",
                        message=(
                            f"Computed at {snap.computed_at.isoformat()}, "
                            f"older than {self._max_age_days} days"
                        ),
                    )
                )

    def _check_version_mismatches(
        self, fs: CompanyFeatureSet, report: FeatureValidationReport
    ) -> None:
        for snap in fs.features.values():
            if snap.status != FeatureStatus.COMPUTED:
                continue
            expected = self._computation_version
            defn = self._registry.get(snap.feature_id)
            if defn is not None and defn.computation_version == expected:
                defn_expected = expected
            else:
                defn_expected = expected
            if snap.computation_version != defn_expected:
                report.issues.append(
                    ValidationIssue(
                        feature_id=snap.feature_id,
                        severity=Severity.WARNING,
                        category="version_mismatch",
                        message=(
                            f"Snap version '{snap.computation_version}' "
                            f"!= expected version '{defn_expected}'"
                        ),
                    )
                )

    def _check_dependency_integrity(
        self, fs: CompanyFeatureSet, report: FeatureValidationReport
    ) -> None:
        for snap in fs.features.values():
            if snap.status != FeatureStatus.COMPUTED:
                continue
            defn = self._registry.get(snap.feature_id)
            if defn is None:
                continue
            for dep_id in defn.dependencies:
                dep_snap = fs.features.get(dep_id)
                if dep_snap is None:
                    report.issues.append(
                        ValidationIssue(
                            feature_id=snap.feature_id,
                            severity=Severity.WARNING,
                            category="dependency_integrity",
                            message=f"Dependency '{dep_id}' missing from feature set",
                        )
                    )
                elif dep_snap.status != FeatureStatus.COMPUTED:
                    report.issues.append(
                        ValidationIssue(
                            feature_id=snap.feature_id,
                            severity=Severity.WARNING,
                            category="dependency_integrity",
                            message=(
                                f"Dependency '{dep_id}' has status "
                                f"'{dep_snap.status.value}'"
                            ),
                        )
                    )

    def _check_evidence_references(
        self, fs: CompanyFeatureSet, report: FeatureValidationReport
    ) -> None:
        for snap in fs.features.values():
            if snap.status != FeatureStatus.COMPUTED:
                continue
            if snap.value is not None and not snap.evidence_references:
                report.issues.append(
                    ValidationIssue(
                        feature_id=snap.feature_id,
                        severity=Severity.INFO,
                        category="missing_evidence",
                        message="Computed feature has no evidence references",
                    )
                )
