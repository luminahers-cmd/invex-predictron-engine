"""Report validator — validates the final assembled Report."""

from __future__ import annotations

from predictron_engine.validation.validators.pipeline_validator import (
    ValidationFinding,
)


class ReportValidator:
    """Validates the final Report object for structural integrity.

    Checks that all Report fields are internally consistent,
    that cross-references between components are valid, and that
    metadata is populated. Never modifies the report.
    """

    def validate(self, report: object) -> list[ValidationFinding]:
        """Run all report validation checks."""
        findings: list[ValidationFinding] = []
        findings.extend(self._validate_metadata(report))
        findings.extend(self._validate_scores(report))
        findings.extend(self._validate_recommendations(report))
        findings.extend(self._validate_confidence(report))
        findings.extend(self._validate_cross_references(report))
        return findings

    def _validate_metadata(
        self, report: object
    ) -> list[ValidationFinding]:
        """Validate analysis metadata."""
        findings: list[ValidationFinding] = []
        meta = getattr(report, "analysis_metadata", None)
        if meta is None:
            findings.append(
                ValidationFinding(
                    validator="report_validator",
                    severity="error",
                    category="missing_metadata",
                    message="Report is missing analysis_metadata",
                    artifact_type="Report",
                )
            )
            return findings

        version = getattr(meta, "engine_version", "")
        if not version or version == "0.1.0":
            findings.append(
                ValidationFinding(
                    validator="report_validator",
                    severity="warning",
                    category="metadata",
                    message=(
                        f"Engine version '{version}' may be "
                        f"outdated or unset"
                    ),
                    artifact_type="AnalysisMetadata",
                )
            )

        stages = getattr(meta, "pipeline_stages_completed", [])
        if len(stages) < 8:
            findings.append(
                ValidationFinding(
                    validator="report_validator",
                    severity="info",
                    category="metadata",
                    message=(
                        f"Only {len(stages)} pipeline stages "
                        f"reported as completed"
                    ),
                    artifact_type="AnalysisMetadata",
                )
            )
        return findings

    def _validate_scores(
        self, report: object
    ) -> list[ValidationFinding]:
        """Validate score consistency."""
        findings: list[ValidationFinding] = []
        scores = getattr(report, "scores", [])
        overall = getattr(report, "overall_score", 0.0)
        if scores and overall > 0:
            avg = sum(s.score for s in scores) / len(scores)
            if abs(avg - overall) > 1.0:
                findings.append(
                    ValidationFinding(
                        validator="report_validator",
                        severity="info",
                        category="consistency",
                        message=(
                            f"Overall score ({overall:.1f}) differs "
                            f"from average of individual scores "
                            f"({avg:.1f})"
                        ),
                        artifact_type="Report",
                    )
                )
        return findings

    def _validate_recommendations(
        self, report: object
    ) -> list[ValidationFinding]:
        """Validate recommendation structure."""
        findings: list[ValidationFinding] = []
        recs = getattr(report, "recommendations", [])
        for i, rec in enumerate(recs):
            action = getattr(rec, "action", "")
            if not action:
                findings.append(
                    ValidationFinding(
                        validator="report_validator",
                        severity="info",
                        category="missing_field",
                        message=f"Recommendation {i} has empty action",
                        artifact_type="Recommendation",
                    )
                )
            priority = getattr(rec, "priority", "")
            if priority not in ("high", "medium", "low", ""):
                findings.append(
                    ValidationFinding(
                        validator="report_validator",
                        severity="info",
                        category="invalid_value",
                        message=(
                            f"Recommendation {i} has unexpected "
                            f"priority '{priority}'"
                        ),
                        artifact_type="Recommendation",
                    )
                )
        return findings

    def _validate_confidence(
        self, report: object
    ) -> list[ValidationFinding]:
        """Validate confidence assessments."""
        findings: list[ValidationFinding] = []
        conf = getattr(report, "confidence", [])
        overall = getattr(report, "overall_confidence", 0.0)
        if conf and overall > 0:
            avg = sum(c.confidence for c in conf) / len(conf)
            if abs(avg - overall) > 0.1:
                findings.append(
                    ValidationFinding(
                        validator="report_validator",
                        severity="info",
                        category="consistency",
                        message=(
                            f"Overall confidence ({overall:.3f}) "
                            f"differs from average of individual "
                            f"confidences ({avg:.3f})"
                        ),
                        artifact_type="Report",
                    )
                )
        return findings

    def _validate_cross_references(
        self, report: object
    ) -> list[ValidationFinding]:
        """Validate cross-references between report components."""
        findings: list[ValidationFinding] = []
        assessments = getattr(report, "dimension_assessments", [])
        obs = getattr(report, "observations", [])

        assess_dims = {a.dimension for a in assessments}
        obs_dims = {o.dimension for o in obs}

        for dim in obs_dims:
            if dim not in assess_dims:
                findings.append(
                    ValidationFinding(
                        validator="report_validator",
                        severity="info",
                        category="cross_reference",
                        message=(
                            f"Observations exist for dimension "
                            f"'{dim}' but no corresponding "
                            f"assessment in report"
                        ),
                        artifact_type="Report",
                    )
                )
        return findings
