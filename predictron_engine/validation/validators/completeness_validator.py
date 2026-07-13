"""Completeness validator — checks for missing data and unused artifacts."""

from __future__ import annotations

from predictron_engine.validation.validators.pipeline_validator import (
    ValidationFinding,
)


class CompletenessValidator:
    """Validates data completeness across the pipeline.

    Checks for unused evidence, unused observations, missing inputs,
    and coverage gaps. Never modifies outputs.
    """

    def validate(
        self,
        features: object,
        evidence: list[object],
        observations: list[object],
        assessments: list[object],
        scores: list[object],
        recommendations: list[object],
    ) -> list[ValidationFinding]:
        """Run all completeness checks."""
        findings: list[ValidationFinding] = []
        findings.extend(self._check_unused_evidence(evidence, observations))
        findings.extend(
            self._check_unused_observations(observations, assessments, scores)
        )
        findings.extend(
            self._check_dimension_coverage(observations, assessments, scores)
        )
        findings.extend(self._check_feature_coverage(features))
        return findings

    def _check_unused_evidence(
        self,
        evidence: list[object],
        observations: list[object],
    ) -> list[ValidationFinding]:
        """Check for evidence items not referenced by any observation."""
        findings: list[ValidationFinding] = []
        if not evidence:
            return findings

        referenced = set()
        for obs in observations:
            for ref in getattr(obs, "evidence", []):
                referenced.add(ref)

        for i, item in enumerate(evidence):
            stmt = getattr(item, "statement", "")
            domain = getattr(item, "domain", "")
            is_used = any(
                stmt in ref or domain in ref for ref in referenced
            )
            if not is_used:
                findings.append(
                    ValidationFinding(
                        validator="completeness_validator",
                        severity="info",
                        category="unused_evidence",
                        message=(
                            f"Evidence item {i} (domain='{domain}') "
                            f"was not referenced by any observation"
                        ),
                        artifact_type="EvidenceItem",
                    )
                )
        return findings

    def _check_unused_observations(
        self,
        observations: list[object],
        assessments: list[object],
        scores: list[object],
    ) -> list[ValidationFinding]:
        """Check for observations not used by any assessment or score."""
        findings: list[ValidationFinding] = []
        if not observations:
            return findings

        assessed_dims = {
            getattr(a, "dimension", "") for a in assessments
        }
        scored_dims = {getattr(s, "dimension", "") for s in scores}
        used_dims = assessed_dims | scored_dims

        for i, obs in enumerate(observations):
            dim = getattr(obs, "dimension", "")
            if dim not in used_dims:
                findings.append(
                    ValidationFinding(
                        validator="completeness_validator",
                        severity="info",
                        category="unused_observations",
                        message=(
                            f"Observation {i} (dimension='{dim}') "
                            f"was not used by any assessment or score"
                        ),
                        artifact_type="Observation",
                    )
                )
        return findings

    def _check_dimension_coverage(
        self,
        observations: list[object],
        assessments: list[object],
        scores: list[object],
    ) -> list[ValidationFinding]:
        """Check that all dimensions with observations have assessments."""
        findings: list[ValidationFinding] = []
        obs_dims = {getattr(o, "dimension", "") for o in observations}
        assessed_dims = {getattr(a, "dimension", "") for a in assessments}
        scored_dims = {getattr(s, "dimension", "") for s in scores}

        uncovered = obs_dims - assessed_dims
        for dim in uncovered:
            if dim:
                findings.append(
                    ValidationFinding(
                        validator="completeness_validator",
                        severity="info",
                        category="coverage_gap",
                        message=(
                            f"Dimension '{dim}' has observations "
                            f"but no assessment"
                        ),
                        artifact_type="DimensionAssessment",
                    )
                )

        unscored = obs_dims - scored_dims
        for dim in unscored:
            if dim:
                findings.append(
                    ValidationFinding(
                        validator="completeness_validator",
                        severity="info",
                        category="coverage_gap",
                        message=(
                            f"Dimension '{dim}' has observations "
                            f"but no score"
                        ),
                        artifact_type="ScoreResult",
                    )
                )
        return findings

    def _check_feature_coverage(
        self, features: object
    ) -> list[ValidationFinding]:
        """Check for missing critical features."""
        findings: list[ValidationFinding] = []
        critical_fields = {
            "industry": "Industry classification",
            "business_model": "Business model",
            "funding_stage": "Funding stage",
        }
        for field, label in critical_fields.items():
            value = getattr(features, field, None)
            if value is None:
                findings.append(
                    ValidationFinding(
                        validator="completeness_validator",
                        severity="info",
                        category="missing_input",
                        message=f"{label} was not determined",
                        artifact_type="ExtractedFeatures",
                    )
                )
        completeness = getattr(features, "data_completeness", 0.0)
        if completeness < 0.3:
            findings.append(
                ValidationFinding(
                    validator="completeness_validator",
                    severity="warning",
                    category="low_coverage",
                    message=(
                        f"Overall data completeness is low "
                        f"({completeness:.0%}). Analysis may "
                        f"have significant gaps."
                    ),
                    artifact_type="ExtractedFeatures",
                )
            )
        return findings
