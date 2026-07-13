"""Pipeline validator — verifies all expected outputs exist."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ValidationFinding(BaseModel):
    """A single validation finding."""

    validator: str = Field(
        ..., description="Name of the validator that produced this finding"
    )
    severity: str = Field(
        ...,
        description="Severity level: 'error', 'warning', or 'info'",
    )
    category: str = Field(
        ..., description="Finding category (e.g. 'missing_data', 'consistency')"
    )
    message: str = Field(..., description="Human-readable description")
    artifact_type: str = Field(
        default="",
        description="Type of artifact this finding relates to",
    )
    artifact_id: str = Field(
        default="",
        description="ID of the specific artifact, if applicable",
    )


class PipelineValidator:
    """Validates that all pipeline stages produced expected outputs.

    Checks for empty outputs, missing required fields, and structural
    integrity of pipeline results. Never modifies outputs — only reports
    findings.
    """

    def validate(
        self,
        startup: object,
        features: object,
        evidence: list[object],
        observations: list[object],
        assessments: list[object],
        scores: list[object],
        recommendations: list[object],
        confidence: list[object],
    ) -> list[ValidationFinding]:
        """Run all pipeline validation checks."""
        findings: list[ValidationFinding] = []
        findings.extend(self._validate_startup(startup))
        findings.extend(self._validate_features(features))
        findings.extend(self._validate_evidence(evidence))
        findings.extend(self._validate_observations(observations))
        findings.extend(self._validate_assessments(assessments))
        findings.extend(self._validate_scores(scores))
        findings.extend(self._validate_recommendations(recommendations))
        findings.extend(self._validate_confidence(confidence))
        return findings

    def _validate_startup(self, startup: object) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        name = getattr(startup, "name", None)
        if not name:
            findings.append(
                ValidationFinding(
                    validator="pipeline_validator",
                    severity="error",
                    category="missing_data",
                    message="Startup name is empty or missing",
                    artifact_type="Startup",
                )
            )
        website = getattr(startup, "website", None)
        if not website:
            findings.append(
                ValidationFinding(
                    validator="pipeline_validator",
                    severity="warning",
                    category="missing_data",
                    message="Startup website is missing",
                    artifact_type="Startup",
                )
            )
        description = getattr(startup, "description", None)
        if not description or len(description) < 10:
            findings.append(
                ValidationFinding(
                    validator="pipeline_validator",
                    severity="warning",
                    category="missing_data",
                    message="Startup description is very short or missing",
                    artifact_type="Startup",
                )
            )
        return findings

    def _validate_features(self, features: object) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        completeness = getattr(features, "data_completeness", 0.0)
        if completeness < 0.1:
            findings.append(
                ValidationFinding(
                    validator="pipeline_validator",
                    severity="warning",
                    category="missing_data",
                    message=(
                        f"Data completeness is very low "
                        f"({completeness:.0%}). Most features "
                        f"could not be extracted."
                    ),
                    artifact_type="ExtractedFeatures",
                )
            )
        industry = getattr(features, "industry", None)
        if industry is None:
            findings.append(
                ValidationFinding(
                    validator="pipeline_validator",
                    severity="info",
                    category="missing_data",
                    message="Industry could not be determined from input",
                    artifact_type="ExtractedFeatures",
                )
            )
        return findings

    def _validate_evidence(
        self, evidence: list[object]
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if not evidence:
            findings.append(
                ValidationFinding(
                    validator="pipeline_validator",
                    severity="warning",
                    category="missing_data",
                    message="No evidence items were gathered",
                    artifact_type="EvidenceSet",
                )
            )
        return findings

    def _validate_observations(
        self, observations: list[object]
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if not observations:
            findings.append(
                ValidationFinding(
                    validator="pipeline_validator",
                    severity="warning",
                    category="missing_data",
                    message="No observations were generated",
                    artifact_type="list[Observation]",
                )
            )
        for i, obs in enumerate(observations):
            confidence = getattr(obs, "confidence", 0.0)
            if confidence < 0.2:
                findings.append(
                    ValidationFinding(
                        validator="pipeline_validator",
                        severity="info",
                        category="low_confidence",
                        message=(
                            f"Observation {i} has very low "
                            f"confidence ({confidence:.2f})"
                        ),
                        artifact_type="Observation",
                    )
                )
        return findings

    def _validate_assessments(
        self, assessments: list[object]
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if not assessments:
            findings.append(
                ValidationFinding(
                    validator="pipeline_validator",
                    severity="warning",
                    category="missing_data",
                    message="No dimension assessments were produced",
                    artifact_type="EvaluationResult",
                )
            )
        return findings

    def _validate_scores(
        self, scores: list[object]
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if not scores:
            findings.append(
                ValidationFinding(
                    validator="pipeline_validator",
                    severity="warning",
                    category="missing_data",
                    message="No dimension scores were produced",
                    artifact_type="list[ScoreResult]",
                )
            )
        return findings

    def _validate_recommendations(
        self, recommendations: list[object]
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if not recommendations:
            findings.append(
                ValidationFinding(
                    validator="pipeline_validator",
                    severity="info",
                    category="missing_data",
                    message="No recommendations were generated",
                    artifact_type="list[Recommendation]",
                )
            )
        return findings

    def _validate_confidence(
        self, confidence: list[object]
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if not confidence:
            findings.append(
                ValidationFinding(
                    validator="pipeline_validator",
                    severity="warning",
                    category="missing_data",
                    message="No confidence assessments were produced",
                    artifact_type="list[ConfidenceAssessment]",
                )
            )
        for c in confidence:
            conf = getattr(c, "confidence", 0.0)
            if conf < 0.2:
                dim = getattr(c, "dimension", "unknown")
                findings.append(
                    ValidationFinding(
                        validator="pipeline_validator",
                        severity="warning",
                        category="low_confidence",
                        message=(
                            f"Very low confidence ({conf:.2f}) "
                            f"for dimension '{dim}'"
                        ),
                        artifact_type="ConfidenceAssessment",
                    )
                )
        return findings
