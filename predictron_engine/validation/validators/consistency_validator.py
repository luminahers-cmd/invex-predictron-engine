"""Consistency validator — checks for conflicting observations and consistent assessments."""

from __future__ import annotations

from predictron_engine.validation.validators.pipeline_validator import (
    ValidationFinding,
)


class ConsistencyValidator:
    """Validates consistency across pipeline outputs.

    Checks for conflicting observations, assessment-score alignment,
    and recommendation-assessment consistency. Never modifies outputs.
    """

    def validate(
        self,
        observations: list[object],
        assessments: list[object],
        scores: list[object],
        recommendations: list[object],
    ) -> list[ValidationFinding]:
        """Run all consistency checks."""
        findings: list[ValidationFinding] = []
        findings.extend(
            self._check_observation_conflicts(observations)
        )
        findings.extend(
            self._check_assessment_score_alignment(assessments, scores)
        )
        findings.extend(
            self._check_recommendation_consistency(
                recommendations, assessments
            )
        )
        return findings

    def _check_observation_conflicts(
        self, observations: list[object]
    ) -> list[ValidationFinding]:
        """Check for observations with conflicting signals."""
        findings: list[ValidationFinding] = []
        dim_observations: dict[str, list[object]] = {}
        for obs in observations:
            dim = getattr(obs, "dimension", "")
            dim_observations.setdefault(dim, []).append(obs)

        for dim, obs_list in dim_observations.items():
            if len(obs_list) < 2:
                continue
            confidences = [getattr(o, "confidence", 0.0) for o in obs_list]
            statements = [getattr(o, "statement", "") for o in obs_list]
            high_conf = [
                (s, c)
                for s, c in zip(statements, confidences)
                if c > 0.7
            ]
            low_conf = [
                (s, c)
                for s, c in zip(statements, confidences)
                if c < 0.3
            ]
            if high_conf and low_conf:
                findings.append(
                    ValidationFinding(
                        validator="consistency_validator",
                        severity="info",
                        category="conflicting_observations",
                        message=(
                            f"Dimension '{dim}' has both high-confidence "
                            f"({len(high_conf)}) and low-confidence "
                            f"({len(low_conf)}) observations"
                        ),
                        artifact_type="Observation",
                    )
                )
        return findings

    def _check_assessment_score_alignment(
        self,
        assessments: list[object],
        scores: list[object],
    ) -> list[ValidationFinding]:
        """Check that assessments and scores align on dimensions."""
        findings: list[ValidationFinding] = []
        assess_dims = sorted(
            {getattr(a, "dimension", "") for a in assessments}
        )
        score_dims = sorted(
            {getattr(s, "dimension", "") for s in scores}
        )

        unscored = sorted(set(assess_dims) - set(score_dims))
        for dim in unscored:
            findings.append(
                ValidationFinding(
                    validator="consistency_validator",
                    severity="info",
                    category="alignment",
                    message=(
                        f"Dimension '{dim}' has an assessment "
                        f"but no corresponding score"
                    ),
                    artifact_type="DimensionAssessment",
                )
            )

        unassessed = sorted(set(score_dims) - set(assess_dims))
        for dim in unassessed:
            findings.append(
                ValidationFinding(
                    validator="consistency_validator",
                    severity="info",
                    category="alignment",
                    message=(
                        f"Dimension '{dim}' has a score "
                        f"but no corresponding assessment"
                    ),
                    artifact_type="ScoreResult",
                )
            )
        return findings

    def _check_recommendation_consistency(
        self,
        recommendations: list[object],
        assessments: list[object],
    ) -> list[ValidationFinding]:
        """Check that recommendations reference valid assessments."""
        findings: list[ValidationFinding] = []
        assess_dims = sorted(
            {getattr(a, "dimension", "") for a in assessments}
        )
        for i, rec in enumerate(recommendations):
            rec_assessments = getattr(
                rec, "supporting_assessments", []
            )
            for assess in rec_assessments:
                adim = getattr(assess, "dimension", "")
                if adim and adim not in assess_dims:
                    findings.append(
                        ValidationFinding(
                            validator="consistency_validator",
                            severity="info",
                            category="reference_integrity",
                            message=(
                                f"Recommendation {i} references "
                                f"assessment dimension '{adim}' "
                                f"not found in pipeline assessments"
                            ),
                            artifact_type="Recommendation",
                        )
                    )
        return findings
