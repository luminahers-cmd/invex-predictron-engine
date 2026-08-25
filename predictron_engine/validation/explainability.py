"""Structured explanations for each pipeline stage.

Each explanation captures what happened, why it happened, which inputs
were used, which upstream outputs contributed, confidence, and
missing information. This enables full transparency for debugging
and trust verification.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from predictron_engine.models.report import (
    ConfidenceAssessment,
    DimensionAssessment,
    EvidenceItem,
    Observation,
    Recommendation,
)


class StageExplanation(BaseModel):
    """A structured explanation for a single pipeline stage's output."""

    stage: str = Field(..., description="Pipeline stage name")
    artifact_type: str = Field(
        ..., description="Type of output produced"
    )
    what_happened: str = Field(
        ..., description="Summary of what the stage did"
    )
    why_it_happened: str = Field(
        ..., description="Why this particular output was produced"
    )
    inputs_used: list[str] = Field(
        default_factory=list,
        description="Description of inputs consumed by this stage",
    )
    upstream_contributors: list[str] = Field(
        default_factory=list,
        description="Specific upstream artifacts that contributed",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this explanation",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Information that was unavailable or missing",
    )


class ConclusionTrace(BaseModel):
    """Traces a specific conclusion back to its supporting evidence.

    This provides end-to-end explainability from a conclusion
    (recommendation, score, or assessment) back to the original
    features and evidence that support it.
    """

    conclusion_type: str = Field(
        ..., description="Type of conclusion (recommendation, score, assessment)"
    )
    conclusion_text: str = Field(
        ..., description="The conclusion statement"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this conclusion",
    )
    supporting_observations: list[str] = Field(
        default_factory=list,
        description="Observations that support this conclusion",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence items that support this conclusion",
    )
    feature_references: list[str] = Field(
        default_factory=list,
        description="Feature references that contributed to this conclusion",
    )
    reasoning_chain: list[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning from evidence to conclusion",
    )
    data_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Data completeness at the time of this conclusion",
    )


class ExplanationBuilder:
    """Builds structured explanations for pipeline stage outputs."""

    def explain_extraction(
        self, features: object, startup_name: str
    ) -> StageExplanation:
        """Explain the extraction stage output."""
        populated = []
        missing = []
        for field_name in [
            "industry", "business_model", "funding_stage",
            "geography", "technology_stack", "customer_type",
            "founded_year", "has_revenue", "founder_profile_count",
        ]:
            value = getattr(features, field_name, None)
            if value is not None and value != [] and value != 0:
                populated.append(field_name)
            else:
                missing.append(field_name)

        completeness = getattr(features, "data_completeness", 0.0)
        return StageExplanation(
            stage="extract",
            artifact_type="ExtractedFeatures",
            what_happened=(
                f"Extracted {len(populated)} features from {startup_name}. "
                f"Data completeness: {completeness:.0%}."
            ),
            why_it_happened=(
                "Domain extractors analyzed collected data and "
                "classified startup attributes using keyword taxonomies."
            ),
            inputs_used=["Startup model", "CollectedData"],
            upstream_contributors=[
                f"Populated: {', '.join(populated)}" if populated else "No features populated",
            ],
            confidence=completeness,
            missing_information=[
                f"Field '{f}' could not be determined" for f in missing
            ] if missing else [],
        )

    def explain_evidence(
        self,
        evidence_items: list[EvidenceItem],
        features: object,
    ) -> StageExplanation:
        """Explain the evidence gathering stage output."""
        domains = sorted({e.domain for e in evidence_items})
        completeness = getattr(features, "data_completeness", 0.0)

        missing_domains = set()
        for domain in [
            "industry", "business_model", "funding_stage",
            "technology_stack", "geography",
        ]:
            value = getattr(features, domain, None)
            if value is None or value == [] or value == 0:
                missing_domains.add(domain)

        return StageExplanation(
            stage="evidence",
            artifact_type="EvidenceSet",
            what_happened=(
                f"Gathered {len(evidence_items)} evidence items from "
                f"{len(domains)} domain(s): {', '.join(domains)}."
            ),
            why_it_happened=(
                "Evidence providers retrieved domain-specific facts "
                "from the knowledge base based on extracted features."
            ),
            inputs_used=["ExtractedFeatures"],
            upstream_contributors=[
                f"Feature coverage: {completeness:.0%}",
            ],
            confidence=completeness,
            missing_information=[
                f"Missing feature for domain '{d}'" for d in sorted(missing_domains)
            ] if missing_domains else [],
        )

    def explain_reasoning(
        self,
        observations: list[Observation],
        features: object,
        evidence: list[EvidenceItem],
    ) -> StageExplanation:
        """Explain the reasoning stage output."""
        rules_used = sorted({obs.source_rule for obs in observations})
        dimensions = sorted({obs.dimension for obs in observations})
        avg_confidence = (
            sum(o.confidence for o in observations) / len(observations)
            if observations else 0.0
        )
        missing = []
        completeness = getattr(features, "data_completeness", 0.0)
        if completeness < 0.3:
            missing.append(
                "Low data completeness limits observation quality"
            )

        return StageExplanation(
            stage="reason",
            artifact_type="list[Observation]",
            what_happened=(
                f"Generated {len(observations)} observations from "
                f"{len(rules_used)} reasoning rules across "
                f"{len(dimensions)} dimension(s)."
            ),
            why_it_happened=(
                "Reasoning rules evaluated extracted features against "
                "domain evidence to produce explainable observations."
            ),
            inputs_used=["ExtractedFeatures", "EvidenceSet"],
            upstream_contributors=[
                f"Rules applied: {', '.join(sorted(rules_used))}",
                f"Dimensions covered: {', '.join(sorted(dimensions))}",
            ],
            confidence=avg_confidence,
            missing_information=missing,
        )

    def explain_evaluation(
        self,
        assessments: list[DimensionAssessment],
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> StageExplanation:
        """Explain the evaluation stage output."""
        dimensions = [a.dimension for a in assessments]
        avg_confidence = (
            sum(a.confidence for a in assessments) / len(assessments)
            if assessments else 0.0
        )
        obs_dims = sorted({o.dimension for o in observations})
        assess_dims = sorted({a.dimension for a in assessments})
        uncovered = sorted(set(obs_dims) - set(assess_dims))

        return StageExplanation(
            stage="evaluate",
            artifact_type="EvaluationResult",
            what_happened=(
                f"Produced {len(assessments)} dimension assessments "
                f"from {len(observations)} observations and "
                f"{len(evidence)} evidence items."
            ),
            why_it_happened=(
                "Evaluators synthesized observations and evidence "
                "into structured dimension assessments with rationale."
            ),
            inputs_used=["ExtractedFeatures", "list[Observation]", "list[EvidenceItem]"],
            upstream_contributors=[
                f"Dimensions assessed: {', '.join(dimensions)}",
            ],
            confidence=avg_confidence,
            missing_information=[
                f"Dimension '{d}' had observations but no assessment"
                for d in uncovered
            ] if uncovered else [],
        )

    def explain_recommendations(
        self,
        recommendations: list[Recommendation],
        assessments: list[DimensionAssessment],
        observations: list[Observation],
    ) -> StageExplanation:
        """Explain the recommendation stage output."""
        categories = sorted({r.category for r in recommendations})
        strategies = sorted({
            r.metadata.get("strategy", "unknown")
            for r in recommendations
            if isinstance(r.metadata, dict)
        })
        avg_confidence = (
            sum(r.confidence for r in recommendations) / len(recommendations)
            if recommendations else 0.0
        )
        with_support = sum(
            1 for r in recommendations
            if r.supporting_observations or r.supporting_assessments
        )

        return StageExplanation(
            stage="recommend",
            artifact_type="list[Recommendation]",
            what_happened=(
                f"Generated {len(recommendations)} recommendations "
                f"across {len(categories)} category(ies) from "
                f"{len(strategies)} strategy/strategies."
            ),
            why_it_happened=(
                "Domain recommendation strategies analyzed features, "
                "observations, and assessments to produce actionable "
                "recommendations for the investor."
            ),
            inputs_used=[
                "ExtractedFeatures",
                "list[Observation]",
                "list[ScoreResult]",
                "list[DimensionAssessment]",
            ],
            upstream_contributors=[
                f"Strategies used: {', '.join(sorted(strategies))}",
                f"Categories: {', '.join(sorted(categories))}",
                f"Recommendations with supporting data: {with_support}",
            ],
            confidence=avg_confidence,
            missing_information=[],
        )

    def explain_confidence(
        self,
        confidence: list[ConfidenceAssessment],
        features: object,
    ) -> StageExplanation:
        """Explain the confidence assessment stage output."""
        avg_confidence = (
            sum(c.confidence for c in confidence) / len(confidence)
            if confidence else 0.0
        )
        completeness = getattr(features, "data_completeness", 0.0)
        low_conf = [
            c.dimension for c in confidence if c.confidence < 0.4
        ]

        return StageExplanation(
            stage="confidence",
            artifact_type="list[ConfidenceAssessment]",
            what_happened=(
                f"Assessed confidence for {len(confidence)} dimensions. "
                f"Average confidence: {avg_confidence:.2f}."
            ),
            why_it_happened=(
                "Confidence was computed from data completeness, "
                "observation coverage, and dimension assessment "
                "confidence for each scored dimension."
            ),
            inputs_used=[
                "ExtractedFeatures",
                "list[Observation]",
                "list[ScoreResult]",
                "list[DimensionAssessment]",
            ],
            upstream_contributors=[
                f"Data completeness: {completeness:.0%}",
            ],
            confidence=avg_confidence,
            missing_information=[
                f"Low confidence in dimension '{d}'"
                for d in low_conf
            ] if low_conf else [],
        )

    def trace_recommendation(
        self,
        recommendation: Recommendation,
        observations: list[Observation],
        evidence: list[EvidenceItem],
        features: object,
    ) -> ConclusionTrace:
        """Trace a recommendation back to its supporting evidence.

        Provides end-to-end explainability by connecting a recommendation
        to the observations, evidence, and features that support it.
        """
        completeness = getattr(features, "data_completeness", 0.0)

        supporting_obs_statements = [
            o.statement for o in recommendation.supporting_observations
        ]

        supporting_evidence_statements = [
            e.statement for e in recommendation.supporting_evidence
        ]

        feature_refs = []
        for obs in recommendation.supporting_observations:
            feature_refs.extend(obs.evidence)

        reasoning_chain = self._build_reasoning_chain(
            recommendation, observations, evidence
        )

        return ConclusionTrace(
            conclusion_type="recommendation",
            conclusion_text=recommendation.action,
            confidence=recommendation.confidence,
            supporting_observations=supporting_obs_statements,
            supporting_evidence=supporting_evidence_statements,
            feature_references=feature_refs,
            reasoning_chain=reasoning_chain,
            data_completeness=completeness,
        )

    def trace_assessment(
        self,
        assessment: DimensionAssessment,
        observations: list[Observation],
        evidence: list[EvidenceItem],
        features: object,
    ) -> ConclusionTrace:
        """Trace a dimension assessment back to its supporting evidence."""
        completeness = getattr(features, "data_completeness", 0.0)

        supporting_obs_statements = [
            o.statement for o in assessment.supporting_observations
        ]

        supporting_evidence_statements = [
            e.statement for e in assessment.supporting_evidence
        ]

        feature_refs = []
        for obs in assessment.supporting_observations:
            feature_refs.extend(obs.evidence)

        reasoning_chain = [
            f"Dimension '{assessment.dimension}' assessed with "
            f"{len(assessment.supporting_observations)} observations "
            f"and {len(assessment.supporting_evidence)} evidence items.",
            f"Rationale: {assessment.rationale[:200]}..."
            if len(assessment.rationale) > 200
            else f"Rationale: {assessment.rationale}",
        ]

        return ConclusionTrace(
            conclusion_type="assessment",
            conclusion_text=assessment.summary,
            confidence=assessment.confidence,
            supporting_observations=supporting_obs_statements,
            supporting_evidence=supporting_evidence_statements,
            feature_references=feature_refs,
            reasoning_chain=reasoning_chain,
            data_completeness=completeness,
        )

    @staticmethod
    def _build_reasoning_chain(
        recommendation: Recommendation,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> list[str]:
        """Build a step-by-step reasoning chain for a recommendation."""
        chain: list[str] = []

        chain.append(
            f"Recommendation: {recommendation.action}"
        )

        if recommendation.supporting_observations:
            obs_summary = (
                f"Based on {len(recommendation.supporting_observations)} "
                f"observation(s)"
            )
            chain.append(obs_summary)

        if recommendation.supporting_evidence:
            evidence_summary = (
                f"Supported by {len(recommendation.supporting_evidence)} "
                f"evidence item(s)"
            )
            chain.append(evidence_summary)

        if recommendation.rationale:
            chain.append(f"Rationale: {recommendation.rationale}")

        return chain
