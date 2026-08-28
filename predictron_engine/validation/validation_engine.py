"""Validation engine — orchestrates all validators and builds debug reports.

This module composes all validators, explainers, and trace builders
into a single entry point. It never modifies pipeline outputs —
it only reads them and produces diagnostic information.
"""

from __future__ import annotations

import logging
from typing import cast

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.models.report import (
    ConfidenceAssessment,
    DimensionAssessment,
    Observation,
    Recommendation,
)
from predictron_engine.validation.debug_report import (
    ConfidenceSummary,
    DebugReport,
    MissingInputs,
    PipelineSummary,
    StageOutput,
)
from predictron_engine.validation.explainability import ExplanationBuilder
from predictron_engine.validation.trace import TraceGraphBuilder
from predictron_engine.validation.validators.completeness_validator import (
    CompletenessValidator,
)
from predictron_engine.validation.validators.consistency_validator import (
    ConsistencyValidator,
)
from predictron_engine.validation.validators.pipeline_validator import (
    PipelineValidator,
    ValidationFinding,
)
from predictron_engine.validation.validators.report_validator import (
    ReportValidator,
)

logger = logging.getLogger(__name__)


class ValidationEngine:
    """Orchestrates all validators and explanation builders.

    Composes pipeline, consistency, completeness, and report validators
    with explanation builders and the trace graph builder to produce
    a complete DebugReport.
    """

    def __init__(
        self,
        pipeline_validator: PipelineValidator | None = None,
        consistency_validator: ConsistencyValidator | None = None,
        completeness_validator: CompletenessValidator | None = None,
        report_validator: ReportValidator | None = None,
        explanation_builder: ExplanationBuilder | None = None,
        trace_builder: TraceGraphBuilder | None = None,
    ) -> None:
        self._pipeline = pipeline_validator or PipelineValidator()
        self._consistency = consistency_validator or ConsistencyValidator()
        self._completeness = completeness_validator or CompletenessValidator()
        self._report = report_validator or ReportValidator()
        self._explainer = explanation_builder or ExplanationBuilder()
        self._tracer = trace_builder or TraceGraphBuilder()

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
    ) -> DebugReport:
        """Run all validations and build a complete DebugReport."""
        logger.info("Starting validation engine")

        # Run validators
        pipeline_findings = self._pipeline.validate(
            startup, features, evidence, observations,
            assessments, scores, recommendations, confidence,
        )
        consistency_findings = self._consistency.validate(
            observations, assessments, scores, recommendations
        )
        completeness_findings = self._completeness.validate(
            features, evidence, observations, assessments,
            scores, recommendations,
        )

        # Build explanations
        stage_outputs = self._build_stage_outputs(
            startup, features, evidence, observations,
            assessments, scores, recommendations, confidence,
        )

        # Build trace graph
        startup_name = getattr(startup, "name", "unknown")
        trace_graph = self._tracer.build(
            startup_name=startup_name,
            features=features,
            evidence=evidence,
            observations=observations,
            assessments=assessments,
            scores=scores,
            recommendations=recommendations,
            confidence=confidence,
        )

        # Build pipeline summary
        meta = getattr(features, "data_completeness", 0.0)
        pipeline_summary = PipelineSummary(
            startup_name=startup_name,
            engine_version="",
            processing_time_ms=0.0,
            stages_completed=[
                "normalize", "collect", "extract", "evidence",
                "reason", "evaluate", "score", "recommend",
                "confidence", "build_report",
            ],
            total_artifacts=(
                len(evidence) + len(observations) + len(assessments)
                + len(scores) + len(recommendations) + len(confidence)
                + 2
            ),
            data_completeness=meta,
        )

        # Aggregate all findings
        all_findings = (
            pipeline_findings + consistency_findings + completeness_findings
        )

        # Build missing inputs summary
        missing = self._build_missing_inputs(
            features, evidence, observations, completeness_findings
        )

        # Build confidence summary
        conf_summary = self._build_confidence_summary(confidence)

        # Key artifacts
        key_artifacts = self._build_key_artifacts(trace_graph)

        report = DebugReport(
            pipeline_summary=pipeline_summary,
            stage_outputs=stage_outputs,
            warnings=all_findings,
            validation_results={
                "pipeline": pipeline_findings,
                "consistency": consistency_findings,
                "completeness": completeness_findings,
            },
            trace_graph=trace_graph,
            missing_inputs=missing,
            confidence_summary=conf_summary,
            key_artifacts=key_artifacts,
        )

        logger.info(
            "Validation complete: %d findings (%d errors, %d warnings, %d info)",
            len(all_findings),
            report.error_count,
            report.warning_count,
            report.info_count,
        )
        return report

    def validate_report(self, report: object) -> DebugReport:
        """Validate a completed Report and produce a DebugReport."""
        report_findings = self._report.validate(report)
        startup = getattr(report, "startup", None)
        features = getattr(report, "features", None)
        evidence = getattr(report, "evidence", [])
        observations = getattr(report, "observations", [])
        assessments = getattr(report, "dimension_assessments", [])
        scores = getattr(report, "scores", [])
        recommendations = getattr(report, "recommendations", [])
        confidence = getattr(report, "confidence", [])

        debug = self.validate(
            startup, features, evidence, observations,
            assessments, scores, recommendations, confidence,
        )

        debug.warnings.extend(report_findings)
        debug.validation_results["report"] = report_findings

        meta = getattr(report, "analysis_metadata", None)
        if meta:
            debug.pipeline_summary.engine_version = getattr(
                meta, "engine_version", ""
            )
            debug.pipeline_summary.processing_time_ms = getattr(
                meta, "processing_time_ms", 0.0
            )

        return debug

    def _build_stage_outputs(
        self,
        startup: object,
        features: object,
        evidence: list[object],
        observations: list[object],
        assessments: list[object],
        scores: list[object],
        recommendations: list[object],
        confidence: list[object],
    ) -> list[StageOutput]:
        """Build per-stage output summaries with explanations."""
        startup_name = getattr(startup, "name", "unknown")
        typed_evidence = cast(list[EvidenceItem], evidence)
        typed_observations = cast(list[Observation], observations)
        typed_assessments = cast(list[DimensionAssessment], assessments)
        typed_recommendations = cast(list[Recommendation], recommendations)
        typed_confidence = cast(list[ConfidenceAssessment], confidence)
        return [
            StageOutput(
                stage="collect",
                artifact_type="CollectedData",
                artifact_count=1,
            ),
            StageOutput(
                stage="extract",
                artifact_type="ExtractedFeatures",
                artifact_count=1,
                explanation=self._explainer.explain_extraction(
                    features, startup_name
                ),
            ),
            StageOutput(
                stage="evidence",
                artifact_type="EvidenceSet",
                artifact_count=len(evidence),
                explanation=self._explainer.explain_evidence(
                    typed_evidence, features
                ),
            ),
            StageOutput(
                stage="reason",
                artifact_type="list[Observation]",
                artifact_count=len(observations),
                explanation=self._explainer.explain_reasoning(
                    typed_observations, features, typed_evidence
                ),
            ),
            StageOutput(
                stage="evaluate",
                artifact_type="EvaluationResult",
                artifact_count=len(assessments),
                explanation=self._explainer.explain_evaluation(
                    typed_assessments, typed_observations, typed_evidence
                ),
            ),
            StageOutput(
                stage="recommend",
                artifact_type="list[Recommendation]",
                artifact_count=len(recommendations),
                explanation=self._explainer.explain_recommendations(
                    typed_recommendations, typed_assessments,
                    typed_observations,
                ),
            ),
            StageOutput(
                stage="confidence",
                artifact_type="list[ConfidenceAssessment]",
                artifact_count=len(confidence),
                explanation=self._explainer.explain_confidence(
                    typed_confidence, features
                ),
            ),
        ]

    def _build_missing_inputs(
        self,
        features: object,
        evidence: list[object],
        observations: list[object],
        completeness_findings: list[ValidationFinding],
    ) -> MissingInputs:
        """Build missing inputs summary."""
        missing_features = []
        for field in [
            "industry", "business_model", "funding_stage",
            "geography", "technology_stack",
        ]:
            value = getattr(features, field, None)
            if value is None or value == [] or value == 0:
                missing_features.append(field)

        evidence_domains = {getattr(e, "domain", "") for e in evidence}
        all_expected = {
            "industry", "business_model", "funding_stage",
            "technology_stack", "geography",
        }
        missing_domains = sorted(all_expected - evidence_domains)

        low_conf_dims: list[str] = []
        for obs in observations:
            if getattr(obs, "confidence", 1.0) < 0.3:
                dim = getattr(obs, "dimension", "")
                if dim and dim not in low_conf_dims:
                    low_conf_dims.append(dim)

        unused_evidence = sum(
            1 for f in completeness_findings
            if f.category == "unused_evidence"
        )
        unused_obs = sum(
            1 for f in completeness_findings
            if f.category == "unused_observations"
        )

        return MissingInputs(
            missing_features=missing_features,
            missing_evidence_domains=missing_domains,
            low_confidence_dimensions=low_conf_dims,
            unused_evidence_count=unused_evidence,
            unused_observation_count=unused_obs,
        )

    def _build_confidence_summary(
        self, confidence: list[object]
    ) -> ConfidenceSummary:
        """Build confidence summary from confidence assessments."""
        if not confidence:
            return ConfidenceSummary()

        per_dim: dict[str, float] = {}
        for c in confidence:
            dim = getattr(c, "dimension", "")
            conf = getattr(c, "confidence", 0.0)
            if dim:
                per_dim[dim] = conf

        overall = (
            sum(per_dim.values()) / len(per_dim) if per_dim else 0.0
        )
        lowest = min(per_dim, key=lambda d: per_dim[d]) if per_dim else ""
        highest = max(per_dim, key=lambda d: per_dim[d]) if per_dim else ""

        return ConfidenceSummary(
            overall_confidence=overall,
            per_dimension=per_dim,
            lowest_confidence_dimension=lowest,
            highest_confidence_dimension=highest,
        )

    def _build_key_artifacts(
        self, trace_graph: object
    ) -> dict[str, str]:
        """Extract key artifact IDs from the trace graph."""
        nodes = getattr(trace_graph, "nodes", {})
        result: dict[str, str] = {}
        for node_id, node in nodes.items():
            atype = getattr(node, "artifact_type", "")
            if atype and atype not in result:
                result[atype] = node_id
        return result
