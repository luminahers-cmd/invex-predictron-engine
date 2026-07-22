"""Report Builder — assembles all pipeline outputs into a single Report.

The report builder is the final stage of the pipeline. It receives
outputs from every preceding stage and assembles them into a single
structured Report object.

Key principles:
  - Score aggregation is straightforward (average)
  - Confidence aggregation is straightforward (average)
  - Evidence is passed through without modification
  - Metadata is populated here (timing, version, completeness)
  - Investment readiness is computed from all available signals (Sprint 8)
  - Investment decision is computed from all signals (Sprint 13)
  - The Report is the single output consumed by the adapter layer
"""

import logging
from datetime import UTC, datetime

from predictron_engine.evaluation.evaluation_models import DimensionAssessment
from predictron_engine.evaluation.investment_readiness import (
    compute_investment_readiness,
)
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    AnalysisMetadata,
    ConfidenceAssessment,
    EvidenceItem,
    InvestmentDecision,
    InvestmentReadiness,
    Observation,
    Recommendation,
    Report,
    ScoreResult,
)
from predictron_engine.models.startup import Startup

logger = logging.getLogger(__name__)

ENGINE_VERSION = "0.10.0"


class DefaultReportBuilder:
    """Standard implementation of the ReportBuilder protocol.

    Assembles pipeline outputs into a Report with aggregated scores,
    confidence levels, and investment readiness assessment.
    """

    def build(
        self,
        startup: Startup,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
        observations: list[Observation],
        scores: list[ScoreResult],
        recommendations: list[Recommendation],
        confidence: list[ConfidenceAssessment],
        dimension_assessments: list[DimensionAssessment] | None = None,
        decision: InvestmentDecision | None = None,
        investment_readiness: InvestmentReadiness | None = None,
    ) -> Report:
        """Assemble the final report from all pipeline outputs."""
        logger.info("Building analysis report")

        overall_score = self._aggregate_scores(scores)
        overall_confidence = self._aggregate_confidence(confidence)
        assessments = dimension_assessments or []

        if investment_readiness is None:
            investment_readiness = compute_investment_readiness(
                features, observations, scores, assessments,
            )

        signal_relationships = investment_readiness.signal_relationships

        pipeline_stages = [
            "normalize",
            "collect",
            "extract",
            "evidence",
            "reason",
            "evaluate",
            "score",
            "recommend",
            "confidence",
        ]
        if decision is not None:
            pipeline_stages.append("decide")
        pipeline_stages.append("build_report")

        return Report(
            startup=startup,
            features=features,
            evidence=evidence,
            observations=observations,
            dimension_assessments=assessments,
            scores=scores,
            overall_score=overall_score,
            recommendations=recommendations,
            confidence=confidence,
            overall_confidence=overall_confidence,
            investment_readiness=investment_readiness,
            investment_decision=decision,
            signal_relationships=signal_relationships,
            analysis_metadata=AnalysisMetadata(
                engine_version=ENGINE_VERSION,
                pipeline_stages_completed=pipeline_stages,
                processing_time_ms=0.0,
                timestamp=datetime.now(UTC),
                data_completeness=features.data_completeness,
            ),
        )

    def _aggregate_scores(self, scores: list[ScoreResult]) -> float:
        """Compute the overall score as the mean of dimension scores."""
        if not scores:
            return 0.0
        return round(sum(s.score for s in scores) / len(scores), 2)

    def _aggregate_confidence(
        self, confidence: list[ConfidenceAssessment]
    ) -> float:
        """Compute the overall confidence as the mean of dimension confidences."""
        if not confidence:
            return 0.0
        return round(
            sum(c.confidence for c in confidence) / len(confidence), 4
        )
