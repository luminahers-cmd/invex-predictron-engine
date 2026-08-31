"""Report Builder — assembles all pipeline outputs into a single Report.

The report builder is the final stage of the pipeline. It receives
outputs from every preceding stage and assembles them into a single
structured Report object.

Key principles:
  - Score aggregation is straightforward (average)
  - Confidence aggregation is straightforward (average)
  - Evidence is passed through without modification
  - Metadata is populated here (timing, version, completeness)
  - Investment readiness is provided by the pipeline (canonical
    computation in ``PredictronEngine``); the builder reuses it. A
    fallback recompute remains only for direct callers that do not
    supply readiness, preserving the standalone build() API.
  - The Report is the single output consumed by the adapter layer
"""

import logging
from datetime import UTC, datetime

from predictron_engine.decision.models import CalibrationSummary, DecisionConfidence
from predictron_engine.evaluation.aggregation import mean_score
from predictron_engine.evaluation.evaluation_models import DimensionAssessment
from predictron_engine.evaluation.investment_readiness import (
    compute_investment_readiness,
)
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    AnalysisMetadata,
    ConfidenceAssessment,
    DecisionSynthesis,
    EvidenceCollectionMetadata,
    EvidenceItem,
    InvestmentDecision,
    InvestmentReadiness,
    Observation,
    Recommendation,
    Report,
    ScoreResult,
)
from predictron_engine.models.startup import Startup
from predictron_engine.version import ENGINE_VERSION

logger = logging.getLogger(__name__)


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
        evidence_collection: EvidenceCollectionMetadata | None = None,
        decision_confidence: DecisionConfidence | None = None,
        calibration_summary: CalibrationSummary | None = None,
        decision_synthesis: DecisionSynthesis | None = None,
        processing_time_ms: float = 0.0,
    ) -> Report:
        """Assemble the final report from all pipeline outputs.

        When supplied, ``decision_confidence``, ``calibration_summary``,
        ``decision_synthesis``, and ``processing_time_ms`` are embedded in
        the Report so that ``build()`` can produce a complete Report on
        its own.  All parameters remain optional so existing callers are
        unaffected; the engine supplies them to avoid post-construction
        mutation.
        """
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
            "collect_evidence",
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
            decision_confidence=decision_confidence,
            calibration_summary=calibration_summary,
            decision_synthesis=decision_synthesis,
            signal_relationships=signal_relationships,
            analysis_metadata=AnalysisMetadata(
                engine_version=ENGINE_VERSION,
                pipeline_stages_completed=pipeline_stages,
                processing_time_ms=processing_time_ms,
                timestamp=datetime.now(UTC),
                data_completeness=features.data_completeness,
                evidence_collection=evidence_collection,
            ),
        )

    def _aggregate_scores(self, scores: list[ScoreResult]) -> float:
        """Compute the overall score via the canonical mean aggregation.

        ``mean_score`` is the same primitive consumed by the decision score
        factor, so the report and decision cannot drift apart.
        """
        return round(mean_score(scores), 2)

    def _aggregate_confidence(
        self, confidence: list[ConfidenceAssessment]
    ) -> float:
        """Compute the overall confidence as the mean of dimension confidences."""
        if not confidence:
            return 0.0
        return round(
            sum(c.confidence for c in confidence) / len(confidence), 4
        )
