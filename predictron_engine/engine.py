"""PredictronEngine — core intelligence pipeline orchestrator.

This module exposes the PredictronEngine class, which composes all
pipeline stages into a cohesive analysis flow.

Pipeline:
  normalize -> collect -> extract -> evidence -> reason ->
  evaluate -> score -> recommend -> confidence -> decide -> build_report

All dependencies are injected via the constructor. Every stage can be
replaced independently without modifying the engine or any other stage.
Default implementations are provided when no explicit dependency is given.

Usage:
    engine = PredictronEngine()
    report = engine.analyze(request)

    # Or with debug report:
    report, debug = engine.analyze_with_debug(request)

    # Or with custom implementations:
    engine = PredictronEngine(
        scoring=MyCustomScoringEngine(),
        reasoning=MyCustomReasoningEngine(),
    )
"""

import logging
import time

from app.schemas.analysis import StartupAnalysisRequest
from predictron_engine.collection.collector import DefaultDataCollector
from predictron_engine.confidence.confidence_engine import DefaultConfidenceEngine
from predictron_engine.decision.decision_engine import DefaultDecisionEngine
from predictron_engine.evaluation.composite import CompositeEvaluator
from predictron_engine.evaluation.investment_readiness import (
    compute_investment_readiness,
)
from predictron_engine.evidence.evidence_engine import DefaultEvidenceEngine
from predictron_engine.extraction.composite import CompositeExtractor
from predictron_engine.ingest.normalizer import DefaultNormalizer
from predictron_engine.models.report import Report
from predictron_engine.reasoning.reasoning_engine import DefaultReasoningEngine
from predictron_engine.recommendations.composite import (
    CompositeRecommendationEngine,
)
from predictron_engine.report.report_builder import DefaultReportBuilder
from predictron_engine.scoring.scoring_engine import DefaultScoringEngine

logger = logging.getLogger(__name__)

ENGINE_VERSION = "0.9.0"


class PredictronEngine:
    """The core intelligence pipeline orchestrator.

    Composes independent pipeline stages into a cohesive analysis flow.
    All dependencies are injected — no module is hardcoded.

    Each parameter accepts a protocol-compatible implementation.
    When omitted, the default implementation for that stage is used.
    """

    def __init__(
        self,
        normalizer: DefaultNormalizer | None = None,
        collector: DefaultDataCollector | None = None,
        extractor: CompositeExtractor | None = None,
        evidence: DefaultEvidenceEngine | None = None,
        reasoning: DefaultReasoningEngine | None = None,
        evaluation: CompositeEvaluator | None = None,
        scoring: DefaultScoringEngine | None = None,
        recommendations: CompositeRecommendationEngine | None = None,
        confidence: DefaultConfidenceEngine | None = None,
        decision: DefaultDecisionEngine | None = None,
        report_builder: DefaultReportBuilder | None = None,
    ) -> None:
        self._normalizer = normalizer or DefaultNormalizer()
        self._collector = collector or DefaultDataCollector()
        self._extractor = extractor or CompositeExtractor()
        self._evidence = evidence or DefaultEvidenceEngine()
        self._reasoning = reasoning or DefaultReasoningEngine()
        self._evaluation = evaluation or CompositeEvaluator()
        self._scoring = scoring or DefaultScoringEngine()
        self._recommendations = recommendations or CompositeRecommendationEngine()
        self._confidence = confidence or DefaultConfidenceEngine()
        self._decision = decision or DefaultDecisionEngine()
        self._report_builder = report_builder or DefaultReportBuilder()
        logger.info("PredictronEngine initialized")

    def analyze(self, request: StartupAnalysisRequest) -> Report:
        """Execute the full analysis pipeline and return a structured Report.

        This is the single public entry point for the engine.
        It orchestrates every pipeline stage in sequence.
        """
        start_time = time.perf_counter()
        logger.info("Starting analysis pipeline")

        # Stage 1: Normalize
        startup = self._normalizer.normalize(request)

        # Stage 2: Collect
        collected_data = self._collector.collect(startup)

        # Stage 3: Extract features
        features = self._extractor.extract(startup, collected_data)

        # Stage 4: Gather evidence
        evidence_set = self._evidence.gather(features)

        # Stage 5: Reason
        observations = self._reasoning.reason(features, evidence_set.items)

        # Stage 6: Evaluate
        evaluation_result = self._evaluation.evaluate(
            features, observations, evidence_set.items
        )

        # Stage 7: Score
        scores = self._scoring.score(features, observations)

        # Stage 7b: Compute investment readiness
        assessments = evaluation_result.assessments
        investment_readiness = compute_investment_readiness(
            features, observations, scores, assessments,
        )

        # Stage 8: Recommend
        recs = self._recommendations.recommend(
            features, observations, scores, assessments
        )

        # Stage 9: Assess confidence
        conf = self._confidence.assess(
            features, observations, scores, assessments
        )

        # Stage 10: Decide
        decision = self._decision.decide(
            features,
            observations,
            scores,
            conf,
            assessments,
            investment_readiness.signal_relationships,
        )

        # Stage 11: Build report
        report = self._report_builder.build(
            startup,
            features,
            evidence_set.items,
            observations,
            scores,
            recs,
            conf,
            assessments,
            decision,
            investment_readiness,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        report.analysis_metadata.processing_time_ms = round(elapsed_ms, 2)

        logger.info("Analysis complete in %.1fms", elapsed_ms)
        return report

    def analyze_with_debug(
        self, request: StartupAnalysisRequest
    ) -> tuple[Report, object]:
        """Execute analysis and return a DebugReport alongside the Report.

        Returns:
            A tuple of (Report, DebugReport). The DebugReport contains
            traceability, explanations, validation findings, and
            confidence summaries.
        """
        from predictron_engine.validation.validation_engine import (
            ValidationEngine,
        )

        report = self.analyze(request)
        val_engine = ValidationEngine()
        debug = val_engine.validate_report(report)
        return report, debug
