"""PredictronEngine — core intelligence pipeline orchestrator.

This module exposes the PredictronEngine class, which composes all
pipeline stages into a cohesive analysis flow.

Pipeline:
  normalize -> collect -> collect_evidence -> extract -> evidence ->
  reason -> evaluate -> score -> recommend -> confidence -> decide ->
  calibrate -> synthesize -> build_report

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
import uuid

from app.schemas.analysis import StartupAnalysisRequest
from predictron_engine.collection.collector import DefaultDataCollector
from predictron_engine.confidence.confidence_engine import DefaultConfidenceEngine
from predictron_engine.context import AnalysisContext
from predictron_engine.decision.calibration import (
    apply_recommendation_risk,
    build_calibration_summary,
    compute_decision_confidence,
)
from predictron_engine.decision.decision_engine import DefaultDecisionEngine
from predictron_engine.evaluation.composite import CompositeEvaluator
from predictron_engine.evaluation.investment_readiness import (
    compute_investment_readiness,
)
from predictron_engine.evidence.evidence_engine import DefaultEvidenceEngine
from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.evidence.orchestrator import EvidenceOrchestrator
from predictron_engine.evidence.runner import (
    EvidenceOrchestratorProtocol,
    collect_evidence_sync,
)
from predictron_engine.extraction.composite import CompositeExtractor
from predictron_engine.ingest.normalizer import DefaultNormalizer
from predictron_engine.models.report import (
    EvidenceCollectionMetadata,
    Report,
)
from predictron_engine.reasoning.reasoning_engine import DefaultReasoningEngine
from predictron_engine.recommendations.composite import (
    CompositeRecommendationEngine,
)
from predictron_engine.report.report_builder import DefaultReportBuilder
from predictron_engine.scoring.scoring_engine import DefaultScoringEngine
from predictron_engine.synthesis.engine import DecisionSynthesisEngine

logger = logging.getLogger(__name__)

ENGINE_VERSION = "0.12.1"


def _evidence_metadata(bundle: EvidenceBundle) -> EvidenceCollectionMetadata:
    """Derive report metadata from a collected evidence bundle."""
    successful = sum(1 for source in bundle.sources if source.success)
    return EvidenceCollectionMetadata(
        website=str(bundle.website) if bundle.website else None,
        pages_discovered=bundle.attempted_pages,
        pages_fetched=bundle.total_pages,
        successful_sources=successful,
        failed_sources=len(bundle.sources) - successful,
        collection_time_ms=bundle.duration_ms,
    )


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
        evidence_collector: EvidenceOrchestratorProtocol | None = None,
        evidence: DefaultEvidenceEngine | None = None,
        reasoning: DefaultReasoningEngine | None = None,
        evaluation: CompositeEvaluator | None = None,
        scoring: DefaultScoringEngine | None = None,
        recommendations: CompositeRecommendationEngine | None = None,
        confidence: DefaultConfidenceEngine | None = None,
        decision: DefaultDecisionEngine | None = None,
        report_builder: DefaultReportBuilder | None = None,
        synthesis: DecisionSynthesisEngine | None = None,
    ) -> None:
        self._normalizer = normalizer or DefaultNormalizer()
        self._collector = collector or DefaultDataCollector()
        self._extractor = extractor or CompositeExtractor()
        self._evidence_collector = evidence_collector or EvidenceOrchestrator()
        self._evidence = evidence or DefaultEvidenceEngine()
        self._reasoning = reasoning or DefaultReasoningEngine()
        self._evaluation = evaluation or CompositeEvaluator()
        self._scoring = scoring or DefaultScoringEngine()
        self._recommendations = recommendations or CompositeRecommendationEngine()
        self._confidence = confidence or DefaultConfidenceEngine()
        self._decision = decision or DefaultDecisionEngine()
        self._report_builder = report_builder or DefaultReportBuilder()
        self._synthesis = synthesis or DecisionSynthesisEngine()
        logger.info("PredictronEngine initialized")

    def analyze(
        self,
        request: StartupAnalysisRequest,
        request_id: str | None = None,
    ) -> Report:
        """Execute the full analysis pipeline and return a structured Report.

        This is the single public entry point for the engine.
        It orchestrates every pipeline stage in sequence.

        Parameters
        ----------
        request:
            The analysis request to process.
        request_id:
            Optional correlation id for logging and tracing. A random id
            is generated when omitted.
        """
        start_time = time.perf_counter()
        request_id = request_id or uuid.uuid4().hex
        logger.info("Starting analysis pipeline", extra={"request_id": request_id})

        # Stage 1: Normalize
        startup = self._normalizer.normalize(request)

        # Stage 2: Collect
        collected_data = self._collector.collect(startup)

        # Stage 3: Collect website evidence (best-effort, never fatal)
        evidence_bundle = self._collect_evidence(startup, request_id)

        context = AnalysisContext(
            request_id=request_id,
            startup=startup,
            evidence_bundle=evidence_bundle,
        )

        # Stage 4: Extract features (enriched with website evidence)
        features = self._extractor.extract(
            startup, collected_data, context.evidence_bundle
        )

        # Stage 5: Gather evidence
        evidence_set = self._evidence.gather(features)

        # Stage 6: Reason
        observations = self._reasoning.reason(features, evidence_set.items)

        # Stage 7: Evaluate
        evaluation_result = self._evaluation.evaluate(
            features, observations, evidence_set.items
        )

        # Stage 8: Score
        scores = self._scoring.score(features, observations)

        # Stage 8b: Compute investment readiness
        assessments = evaluation_result.assessments
        investment_readiness = compute_investment_readiness(
            features, observations, scores, assessments,
        )

        # Stage 9: Recommend
        recs = self._recommendations.recommend(
            features, observations, scores, assessments
        )

        # Stage 10: Assess confidence
        conf = self._confidence.assess(
            features, observations, scores, assessments
        )

        # Stage 11: Decide
        decision = self._decision.decide(
            features,
            observations,
            scores,
            conf,
            assessments,
            investment_readiness.signal_relationships,
        )

        # Stage 11b: Calibrate decision confidence (Sprint 6B).
        # Combines existing pipeline outputs only — no new evidence,
        # reasoning, or evaluation work is performed here.
        decision_confidence = compute_decision_confidence(
            bundle=evidence_bundle,
            observations=observations,
            assessments=assessments,
            features=features,
            scores=scores,
        )
        recs = apply_recommendation_risk(recs, decision_confidence)
        calibration_summary = build_calibration_summary(
            decision_confidence,
            observation_count=len(observations),
            assessed_dimension_count=len(assessments),
            evidence_document_count=evidence_bundle.total_pages,
        )

        # Stage 11c: Decision synthesis (Sprint 6C).
        # Aggregates already-produced outputs only — no recomputation of
        # scores, confidence, trust, evidence, reasoning, or evaluations.
        synthesis = self._synthesis.synthesize(
            features=features,
            observations=observations,
            assessments=assessments,
            scores=scores,
            recommendations=recs,
            readiness=investment_readiness,
            decision=decision,
            decision_confidence=decision_confidence,
            calibration_summary=calibration_summary,
            consistency=getattr(self._reasoning, "last_consistency", None),
        )

        # Stage 12: Build report
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
            evidence_collection=_evidence_metadata(evidence_bundle),
        )
        report.decision_confidence = decision_confidence
        report.calibration_summary = calibration_summary
        report.decision_synthesis = synthesis

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        report.analysis_metadata.processing_time_ms = round(elapsed_ms, 2)

        logger.info(
            "Analysis complete in %.1fms",
            elapsed_ms,
            extra={"request_id": request_id},
        )
        return report

    def _collect_evidence(
        self, startup: object, request_id: str
    ) -> EvidenceBundle:
        """Run website evidence collection for the startup, never raising.

        Returns an empty bundle when no website is available or collection
        fails for any reason, so analysis always proceeds.
        """
        website = getattr(startup, "website", "") or ""
        if not website:
            logger.info(
                "Skipping evidence collection: no website",
                extra={"request_id": request_id},
            )
            return EvidenceBundle.empty(getattr(startup, "name", ""))

        try:
            bundle = collect_evidence_sync(
                self._evidence_collector,
                getattr(startup, "name", ""),
                website,
            )
            logger.info(
                "Evidence collected: %d documents from %d pages in %d ms",
                bundle.total_pages,
                bundle.attempted_pages,
                bundle.duration_ms,
                extra={"request_id": request_id},
            )
            return bundle
        except Exception:  # noqa: BLE001 - evidence must never break analysis
            logger.exception(
                "Evidence collection failed for %s",
                website,
                extra={"request_id": request_id},
            )
            return EvidenceBundle.empty(getattr(startup, "name", ""))

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
