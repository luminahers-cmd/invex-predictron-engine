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
    DimensionAssessment,
    EvidenceCollectionMetadata,
    Report,
    ScoreResult,
)
from predictron_engine.reasoning.reasoning_engine import DefaultReasoningEngine
from predictron_engine.recommendations.composite import (
    CompositeRecommendationEngine,
)
from predictron_engine.report.report_builder import DefaultReportBuilder
from predictron_engine.scoring.scoring_engine import DefaultScoringEngine
from predictron_engine.synthesis.engine import DecisionSynthesisEngine
from predictron_engine.validation.debug_report import DebugReport
from predictron_engine.version import ENGINE_VERSION

logger = logging.getLogger(__name__)

__all__ = ["PredictronEngine", "ENGINE_VERSION"]


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

        # Stage 6: Reason (thread the collected EvidenceBundle through so
        # reasoning consumes trust, provenance, authority, quality, and
        # citations rather than only extracted features and static evidence).
        observations = self._reasoning.reason(
            features, evidence_set.items, context.evidence_bundle
        )

        # H5 (Sprint P8D): the contradiction graph is built exactly once
        # by the reasoning layer from the observations and reused by every
        # downstream stage — no duplicate contradiction detection or
        # recomputation of observations.
        contradiction_graph = getattr(
            self._reasoning, "last_contradiction_graph", None
        )

        # Stage 7: Evaluate
        evaluation_result = self._evaluation.evaluate(
            features, observations, evidence_set.items
        )

        # Stage 8: Score
        scores = self._scoring.score(features, observations)

        # L1: Populate DimensionAssessment.score from the matching ScoreResult
        # so otherwise-dead rationale branches (which filter on assessment
        # scores) become reachable. Reuses existing scoring output — no
        # duplicate computation is performed.
        assessments = evaluation_result.assessments
        self._annex_scores_to_assessments(assessments, scores)

        # Stage 8b: Compute investment readiness
        investment_readiness = compute_investment_readiness(
            features, observations, scores, assessments,
        )

        # Stage 9: Recommend
        recs = self._recommendations.recommend(
            features, observations, scores, assessments
        )

        # Stage 10: Assess confidence
        conf = self._confidence.assess(
            features,
            observations,
            scores,
            assessments,
            contradiction_graph=contradiction_graph,
        )

        # Stage 11: Decide
        # C1: wire the canonical readiness score (Stage 8b output) into the
        # decision instead of falling back to a constant readiness factor.
        # M2: supply the collected evidence bundle so the decision weighs
        # genuine evidence quality rather than an observation-quality proxy.
        decision = self._decision.decide(
            features,
            observations,
            scores,
            conf,
            assessments,
            investment_readiness.signal_relationships,
            readiness_score=investment_readiness.readiness_score,
            evidence_bundle=evidence_bundle,
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
            contradiction_graph=contradiction_graph,
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
            contradiction_graph=contradiction_graph,
        )

        # Stage 12: Build report
        elapsed_ms = (time.perf_counter() - start_time) * 1000
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
            decision_confidence=decision_confidence,
            calibration_summary=calibration_summary,
            decision_synthesis=synthesis,
            processing_time_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "Analysis complete in %.1fms",
            elapsed_ms,
            extra={"request_id": request_id},
        )
        return report

    @staticmethod
    def _annex_scores_to_assessments(
        assessments: list[DimensionAssessment],
        scores: list[ScoreResult],
    ) -> None:
        """Populate each assessment's ``score`` from the matching ScoreResult.

        Reuses the scoring engine's per-dimension output so the assessment
        score field reflects real data (making dead rationale branches
        reachable) without performing any new scoring work.
        """
        score_by_dim = {s.dimension: s.score for s in scores}
        for assessment in assessments:
            if assessment.score is None and assessment.dimension in score_by_dim:
                assessment.score = score_by_dim[assessment.dimension]

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
    ) -> tuple[Report, DebugReport]:
        """Execute analysis and return a DebugReport alongside the Report.

        The debug path runs the identical production pipeline and then
        augments the DebugReport with the reasoning contradiction graph,
        a reasoning trace, and adaptive-budget diagnostics.  The Report
        schema is unchanged.

        Returns:
            A tuple of (Report, DebugReport). The DebugReport contains
            traceability, explanations, validation findings, confidence
            summaries, and (Sprint P8D) reasoning diagnostics.
        """
        from predictron_engine.validation.validation_engine import (
            ValidationEngine,
        )

        report = self.analyze(request)
        val_engine = ValidationEngine()
        debug = val_engine.validate_report(report)
        self._attach_reasoning_debug(debug, report)
        return report, debug

    def _attach_reasoning_debug(self, debug: DebugReport, report: Report) -> None:
        """Attach contradiction graph, trace, and budget to the DebugReport.

        All artifacts are derived deterministically from already-produced
        pipeline outputs (observations, evidence, features, scores) using
        the existing graph/trace/budget implementations.  Nothing here
        touches the Report schema; failures degrade gracefully.
        """
        try:
            from predictron_engine.reasoning.adaptive_budget import (
                compute_reasoning_budget,
            )
            from predictron_engine.reasoning.trace import build_reasoning_trace

            graph = getattr(self._reasoning, "last_contradiction_graph", None)

            # Contradiction graph (already built exactly once by reason()).
            if graph is not None:
                try:
                    debug.contradiction_graph = graph.to_dict()
                except AttributeError:
                    pass

            # Reasoning trace, reusing the already-built graph so it is
            # never recomputed.
            trace = build_reasoning_trace(
                report.observations,
                report.evidence,
                report.scores,
                contradiction_graph=graph,
            )
            debug.reasoning_trace = trace.to_dict()

            # Adaptive-budget diagnostics (reuses existing computation on
            # the produced features/evidence; never changes the pipeline).
            budget = compute_reasoning_budget(report.features, report.evidence)
            debug.reasoning_budget = {
                "budget_fraction": budget.budget_fraction,
                "max_rules": budget.max_rules,
                "skipped_rules": list(budget.skipped_rules),
                "rules_executed": budget.rules_executed,
                "savings_fraction": budget.savings_fraction,
                "confidence_estimate": budget.confidence_estimate,
                "impact_score": budget.impact_score,
                "rationale": budget.rationale,
            }
        except Exception:  # noqa: BLE001 - debug path must never raise
            logger.debug(
                "Failed to attach reasoning debug diagnostics",
                exc_info=True,
            )
