"""Support helpers for the analysis pipeline (Part B).

Contains request-builder and Report-extraction helpers used by
``analysis.AnalysisPipeline``.  Kept as a separate module so the heavy
``app.schemas`` import does not pollute the pipeline module and the pure
extraction functions can be unit-tested directly.
"""

from __future__ import annotations

from typing import Any

from app.schemas.analysis import StartupAnalysisRequest
from predictron_engine.dataset.models import DatasetRecord, PredictionSummary


def build_request(record: DatasetRecord) -> StartupAnalysisRequest:
    """Build a StartupAnalysisRequest from a DatasetRecord.

    Uses ``model_construct`` so historical records whose website is not a
    strictly-valid URL (or is absent) still produce a request.  The
    engine runs offline, best-effort evidence collection and handles a
    missing website gracefully.
    """
    description = record.analysis_metadata.get(
        "description"
    ) or _default_description(record)
    return StartupAnalysisRequest.model_construct(
        startup_name=record.startup_name,
        description=description,
        website=record.website or None,
    )


def _default_description(record: DatasetRecord) -> str:
    return (
        f"{record.startup_name} operates at "
        f"{record.website or 'an unreported website'}. "
        "No additional description was provided for this historical record."
    )


def extract_prediction_summary(report: Any) -> PredictionSummary:
    """Extract a PredictionSummary from an engine Report.

    Uses only existing Report fields.  Falls back to conservative
    defaults for optional fields; the decision, confidence, and
    composite score are always sourced from the report when present.
    """
    from predictron_engine.dataset.models import DecisionLabel

    decision_label = _decision_label(report, DecisionLabel)
    confidence = _overall_confidence(report)
    composite_score = _composite_score(report)
    dimension_scores = _dimension_scores(report)
    readiness = _readiness_score(report)
    recommendation_count = len(getattr(report, "recommendations", []))

    return PredictionSummary(
        decision=decision_label,
        confidence=confidence,
        composite_score=composite_score,
        dimension_scores=dimension_scores,
        investment_readiness_score=readiness,
        recommendation_count=recommendation_count,
    )


def _decision_label(
    report: Any, decision_label_type: type[Any]
) -> Any:
    decision = getattr(report, "investment_decision", None)
    if decision is None:
        return decision_label_type.WATCH
    category = getattr(decision, "category", None)
    if category is None:
        return decision_label_type.WATCH
    category_str = category.value if hasattr(category, "value") else str(category)
    mapping = {
        "strong_invest": decision_label_type.STRONG_INVEST,
        "invest": decision_label_type.INVEST,
        "watch": decision_label_type.WATCH,
        "investigate_further": decision_label_type.INVESTIGATE_FURTHER,
        "pass": decision_label_type.PASS,
    }
    return mapping.get(category_str, decision_label_type.WATCH)


def _overall_confidence(report: Any) -> float:
    value = getattr(report, "overall_confidence", None)
    if value is not None:
        return float(value)
    return 0.0


def _composite_score(report: Any) -> float:
    value = getattr(report, "overall_score", None)
    if value is not None:
        return float(value)
    decision = getattr(report, "investment_decision", None)
    if decision is not None:
        composite = getattr(decision, "composite_score", None)
        if composite is not None:
            return float(composite)
    return 0.0


def _dimension_scores(report: Any) -> dict[str, float]:
    scores = getattr(report, "scores", None)
    if not scores:
        return {}
    result: dict[str, float] = {}
    for score in scores:
        dimension = getattr(score, "dimension", None)
        value = getattr(score, "score", None)
        if dimension is not None and value is not None:
            result[str(dimension)] = float(value)
    return result


def _readiness_score(report: Any) -> float | None:
    readiness = getattr(report, "investment_readiness", None)
    if readiness is None:
        return None
    value = getattr(readiness, "readiness_score", None)
    if value is None:
        return None
    return float(value)


def report_summary(report: Any) -> dict[str, Any]:
    """Compact digest of a Report for provenance/reporting."""
    decision = getattr(report, "investment_decision", None)
    decision_confidence = None
    if decision is not None:
        composite = getattr(decision, "composite_score", None)
        if composite is not None:
            decision_confidence = float(composite)
    return {
        "overall_score": getattr(report, "overall_score", None),
        "overall_confidence": getattr(report, "overall_confidence", None),
        "recommendation_count": len(getattr(report, "recommendations", [])),
        "decision_category": (
            getattr(decision, "category", None) if decision is not None else None
        ),
        "decision_confidence": decision_confidence,
    }
