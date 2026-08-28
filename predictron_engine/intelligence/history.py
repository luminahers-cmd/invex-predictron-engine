"""Sprint 9 — Historical performance tracking.

Encodes :class:`Report` objects into lightweight :class:`PerformanceSnapshot`
records and computes :class:`PerformanceTrend` analytics over ordered
collections (ordered by ``version``).  Purely deterministic and pure — the
caller supplies the records; no global state or hidden caches are used.
"""

from __future__ import annotations

from predictron_engine.intelligence.models import (
    PerformanceSnapshot,
    PerformanceTrend,
    TrendDirection,
)
from predictron_engine.models.report import (
    ConfidenceAssessment,
    Report,
    ScoreResult,
)
from predictron_engine.version import ENGINE_VERSION


def _overall(records: list[PerformanceSnapshot], metric: str) -> float:
    """Return the population mean of a snapshot metric, or 0.0 when empty."""
    values = [getattr(r, metric) for r in records]
    if not values:
        return 0.0
    return round(float(sum(values)) / len(values), 4)


def _direction(first: float, last: float) -> TrendDirection:
    """Deterministic trend direction given endpoints."""
    if first == last:
        return TrendDirection.FLAT
    return TrendDirection.UP if last > first else TrendDirection.DOWN


def snapshot_from_report(
    report: Report,
    *,
    version: str | None = None,
) -> PerformanceSnapshot:
    """Encode a single pipeline ``Report`` into a ``PerformanceSnapshot``.

    This is the only place pipeline outputs are projected into the
    analytics layer.  It derives scalars and per-dimension maps from the
    existing report fields without recomputing anything.
    """
    version = version or ENGINE_VERSION
    decision_confidence = report.decision_confidence

    rule_hits: dict[str, int] = {}
    for obs in report.observations:
        rule_hits[obs.source_rule] = rule_hits.get(obs.source_rule, 0) + 1

    dimension_scores: dict[str, float] = {
        s.dimension: s.score for s in report.scores
    }
    assessed_confidences: dict[str, float] = {
        a.dimension: a.confidence for a in report.dimension_assessments
    }

    return PerformanceSnapshot(
        version=version,
        overall_score=report.overall_score,
        overall_confidence=report.overall_confidence,
        data_completeness=report.features.data_completeness,
        decision_confidence=(
            decision_confidence.confidence if decision_confidence else 0.0
        ),
        uncertainty_score=(
            decision_confidence.uncertainty_score if decision_confidence else 0.0
        ),
        observation_count=len(report.observations),
        evidence_count=len(report.evidence),
        recommendation_count=len(report.recommendations),
        assessment_count=len(report.dimension_assessments),
        dimension_scores=dimension_scores,
        assessed_confidences=assessed_confidences,
        rule_hits=rule_hits,
    )


def snapshot_from_confidence(
    overall_score: float,
    overall_confidence: float,
    confidence_assessments: list[ConfidenceAssessment],
    scores: list[ScoreResult],
    *,
    version: str | None = None,
    decision_confidence: float = 0.0,
    uncertainty_score: float = 0.0,
    data_completeness: float = 0.0,
    observation_count: int = 0,
    evidence_count: int = 0,
) -> PerformanceSnapshot:
    """Build a snapshot from scalars, for callers without a full Report.

    Accepts only primitive values and lightweight model lists, mirroring
    the ``snapshot_from_report`` projection.  Missing data defaults to
    zero so the snapshot remains constructible and backward compatible.
    """
    version = version or ENGINE_VERSION
    dimension_scores = {s.dimension: s.score for s in scores}
    assessed_confidences = {
        c.dimension: c.confidence for c in confidence_assessments
    }
    return PerformanceSnapshot(
        version=version,
        overall_score=overall_score,
        overall_confidence=overall_confidence,
        data_completeness=data_completeness,
        decision_confidence=decision_confidence,
        uncertainty_score=uncertainty_score,
        observation_count=observation_count,
        evidence_count=evidence_count,
        assessment_count=len(confidence_assessments),
        dimension_scores=dimension_scores,
        assessed_confidences=assessed_confidences,
        rule_hits={},
    )


def compute_trend(records: list[PerformanceSnapshot]) -> PerformanceTrend:
    """Compute a :class:`PerformanceTrend` from an ordered snapshot list.

    Records are returned ordered by ``version`` (the chronological/engine
    ordering encoded by the caller) for deterministic, meaningful trends.
    """
    ordered = sorted(records, key=lambda r: r.version)

    metric_trends: dict[str, TrendDirection] = {}
    if ordered:
        for metric in (
            "overall_score",
            "overall_confidence",
            "decision_confidence",
            "data_completeness",
        ):
            vals = [getattr(r, metric) for r in ordered]
            metric_trends[metric] = _direction(vals[0], vals[-1])

    return PerformanceTrend(
        records=ordered,
        metric_trends=metric_trends,
        mean_overall_score=_overall(ordered, "overall_score"),
        mean_overall_confidence=_overall(ordered, "overall_confidence"),
        mean_decision_confidence=_overall(ordered, "decision_confidence"),
        mean_data_completeness=_overall(ordered, "data_completeness"),
        record_count=len(ordered),
    )
