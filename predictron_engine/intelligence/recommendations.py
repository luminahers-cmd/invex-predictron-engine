"""Sprint 9 — Recommendation effectiveness metrics.

Computes effectiveness analytics over existing :class:`Recommendation`
objects (from ``predictron_engine.models.report``).  These are aggregate
views — priority/action coverage, confidence behaviour, and how
recommendation confidence tracks the overall confidence.  No new
recommendation logic is introduced; the existing recommendation list is
consumed as-is.
"""

from __future__ import annotations

from predictron_engine.intelligence.models import RecommendationEffectiveness
from predictron_engine.models.report import Report


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Deterministic Pearson correlation, with zero-variance fallback."""
    n = len(xs)
    if n == 0:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    if var_x == 0 or var_y == 0 or denom == 0:
        return 0.0
    return round(float(max(-1.0, min(1.0, cov / denom))), 4)


def recommendation_effectiveness(
    reports: list[Report],
) -> RecommendationEffectiveness:
    """Compute recommendation effectiveness across a collection of reports.

    ``confidence_calibration`` measures how recommendation confidence
    correlates with each report's overall calibrated confidence (across
    all reports, treating every recommendation as a sample).  A positive
    value indicates higher-confidence recommendations accompany
    higher-confidence analyses.
    """
    total = 0
    with_priority = 0
    with_action = 0
    confidences: list[float] = []
    report_confs: list[float] = []
    categories: set[str] = set()

    for report in reports:
        overall_conf = report.overall_confidence
        for rec in report.recommendations:
            total += 1
            if rec.priority:
                with_priority += 1
            if rec.action:
                with_action += 1
            categories.add(rec.category)
            confidences.append(rec.confidence)
            report_confs.append(overall_conf)

    avg_confidence = (
        round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    )
    calibration = _pearson(confidences, report_confs)

    return RecommendationEffectiveness(
        total_recommendations=total,
        categories=sorted(categories),
        priority_coverage=round(with_priority / total, 4) if total else 0.0,
        action_coverage=round(with_action / total, 4) if total else 0.0,
        avg_confidence=avg_confidence,
        confidence_calibration=calibration,
    )
