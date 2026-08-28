"""Sprint 9 — Regression detection.

Detects regressions between two collections of performance snapshots
(e.g. two engine versions or two time windows).  A regression is a metric
whose comparison mean is worse than the baseline mean by more than a
caller-supplied tolerance.

The existing ``benchmarks.benchmark_report.generate_regression_diff``
compares *case-by-case snapshot JSON*; this module operates on the
intelligence layer's :class:`PerformanceSnapshot` projections and is
genuinely additive (no duplication of the benchmark pipeline).
"""

from __future__ import annotations

from predictron_engine.intelligence.models import (
    PerformanceSnapshot,
    RegressionFinding,
)

_TRACKED_METRICS: tuple[str, ...] = (
    "overall_score",
    "overall_confidence",
    "decision_confidence",
    "data_completeness",
)


def _mean(records: list[PerformanceSnapshot], metric: str) -> float:
    values = [getattr(r, metric) for r in records]
    if not values:
        return 0.0
    return round(float(sum(values)) / len(values), 4)


def detect_regressions(
    snapshots_a: list[PerformanceSnapshot],
    snapshots_b: list[PerformanceSnapshot],
    *,
    tolerance: float = 0.01,
) -> list[RegressionFinding]:
    """Return regression findings where the comparison is worse.

    ``tolerance`` is the absolute threshold (in the metric's own units)
    below which a change is considered noise.  Only metrics that
    *decreased* by more than ``tolerance`` are reported as regressions.
    """
    findings: list[RegressionFinding] = []
    for metric in _TRACKED_METRICS:
        a_mean = _mean(snapshots_a, metric)
        b_mean = _mean(snapshots_b, metric)
        delta = round(b_mean - a_mean, 4)
        if a_mean == 0.0 and b_mean == 0.0:
            continue
        if b_mean < a_mean - tolerance:
            findings.append(
                RegressionFinding(
                    metric=metric,
                    from_value=a_mean,
                    to_value=b_mean,
                    delta=delta,
                )
            )
    return findings
