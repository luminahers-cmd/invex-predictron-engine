"""Sprint 9 — IntelligenceAnalyzer composition root.

A thin, dependency-injected facade that wires together the pure Sprint 9
analytics modules into a single :class:`IntelligenceDashboard` (the
code-model "performance dashboard").  Every dependency is an injectable
callable; when omitted, the deterministic default is used.

This analyzer only *aggregates* analytics over caller-supplied data.  It
never mutates pipeline state and holds no global state.
"""

from __future__ import annotations

from collections.abc import Callable

from predictron_engine.intelligence.confidence_drift import (
    compute_confidence_drift,
)
from predictron_engine.intelligence.history import (
    compute_trend,
    snapshot_from_report,
)
from predictron_engine.intelligence.models import (
    ConfidenceDrift,
    EngineQualityMetrics,
    IntelligenceDashboard,
    PerformanceSnapshot,
    PerformanceTrend,
    RecommendationEffectiveness,
    RegressionFinding,
    RuleEffectiveness,
)
from predictron_engine.intelligence.quality import engine_quality
from predictron_engine.intelligence.recommendations import (
    recommendation_effectiveness,
)
from predictron_engine.intelligence.regression import detect_regressions
from predictron_engine.intelligence.rules import rule_effectiveness_from_reports
from predictron_engine.models.report import Report
from predictron_engine.version import ENGINE_VERSION


class IntelligenceAnalyzer:
    """Wires the adaptive-intelligence analytics together.

    Parameters
    ----------
    version:
        Engine version label stamped onto the dashboard.  Defaults to the
        engine version.
    trend_fn, regression_fn, rules_fn, recommendation_fn, drift_fn, quality_fn:
        Injectables for each analytics operation (dependency-injected
        design).  Defaults are the deterministic pure implementations.
    """

    def __init__(
        self,
        *,
        version: str | None = None,
        trend_fn: Callable[[list[PerformanceSnapshot]], PerformanceTrend]
        | None = None,
        regression_fn: Callable[
            [list[PerformanceSnapshot], list[PerformanceSnapshot]],
            list[RegressionFinding],
        ]
        | None = None,
        rules_fn: Callable[[list[Report]], list[RuleEffectiveness]] | None = None,
        recommendation_fn: Callable[
            [list[Report]], RecommendationEffectiveness
        ]
        | None = None,
        drift_fn: Callable[..., ConfidenceDrift] | None = None,
        quality_fn: Callable[..., EngineQualityMetrics] | None = None,
    ) -> None:
        self._version = version or ENGINE_VERSION
        self._trend_fn = trend_fn or compute_trend
        self._regression_fn = regression_fn or detect_regressions
        self._rules_fn = rules_fn or rule_effectiveness_from_reports
        self._recommendation_fn = (
            recommendation_fn or recommendation_effectiveness
        )
        self._drift_fn = drift_fn or compute_confidence_drift
        self._quality_fn = quality_fn or engine_quality

    def snapshot(self, report: Report) -> PerformanceSnapshot:
        """Project a single report into a performance snapshot."""
        return snapshot_from_report(report, version=self._version)

    def trend(
        self, records: list[PerformanceSnapshot]
    ) -> PerformanceTrend:
        return self._trend_fn(records)

    def regressions(
        self,
        baseline: list[PerformanceSnapshot],
        comparison: list[PerformanceSnapshot],
        *,
        tolerance: float = 0.01,
    ) -> list[RegressionFinding]:
        return self._regression_fn(baseline, comparison)

    def rules(self, reports: list[Report]) -> list[RuleEffectiveness]:
        return self._rules_fn(reports)

    def recommendations(
        self, reports: list[Report]
    ) -> RecommendationEffectiveness:
        return self._recommendation_fn(reports)

    def confidence_drift(
        self,
        baseline: list[PerformanceSnapshot],
        comparison: list[PerformanceSnapshot],
        *,
        overconfident: bool = False,
    ) -> ConfidenceDrift:
        return self._drift_fn(
            baseline,
            comparison,
            overconfident=overconfident,
        )

    def quality(
        self,
        reports: list[Report],
        *,
        baseline: list[PerformanceSnapshot] | None = None,
        comparison: list[PerformanceSnapshot] | None = None,
        overconfident: bool | None = None,
    ) -> EngineQualityMetrics:
        return self._quality_fn(
            reports,
            snapshots_a=baseline,
            snapshots_b=comparison,
            overconfident=overconfident,
        )

    def dashboard(
        self,
        *,
        records: list[PerformanceSnapshot] | None = None,
        reports: list[Report] | None = None,
        baseline: list[PerformanceSnapshot] | None = None,
        comparison: list[PerformanceSnapshot] | None = None,
        overconfident: bool | None = None,
        regression_tolerance: float = 0.01,
    ) -> IntelligenceDashboard:
        """Produce the full :class:`IntelligenceDashboard`.

        Parameters
        ----------
        records:
            Historical snapshots for the trend.  When None and ``reports``
            is provided, the reports are projected into snapshots first.
        reports:
            Reports to source rule/recommendation/quality analytics and
            (when ``records`` is None) the trend.
        baseline, comparison:
            Optional collections for regression and confidence-drift.
        overconfident:
            Overconfidence flag for drift/quality; delegated to the Sprint
            8 calibration monitor when None.
        regression_tolerance:
            Absolute threshold for regression detection.
        """
        snapshots = list(records) if records is not None else (
            [self.snapshot(r) for r in (reports or [])]
        )

        trend = self._trend_fn(snapshots)

        regressions: list[RegressionFinding] = []
        drift: ConfidenceDrift | None = None
        if baseline is not None and comparison is not None:
            regressions = self._regression_fn(baseline, comparison)
            drift = self._drift_fn(
                baseline,
                comparison,
                overconfident=bool(overconfident),
            )

        reports_list = reports or []

        quality = self._quality_fn(
            reports_list,
            snapshots_a=baseline,
            snapshots_b=comparison,
            overconfident=overconfident,
        )

        return IntelligenceDashboard(
            version=self._version,
            trend=trend,
            regressions=regressions,
            rule_effectiveness=self._rules_fn(reports_list),
            recommendation_effectiveness=self._recommendation_fn(reports_list),
            confidence_drift=drift,
            quality=quality,
        )
