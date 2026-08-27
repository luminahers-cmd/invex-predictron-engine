"""Sprint 9 — Engine quality metrics.

Consolidates deterministic quality analytics over a collection of
:class:`Report` objects.

Delegation rules (no duplication):
* ``explainability_coverage``, ``recommendation_quality`` and
  ``data_completeness`` are **reused** from the existing
  :class:`BenchmarkMetrics` engine (the canonical implementation), by
  projecting each report into a lightweight ``CaseResult`` and reading the
  computed metric values.
* ``calibration_error`` / ``calibration_quality`` / ``overconfidence_detected``
  are **delegated** to the Sprint 8 module
  (``calibration_sprint8.build_calibration_report``).
* ``rule_effectiveness`` and ``confidence_drift`` are genuinely additive
  analytics computed by the intelligence layer.

This module adds no reimplementation of logic that already exists
elsewhere.
"""

from __future__ import annotations

from benchmarks.benchmark_metrics import BenchmarkMetrics
from benchmarks.benchmark_runner import CaseResult
from predictron_engine.decision.calibration_sprint8 import (
    build_calibration_report,
)
from predictron_engine.intelligence.confidence_drift import (
    compute_confidence_drift,
)
from predictron_engine.intelligence.models import (
    EngineQualityMetrics,
    PerformanceSnapshot,
)
from predictron_engine.intelligence.rules import rule_effectiveness_from_reports
from predictron_engine.models.report import Report


def _case_results(reports: list[Report]) -> list[CaseResult]:
    """Project reports into lightweight benchmark case results.

    This enables reuse of the canonical :class:`BenchmarkMetrics` engine
    for coverage/quality/completeness without running the whole pipeline.
    Only the fields the reused metrics read are populated.
    """
    return [
        CaseResult(
            case_id=f"quality-{i}",
            case_label="",
            request={},
            success=True,
            processing_time_ms=0.0,
            report=r,
        )
        for i, r in enumerate(reports)
    ]


def engine_quality(
    reports: list[Report],
    *,
    snapshots_a: list[PerformanceSnapshot] | None = None,
    snapshots_b: list[PerformanceSnapshot] | None = None,
    overconfident: bool | None = None,
) -> EngineQualityMetrics:
    """Compute consolidated engine-quality metrics for a report collection.

    Parameters
    ----------
    reports:
        The reports to analyse.
    snapshots_a, snapshots_b:
        Optional baseline/comparison snapshot collections for drift
        analysis.  When ``snapshots_a``/``snapshots_b`` are given, a
        :class:`ConfidenceDrift` is attached.
    overconfident:
        Overconfidence flag for the comparison run.  When None, it is
        taken from the Sprint 8 calibration monitor's report on ``reports``.
    """
    if not reports:
        confidence_values: list[float] = []
        completeness_values: list[float] = []
    else:
        confidence_values = [r.overall_confidence for r in reports]
        completeness_values = [r.features.data_completeness for r in reports]

    cal = build_calibration_report(confidence_values, completeness_values)

    bench = BenchmarkMetrics().compute(_case_results(reports))

    eff = rule_effectiveness_from_reports(reports)

    drift = None
    if snapshots_a is not None and snapshots_b is not None:
        oc = (
            cal.overconfidence_detected
            if overconfident is None
            else overconfident
        )
        drift = compute_confidence_drift(
            snapshots_a,
            snapshots_b,
            overconfident=oc,
        )

    return EngineQualityMetrics(
        report_count=len(reports),
        calibration_error=cal.expected_calibration_error,
        calibration_quality=cal.calibration_quality,
        overconfidence_detected=cal.overconfidence_detected,
        explainability_coverage=bench.by_name["explainability_coverage"].value,
        recommendation_quality=bench.by_name["recommendation_quality"].value,
        data_completeness=bench.by_name["extraction_completeness"].value,
        rule_effectiveness=eff,
        drift=drift,
    )
