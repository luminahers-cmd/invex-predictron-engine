"""Sprint 9 — Benchmark-driven evaluation.

Adapters that connect the existing benchmark framework
(``benchmarks.benchmark_metrics.BenchmarkMetrics`` and
``benchmarks.benchmark_validator.BenchmarkValidator``) to the
intelligence layer.  No benchmark logic is reimplemented — these are thin
wrappers plus report→snapshot projection for trend/drift analysis.

All functions are deterministic; the benchmark results are supplied by
the caller (via ``benchmarks.benchmark_runner.run_benchmark``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks.benchmark_metrics import BenchmarkMetrics, MetricsReport
from benchmarks.benchmark_runner import CaseResult
from benchmarks.benchmark_validator import (
    BenchmarkValidator,
    CaseValidationResult,
)
from predictron_engine.intelligence.history import snapshot_from_report
from predictron_engine.intelligence.models import PerformanceSnapshot


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """Aggregated benchmark evaluation for one set of case results.

    ``metrics`` is the reused :class:`MetricsReport` from
    :class:`BenchmarkMetrics`; ``validations`` reuses
    :class:`BenchmarkValidator` findings; ``snapshots`` are the
    *projections* of successful reports for trend/drift analysis.
    """

    metrics: MetricsReport
    validations: list[CaseValidationResult]
    snapshots: list[PerformanceSnapshot] = field(default_factory=list)

    @property
    def passed_cases(self) -> int:
        return sum(1 for v in self.validations if v.passed)

    @property
    def failed_cases(self) -> int:
        return sum(1 for v in self.validations if not v.passed)

    def metrics_dict(self) -> dict:
        return self.metrics.to_dict()


def evaluate_benchmark(
    results: list[CaseResult],
    *,
    validator: BenchmarkValidator | None = None,
) -> BenchmarkEvaluation:
    """Evaluate a set of benchmark case results using the existing tools.

    Parameters
    ----------
    results:
        Case results from ``benchmarks.benchmark_runner.run_benchmark``.
    validator:
        Reserved for a :class:`BenchmarkValidator` — kept for interface
        symmetry with :func:`evaluate_benchmark_with_validation`; it is not
        used by this metrics-only evaluation.

    Returns
    -------
    A :class:`BenchmarkEvaluation` reusing the existing metrics engine
    (validations are empty here).
    """
    metrics = BenchmarkMetrics().compute(results)
    snapshots = [
        snapshot_from_report(r.report, version=r.report.analysis_metadata.engine_version)
        for r in results
        if r.success and r.report is not None
    ]
    return BenchmarkEvaluation(
        metrics=metrics,
        validations=[],
        snapshots=snapshots,
    )


def evaluate_benchmark_with_validation(
    results: list[CaseResult],
    cases: list[dict],
    *,
    validator: BenchmarkValidator | None = None,
) -> BenchmarkEvaluation:
    """Evaluate benchmark results including case validation.

    This is a convenience that also runs the existing
    :class:`BenchmarkValidator` over :class:`CaseResult` objects.  It is
    additive to :func:`evaluate_benchmark`.
    """
    base = evaluate_benchmark(results)
    v = validator or BenchmarkValidator()
    validations = v.validate_all(cases, results)
    return BenchmarkEvaluation(
        metrics=base.metrics,
        validations=validations,
        snapshots=base.snapshots,
    )
