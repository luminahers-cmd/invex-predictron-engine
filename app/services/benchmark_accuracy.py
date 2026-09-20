"""Read-only benchmark accuracy service (Milestone V1.4).

Exposes the *latest recorded benchmark run* for the Ground Truth Evaluation
Platform as a typed, read-only view.  Everything is read from the append-only
benchmark history (``benchmarks/benchmark_history``): nothing is recomputed,
and metrics are surfaced from the run's stored metric block (reconstructed
via ``metrics_from_dict`` — absent numbers stay ``None``, never fabricated).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.schemas.evaluation import (
    BenchmarkAccuracyResponse,
    BenchmarkCompanyEntry,
    BenchmarkMetricsSummary,
)
from benchmarks.ground_truth_eval.history import DEFAULT_HISTORY_DIR, BenchmarkHistory
from benchmarks.ground_truth_eval.metrics import GroundTruthMetrics, metrics_from_dict


class BenchmarkAccuracyError(RuntimeError):
    """Raised when the benchmark history store misbehaves."""


class BenchmarkAccuracyService:
    """Latest-run accuracy reader over the append-only benchmark history."""

    def __init__(self, history_root: Path | str | None = None) -> None:
        self._root = Path(history_root) if history_root is not None else DEFAULT_HISTORY_DIR

    def latest(self) -> BenchmarkAccuracyResponse | None:
        """Return the accuracy view for the most recent benchmark run."""
        history = BenchmarkHistory(self._root)
        summaries = history.list_summaries()
        if not summaries:
            return None
        latest_summary = max(summaries, key=lambda s: (s.created_at, s.run_id))

        run = history.load(latest_summary.run_id)
        stored_block = history.load_metrics(latest_summary.run_id)
        metrics = metrics_from_dict(stored_block) if stored_block else None

        companies = [
            BenchmarkCompanyEntry(
                company_id=entry.company_id,
                success=entry.success,
                decision=entry.decision,
                overall_score=entry.overall_score,
                overall_confidence=entry.overall_confidence,
                error=entry.error,
            )
            for entry in sorted(run.entries, key=lambda e: e.company_id)
        ]
        return BenchmarkAccuracyResponse(
            run_id=run.run_id,
            dataset_name=run.dataset_name,
            benchmark_version=run.benchmark_version,
            engine_version=run.engine_version,
            run_created_at=run.created_at,
            dataset_hash=run.dataset_hash,
            result_hash=run.result_hash,
            entry_count=len(run.entries),
            successful_count=len(run.successful_entries),
            has_metrics=metrics is not None,
            metrics=_summarize(metrics) if metrics is not None else None,
            companies=companies,
            generated_at=datetime.now(UTC),
        )


def _summarize(metrics: GroundTruthMetrics) -> BenchmarkMetricsSummary:
    confusion = metrics.confusion
    calibration = metrics.calibration
    base_rates = metrics.base_rates
    return BenchmarkMetricsSummary(
        total=confusion.total,
        scoreable=confusion.scoreable,
        insufficient_ground_truth=max(confusion.total - confusion.scoreable, 0),
        true_positives=confusion.true_positives,
        true_negatives=confusion.true_negatives,
        false_positives=confusion.false_positives,
        false_negatives=confusion.false_negatives,
        accuracy=confusion.accuracy,
        precision=confusion.precision,
        recall=confusion.recall,
        specificity=confusion.specificity,
        f1=confusion.f1,
        f0_5=confusion.f05,
        balanced_accuracy=confusion.balanced_accuracy,
        false_positive_rate=confusion.false_positive_rate,
        false_negative_rate=confusion.false_negative_rate,
        base_rate_positive=base_rates.positive_rate,
        base_rate_negative=base_rates.negative_rate,
        roc_auc=metrics.roc_auc,
        average_precision=metrics.average_precision,
        brier_score=metrics.brier,
        expected_calibration_error=calibration.expected_calibration_error,
        overconfidence_detected=calibration.overconfidence_detected,
        coverage=metrics.coverage.scoreability,
    )


__all__ = [
    "BenchmarkAccuracyError",
    "BenchmarkAccuracyService",
]
