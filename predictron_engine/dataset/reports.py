"""Dataverse JSON reports (Part E).

Generates deterministic JSON reports comprising:
  - dataset summary
  - coverage
  - outcome distribution
  - evaluation metrics
  - version metadata
  - timestamp
  - engine version
  - benchmark version

The report is produced from data already present in the store and the
aggregate metrics already computed; it never fabricates labels.  Apart
from an explicit ``generated_at`` timestamp, all fields are derived
deterministically from the underlying data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from predictron_engine.dataset.metrics import EvaluationMetrics
from predictron_engine.dataset.outcomes import OutcomeRecord, OutcomeVerdict
from predictron_engine.dataset.store import DatasetStore


class DatasetReportBuilder:
    """Builds a deterministic dataset report from a store."""

    def __init__(
        self,
        store: DatasetStore,
        engine_version: str,
        benchmark_version: str | None = None,
    ) -> None:
        self._store = store
        self._engine_version = engine_version
        self._benchmark_version = benchmark_version

    def build(
        self,
        metrics: EvaluationMetrics | None = None,
        generated_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Assemble the report dictionary.

        ``metrics`` may be supplied or computed from the store on demand.
        ``generated_at`` defaults to the current UTC time; pass an
        explicit value to make the output fully deterministic.
        """
        if metrics is None:
            metrics = self._compute_metrics()

        record_ids = self._store.list_records()
        outcome_ids = self._store.list_outcomes()
        evaluation_ids = self._store.list_evaluations()

        loaded_outcomes: list[OutcomeRecord | None] = [
            self._store.load_outcome(oid) for oid in outcome_ids
        ]
        outcomes = [o for o in loaded_outcomes if o is not None]
        outcome_distribution = _outcome_distribution(outcomes)

        return {
            "dataset_summary": {
                "record_count": len(record_ids),
                "outcome_count": len(outcome_ids),
                "evaluation_count": len(evaluation_ids),
                "startup_count": len(
                    self._store.find_distinct_startups()
                ),
                "with_outcome_count": sum(
                    1
                    for rid in record_ids
                    if self._store.find_outcome_by_record(rid) is not None
                ),
            },
            "coverage": {
                "outcome_coverage": _ratio(
                    len(outcome_ids), len(record_ids)
                ),
                "evaluation_coverage": _ratio(
                    len(evaluation_ids), len(record_ids)
                ),
                "scoreable_metric_coverage": metrics.coverage,
                "insufficient_ground_truth_rate": (
                    metrics.insufficient_ground_truth_rate
                ),
            },
            "outcome_distribution": outcome_distribution,
            "evaluation_metrics": metrics.summary(),
            "version_metadata": {
                "engine_version": self._engine_version,
                "benchmark_version": self._benchmark_version,
                "dataset_builder_version": _DATASET_BUILDER_VERSION,
                "report_schema_version": _REPORT_SCHEMA_VERSION,
            },
            "generated_at": (
                generated_at or datetime.now(UTC)
            ).isoformat(),
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the report to a JSON string."""
        import json

        return json.dumps(self.build(**kwargs), indent=2, default=str)

    def _compute_metrics(self) -> EvaluationMetrics:
        from predictron_engine.dataset.metrics import compute_evaluation_metrics

        evaluations = [
            self._store.load_evaluation(eid)
            for eid in self._store.list_evaluations()
        ]
        valid = [e for e in evaluations if e is not None]
        return compute_evaluation_metrics(valid)


_DATASET_BUILDER_VERSION = "2.0.0"
_REPORT_SCHEMA_VERSION = 1


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _outcome_distribution(
    outcomes: list[OutcomeRecord],
) -> dict[str, int]:
    """Count outcome verdicts across the provided outcome records.

    Never fabricates labels: records whose verdict is UNKNOWN are counted
    explicitly and separately.
    """
    distribution: dict[str, int] = {
        OutcomeVerdict.SUCCESS.value: 0,
        OutcomeVerdict.PARTIAL_SUCCESS.value: 0,
        OutcomeVerdict.FAILURE.value: 0,
        OutcomeVerdict.INCONCLUSIVE.value: 0,
        OutcomeVerdict.UNKNOWN.value: 0,
    }
    for outcome in outcomes:
        verdict = outcome.derive_verdict()
        distribution[verdict.value] = distribution.get(verdict.value, 0) + 1
    return distribution
