"""Population run metrics (Project V4).

Tracks operational metrics for dataset population runs:

  - import rate (records per second)
  - records per hour
  - validation success rate
  - duplicate rate
  - source contribution
  - dataset growth
  - import history

These metrics are derived deterministically from the acquisition
results, the destination store, and the run's own timings.  They are
additive and never modify the acquisition framework or the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from predictron_engine.dataset.acquisition.pipeline import (
    AcquisitionMetrics,
    AcquisitionResult,
)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _rate(records: int, elapsed_seconds: float) -> float | None:
    if elapsed_seconds <= 0:
        return None
    return round(records / elapsed_seconds, 6)


@dataclass
class PopulationRunMetrics:
    """Metrics collected for a single population run.

    ``sources`` holds per-source contribution keyed by source name.
    ``history`` records the number of records imported by each prior
    completed run, in completion order.
    """

    total_processed: int = 0
    total_imported: int = 0
    total_skipped: int = 0
    total_duplicates: int = 0
    total_failed: int = 0
    elapsed_seconds: float = 0.0
    start_record_count: int = 0
    end_record_count: int = 0
    sources: dict[str, dict[str, int]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    duplicate_rate: float | None = None
    validation_success_rate: float | None = None

    def import_rate(self) -> float | None:
        """Import rate in records per second."""
        return _rate(self.total_imported, self.elapsed_seconds)

    def records_per_hour(self) -> float | None:
        """Throughput in records per hour."""
        rate = self.import_rate()
        if rate is None:
            return None
        return round(rate * 3600, 6)

    def dataset_growth(self) -> int:
        """Net growth in the store's record count during the run."""
        return max(0, self.end_record_count - self.start_record_count)

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict view."""
        return {
            "total_processed": self.total_processed,
            "total_imported": self.total_imported,
            "total_skipped": self.total_skipped,
            "total_duplicates": self.total_duplicates,
            "total_failed": self.total_failed,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "import_rate": self.import_rate(),
            "records_per_hour": self.records_per_hour(),
            "validation_success_rate": self.validation_success_rate,
            "duplicate_rate": self.duplicate_rate,
            "dataset_growth": self.dataset_growth(),
            "start_record_count": self.start_record_count,
            "end_record_count": self.end_record_count,
            "sources": self.sources,
            "import_history": self.history,
        }


def compute_population_metrics(
    results: list[AcquisitionResult],
    *,
    start_record_count: int,
    end_record_count: int,
    elapsed_seconds: float,
    duplicate_count: int = 0,
    sources: dict[str, dict[str, int]] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> PopulationRunMetrics:
    """Compute aggregate population metrics from acquisition results.

    Parameters
    ----------
    results :
        Acquisition results produced by the orchestrator.
    start_record_count :
        Store record count before the run.
    end_record_count :
        Store record count after the run.
    elapsed_seconds :
        Total wall-clock time of the run.
    duplicate_count :
        Number of records deduplicated (skipped as duplicates).
    sources :
        Optional override of per-source contribution.  When not
        supplied, it is derived from ``results``.
    history :
        Optional import history from prior runs.
    """
    total_processed = 0
    total_imported = 0
    total_skipped = 0
    total_failed = 0
    derived_sources: dict[str, dict[str, int]] = {}

    for result in results:
        m = result.metrics
        total_processed += m.total_raw_records
        total_imported += m.imported
        total_skipped += m.skipped
        total_failed += m.failed

        entry = derived_sources.setdefault(
            result.source_name or "unknown",
            {"processed": 0, "imported": 0, "skipped": 0, "failed": 0},
        )
        entry["processed"] += m.total_raw_records
        entry["imported"] += m.imported
        entry["skipped"] += m.skipped
        entry["failed"] += m.failed

    if sources is not None:
        derived_sources = sources

    total_processed = max(total_processed, total_imported + total_failed + total_skipped)

    return PopulationRunMetrics(
        total_processed=total_processed,
        total_imported=total_imported,
        total_skipped=total_skipped,
        total_duplicates=duplicate_count,
        total_failed=total_failed,
        elapsed_seconds=elapsed_seconds,
        start_record_count=start_record_count,
        end_record_count=end_record_count,
        sources=derived_sources,
        history=history or [],
        duplicate_rate=_ratio(duplicate_count, total_processed),
        validation_success_rate=_ratio(
            total_imported + duplicate_count, total_processed
        ),
    )


def load_import_history(store_root: str | Path) -> list[dict[str, Any]]:
    """Load import history from the acquisition state.

    Reads the completed acquisition batches recorded by the
    forestate state manager and returns them in completion order.
    """
    from predictron_engine.dataset.acquisition.state import (
        AcquisitionStateManager,
    )

    manager = AcquisitionStateManager(store_root)
    batches = manager.list_batches()
    history: list[dict[str, Any]] = []
    for batch in batches:
        history.append({
            "batch_id": batch.batch_id,
            "source_name": batch.source_name,
            "file_path": batch.file_path,
            "record_count": batch.record_count,
            "imported_at": batch.imported_at,
        })
    return history


def _aggregate_metrics(metrics: list[AcquisitionMetrics]) -> AcquisitionMetrics:
    """Combine a list of acquisition metrics into one aggregate."""
    combined = AcquisitionMetrics()
    for m in metrics:
        combined.total_files += m.total_files
        combined.total_raw_records += m.total_raw_records
        combined.imported += m.imported
        combined.failed += m.failed
        combined.skipped += m.skipped
        combined.skipped_idempotent += m.skipped_idempotent
        combined.errors.extend(m.errors)
        combined.batch_ids.extend(m.batch_ids)
    return combined
