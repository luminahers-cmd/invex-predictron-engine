"""Population run reports (Project V4).

Generates a deterministic, serializable report for every population
run.  The report captures:

  - companies processed / imported / skipped / duplicated / failed
  - validation failures and failure reasons
  - source breakdown
  - processing time and throughput
  - quality findings

Everything except an explicit ``generated_at`` timestamp is derived
deterministically from the run's inputs, so identical runs (with a
pinned timestamp) produce identical reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from predictron_engine.dataset.acquisition.pipeline import AcquisitionResult
from predictron_engine.dataset.population_metrics import (
    compute_population_metrics,
)
from predictron_engine.dataset.quality import QualityReport, generate_quality_report
from predictron_engine.dataset.store import DatasetStore

_POPULATION_REPORT_VERSION = "1.0.0"
_POPULATION_REPORT_SCHEMA_VERSION = 1


@dataclass
class FailureReason:
    """A validation/import failure aggregated by reason string."""

    reason: str
    count: int = 1

    def to_dict(self) -> dict[str, object]:
        return {"reason": self.reason, "count": self.count}


@dataclass
class PopulationReport:
    """Deterministic, serializable report for a population run.

    Attributes
    ----------
    run_id :
        Stable identifier for this run.
    source_names :
        Sources included in this run.
    counts :
        Summary counts (processed, imported, skipped, duplicates,
        failed, validation failures).
    metrics :
        Population metrics for the run.
    source_breakdown :
        Per-source acquisition metrics dicts.
    failure_reasons :
        Aggregated failure reasons, sorted by count then reason.
    quality :
        Quality report for the affected records.
    dry_run :
        Whether this was a dry run.
    generated_at :
        Timestamp of report generation (pinned for determinism).
    """

    run_id: str = ""
    source_names: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    source_breakdown: dict[str, Any] = field(default_factory=dict)
    failure_reasons: list[dict[str, object]] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict view."""
        return {
            "report_type": "population_report",
            "report_version": _POPULATION_REPORT_VERSION,
            "schema_version": _POPULATION_REPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "source_names": list(self.source_names),
            "counts": dict(self.counts),
            "metrics": self.metrics,
            "source_breakdown": self.source_breakdown,
            "failure_reasons": self.failure_reasons,
            "quality": self.quality,
            "dry_run": self.dry_run,
            "generated_at": self.generated_at,
        }

    def to_json(self, indent: int = 2, *, default: Any = str) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=default)

    def write(self, path: str | Path) -> Path:
        """Write the report to a JSON file and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out


def build_population_report(
    results: list[AcquisitionResult],
    *,
    store: DatasetStore,
    start_record_count: int,
    elapsed_seconds: float,
    duplicate_count: int = 0,
    run_id: str = "",
    dry_run: bool = False,
    quality: QualityReport | None = None,
    history: list[dict[str, Any]] | None = None,
    generated_at: datetime | None = None,
) -> PopulationReport:
    """Build a population report from acquisition results.

    Parameters
    ----------
    results :
        Acquisition results to report on.
    store :
        The destination store (used for end counts and quality).
    start_record_count :
        Store record count before the run.
    elapsed_seconds :
        Wall-clock runtime of the run.
    duplicate_count :
        Number of records skipped as duplicates.
    run_id :
        Optional stable run identifier.
    dry_run :
        Whether this was a dry run.
    quality :
        Optional precomputed quality report.  Computed when omitted.
    history :
        Optional import history from prior runs.  Included in metrics.
    generated_at :
        Optional timestamp.  Defaults to UTC now; pass a value to make
        the output fully deterministic.
    """
    end_record_count = store.count_records()

    metrics = compute_population_metrics(
        results,
        start_record_count=start_record_count,
        end_record_count=end_record_count,
        elapsed_seconds=elapsed_seconds,
        duplicate_count=duplicate_count,
        history=history,
    )

    failure_reasons = _collect_failure_reasons(results)
    source_breakdown = {
        result.source_name or "unknown": result.to_dict()
        for result in results
    }

    if quality is None:
        quality = generate_quality_report(store)

    return PopulationReport(
        run_id=run_id or _default_run_id(),
        source_names=_ordered_source_names(results),
        counts={
            "processed": metrics.total_processed,
            "imported": metrics.total_imported,
            "skipped": metrics.total_skipped,
            "duplicates": duplicate_count,
            "failed": metrics.total_failed,
            "validation_failures": sum(
                m.failed + m.skipped for r in results for m in [r.metrics]
            ),
        },
        metrics=metrics.to_dict(),
        source_breakdown=source_breakdown,
        failure_reasons=[f.to_dict() for f in failure_reasons],
        quality=quality.to_dict(),
        dry_run=dry_run,
        generated_at=(generated_at or datetime.now(UTC)).isoformat(),
    )


def _ordered_source_names(results: list[AcquisitionResult]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for result in results:
        name = result.source_name or "unknown"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _collect_failure_reasons(
    results: list[AcquisitionResult],
) -> list[FailureReason]:
    """Aggregate failure reasons across all results, deterministically."""
    counts: dict[str, int] = {}
    for result in results:
        for error in result.metrics.errors:
            reason = error or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
    # Deterministic: sort by count (desc) then reason (asc).
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [FailureReason(reason=r, count=c) for r, c in ordered]


def _default_run_id() -> str:
    import time
    import uuid

    return f"pop-{int(time.time())}-{uuid.uuid4().hex[:8]}"
