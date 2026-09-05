"""Acquisition pipeline with batching, retry, rate limiting, and progress.

Processes RawImportRecords through the existing ImportPipeline in
configurable batches, with checkpointing, retry logic, and progress
tracking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from predictron_engine.dataset.acquisition.state import (
    AcquisitionBatch,
    AcquisitionStateManager,
    CheckpointData,
)
from predictron_engine.dataset.imports import (
    ImportPipeline,
    ImportResult,
    RawImportRecord,
)
from predictron_engine.dataset.store import DatasetStore

logger = logging.getLogger(__name__)


@dataclass
class RetryPolicy:
    """Configuration for retry behaviour."""

    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 30.0
    retry_on: tuple[type[Exception], ...] = (OSError, TimeoutError)


@dataclass
class AcquisitionMetrics:
    """Metrics collected during an acquisition run."""

    total_files: int = 0
    total_raw_records: int = 0
    imported: int = 0
    failed: int = 0
    skipped: int = 0
    skipped_idempotent: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    batch_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_raw_records": self.total_raw_records,
            "imported": self.imported,
            "failed": self.failed,
            "skipped": self.skipped,
            "skipped_idempotent": self.skipped_idempotent,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "error_count": len(self.errors),
            "batch_ids": self.batch_ids,
        }


@dataclass
class AcquisitionResult:
    """Final result of an acquisition run."""

    metrics: AcquisitionMetrics
    dry_run: bool = False
    source_name: str = ""
    files_processed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metrics.to_dict(),
            "dry_run": self.dry_run,
            "source_name": self.source_name,
            "files_processed": self.files_processed,
        }


class AcquisitionPipeline:
    """Processes acquisition source data through batching and retry.

    This pipeline reads RawImportRecords from a source connector,
    processes them through the existing ImportPipeline in batches,
    deduplicates against the store, and persists results with
    checkpointing.
    """

    def __init__(
        self,
        store: DatasetStore,
        *,
        batch_size: int = 1000,
        checkpoint_every: int = 1,
        retry_policy: RetryPolicy | None = None,
        dry_run: bool = False,
        limit: int | None = None,
        since: datetime | None = None,
    ) -> None:
        self._store = store
        self._batch_size = batch_size
        self._checkpoint_every = max(1, checkpoint_every)
        self._retry_policy = retry_policy or RetryPolicy()
        self._dry_run = dry_run
        self._limit = limit
        self._since = since
        self._state_mgr = AcquisitionStateManager(store._root)

    def run_source(
        self,
        connector: Any,
        file_path: str,
        *,
        idempotent: bool = True,
    ) -> AcquisitionResult:
        """Run acquisition for a single source file.

        Parameters
        ----------
        connector :
            A BaseSource connector instance.
        file_path :
            Path to the data file to process.
        idempotent :
            If True, skip files that have already been imported.
        """
        start_time = time.monotonic()
        metrics = AcquisitionMetrics()
        source_name = connector.descriptor.name

        checkpoint_key = connector.checkpoint(file_path)
        if idempotent and self._state_mgr.has_file_been_imported(checkpoint_key):
            metrics.skipped_idempotent = 1
            metrics.elapsed_seconds = time.monotonic() - start_time
            logger.info("Skipping already-imported file: %s", file_path)
            return AcquisitionResult(
                metrics=metrics,
                dry_run=self._dry_run,
                source_name=source_name,
            )

        raw_records = self._fetch_with_retry(connector.normalize, file_path)
        metrics.total_raw_records = len(raw_records)
        metrics.total_files = 1

        if self._since is not None:
            raw_records = [
                r
                for r in raw_records
                if r.analysis_date is None or r.analysis_date >= self._since
            ]

        if self._limit is not None:
            raw_records = raw_records[: self._limit]

        if self._dry_run:
            metrics.elapsed_seconds = time.monotonic() - start_time
            return AcquisitionResult(
                metrics=metrics,
                dry_run=True,
                source_name=source_name,
                files_processed=[file_path],
            )

        import_pipeline = ImportPipeline(
            _PassthroughSource(source_name),
            track_provenance=True,
        )

        batch_id = f"batch-{int(time.time())}-{source_name}"
        checkpoint = CheckpointData(
            batch_id=batch_id,
            source_name=source_name,
            file_path=file_path,
            file_hash=checkpoint_key,
            total_records=len(raw_records),
        )

        self._state_mgr.initialize()

        for batch_start in range(0, len(raw_records), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(raw_records))
            batch_raw = raw_records[batch_start:batch_end]

            result = self._process_batch(
                batch_raw, import_pipeline, source_name
            )
            metrics.imported += result.records_imported
            metrics.failed += result.records_failed
            metrics.skipped += len(result.validation_errors)

            checkpoint.advance(
                imported=result.records_imported,
                failed=result.records_failed,
                skipped=len(result.validation_errors),
            )

            if (batch_start // self._batch_size) % self._checkpoint_every == 0:
                self._state_mgr.save_checkpoint(checkpoint)

        self._state_mgr.clear_checkpoint(batch_id)

        batch = AcquisitionBatch(
            batch_id=batch_id,
            source_name=source_name,
            file_path=file_path,
            file_hash=checkpoint_key,
            source_version=connector.descriptor.name,
            record_count=metrics.imported,
        )
        self._state_mgr.record_batch(batch)

        metrics.elapsed_seconds = time.monotonic() - start_time
        metrics.batch_ids.append(batch_id)

        logger.info(
            "Acquisition complete: %d imported, %d failed, %d skipped",
            metrics.imported,
            metrics.failed,
            metrics.skipped,
        )

        return AcquisitionResult(
            metrics=metrics,
            dry_run=self._dry_run,
            source_name=source_name,
            files_processed=[file_path],
        )

    def run_source_batch(
        self,
        connector: Any,
        file_paths: list[str],
        *,
        idempotent: bool = True,
    ) -> AcquisitionResult:
        """Run acquisition for multiple files from the same source."""
        start_time = time.monotonic()
        combined = AcquisitionMetrics()
        all_files: list[str] = []
        source_name = connector.descriptor.name

        for fp in file_paths:
            result = self.run_source(
                connector, fp, idempotent=idempotent
            )
            combined.total_files += result.metrics.total_files
            combined.total_raw_records += result.metrics.total_raw_records
            combined.imported += result.metrics.imported
            combined.failed += result.metrics.failed
            combined.skipped += result.metrics.skipped
            combined.skipped_idempotent += result.metrics.skipped_idempotent
            combined.errors.extend(result.metrics.errors)
            combined.batch_ids.extend(result.metrics.batch_ids)
            all_files.extend(result.files_processed)

        combined.elapsed_seconds = time.monotonic() - start_time
        return AcquisitionResult(
            metrics=combined,
            dry_run=self._dry_run,
            source_name=source_name,
            files_processed=all_files,
        )

    def _process_batch(
        self,
        raw_records: list[RawImportRecord],
        import_pipeline: ImportPipeline,
        source_name: str,
    ) -> ImportResult:
        """Process a batch of raw records through the import pipeline.

        Mirrors :meth:`ImportPipeline.run` including field-level
        provenance attachment, so records acquired here carry the same
        audit trail as records imported directly.
        """
        from predictron_engine.dataset.provenance import ProvenanceTracker

        result = ImportResult()
        tracker = ProvenanceTracker()

        for raw in raw_records:
            try:
                errors = _SOURCE_ADAPTER.validate(raw) if _SOURCE_ADAPTER else []
                if not errors:
                    errors = _validate_raw(raw)
            except Exception:  # noqa: BLE001
                errors = [f"validation error: {raw.startup_name}"]

            if errors:
                result.records_failed += 1
                result.validation_errors.append(
                    (result.records_failed, errors)
                )
                continue

            dataset_record = _normalize_dataset(raw, source_name)
            dataset_record = _attach_provenance(
                tracker, raw, dataset_record, source_name
            )
            outcome_record = _normalize_outcome(raw, dataset_record.record_id)

            self._store.save_record(dataset_record)
            outcome_record.verdict = outcome_record.derive_verdict()
            self._store.save_outcome(outcome_record)

            result.imported_records.append(dataset_record)
            result.imported_outcomes.append(outcome_record)
            result.records_imported += 1

        return result

    def _fetch_with_retry(
        self, fn: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a function with retry logic and exponential backoff."""
        policy = self._retry_policy
        last_error: Exception | None = None

        for attempt in range(policy.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except policy.retry_on as exc:
                last_error = exc
                if attempt < policy.max_retries:
                    delay = min(
                        policy.backoff_base * (2**attempt),
                        policy.backoff_max,
                    )
                    logger.warning(
                        "Attempt %d failed (%s), retrying in %.1fs",
                        attempt + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        if last_error is not None:
            raise last_error
        return None


class _PassthroughSource:
    """Minimal ImportSource that accepts raw records without validation."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def source_name(self) -> str:
        return self._name

    def read(self, path: str) -> list[RawImportRecord]:  # noqa: ARG002
        return []

    def validate(self, record: RawImportRecord) -> list[str]:  # noqa: ARG002
        return []


def _attach_provenance(
    tracker: Any,
    raw: RawImportRecord,
    record: Any,
    source_name: str,
) -> Any:
    """Attach field-level provenance to a normalized record.

    Uses the same ProvenanceTracker mechanics as
    :meth:`ImportPipeline._attach_provenance`.  When the pipeline
    assigns a default engine version (raw did not supply one), that
    value is attributed to the acquisition pipeline itself so the
    provenance trail stays complete and honest.
    """
    from predictron_engine.dataset.provenance import ProvenanceTracker

    if not isinstance(tracker, ProvenanceTracker):
        return record

    tracked_fields = (
        "startup_name",
        "website",
        "analysis_date",
        "engine_version",
        "benchmark_version",
    )
    provenance = tracker.build_provenance(raw, list(tracked_fields))

    engine_version = getattr(raw, "engine_version", "")
    record_engine_version = getattr(record, "engine_version", "")
    if not engine_version and record_engine_version:
        provenance["engine_version"] = {
            "source": f"{source_name}/pipeline",
            "retrieval_date": tracker.retrieval_date,
        }

    return tracker.attach_to_record(record, provenance)


_SOURCE_ADAPTER = _PassthroughSource("passthrough")


def _validate_raw(raw: RawImportRecord) -> list[str]:
    """Validate a raw record has minimum required fields."""
    errors: list[str] = []
    if not raw.startup_name:
        errors.append("startup_name is required")
    return errors


def _normalize_dataset(
    raw: RawImportRecord, source_name: str
) -> Any:
    """Normalize a raw record into a DatasetRecord."""
    from predictron_engine.dataset.models import (
        DatasetRecord,
        DecisionLabel,
        FundingStage,
        PredictionSummary,
    )

    prediction_data = raw.prediction_data
    decision_str = str(prediction_data.get("decision", "unknown"))
    try:
        decision = DecisionLabel(decision_str)
    except ValueError:
        decision = DecisionLabel.WATCH

    funding_stage_str = str(
        raw.metadata.get("funding_stage_at_analysis", "unknown")
    )
    try:
        funding_stage = FundingStage(funding_stage_str)
    except ValueError:
        funding_stage = FundingStage.UNKNOWN

    dimension_scores: dict[str, float] = {}
    ds = prediction_data.get("dimension_scores")
    if isinstance(ds, dict):
        for k, v in ds.items():
            if isinstance(v, int | float):
                dimension_scores[str(k)] = float(v)
            elif isinstance(v, str):
                try:
                    dimension_scores[str(k)] = float(v)
                except ValueError:
                    pass

    confidence = 0.0
    c = prediction_data.get("confidence")
    if isinstance(c, int | float):
        confidence = float(c)
    elif isinstance(c, str):
        try:
            confidence = float(c)
        except ValueError:
            pass

    composite = 0.0
    cs = prediction_data.get("composite_score")
    if isinstance(cs, int | float):
        composite = float(cs)
    elif isinstance(cs, str):
        try:
            composite = float(cs)
        except ValueError:
            pass

    irs = prediction_data.get("investment_readiness_score")
    investment_readiness: float | None = None
    if isinstance(irs, int | float):
        investment_readiness = float(irs)
    elif isinstance(irs, str):
        try:
            investment_readiness = float(irs)
        except ValueError:
            pass

    rc = prediction_data.get("recommendation_count", 0)
    recommendation_count = int(rc) if isinstance(rc, int | float) else 0

    prediction = PredictionSummary(
        decision=decision,
        confidence=confidence,
        composite_score=composite,
        dimension_scores=dimension_scores,
        investment_readiness_score=investment_readiness,
        recommendation_count=recommendation_count,
    )

    return DatasetRecord(
        startup_name=raw.startup_name,
        website=raw.website or "",
        analysis_date=raw.analysis_date or datetime.now(UTC),
        engine_version=raw.engine_version or "0.12.1",
        benchmark_version=raw.benchmark_version,
        prediction=prediction,
        funding_stage_at_analysis=funding_stage,
        analysis_metadata=raw.metadata,
        tags=raw.tags,
        source=source_name,
    )


def _normalize_outcome(raw: RawImportRecord, record_id: str) -> Any:
    """Normalize outcome data into an OutcomeRecord."""
    from predictron_engine.dataset.outcomes import (
        OutcomeRecord,
        OutcomeStatus,
        StartupOutcome,
    )

    outcome_data = raw.outcome_data
    status_str = str(outcome_data.get("status", "unknown"))
    try:
        status = OutcomeStatus(status_str)
    except ValueError:
        status = OutcomeStatus.UNKNOWN

    acquisition = outcome_data.get("acquisition")
    shutdown = outcome_data.get("shutdown")
    bankruptcy = outcome_data.get("bankruptcy")
    total_funding = outcome_data.get("total_funding_usd")
    investors_raw = outcome_data.get("investors")
    exit_type = outcome_data.get("exit_type")

    investors: list[str] = []
    if isinstance(investors_raw, list):
        investors = [str(i) for i in investors_raw]
    elif isinstance(investors_raw, str):
        investors = [investors_raw]

    return OutcomeRecord(
        record_id=record_id,
        outcome=StartupOutcome(
            acquisition=str(acquisition) if acquisition else None,
            shutdown=bool(shutdown) if shutdown is not None else None,
            bankruptcy=bool(bankruptcy) if bankruptcy is not None else None,
            total_funding_usd=(
                float(total_funding)
                if isinstance(total_funding, int | float)
                else None
            ),
            investors=investors,
            exit_type=str(exit_type) if exit_type else None,
            status=status,
        ),
    )
