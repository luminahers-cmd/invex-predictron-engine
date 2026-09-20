"""Population orchestrator (Project V4).

The production workflow that populates the Predictron historical
dataset with real companies.  It is a thin orchestration layer over
the existing acquisition framework — it does not replace or redesign
anything.

The orchestrator:

  1. iterates over configured sources in order
  2. acquires records via :class:`AcquisitionManager`
  3. validates each record (import-time and field-level)
  4. deduplicates against the existing store
  5. imports into the :class:`DatasetStore`
  6. records provenance and checkpoint progress
  7. resumes after interruption without re-importing completed records

Everything here is additive and deterministic.  No engine, benchmark,
API, or scoring behavior is modified.  The acquisition framework is
reused as-is.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from predictron_engine.dataset.acquisition.pipeline import (
    AcquisitionPipeline,
    AcquisitionResult,
)
from predictron_engine.dataset.acquisition.sources.registry import (
    SourceConnectorRegistry,
)
from predictron_engine.dataset.acquisition.state import AcquisitionStateManager
from predictron_engine.dataset.imports import RawImportRecord
from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.population_config import (
    PopulationConfig,
    SourceConfig,
    build_default_config,
)
from predictron_engine.dataset.population_metrics import load_import_history
from predictron_engine.dataset.population_report import (
    PopulationReport,
    build_population_report,
)
from predictron_engine.dataset.quality import generate_quality_report
from predictron_engine.dataset.store import DatasetStore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class PopulateOptions:
    """Options for a population run."""

    source_names: list[str] = field(default_factory=list)
    all_sources: bool = False
    resume: bool = False
    limit: int | None = None
    dry_run: bool = False
    report_path: str | None = None
    since: datetime | None = None
    idempotent: bool = True
    batch_size: int = 1000
    checkpoint_every: int = 1
    config: PopulationConfig | None = None


class PopulationOrchestrator:
    """Orchestrates dataset population across configured sources.

    Parameters
    ----------
    store :
        The destination :class:`DatasetStore`.
    config :
        The population configuration.  Defaults to
        :func:`build_default_config` with all registered sources.
    """

    def __init__(
        self,
        store: DatasetStore,
        config: PopulationConfig | None = None,
    ) -> None:
        self._store = store
        self._config = config or build_default_config()
        self._registry = SourceConnectorRegistry.default()

    @property
    def config(self) -> PopulationConfig:
        """The active population configuration."""
        return self._config

    def populate(self, options: PopulateOptions | None = None) -> PopulationReport:
        """Run a population pass and return a report.

        Parameters
        ----------
        options :
            Options controlling this run's sources, resume behaviour,
            limit, and reporting.

        Returns
        -------
        A :class:`PopulationReport` summarising the run.
        """
        options = options or PopulateOptions()
        start = time.monotonic()
        start_record_count = self._store.count_records()
        self._store.initialize()

        source_configs = self._select_sources(options)
        results: list[AcquisitionResult] = []
        duplicate_count = 0
        run_id = _default_run_id()

        for source_config in source_configs:
            result, dupes = self._populate_source(source_config, options)
            results.append(result)
            duplicate_count += dupes

        elapsed = time.monotonic() - start

        history = load_import_history(self._store._root)

        report = build_population_report(
            results,
            store=self._store,
            start_record_count=start_record_count,
            elapsed_seconds=elapsed,
            duplicate_count=duplicate_count,
            run_id=run_id,
            dry_run=options.dry_run,
            quality=generate_quality_report(self._store),
            history=history,
        )

        if options.report_path:
            report.write(options.report_path)
            logger.info("Population report written to %s", options.report_path)

        if report.counts["failed"] > 0 or not report.quality.get("clean", True):
            logger.warning("Population finished with issues; inspect the report")

        return report

    def _select_sources(
        self, options: PopulateOptions
    ) -> list[SourceConfig]:
        """Select the ordered source configurations for this run."""
        if options.source_names:
            selected = [
                self._config.get(name)
                for name in options.source_names
            ]
            return [s for s in selected if s is not None]

        if options.all_sources or not self._config.sources:
            return list(self._config.sources)

        # Default: every configured source (same as --all).
        return list(self._config.sources)

    def _populate_source(
        self,
        source_config: SourceConfig,
        options: PopulateOptions,
    ) -> tuple[AcquisitionResult, int]:
        """Populate a single configured source.

        Returns a tuple of (AcquisitionResult, duplicate_count).  The
        duplicate count is tracked by matching each raw record against
        the store before import.
        """
        connector = self._registry.get(source_config.name)
        if connector is None:
            return _error_result(source_config.name, "no matching source connector"), 0

        file_paths = self._resolve_files(connector, source_config, options)
        if not file_paths:
            return _error_result(source_config.name, "no files found"), 0

        if options.resume:
            return self._resume_source(source_config, connector, file_paths, options)

        pipeline = AcquisitionPipeline(
            self._store,
            batch_size=options.batch_size,
            checkpoint_every=options.checkpoint_every,
            dry_run=options.dry_run,
            limit=options.limit,
            since=options.since,
        )

        duplicate_count = self._count_duplicates(connector, file_paths, options.limit)

        result = pipeline.run_source_batch(
            connector,
            file_paths,
            idempotent=options.idempotent,
        )
        result.source_name = source_config.name

        return result, duplicate_count

    def _resume_source(
        self,
        source_config: SourceConfig,
        connector: Any,
        file_paths: list[str],
        options: PopulateOptions,
    ) -> tuple[AcquisitionResult, int]:
        """Resume an interrupted source at record granularity.

        Only files that are not fully imported and have no completed
        batch are processed.  Each raw record is matched against the
        store before import so completed records are never re-imported;
        provenance is preserved via the existing import pipeline.
        """
        from predictron_engine.dataset.acquisition.pipeline import (
            AcquisitionMetrics,
        )
        from predictron_engine.dataset.acquisition.state import AcquisitionBatch
        from predictron_engine.dataset.dedup import match_record_to_store
        from predictron_engine.dataset.imports import ImportPipeline

        state = AcquisitionStateManager(self._store._root)
        state.initialize()

        existing = self._load_existing_records()
        metrics = AcquisitionMetrics()
        metrics.total_files = len(file_paths)
        batch_ids: list[str] = []
        source_name = source_config.name or connector.descriptor.name

        for fp in file_paths:
            file_hash = connector.checkpoint(fp)
            if state.has_file_been_imported(file_hash):
                metrics.skipped_idempotent += 1
                continue

            raw_records = connector.normalize(fp)
            metrics.total_raw_records += len(raw_records)

            if options.dry_run:
                continue

            new_raw: list[RawImportRecord] = []
            duplicate_this_file = 0
            for raw in raw_records:
                if not raw.startup_name:
                    continue
                candidate = _raw_to_record(raw)
                matched, _method = match_record_to_store(candidate, existing)
                if matched is not None:
                    duplicate_this_file += 1
                    continue
                new_raw.append(raw)

            resume_source = _MemorySource(source_name, new_raw)
            pipeline = ImportPipeline(
                resume_source,
                track_provenance=True,
            )
            result = pipeline.run("memory://resume")
            for record in result.imported_records:
                self._store.save_record(record)
                existing.append(record)
            for outcome in result.imported_outcomes:
                outcome.verdict = outcome.derive_verdict()
                self._store.save_outcome(outcome)

            metrics.imported += result.records_imported
            metrics.failed += result.records_failed
            metrics.skipped += duplicate_this_file + len(result.validation_errors)

            if result.records_imported or duplicate_this_file:
                batch_id = _resume_batch_id(source_name)
                batch_ids.append(batch_id)
                state.record_batch(AcquisitionBatch(
                    batch_id=batch_id,
                    source_name=source_name,
                    file_path=fp,
                    file_hash=file_hash,
                    source_version=connector.descriptor.name,
                    record_count=result.records_imported,
                ))
            self._clear_file_checkpoint(state, source_name, file_hash)

        metrics.batch_ids = batch_ids
        result_outer = AcquisitionResult(
            metrics=metrics,
            dry_run=options.dry_run,
            source_name=source_config.name,
            files_processed=file_paths if options.dry_run else [],
        )
        return result_outer, 0

    @staticmethod
    def _clear_file_checkpoint(
        state: AcquisitionStateManager, source_name: str, file_hash: str
    ) -> None:
        for cp in state.list_checkpoints():
            if cp.source_name == source_name and cp.file_hash == file_hash:
                state.clear_checkpoint(cp.batch_id)

    def _load_existing_records(self) -> list[DatasetRecord]:
        loaded = [
            self._store.load_record(rid)
            for rid in self._store.list_records()
        ]
        return [r for r in loaded if r is not None]

    def _resolve_files(
        self,
        connector: Any,
        source_config: SourceConfig,
        options: PopulateOptions,
    ) -> list[str]:
        """Resolve file paths for a source from config + CLI options.

        Explicit ``file_paths`` are returned as-is (existing files only).
        ``directories`` are scanned for files the connector can consume.
        """
        files: list[str] = []
        for fp in source_config.file_paths:
            if Path(fp).exists():
                files.append(fp)

        for directory in source_config.directories:
            files.extend(self._discover_in_directory(connector, directory))

        # De-duplicate while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for fp in files:
            if fp not in seen:
                seen.add(fp)
                ordered.append(fp)
        return ordered

    def _discover_in_directory(
        self, connector: Any, directory: str
    ) -> list[str]:
        """Scan a directory using the connector's discovery logic."""
        d = Path(directory)
        if not d.is_dir():
            return []
        found = connector.discover({"directory": directory})
        if isinstance(found, list):
            return [str(f) for f in found]
        return []

    def _count_duplicates(
        self, connector: Any, file_paths: list[str], limit: int | None
    ) -> int:
        """Count records that would be rejected as duplicates.

        Uses the existing ``match_record_to_store`` logic against the
        store's current records.  This is deterministic and
        non-destructive.
        """
        from predictron_engine.dataset.dedup import match_record_to_store

        loaded = [
            self._store.load_record(rid)
            for rid in self._store.list_records()
        ]
        existing: list[DatasetRecord] = [r for r in loaded if r is not None]

        count = 0
        processed = 0
        for fp in file_paths:
            raw_records = connector.normalize(fp)
            for raw in raw_records:
                if limit is not None and processed >= limit:
                    break
                processed += 1
                if not raw.startup_name:
                    continue
                candidate = _raw_to_record(raw)
                matched, _method = match_record_to_store(candidate, existing)
                if matched is not None:
                    count += 1
        return count


def _raw_to_record(raw: Any) -> Any:
    """Build a DatasetRecord from a raw record for dedupe matching.

    This mirrors the acquisition pipeline's normalization so that
    duplicate detection against the store uses equivalent identity
    keys.  It never writes to the store.
    """
    from predictron_engine.dataset.imports import _coerce_float
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

    funding_str = str(raw.metadata.get("funding_stage_at_analysis", "unknown"))
    try:
        funding_stage = FundingStage(funding_str)
    except ValueError:
        funding_stage = FundingStage.UNKNOWN

    dimension_scores: dict[str, float] = {}
    ds = prediction_data.get("dimension_scores")
    if isinstance(ds, dict):
        for k, v in ds.items():
            if isinstance(v, int | float):
                dimension_scores[str(k)] = float(v)

    prediction = PredictionSummary(
        decision=decision,
        confidence=_coerce_float(prediction_data.get("confidence"), 0.0),
        composite_score=_coerce_float(
            prediction_data.get("composite_score"), 0.0
        ),
        dimension_scores=dimension_scores,
    )
    return DatasetRecord(
        startup_name=raw.startup_name,
        website=raw.website or "",
        analysis_date=raw.analysis_date or datetime.now(UTC),
        engine_version=raw.engine_version or "",
        prediction=prediction,
        funding_stage_at_analysis=funding_stage,
        analysis_metadata=raw.metadata,
        tags=raw.tags,
        source=raw.source,
    )


def _default_run_id() -> str:
    import time
    import uuid

    return f"pop-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _error_result(source_name: str, error: str) -> AcquisitionResult:
    from predictron_engine.dataset.acquisition.pipeline import (
        AcquisitionMetrics,
    )

    return AcquisitionResult(
        metrics=AcquisitionMetrics(errors=[error]),
        source_name=source_name,
    )


class _MemorySource:
    """In-memory ImportSource that yields pre-filtered raw records.

    Used during resume so the existing :class:`ImportPipeline` can
    normalize and attach provenance to a filtered subset without
    needing a file on disk.
    """

    def __init__(
        self, name: str, records: list[RawImportRecord] | None = None
    ) -> None:
        self._name = name
        self._records = records or []

    @property
    def source_name(self) -> str:
        return self._name

    def read(self, path: str) -> list[RawImportRecord]:  # noqa: ARG002
        return self._records

    def validate(self, record: RawImportRecord) -> list[str]:  # noqa: ARG002
        return []


def _resume_batch_id(source_name: str) -> str:
    import time
    import uuid

    return f"batch-resume-{int(time.time())}-{source_name}-{uuid.uuid4().hex[:4]}"
