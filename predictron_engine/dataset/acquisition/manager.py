"""Acquisition manager — top-level orchestrator.

Coordinates source discovery, fetching, caching, normalization, and
storage.  This is the primary entry point for acquisition operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from predictron_engine.dataset.acquisition.cache import AcquisitionCache
from predictron_engine.dataset.acquisition.pipeline import (
    AcquisitionMetrics,
    AcquisitionPipeline,
    AcquisitionResult,
    RetryPolicy,
)
from predictron_engine.dataset.acquisition.sources.base import BaseSource
from predictron_engine.dataset.acquisition.sources.registry import (
    SourceConnectorRegistry,
)
from predictron_engine.dataset.acquisition.state import AcquisitionStateManager
from predictron_engine.dataset.store import DatasetStore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class AcquireOptions:
    """Options for an acquisition run."""

    source_name: str | None = None
    file_path: str | None = None
    directory: str | None = None
    resume: bool = False
    since: datetime | None = None
    limit: int | None = None
    dry_run: bool = False
    idempotent: bool = True
    batch_size: int = 1000
    checkpoint_every: int = 1
    max_retries: int = 3


class AcquisitionManager:
    """Top-level orchestrator for data acquisition.

    Coordinates the full acquisition workflow:
    1. Discover available data from source connectors
    2. Fetch data files (with caching)
    3. Normalize via source-specific parsers
    4. Batch-process through the import pipeline
    5. Deduplicate against existing store
    6. Persist with checkpointing
    """

    def __init__(self, store: DatasetStore) -> None:
        self._store = store
        self._registry = SourceConnectorRegistry.default()
        self._cache = AcquisitionCache(store._root)
        self._state = AcquisitionStateManager(store._root)

    @property
    def registry(self) -> SourceConnectorRegistry:
        """Access the source connector registry."""
        return self._registry

    def list_sources(self) -> list[dict[str, object]]:
        """List all available source connectors."""
        return self._registry.list_descriptors()

    def acquire(self, options: AcquireOptions) -> AcquisitionResult:
        """Run an acquisition with the given options.

        This is the primary entry point for acquisition operations.
        """
        self._state.initialize()
        self._cache.initialize()

        connector = self._resolve_connector(options)
        if connector is None:
            return AcquisitionResult(
                metrics=AcquisitionMetrics(errors=["no matching source found"]),
                source_name=options.source_name or "",
            )

        file_paths = self._resolve_files(connector, options)
        if not file_paths:
            return AcquisitionResult(
                metrics=AcquisitionMetrics(errors=["no files found"]),
                source_name=connector.descriptor.name,
            )

        if options.resume:
            file_paths = self._filter_resumable(file_paths, connector)

        pipeline = AcquisitionPipeline(
            self._store,
            batch_size=options.batch_size,
            checkpoint_every=options.checkpoint_every,
            retry_policy=RetryPolicy(max_retries=options.max_retries),
            dry_run=options.dry_run,
            limit=options.limit,
            since=options.since,
        )

        result = pipeline.run_source_batch(
            connector,
            file_paths,
            idempotent=options.idempotent,
        )

        logger.info(
            "Acquisition complete for %s: %s",
            connector.descriptor.name,
            result.to_dict(),
        )
        return result

    def acquire_source(
        self, source_name: str, config: dict[str, object] | None = None
    ) -> AcquisitionResult:
        """Run acquisition for a named source with a config dict.

        Convenience method for programmatic access.
        """
        options = AcquireOptions(source_name=source_name)
        if config:
            options.file_path = str(config.get("file_path", ""))
            options.directory = str(config.get("directory", ""))
        return self.acquire(options)

    def resume(self, source_name: str | None = None) -> list[AcquisitionResult]:
        """Resume any incomplete acquisitions."""
        self._state.initialize()
        checkpoints = self._state.list_checkpoints()
        results: list[AcquisitionResult] = []

        for cp in checkpoints:
            if source_name and cp.source_name != source_name:
                continue
            connector = self._registry.get(cp.source_name)
            if connector is None:
                continue
            if not Path(cp.file_path).exists():
                continue

            pipeline = AcquisitionPipeline(
                self._store,
                dry_run=False,
            )
            result = pipeline.run_source(
                connector,
                cp.file_path,
                idempotent=False,
            )
            results.append(result)

        return results

    def status(self) -> dict[str, Any]:
        """Return the current acquisition status."""
        self._state.initialize()
        batches = self._state.list_batches()
        checkpoints = self._state.list_checkpoints()

        source_stats: dict[str, dict[str, Any]] = {}
        for batch in batches:
            sn = batch.source_name
            if sn not in source_stats:
                source_stats[sn] = {
                    "batch_count": 0,
                    "total_records": 0,
                    "last_imported": None,
                }
            source_stats[sn]["batch_count"] += 1
            source_stats[sn]["total_records"] += batch.record_count
            if batch.imported_at:
                source_stats[sn]["last_imported"] = batch.imported_at

        return {
            "record_count": self._store.count_records(),
            "outcome_count": self._store.count_outcomes(),
            "evaluation_count": self._store.count_evaluations(),
            "total_batches": len(batches),
            "pending_checkpoints": len(checkpoints),
            "sources": source_stats,
        }

    def _resolve_connector(
        self, options: AcquireOptions
    ) -> BaseSource | None:
        """Resolve the source connector from options."""
        if options.source_name:
            return self._registry.get(options.source_name)

        if options.file_path:
            lower = options.file_path.lower()
            if "sec_edgar" in lower or "edgar" in lower:
                return self._registry.get("sec_edgar")
            if "companies_house" in lower or "ch_" in lower:
                return self._registry.get("companies_house")
            if "yc_oss" in lower or "yc" in lower:
                return self._registry.get("yc_oss")
            return self._registry.get("csv_export")

        if options.directory:
            return self._registry.get("csv_export")

        return None

    def _resolve_files(
        self, connector: BaseSource, options: AcquireOptions
    ) -> list[str]:
        """Resolve the list of files to process."""
        config: dict[str, object] = {}
        if options.file_path:
            config["file_path"] = options.file_path
        if options.directory:
            config["directory"] = options.directory

        return connector.discover(config or None)

    def _filter_resumable(
        self, file_paths: list[str], connector: BaseSource
    ) -> list[str]:
        """Filter files to only those with active checkpoints."""
        filtered: list[str] = []
        for fp in file_paths:
            file_hash = connector.checkpoint(fp)
            existing = self._state.find_resume_checkpoint(
                connector.descriptor.name, file_hash
            )
            if existing is not None:
                filtered.append(fp)
            elif not self._state.has_file_been_imported(file_hash):
                filtered.append(fp)
        return filtered if filtered else file_paths
