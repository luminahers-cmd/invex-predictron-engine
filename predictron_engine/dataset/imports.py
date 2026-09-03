"""Pluggable import pipeline interfaces (Part C).

Defines the architecture for importing historical datasets from
external sources.  No scraping or downloading is implemented — only
the import interfaces and pipeline framework.

Supported sources are pluggable through the ImportSource protocol.
Each source adapter implements the protocol to parse its specific
format into the canonical DatasetRecord and OutcomeRecord models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.outcomes import OutcomeRecord


def _coerce_float(value: object, default: float) -> float:
    """Coerce an object to float with a fallback default."""
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_optional_float(value: object) -> float | None:
    """Coerce an object to float or None."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: object, default: int) -> int:
    """Coerce an object to int with a fallback default."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_optional_str(value: object) -> str | None:
    """Coerce an object to str or None."""
    if value is None:
        return None
    return str(value)


def _coerce_optional_bool(value: object) -> bool | None:
    """Coerce an object to bool or None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        return None
    return bool(value)


def _coerce_str_list(value: object) -> list[str]:
    """Coerce an object to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_dimension_scores(value: object) -> dict[str, float]:
    """Coerce an object to a dict of string-to-float."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for k, v in value.items():
        coerced = _coerce_optional_float(v)
        if coerced is not None:
            result[str(k)] = coerced
    return result


@dataclass
class RawImportRecord:
    """Intermediate representation from an import source.

    Import adapters parse their source format into this structure,
    which is then normalized into DatasetRecord / OutcomeRecord by
    the pipeline.
    """

    startup_name: str = ""
    website: str = ""
    analysis_date: datetime | None = None
    engine_version: str = ""
    benchmark_version: str | None = None
    prediction_data: dict[str, object] = field(default_factory=dict)
    outcome_data: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source: str = "unknown"


@runtime_checkable
class ImportSource(Protocol):
    """Protocol for pluggable data source adapters.

    Each adapter reads from a specific source format (JSON, CSV,
    database, API) and yields RawImportRecords that the pipeline
    normalizes.

    Implementations must NOT:
    - Scrape websites
    - Download data from the internet
    - Fabricate any data
    """

    @property
    def source_name(self) -> str:
        """Unique identifier for this import source."""
        ...

    def read(self, path: str) -> list[RawImportRecord]:
        """Read raw records from the given path.

        Parameters
        ----------
        path :
            File path or connection string to read from.

        Returns
        -------
        List of raw import records, possibly empty if no data found.
        """
        ...

    def validate(self, record: RawImportRecord) -> list[str]:
        """Validate a raw record, returning any validation errors.

        Returns an empty list if the record is valid.
        """
        ...


class JsonFileSource:
    """Import adapter for JSON file datasets.

    Reads a JSON file containing an array of record objects and
    yields RawImportRecords.  Each object in the array is expected
    to have at minimum 'startup_name' and 'website' fields.
    """

    @property
    def source_name(self) -> str:
        return "json_file"

    def read(self, path: str) -> list[RawImportRecord]:
        """Read records from a JSON file."""
        import json
        from pathlib import Path

        file_path = Path(path)
        if not file_path.exists():
            return []

        with file_path.open(encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        records: list[RawImportRecord] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            analysis_date = None
            if "analysis_date" in item and item["analysis_date"]:
                try:
                    analysis_date = datetime.fromisoformat(str(item["analysis_date"]))
                except (ValueError, TypeError):
                    pass
            records.append(
                RawImportRecord(
                    startup_name=str(item.get("startup_name", "")),
                    website=str(item.get("website", "")),
                    analysis_date=analysis_date,
                    engine_version=str(item.get("engine_version", "")),
                    benchmark_version=item.get("benchmark_version"),
                    prediction_data=item.get("prediction", {}),
                    outcome_data=item.get("outcome", {}),
                    metadata=item.get("metadata", {}),
                    tags=item.get("tags", []),
                    source="json_file",
                )
            )
        return records

    def validate(self, record: RawImportRecord) -> list[str]:
        errors: list[str] = []
        if not record.startup_name:
            errors.append("startup_name is required")
        if not record.website:
            errors.append("website is required")
        return errors


@dataclass
class ImportResult:
    """Result of an import pipeline run."""

    records_imported: int = 0
    records_failed: int = 0
    validation_errors: list[tuple[int, list[str]]] = field(default_factory=list)
    imported_records: list[DatasetRecord] = field(default_factory=list)
    imported_outcomes: list[OutcomeRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ImportPipeline:
    """Orchestrates the import of historical data from a source.

    The pipeline reads raw records from an ImportSource, validates
    them, normalizes them into DatasetRecord and OutcomeRecord
    instances, and returns the results for storage by the DatasetStore.
    """

    def __init__(
        self,
        source: ImportSource,
        *,
        track_provenance: bool = True,
    ) -> None:
        self._source = source
        self._track_provenance = track_provenance

    @property
    def source_name(self) -> str:
        return self._source.source_name

    def run(
        self,
        path: str,
        *,
        retrieval_date: datetime | None = None,
    ) -> ImportResult:
        """Execute the full import pipeline.

        Parameters
        ----------
        path :
            Path to the data source.
        retrieval_date :
            Override the retrieval date recorded in field provenance.
            Defaults to the current UTC time.

        Returns
        -------
        ImportResult with imported records and any errors.
        """
        from predictron_engine.dataset.provenance import ProvenanceTracker

        tracker = ProvenanceTracker(retrieval_date)
        result = ImportResult()
        raw_records = self._source.read(path)

        for idx, raw in enumerate(raw_records):
            errors = self._source.validate(raw)
            if errors:
                result.records_failed += 1
                result.validation_errors.append((idx, errors))
                continue

            dataset_record = self._normalize_dataset(raw)

            if self._track_provenance:
                dataset_record = self._attach_provenance(
                    tracker, raw, dataset_record
                )

            outcome_record = self._normalize_outcome(raw, dataset_record.record_id)

            result.imported_records.append(dataset_record)
            result.imported_outcomes.append(outcome_record)
            result.records_imported += 1

        return result

    @staticmethod
    def _attach_provenance(
        tracker: object,
        raw: RawImportRecord,
        record: DatasetRecord,
    ) -> DatasetRecord:
        """Attach field-level provenance to an imported record.

        Provenance is stored under ``analysis_metadata["provenance"]``
        without mutating any other fields.
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
        return tracker.attach_to_record(record, provenance)

    def _normalize_dataset(self, raw: RawImportRecord) -> DatasetRecord:
        """Normalize a raw record into a DatasetRecord."""
        from predictron_engine.dataset.models import (
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

        dimension_scores = _coerce_dimension_scores(
            prediction_data.get("dimension_scores")
        )

        prediction = PredictionSummary(
            decision=decision,
            confidence=_coerce_float(prediction_data.get("confidence"), 0.0),
            composite_score=_coerce_float(
                prediction_data.get("composite_score"), 0.0
            ),
            dimension_scores=dimension_scores,
            investment_readiness_score=_coerce_optional_float(
                prediction_data.get("investment_readiness_score")
            ),
            recommendation_count=_coerce_int(
                prediction_data.get("recommendation_count"), 0
            ),
        )

        return DatasetRecord(
            startup_name=raw.startup_name,
            website=raw.website,
            analysis_date=raw.analysis_date or datetime.now(UTC),
            engine_version=raw.engine_version,
            benchmark_version=raw.benchmark_version,
            prediction=prediction,
            funding_stage_at_analysis=funding_stage,
            analysis_metadata=raw.metadata,
            tags=raw.tags,
            source=raw.source,
        )

    def _normalize_outcome(
        self, raw: RawImportRecord, record_id: str
    ) -> OutcomeRecord:
        """Normalize outcome data into an OutcomeRecord."""
        from predictron_engine.dataset.outcomes import OutcomeStatus, StartupOutcome

        outcome_data = raw.outcome_data
        status_str = str(outcome_data.get("status", "unknown"))
        try:
            status = OutcomeStatus(status_str)
        except ValueError:
            status = OutcomeStatus.UNKNOWN

        return OutcomeRecord(
            record_id=record_id,
            outcome=StartupOutcome(
                acquisition=_coerce_optional_str(
                    outcome_data.get("acquisition")
                ),
                shutdown=_coerce_optional_bool(outcome_data.get("shutdown")),
                bankruptcy=_coerce_optional_bool(
                    outcome_data.get("bankruptcy")
                ),
                total_funding_usd=_coerce_optional_float(
                    outcome_data.get("total_funding_usd")
                ),
                investors=_coerce_str_list(outcome_data.get("investors")),
                exit_type=_coerce_optional_str(outcome_data.get("exit_type")),
                status=status,
            ),
        )


class ImportSourceRegistry:
    """Registry of available import source adapters.

    Maintains a mapping of source names to ImportSource instances,
    allowing the pipeline to select the appropriate adapter for a
    given data format.
    """

    def __init__(self) -> None:
        self._sources: dict[str, ImportSource] = {}

    def register(self, source: ImportSource) -> None:
        """Register an import source adapter."""
        self._sources[source.source_name] = source

    def get(self, source_name: str) -> ImportSource | None:
        """Retrieve a registered source adapter by name."""
        return self._sources.get(source_name)

    def list_sources(self) -> list[str]:
        """List all registered source adapter names."""
        return list(self._sources.keys())

    @classmethod
    def default(cls) -> ImportSourceRegistry:
        """Create a registry with all built-in source adapters."""
        from predictron_engine.dataset.csv_import import CsvFileSource
        from predictron_engine.dataset.sources import (
            GovRegistrySource,
            SecEdgarSource,
            YcOssSource,
        )

        registry = cls()
        registry.register(JsonFileSource())
        registry.register(CsvFileSource())
        registry.register(SecEdgarSource())
        registry.register(YcOssSource())
        registry.register(GovRegistrySource())
        return registry
