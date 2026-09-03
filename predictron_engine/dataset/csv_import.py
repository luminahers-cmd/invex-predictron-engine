"""CSV import adapter (Part A).

Reads structured startup records from CSV files and yields
RawImportRecords for the ImportPipeline.  Supports flexible column
mapping and handles common CSV edge cases (quoted fields, BOM,
variable delimiters).

No data is fabricated; columns not present in the CSV are left at
their defaults.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from predictron_engine.dataset.imports import RawImportRecord

# Default column-to-field mapping.  Keys are CSV header patterns
# (lowercased, stripped); values are RawImportRecord attribute names.
_DEFAULT_COLUMN_MAP: dict[str, str] = {
    "startup_name": "startup_name",
    "name": "startup_name",
    "company_name": "startup_name",
    "company": "startup_name",
    "website": "website",
    "url": "website",
    "company_url": "website",
    "analysis_date": "analysis_date",
    "date": "analysis_date",
    "engine_version": "engine_version",
    "benchmark_version": "benchmark_version",
    "decision": "prediction__decision",
    "confidence": "prediction__confidence",
    "composite_score": "prediction__composite_score",
    "investment_readiness_score": "prediction__investment_readiness_score",
    "recommendation_count": "prediction__recommendation_count",
    "outcome_status": "outcome__status",
    "outcome_acquisition": "outcome__acquisition",
    "outcome_shutdown": "outcome__shutdown",
    "outcome_bankruptcy": "outcome__bankruptcy",
    "total_funding_usd": "outcome__total_funding_usd",
    "exit_type": "outcome__exit_type",
    "funding_stage": "metadata__funding_stage_at_analysis",
    "tags": "tags",
    "source": "source",
}


class CsvFileSource:
    """Import adapter for CSV file datasets.

    Reads a CSV file containing one startup record per row and yields
    RawImportRecords.  At minimum, the CSV must contain ``startup_name``
    and ``website`` columns (or mapped equivalents).

    The adapter handles:
      - UTF-8 and UTF-8-BOM encoded files
      - Comma, semicolon, and tab delimiters (auto-detected)
      - Flexible column mapping via the ``column_map`` parameter
      - Nested prediction/outcome fields via ``__`` separator
    """

    def __init__(
        self,
        column_map: dict[str, str] | None = None,
    ) -> None:
        self._column_map = column_map or dict(_DEFAULT_COLUMN_MAP)

    @property
    def source_name(self) -> str:
        return "csv_file"

    def read(self, path: str) -> list[RawImportRecord]:
        """Read records from a CSV file.

        Parameters
        ----------
        path :
            Path to a CSV file.

        Returns
        -------
        List of RawImportRecord instances, one per row.
        """
        file_path = Path(path)
        if not file_path.exists():
            return []

        raw_bytes = file_path.read_bytes()
        # Strip BOM if present
        if raw_bytes.startswith(b"\xef\xbb\xbf"):
            raw_bytes = raw_bytes[3:]

        text = raw_bytes.decode("utf-8")
        delimiter = _detect_delimiter(text)
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)

        if reader.fieldnames is None:
            return []

        mapped_headers = self._map_headers(reader.fieldnames)
        records: list[RawImportRecord] = []

        for row in reader:
            mapped = _apply_mapping(row, mapped_headers)
            analysis_date = _parse_datetime(mapped.get("analysis_date"))

            prediction_data: dict[str, object] = {}
            for key in (
                "decision",
                "confidence",
                "composite_score",
                "investment_readiness_score",
                "recommendation_count",
            ):
                val = mapped.get(f"prediction__{key}")
                if val is not None and val != "":
                    prediction_data[key] = val

            outcome_data: dict[str, object] = {}
            for key in ("status", "acquisition", "shutdown", "bankruptcy",
                        "total_funding_usd", "exit_type"):
                val = mapped.get(f"outcome__{key}")
                if val is not None and val != "":
                    outcome_data[key] = val

            metadata: dict[str, object] = {}
            funding_stage = mapped.get("metadata__funding_stage_at_analysis")
            if funding_stage:
                metadata["funding_stage_at_analysis"] = funding_stage

            tags_raw = mapped.get("tags", "")
            tags = (
                [t.strip() for t in tags_raw.split(",") if t.strip()]
                if tags_raw
                else []
            )

            records.append(
                RawImportRecord(
                    startup_name=mapped.get("startup_name", ""),
                    website=mapped.get("website", ""),
                    analysis_date=analysis_date,
                    engine_version=mapped.get("engine_version", ""),
                    benchmark_version=mapped.get("benchmark_version") or None,
                    prediction_data=prediction_data,
                    outcome_data=outcome_data,
                    metadata=metadata,
                    tags=tags,
                    source="csv_file",
                )
            )

        return records

    def validate(self, record: RawImportRecord) -> list[str]:
        """Validate a raw record, returning any validation errors."""
        errors: list[str] = []
        if not record.startup_name:
            errors.append("startup_name is required")
        if not record.website:
            errors.append("website is required")
        return errors

    def _map_headers(
        self, fieldnames: Sequence[str]
    ) -> dict[str, str]:
        """Map CSV headers to RawImportRecord field paths."""
        result: dict[str, str] = {}
        for header in fieldnames:
            normalized = header.strip().lower().replace(" ", "_")
            mapped = self._column_map.get(normalized)
            if mapped:
                result[header] = mapped
        return result


def _detect_delimiter(text: str) -> str:
    """Auto-detect the CSV delimiter from a sample of the text."""
    first_line = text.split("\n", 1)[0]
    for candidate in ("\t", ";", ","):
        if candidate in first_line:
            return candidate
    return ","


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO format datetime string."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _apply_mapping(
    row: dict[str, str | None],
    mapped_headers: dict[str, str],
) -> dict[str, str]:
    """Apply the header mapping to a CSV row."""
    result: dict[str, str] = {}
    for csv_header, field_path in mapped_headers.items():
        raw_value = row.get(csv_header)
        if raw_value is not None:
            result[field_path] = raw_value.strip()
    return result
