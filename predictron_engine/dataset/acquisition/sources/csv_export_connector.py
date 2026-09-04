"""Generic CSV export acquisition connector.

Handles any startup data exported as CSV with flexible column mapping.
This is the catch-all connector for manually downloaded datasets,
public CSV exports, and custom data files.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from predictron_engine.dataset.acquisition.sources.base import (
    FetchResult,
    SourceDescriptor,
)
from predictron_engine.dataset.acquisition.state import compute_file_hash
from predictron_engine.dataset.csv_import import CsvFileSource
from predictron_engine.dataset.imports import RawImportRecord


class CsvExportConnector:
    """Acquisition connector for generic CSV startup data.

    Uses the existing CsvFileSource adapter for flexible column
    mapping and delimiter detection.
    """

    def __init__(self) -> None:
        self._parser = CsvFileSource()

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name="csv_export",
            description="Generic CSV startup data export",
            source_url="",
            reliability="medium",
            update_frequency="static",
            requires_download=False,
            default_rate_limit=0.0,
            tags=["csv", "generic"],
        )

    def discover(self, config: dict[str, object] | None = None) -> list[str]:
        """Discover available CSV files."""
        if config and "file_path" in config:
            p = Path(str(config["file_path"]))
            if p.exists():
                return [str(p)]
            return []
        if config and "directory" in config:
            d = Path(str(config["directory"]))
            if d.is_dir():
                return sorted(str(f) for f in d.glob("*.csv"))
        return []

    def fetch(
        self,
        target: str,
        dest_dir: str,
        *,
        config: dict[str, object] | None = None,
    ) -> FetchResult:
        """Fetch a CSV file.  Copies local file to dest_dir."""
        src = Path(target)
        if not src.exists():
            msg = f"File not found: {target}"
            raise FileNotFoundError(msg)

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        dest_file = dest / src.name
        shutil.copy2(src, dest_file)

        file_hash = compute_file_hash(dest_file)
        return FetchResult(
            file_path=str(dest_file),
            file_hash=file_hash,
            file_size=dest_file.stat().st_size,
            source_version=f"csv_{file_hash[:8]}",
        )

    def normalize(self, file_path: str) -> list[RawImportRecord]:
        """Normalize a CSV file into RawImportRecords."""
        return self._parser.read(file_path)

    def checkpoint(self, file_path: str) -> str:
        """Compute checkpoint key from file content hash."""
        return compute_file_hash(file_path)
