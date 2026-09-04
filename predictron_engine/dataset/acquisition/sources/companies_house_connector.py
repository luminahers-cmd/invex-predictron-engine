"""Companies House (UK) acquisition connector.

Downloads and normalizes company data from UK Companies House open
data exports.  Companies House publishes bulk CSV files that can be
downloaded freely.

Source: https://download.companieshouse.gov.uk/
Rate limit: API requires key; bulk CSV is unrestricted.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from predictron_engine.dataset.acquisition.sources.base import (
    FetchResult,
    SourceDescriptor,
)
from predictron_engine.dataset.acquisition.state import compute_file_hash
from predictron_engine.dataset.imports import RawImportRecord
from predictron_engine.dataset.sources.gov_registries import CompaniesHouseSource


class CompaniesHouseConnector:
    """Acquisition connector for UK Companies House.

    Reads local CSV exports from Companies House bulk data downloads.
    """

    def __init__(self) -> None:
        self._parser = CompaniesHouseSource()

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name="companies_house",
            description="UK Companies House open company data",
            source_url="https://download.companieshouse.gov.uk/",
            reliability="high",
            update_frequency="daily",
            requires_download=False,
            default_rate_limit=0.0,
            tags=["uk", "government", "registry"],
        )

    def discover(self, config: dict[str, object] | None = None) -> list[str]:
        """Discover available Companies House CSV files."""
        if config and "file_path" in config:
            p = Path(str(config["file_path"]))
            if p.exists():
                return [str(p)]
            return []
        return []

    def fetch(
        self,
        target: str,
        dest_dir: str,
        *,
        config: dict[str, object] | None = None,
    ) -> FetchResult:
        """Fetch a Companies House file.  Copies local file to dest_dir."""
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
            source_version=f"ch_{file_hash[:8]}",
        )

    def normalize(self, file_path: str) -> list[RawImportRecord]:
        """Normalize a Companies House CSV file into RawImportRecords."""
        return self._parser.read(file_path)

    def checkpoint(self, file_path: str) -> str:
        """Compute checkpoint key from file content hash."""
        return compute_file_hash(file_path)
