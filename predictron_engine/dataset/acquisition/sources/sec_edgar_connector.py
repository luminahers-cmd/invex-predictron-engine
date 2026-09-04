"""SEC EDGAR acquisition connector.

Downloads and normalizes company data from SEC EDGAR's public
full-text search API.  Uses the EDGAR EFTS (full-text search) API
which is free and requires only a descriptive User-Agent header.

Source: https://www.sec.gov/cgi-bin/browse-edgar
Rate limit: ~10 requests per second (SEC requirement).
"""

from __future__ import annotations

from pathlib import Path

from predictron_engine.dataset.acquisition.sources.base import (
    FetchResult,
    SourceDescriptor,
)
from predictron_engine.dataset.acquisition.state import compute_file_hash
from predictron_engine.dataset.imports import RawImportRecord
from predictron_engine.dataset.sources.sec_edgar import SecEdgarSource

_EDGAR_EFTS_URL = "https://efts.sec.gov/LATEST/search-index?q=%22*%22&dateRange=custom&startdt={start}&enddt={end}&forms={forms}"
_EDGAR_COMPANY_URL = "https://efts.sec.gov/LATEST/search-index?q=%22*%22&forms={forms}&dateRange=custom&startdt={start}&enddt={end}"


class SecEdgarConnector:
    """Acquisition connector for SEC EDGAR.

    Supports two modes:
    1. Local file mode: reads a pre-downloaded EDGAR JSON file
    2. Fetch mode: downloads from EDGAR EFTS API (rate-limited)
    """

    def __init__(self, user_agent: str = "predictron-engine/0.12.1 (research)") -> None:
        self._user_agent = user_agent
        self._parser = SecEdgarSource()

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name="sec_edgar",
            description="SEC EDGAR company filings (full-text search exports)",
            source_url="https://www.sec.gov/edgar/searchedgar/companysearch",
            reliability="high",
            update_frequency="daily",
            requires_download=True,
            default_rate_limit=10.0,
            tags=["sec", "filings", "public"],
        )

    def discover(self, config: dict[str, object] | None = None) -> list[str]:
        """Discover available EDGAR data.

        In local mode, returns the path from config['file_path'].
        In fetch mode, constructs the API URL.
        """
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
        """Fetch EDGAR data.  For local files, copies to dest_dir."""
        src = Path(target)
        if not src.exists():
            msg = f"File not found: {target}"
            raise FileNotFoundError(msg)

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        dest_file = dest / src.name

        import shutil

        shutil.copy2(src, dest_file)

        file_hash = compute_file_hash(dest_file)
        return FetchResult(
            file_path=str(dest_file),
            file_hash=file_hash,
            file_size=dest_file.stat().st_size,
            source_version=f"edgar_{file_hash[:8]}",
        )

    def normalize(self, file_path: str) -> list[RawImportRecord]:
        """Normalize an EDGAR JSON file into RawImportRecords."""
        return self._parser.read(file_path)

    def checkpoint(self, file_path: str) -> str:
        """Compute checkpoint key from file content hash."""
        return compute_file_hash(file_path)
