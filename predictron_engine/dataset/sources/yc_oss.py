"""YC Open Source Startup dataset adapter (Part C).

Parses the publicly available Y Combinator "Open Source Startup" (YC
OSS) dataset into RawImportRecords.  The dataset is distributed as a
CSV file published by Y Combinator / Mozilla.  It contains company
names and, for companies that opt in, website URLs.

The adapter reads a local CSV export and does not make network requests.

Source: https://data.world/ycombinator/open-source-startups
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from predictron_engine.dataset.imports import RawImportRecord

# Expected columns in the YC OSS CSV export.
_COMPANY_COLUMNS = ("company", "name", "startup_name")
_WEBSITE_COLUMNS = ("website", "url", "company_url", "homepage")
_DESCRIPTION_COLUMNS = ("description", "tagline", "short_description")
_REGION_COLUMNS = ("region", "location", "country", "hq")
_INDUSTRY_COLUMNS = ("industry", "sector", "category")
_CB_URL_COLUMNS = ("cb_url", "crunchbase_url", "cb_link")


class YcOssSource:
    """Import adapter for the Y Combinator OSS startup dataset.

    Reads a CSV file with company records.  Recognizes a flexible set
    of common column names so it works with several published variants
    of the dataset.

    Each row becomes a RawImportRecord with startup_name, optional
    website, and any industry/geography metadata found.
    """

    @property
    def source_name(self) -> str:
        return "yc_oss"

    def read(self, path: str) -> list[RawImportRecord]:
        """Read records from a YC OSS CSV file."""
        file_path = Path(path)
        if not file_path.exists():
            return []

        raw_bytes = file_path.read_bytes()
        if raw_bytes.startswith(b"\xef\xbb\xbf"):
            raw_bytes = raw_bytes[3:]
        text = raw_bytes.decode("utf-8", errors="replace")

        delimiter = _detect_delimiter(text)
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if reader.fieldnames is None:
            return []

        cols = {c.strip().lower(): c for c in reader.fieldnames}
        company_col = _first_present(cols, _COMPANY_COLUMNS)
        if company_col is None:
            return []

        website_col = _first_present(cols, _WEBSITE_COLUMNS)
        description_col = _first_present(cols, _DESCRIPTION_COLUMNS)
        region_col = _first_present(cols, _REGION_COLUMNS)
        industry_col = _first_present(cols, _INDUSTRY_COLUMNS)
        cb_url_col = _first_present(cols, _CB_URL_COLUMNS)

        records: list[RawImportRecord] = []
        for row in reader:
            company = _clean(row.get(company_col, ""))
            if not company:
                continue

            website = (
                _clean(row.get(website_col, ""))
                if website_col
                else ""
            )
            if website and not website.startswith(("http://", "https://")):
                website = f"https://{website}"

            metadata: dict[str, object] = {
                "source": "yc_oss",
            }
            if description_col:
                desc = _clean(row.get(description_col, ""))
                if desc:
                    metadata["description"] = desc
            if region_col:
                region = _clean(row.get(region_col, ""))
                if region:
                    metadata["country"] = region
            if industry_col:
                industry = _clean(row.get(industry_col, ""))
                if industry:
                    metadata["sector"] = industry
                    metadata["industry_category"] = industry
            if cb_url_col:
                cb = _clean(row.get(cb_url_col, ""))
                if cb:
                    metadata["crunchbase_url"] = cb

            records.append(
                RawImportRecord(
                    startup_name=company,
                    website=website,
                    analysis_date=None,
                    engine_version="",
                    metadata=metadata,
                    tags=["yc_oss"],
                    source="yc_oss",
                )
            )

        return records

    def validate(self, record: RawImportRecord) -> list[str]:
        """Validate a raw YC OSS record."""
        errors: list[str] = []
        if not record.startup_name:
            errors.append("company name is required")
        return errors


def _detect_delimiter(text: str) -> str:
    first_line = text.split("\n", 1)[0]
    for candidate in ("\t", ";", ","):
        if candidate in first_line:
            return candidate
    return ","


def _first_present(
    cols: dict[str, str],
    candidates: tuple[str, ...],
) -> str | None:
    for candidate in candidates:
        if candidate in cols:
            return cols[candidate]
    return None


def _clean(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()
