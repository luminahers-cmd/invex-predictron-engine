"""SEC EDGAR source adapter (Part C).

Parses publicly available SEC EDGAR company filing data into
RawImportRecords.  Reads pre-downloaded EDGAR JSON/CSV files —
does not make network requests.

The adapter expects a JSON file with company entries that have
SEC-standard fields (company name, CIK, SIC code, state of
incorporation, etc.).  It maps these to RawImportRecord fields
where applicable.

Source: https://www.sec.gov/edgar/searchedgar/companysearch
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from predictron_engine.dataset.imports import RawImportRecord

# SEC EDGAR standard field name mappings
_CIK_FIELD = "cik"
_COMPANY_NAME_FIELD = "company_name"
_WEBSITE_FIELD = "website_url"
_STATE_FIELD = "state_of_incorporation"
_SIC_CODE_FIELD = "sic_code"
_SIC_DESCRIPTION_FIELD = "sic_description"
_FILING_DATE_FIELD = "latest_filing_date"
_FORM_TYPE_FIELD = "form_type"


class SecEdgarSource:
    """Import adapter for SEC EDGAR company data.

    Reads a JSON file containing an array of company records as
    exported from EDGAR's full-text search or company search APIs.

    Expected JSON structure::

        [
            {
                "cik": "0001234567",
                "company_name": "Example Corp",
                "website_url": "https://example.com",
                "state_of_incorporation": "DE",
                "sic_code": "7372",
                "sic_description": "Prepackaged Software",
                "latest_filing_date": "2024-01-15",
                "form_type": "10-K"
            },
            ...
        ]
    """

    @property
    def source_name(self) -> str:
        return "sec_edgar"

    def read(self, path: str) -> list[RawImportRecord]:
        """Read records from a SEC EDGAR JSON file."""
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

            company_name = str(item.get(_COMPANY_NAME_FIELD, "")).strip()
            website = str(item.get(_WEBSITE_FIELD, "")).strip()
            if not company_name:
                continue

            metadata: dict[str, object] = {
                "source": "sec_edgar",
            }
            cik = item.get(_CIK_FIELD)
            if cik:
                metadata["sec_cik"] = str(cik)

            state = item.get(_STATE_FIELD)
            if state:
                metadata["state_of_incorporation"] = str(state)

            sic_code = item.get(_SIC_CODE_FIELD)
            if sic_code:
                metadata["sic_code"] = str(sic_code)

            sic_desc = item.get(_SIC_DESCRIPTION_FIELD)
            if sic_desc:
                metadata["sic_description"] = str(sic_desc)
                metadata["sector"] = str(sic_desc)

            form_type = item.get(_FORM_TYPE_FIELD)
            if form_type:
                metadata["latest_form_type"] = str(form_type)

            analysis_date: datetime | None = None
            filing_date = item.get(_FILING_DATE_FIELD)
            if filing_date:
                try:
                    analysis_date = datetime.fromisoformat(
                        str(filing_date)
                    ).replace(tzinfo=UTC)
                except (ValueError, TypeError):
                    pass

            records.append(
                RawImportRecord(
                    startup_name=company_name,
                    website=website,
                    analysis_date=analysis_date,
                    engine_version="",
                    metadata=metadata,
                    tags=["sec_edgar"],
                    source="sec_edgar",
                )
            )

        return records

    def validate(self, record: RawImportRecord) -> list[str]:
        """Validate a raw EDGAR record."""
        errors: list[str] = []
        if not record.startup_name:
            errors.append("company_name is required")
        return errors
