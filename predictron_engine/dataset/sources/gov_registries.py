"""Government registry source adapters (Part C).

Parses publicly available government company-registry data into
RawImportRecords.  Reads local CSV/JSON exports and does not make
network requests.

Supported national registries (all adapters parse local files):
  - Companies House (UK)
  - US state incorporation registries
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from pathlib import Path

from predictron_engine.dataset.imports import RawImportRecord


class CompaniesHouseSource:
    """Adapter for UK Companies House open data.

    Companies House publishes open company data as CSV files.  This
    adapter parses a local CSV export with the standard Companies
    House data fields (company number, company name, country of
    origin, incorporation date, SIC codes).

    Expected columns (Companies House standard)::

        company_number, company_name, company_status,
        country_of_origin, incorporation_date, sic_code_1, ...

    Source: https://download.companieshouse.gov.uk/en_accountsdata.html
    """

    @property
    def source_name(self) -> str:
        return "companies_house"

    def read(self, path: str) -> list[RawImportRecord]:
        """Read records from a Companies House CSV file."""
        file_path = Path(path)
        if not file_path.exists():
            return []

        raw_bytes = file_path.read_bytes()
        if raw_bytes.startswith(b"\xef\xbb\xbf"):
            raw_bytes = raw_bytes[3:]
        text = raw_bytes.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            return []

        records: list[RawImportRecord] = []
        for row in reader:
            company_name = (row.get("company_name") or "").strip()
            if not company_name:
                continue

            metadata: dict[str, object] = {
                "source": "companies_house",
            }
            company_number = (row.get("company_number") or "").strip()
            if company_number:
                metadata["company_number"] = company_number
            country = (row.get("country_of_origin") or "").strip()
            if country:
                metadata["country"] = country
                metadata["country_code"] = country

            sic_code = (row.get("sic_code_1") or "").strip()
            if sic_code:
                metadata["sic_code"] = sic_code

            company_status = (row.get("company_status") or "").strip()
            if company_status:
                metadata["company_status"] = company_status
                if company_status.lower() == "dissolved":
                    metadata["dissolved"] = True

            analysis_date: datetime | None = None
            inc_date = (row.get("incorporation_date") or "").strip()
            if inc_date:
                try:
                    analysis_date = datetime.fromisoformat(inc_date).replace(
                        tzinfo=UTC
                    )
                except (ValueError, TypeError):
                    try:
                        analysis_date = datetime.strptime(
                            inc_date, "%Y-%m-%d"
                        ).replace(tzinfo=UTC)
                    except (ValueError, TypeError):
                        pass

            # Companies House does not publish websites directly; leave
            # website empty so the record is flagged as incomplete unless
            # enriched elsewhere.
            records.append(
                RawImportRecord(
                    startup_name=company_name,
                    website="",
                    analysis_date=analysis_date,
                    engine_version="",
                    metadata=metadata,
                    tags=["companies_house"],
                    source="companies_house",
                )
            )

        return records

    def validate(self, record: RawImportRecord) -> list[str]:
        """Validate a raw Companies House record."""
        errors: list[str] = []
        if not record.startup_name:
            errors.append("company_name is required")
        # Website is not provided by Companies House; do not reject.
        return errors


class UsStateRegistrySource:
    """Adapter for US state incorporation registry CSV exports.

    Several US states publish business entity data as CSV exports
    with company name, status, jurisdiction, and filing date.

    Expected columns (varies by state; adapter is lenient)::

        company_name, entity_type, status, jurisdiction,
        formation_date, website

    Source: e.g. Texas SOS, California SOS, Delaware — public exports.
    """

    @property
    def source_name(self) -> str:
        return "us_state_registry"

    def read(self, path: str) -> list[RawImportRecord]:
        """Read records from a US state registry CSV file."""
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

        # Normalized column lookup (lowercase, space->underscore).
        cols: dict[str, str] = {}
        for c in reader.fieldnames:
            key = c.strip().lower().replace(" ", "_")
            cols[key] = c

        company_col = _pick(cols, ("company_name", "name", "entity_name"))
        if company_col is None:
            return []

        website_col = _pick(cols, ("website", "website_url", "url"))
        country_col = _pick(cols, ("jurisdiction", "state", "country"))
        status_col = _pick(cols, ("status", "entity_status"))
        date_col = _pick(cols, ("formation_date", "filing_date", "date"))

        records: list[RawImportRecord] = []
        for row in reader:
            company_name = (row.get(company_col) or "").strip()
            if not company_name:
                continue

            website = ""
            if website_col:
                website = (row.get(website_col) or "").strip()
                if website and not website.startswith(("http://", "https://")):
                    website = f"https://{website}"

            metadata: dict[str, object] = {
                "source": "us_state_registry",
            }
            if country_col:
                jurisdiction = (row.get(country_col) or "").strip()
                if jurisdiction:
                    metadata["country_code"] = jurisdiction
                    metadata["jurisdiction"] = jurisdiction
            if status_col:
                status = (row.get(status_col) or "").strip()
                if status:
                    metadata["registry_status"] = status

            analysis_date: datetime | None = None
            if date_col:
                raw_date = (row.get(date_col) or "").strip()
                if raw_date:
                    try:
                        analysis_date = datetime.fromisoformat(raw_date).replace(
                            tzinfo=UTC
                        )
                    except (ValueError, TypeError):
                        pass

            records.append(
                RawImportRecord(
                    startup_name=company_name,
                    website=website,
                    analysis_date=analysis_date,
                    engine_version="",
                    metadata=metadata,
                    tags=["us_state_registry"],
                    source="us_state_registry",
                )
            )

        return records

    def validate(self, record: RawImportRecord) -> list[str]:
        """Validate a raw US state registry record."""
        errors: list[str] = []
        if not record.startup_name:
            errors.append("company_name is required")
        return errors


class GovRegistrySource:
    """Combined government-registry source exposing all registries.

    Provides a single entry point for the ``gov_registries`` source
    name that dispatches to the appropriate registry adapter based on
    the file content (Companies House vs. US state registry).
    """

    def __init__(self) -> None:
        self._companies_house = CompaniesHouseSource()
        self._us_state = UsStateRegistrySource()

    @property
    def source_name(self) -> str:
        return "gov_registries"

    def read(self, path: str) -> list[RawImportRecord]:
        """Read records from a government registry file.

        Dispatches based on content sniffing: if the file contains a
        ``company_number`` column it is treated as Companies House;
        otherwise it is treated as a US state registry export.
        """
        file_path = Path(path)
        if not file_path.exists():
            return []

        if _looks_like_companies_house(file_path):
            return self._companies_house.read(path)
        return self._us_state.read(path)

    def validate(self, record: RawImportRecord) -> list[str]:
        """Validate a raw government-registry record."""
        errors: list[str] = []
        if not record.startup_name:
            errors.append("company_name is required")
        return errors


def _detect_delimiter(text: str) -> str:
    first_line = text.split("\n", 1)[0]
    for candidate in ("\t", ";", ","):
        if candidate in first_line:
            return candidate
    return ","


def _pick(cols: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in cols:
            return cols[candidate]
    return None


def _looks_like_companies_house(path: Path) -> bool:
    """Sniff a CSV to detect Companies House format (company_number col)."""
    try:
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        text = raw.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            return False
        headers = {h.strip().lower() for h in reader.fieldnames}
        return "company_number" in headers
    except Exception:  # noqa: BLE001
        return False
