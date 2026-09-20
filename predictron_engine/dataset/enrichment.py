"""Company profile enrichment (Project E1).

Promotes raw source metadata into structured, first-class company
profile fields on :class:`DatasetRecord`.  This is the foundation for
industry analysis, geographic analysis, deduplication, and search.

The pipeline is additive and deterministic:

  1. ``normalize_profile`` — build a canonical :class:`CompanyProfile`
     from a source's raw metadata (website, name, and known metadata
     keys from SEC EDGAR, Companies House, YC OSS, US state registries,
     and CSV exports).
  2. ``merge_profiles`` — merge an incoming profile into an existing one,
     filling only empty fields so a canonical record can be enriched
     across multiple sources without clobbering existing data.
  3. ``EnrichmentService.enrich`` — run the normalization over a
     collection of records or an entire :class:`DatasetStore`, persist
     updated records, and report what was populated.

Rules:

  - No fabricating data: only values actually present in a source are
    promoted; every other field stays empty.
  - Deterministic: identical inputs produce identical profiles.
  - Non-destructive: enrichment only fills empty profile fields; it
    never overwrites existing values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from predictron_engine.dataset.imports import RawImportRecord
from predictron_engine.dataset.models import CompanyProfile, DatasetRecord
from predictron_engine.dataset.store import DatasetStore

# Known metadata keys that carry company-attribute values.  Advisory
# order matters: earlier keys are preferred when several are present.
_INDUSTRY_KEYS = ("industry_category", "sector", "sic_description", "industry")
_COUNTRY_KEYS = ("country_code", "country_iso", "country", "jurisdiction")
_STATE_KEYS = (
    "state_of_incorporation",
    "region",
    "state",
    "province",
    "administrative_area",
)
_CITY_KEYS = ("city", "city_name", "locality", "hq_city")
_DESCRIPTION_KEYS = ("description", "tagline", "short_description", "about")
_LEGAL_NAME_KEYS = ("legal_name", "registered_name", "full_company_name")
_STATUS_KEYS = ("company_status", "registry_status", "entity_status")
_FOUNDED_DATE_KEYS = (
    "founding_date",
    "founded_date",
    "incorporation_date",
    "formation_date",
    "founded_at",
)
_FOUNDED_YEAR_KEYS = ("founded_year", "founding_year", "year_founded", "inc_year")
_EMPLOYEE_COUNT_KEYS = (
    "employee_count",
    "num_employees",
    "employees",
    "employee_total",
)
_EMPLOYEE_RANGE_KEYS = ("employee_range", "size_range", "employees_range")

# Two-letter / three-letter ISO country code passthrough.
_ISO_ALPHA2_RE = re.compile(r"^[A-Za-z]{2}$")
_ISO_ALPHA3_RE = re.compile(r"^[A-Za-z]{3}$")

# Coarse country-name -> ISO alpha-2 normalization.  Covers the names
# that actually appear in the supported public sources.  Unknown names
# are left unchanged.
_COUNTRY_ALIASES: dict[str, str] = {
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "u.s.a.": "US",
    "u.s.": "US",
    "us": "US",
    "united kingdom": "GB",
    "great britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "uk": "GB",
    "u.k.": "GB",
    "gb": "GB",
    "gb-eng": "GB",
    "gb-sct": "GB",
    "gb-wls": "GB",
    "gb-nir": "GB",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "deutschland": "DE",
    "france": "FR",
    "netherlands": "NL",
    "the netherlands": "NL",
    "belgium": "BE",
    "switzerland": "CH",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "finland": "FI",
    "ireland": "IE",
    "republic of ireland": "IE",
    "spain": "ES",
    "italy": "IT",
    "portugal": "PT",
    "india": "IN",
    "china": "CN",
    "japan": "JP",
    "singapore": "SG",
    "israel": "IL",
    "south korea": "KR",
    "korea": "KR",
    "brazil": "BR",
    "mexico": "MX",
    "estonia": "EE",
    "poland": "PL",
    "austria": "AT",
    "luxembourg": "LU",
    "hong kong": "HK",
}

_EMPLOYEE_BANDS = (
    ("1-10", (1, 10)),
    ("11-50", (11, 50)),
    ("51-200", (51, 200)),
    ("201-500", (201, 500)),
    ("501-1000", (501, 1000)),
    ("1001-5000", (1001, 5000)),
    ("5001-10000", (5001, 10000)),
    ("10001+", (10001, None)),
)


def extract_domain(website: str | None) -> str | None:
    """Extract a canonical domain (host, no scheme/www) from a website.

    Returns None when the website is empty or has no usable host.
    """
    if not website:
        return None
    value = website.strip()
    if not value:
        return None
    if "://" not in value:
        value = f"http://{value}"
    from urllib.parse import urlparse

    try:
        host = urlparse(value).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def normalize_country_code(value: str | None) -> str | None:
    """Normalize a country name or code to an ISO 3166-1 alpha-2 code.

    Recognizes existing two-letter codes (passed through uppercase),
    three-letter ISO codes (mapped from a small known set), and common
    English country names.  Returns None for empty/unknown input.
    """
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    lowered = raw.lower()
    key = lowered.replace(".", "").strip()

    if _ISO_ALPHA2_RE.match(raw):
        # Existing 2-letter code (e.g. "uk", "us", "de").
        upper = raw.upper()
        if upper == "UK":
            return "GB"
        return upper

    if key in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[key]

    if _ISO_ALPHA3_RE.match(raw):
        mapped = _ALPHA3_TO_ALPHA2.get(raw.upper())
        if mapped:
            return mapped

    return None


def normalize_industry(value: str | None) -> str:
    """Normalize a single industry string into a canonical tag."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_status(value: str | None) -> str | None:
    """Normalize registry status strings into canonical labels."""
    if not value:
        return None
    raw = value.strip().lower()
    if raw in ("", "-"):
        return None
    if raw in ("dissolved", "closed", "liquidation", "struck off"):
        return "dissolved"
    if raw in ("active", "normal", "registered", "in business", "operating"):
        return "active"
    if raw in ("inactive", "inactive - unknown"):
        return "inactive"
    return raw


def employee_range(count: int | None) -> str | None:
    """Map an employee count to a canonical size band."""
    if count is None or count < 0:
        return None
    for label, (low, high) in _EMPLOYEE_BANDS:
        if high is None:
            if count >= low:
                return label
        elif low <= count <= high:
            return label
    return None


def _build_headquarters(
    city: str | None, region: str | None, country: str | None
) -> str | None:
    """Assemble a human-readable headquarters string."""
    parts = [p for p in (city, region, country) if p]
    return ", ".join(parts) if parts else None


def profile_from_metadata(
    metadata: dict[str, Any] | None,
    website: str = "",
    startup_name: str = "",
) -> CompanyProfile:
    """Build a canonical :class:`CompanyProfile` from source metadata.

    Reads the known metadata keys used by the supported connectors and
    adapters (SEC EDGAR, Companies House, YC OSS, US state registries,
    CSV exports).  Values are normalized deterministically; missing
    values stay empty.
    """
    metadata = metadata or {}
    profile = CompanyProfile()

    explicit_domain = _first_present(metadata, ("canonical_domain", "domain"))
    profile.domain = (
        extract_domain(explicit_domain)
        if explicit_domain is not None
        else extract_domain(website)
    )

    industries: list[str] = []
    for key in _INDUSTRY_KEYS:
        value = metadata.get(key)
        if value:
            normalized = normalize_industry(str(value))
            if normalized and normalized not in industries:
                industries.append(normalized)
    profile.industries = industries

    country_raw = _first_present(metadata, _COUNTRY_KEYS)
    profile.country_code = normalize_country_code(
        str(country_raw) if country_raw is not None else None
    )
    profile.city = _first_present(metadata, _CITY_KEYS)
    profile.region = _first_present(metadata, _STATE_KEYS)
    profile.headquarters = _build_headquarters(
        profile.city,
        profile.region,
        str(country_raw) if country_raw is not None else None,
    )

    founded_date = _first_present(metadata, _FOUNDED_DATE_KEYS)
    profile.founded_date = _parse_founded_date(founded_date)
    founded_year = _first_present(metadata, _FOUNDED_YEAR_KEYS)
    year = _parse_year(founded_year)
    if year is not None and profile.founded_year is None:
        profile.founded_year = year
    if profile.founded_date is not None and profile.founded_year is None:
        profile.founded_year = profile.founded_date.year

    profile.employee_count = _parse_int(_first_present(metadata, _EMPLOYEE_COUNT_KEYS))
    range_raw = _first_present(metadata, _EMPLOYEE_RANGE_KEYS)
    profile.employee_range = (
        str(range_raw) if range_raw is not None else employee_range(profile.employee_count)
    )

    description = _first_present(metadata, _DESCRIPTION_KEYS)
    if description:
        profile.description = str(description).strip()

    legal_name = _first_present(metadata, _LEGAL_NAME_KEYS)
    if legal_name:
        profile.legal_name = str(legal_name).strip()

    profile.status = normalize_status(_first_present(metadata, _STATUS_KEYS))

    return profile


def profile_from_raw(raw: RawImportRecord) -> CompanyProfile:
    """Build a profile from a :class:`RawImportRecord`.

    Uses the record's own profile when present (adapters may attach one
    directly); otherwise derives it from the record's metadata, website,
    and name.
    """
    if raw.profile is not None:
        return raw.profile
    return profile_from_metadata(
        raw.metadata,
        website=raw.website,
        startup_name=raw.startup_name,
    )


def merge_profiles(
    base: CompanyProfile,
    incoming: CompanyProfile,
) -> CompanyProfile:
    """Merge an incoming profile into a base profile.

    Incoming values fill empty base fields; existing base values are
    never overwritten (source precedence is already applied by the
    caller ordering).  List fields union deterministically.  Returns a
    new CompanyProfile; neither input is mutated.
    """
    merged = base.model_copy(deep=True)

    def fill(target: str, value: object) -> None:
        if getattr(merged, target) is None and value is not None:
            setattr(merged, target, value)

    fill("domain", incoming.domain)
    for tag in incoming.industries:
        if tag not in merged.industries:
            merged.industries.append(tag)
    if merged.industries:
        merged.industries = _normalized_industry_list(merged.industries)

    fill("headquarters", incoming.headquarters)
    fill("country_code", incoming.country_code)
    fill("city", incoming.city)
    fill("region", incoming.region)
    for attr in (
        "founded_year",
        "founded_date",
        "employee_count",
        "employee_range",
        "description",
        "legal_name",
        "status",
    ):
        fill(attr, getattr(incoming, attr))

    return merged


def enrich_record(
    record: DatasetRecord,
) -> tuple[DatasetRecord, list[str]]:
    """Promote a record's metadata into its profile, filling empty fields.

    Returns a (new_record, fields_added) tuple.  The returned record is
    a copy; the input is never mutated.  When the record already carries
    the canonical structure in ``record.profile``, metadata is only used
    to fill gaps (never to overwrite).
    """
    derived = profile_from_metadata(
        record.analysis_metadata,
        website=record.website,
        startup_name=record.startup_name,
    )
    merged = merge_profiles(record.profile, derived)
    base_populated = set(record.profile.populated_fields())
    merged_populated = set(merged.populated_fields())
    fields_added = sorted(merged_populated - base_populated)
    if not fields_added:
        return record, []

    enriched = record.model_copy(update={"profile": merged})
    return enriched, fields_added


@dataclass
class EnrichmentAction:
    """Record of profile fields added to a single record."""

    record_id: str
    fields_added: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"record_id": self.record_id, "fields_added": list(self.fields_added)}


@dataclass
class EnrichmentReport:
    """Aggregate result of an enrichment pass over a dataset."""

    records_processed: int = 0
    records_updated: int = 0
    fields_populated: dict[str, int] = field(default_factory=dict)
    actions: list[EnrichmentAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "report_type": "enrichment_report",
            "records_processed": self.records_processed,
            "records_updated": self.records_updated,
            "enrichment_rate": (
                round(self.records_updated / self.records_processed, 4)
                if self.records_processed
                else 0.0
            ),
            "fields_populated": dict(sorted(self.fields_populated.items())),
            "actions": [a.to_dict() for a in self.actions],
        }


class EnrichmentService:
    """Runs profile enrichment over records or a dataset store.

    Parameters
    ----------
    store :
        Optional :class:`DatasetStore`.  When supplied, ``enrich_store``
        persists updated records back to disk.
    """

    def __init__(self, store: DatasetStore | None = None) -> None:
        self._store = store

    def enrich(self, records: list[DatasetRecord]) -> tuple[list[DatasetRecord], EnrichmentReport]:
        """Enrich a collection of records in memory and return the results.

        Returns a tuple of (enriched_records, report).  Input records
        that gained no fields are returned unchanged.
        """
        report = EnrichmentReport(records_processed=len(records))
        enriched_records: list[DatasetRecord] = []
        for record in records:
            enriched, fields_added = enrich_record(record)
            enriched_records.append(enriched)
            if fields_added:
                report.records_updated += 1
                for field_name in fields_added:
                    report.fields_populated[field_name] = (
                        report.fields_populated.get(field_name, 0) + 1
                    )
                report.actions.append(
                    EnrichmentAction(record_id=record.record_id, fields_added=fields_added)
                )
        # Deterministic ordering for the action list.
        report.actions.sort(key=lambda action: action.record_id)
        return enriched_records, report

    def enrich_store(self) -> EnrichmentReport:
        """Enrich every record in the store and persist the changes.

        Updated records are written back through the store so the
        canonical profile fields become queryable.  Returns the report.
        """
        if self._store is None:
            msg = "EnrichmentService requires a store to run enrich_store"
            raise ValueError(msg)
        records = [
            r
            for r in (
                self._store.load_record(rid) for rid in self._store.list_records()
            )
            if r is not None
        ]
        enriched, report = self.enrich(records)
        for record in enriched:
            stored = self._store.load_record(record.record_id)
            if stored is not None and record.profile != stored.profile:
                self._store.update_record(record)
        return report


# ---- Internal helpers ----

def _first_present(metadata: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            text = str(value).strip()
            if text and text.lower() not in ("", "unknown", "-", "n/a"):
                return text
    return None


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    raw = str(value).strip().replace(",", "")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_year(value: object) -> int | None:
    year = _parse_int(value)
    if year is None:
        return None
    if 1800 <= year <= 2200:
        return year
    return None


def _parse_founded_date(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if _ISO_DATE_RE.fullmatch(text):
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _normalized_industry_list(values: list[str]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        if normalized and normalized not in seen:
            seen.add(normalized)
            order.append(normalized)
    return order


_ALPHA3_TO_ALPHA2 = {
    "USA": "US",
    "CAN": "CA",
    "AUS": "AU",
    "GBR": "GB",
    "DEU": "DE",
    "FRA": "FR",
    "NLD": "NL",
    "BEL": "BE",
    "CHE": "CH",
    "SWE": "SE",
    "NOR": "NO",
    "DNK": "DK",
    "FIN": "FI",
    "IRL": "IE",
    "ESP": "ES",
    "ITA": "IT",
    "PRT": "PT",
    "IND": "IN",
    "CHN": "CN",
    "JPN": "JP",
    "SGP": "SG",
    "ISR": "IL",
    "KOR": "KR",
    "BRA": "BR",
    "MEX": "MX",
    "EST": "EE",
    "POL": "PL",
    "AUT": "AT",
    "LUX": "LU",
    "HKG": "HK",
}

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
