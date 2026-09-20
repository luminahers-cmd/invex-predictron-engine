"""Tests for canonical company identity model and merge rules."""

from __future__ import annotations

from typing import Any

import pytest

from predictron_engine.dataset.identity import (
    CompanyIdentity,
    CompanyIdentityBuilder,
    extract_identifiers,
)
from predictron_engine.dataset.models import CompanyProfile, DatasetRecord
from tests.dataset.conftest import make_record


def _make_record(
    record_id: str,
    name: str,
    website: str = "",
    *,
    domain: str | None = None,
    country: str | None = None,
    industries: list[str] | None = None,
    legal_name: str | None = None,
    founded_year: int | None = None,
    employee_range: str | None = None,
    city: str | None = None,
    region: str | None = None,
    metadata: dict[str, Any] | None = None,
    source: str = "direct",
) -> DatasetRecord:
    profile = CompanyProfile(
        domain=domain,
        country_code=country,
        industries=industries or [],
        legal_name=legal_name,
        founded_year=founded_year,
        employee_range=employee_range,
        city=city,
        region=region,
    )
    record = make_record(
        startup_name=name,
        website=website or "https://example.com",
        record_id=record_id,
    )
    record.profile = profile
    record.source = source
    if metadata:
        record.analysis_metadata = metadata
    return record


# ---------------------------------------------------------------------------
# extract_identifiers
# ---------------------------------------------------------------------------

class TestExtractIdentifiers:
    def test_sec_cik(self) -> None:
        record = _make_record("a", "Acme", metadata={"sec_cik": "00012345"})
        ids = extract_identifiers(record)
        assert ids["sec_cik"] == "00012345"

    def test_company_number(self) -> None:
        record = _make_record("a", "Acme", metadata={"company_number": "09876543"})
        ids = extract_identifiers(record)
        assert ids["company_number"] == "09876543"

    def test_registration_number(self) -> None:
        record = _make_record("a", "Acme", metadata={"registration_number": "REG-123"})
        ids = extract_identifiers(record)
        assert ids["registration_number"] == "reg-123"

    def test_vat_number(self) -> None:
        record = _make_record("a", "Acme", metadata={"vat_number": "DE123456789"})
        ids = extract_identifiers(record)
        assert ids["vat_number"] == "de123456789"

    def test_crunchbase_url(self) -> None:
        record = _make_record(
            "a", "Acme", metadata={"crunchbase_url": "https://crunchbase.com/acme"}
        )
        ids = extract_identifiers(record)
        assert "crunchbase_url" in ids

    def test_no_metadata(self) -> None:
        record = _make_record("a", "Acme")
        ids = extract_identifiers(record)
        assert ids == {}

    def test_empty_value_ignored(self) -> None:
        record = _make_record("a", "Acme", metadata={"sec_cik": ""})
        ids = extract_identifiers(record)
        assert "sec_cik" not in ids

    def test_multiple_identifiers(self) -> None:
        record = _make_record(
            "a", "Acme", metadata={"sec_cik": "123", "company_number": "456"}
        )
        ids = extract_identifiers(record)
        assert "sec_cik" in ids
        assert "company_number" in ids

    def test_strips_whitespace(self) -> None:
        record = _make_record("a", "Acme", metadata={"sec_cik": "  123  "})
        ids = extract_identifiers(record)
        assert ids["sec_cik"] == "123"


# ---------------------------------------------------------------------------
# CompanyIdentity (model)
# ---------------------------------------------------------------------------

class TestCompanyIdentity:
    def test_defaults(self) -> None:
        identity = CompanyIdentity(canonical_name="Acme")
        assert identity.canonical_name == "Acme"
        assert identity.record_ids == []
        assert identity.aliases == []
        assert identity.alternate_domains == []
        assert identity.identifiers == {}
        assert identity.merge_decision == "singleton"

    def test_to_dict(self) -> None:
        identity = CompanyIdentity(canonical_name="Acme")
        d = identity.to_dict()
        assert d["canonical_name"] == "Acme"
        assert d["record_ids"] == []

    def test_add_record(self) -> None:
        identity = CompanyIdentity(canonical_name="Acme")
        record = _make_record("r1", "Acme Corp")
        identity.add_record(record)
        assert "r1" in identity.record_ids
        assert identity.record_names["r1"] == "Acme Corp"
        # The display name is kept in record_names (provenance).
        assert "Acme Corp" in identity.record_names.values()

    def test_add_record_idempotent(self) -> None:
        identity = CompanyIdentity(canonical_name="Acme")
        record = _make_record("r1", "Acme Corp")
        identity.add_record(record)
        identity.add_record(record)
        assert identity.record_ids.count("r1") == 1

    def test_add_record_preserves_domain(self) -> None:
        identity = CompanyIdentity(canonical_name="Acme")
        record = _make_record("r1", "Acme", domain="acme.com")
        identity.add_record(record)
        assert "acme.com" in identity.alternate_domains

    def test_add_record_preserves_industries(self) -> None:
        identity = CompanyIdentity(canonical_name="Acme")
        record = _make_record("r1", "Acme", industries=["fintech", "saas"])
        identity.add_record(record)
        assert "fintech" in identity.industries
        assert "saas" in identity.industries

    def test_add_record_preserves_country(self) -> None:
        identity = CompanyIdentity(canonical_name="Acme")
        record = _make_record("r1", "Acme", country="US")
        identity.add_record(record)
        assert "US" in identity.country_codes

    def test_add_record_preserves_founded_year(self) -> None:
        identity = CompanyIdentity(canonical_name="Acme")
        record = _make_record("r1", "Acme", founded_year=2010)
        identity.add_record(record)
        assert identity.founded_year == 2010

    def test_add_record_earliest_year(self) -> None:
        identity = CompanyIdentity(canonical_name="Acme", founded_year=2015)
        record = _make_record("r1", "Acme", founded_year=2010)
        identity.add_record(record)
        assert identity.founded_year == 2010

    def test_add_record_preserves_identifiers(self) -> None:
        record = _make_record(
            "r1", "Acme", metadata={"sec_cik": "123"}
        )
        identity = CompanyIdentity(canonical_name="Acme")
        identity.add_record(record)
        assert "sec_cik" in identity.identifiers
        assert "123" in identity.identifiers["sec_cik"]
        assert "r1" in identity.identifier_sources["sec_cik:123"]

    def test_add_record_preserves_source(self) -> None:
        record = _make_record("r1", "Acme", source="sec_edgar")
        identity = CompanyIdentity(canonical_name="Acme")
        identity.add_record(record)
        assert "sec_edgar" in identity.sources

    def test_finalize_deduplicates_aliases(self) -> None:
        identity = CompanyIdentity(
            canonical_name="Acme",
            aliases=["Acme", "Acme Corp", "Acme Corp"],
        )
        identity.finalize()
        # Canonical form and any core-equivalent alias are excluded.
        assert "Acme" not in identity.aliases
        # "Acme Corp" reduces to the same core as "Acme", so it is dropped too.
        assert identity.aliases == []

    def test_finalize_removes_canonical_from_alternate_domains(self) -> None:
        identity = CompanyIdentity(
            canonical_name="Acme",
            canonical_domain="acme.com",
            alternate_domains=["acme.com", "acme.co"],
        )
        identity.finalize()
        assert identity.canonical_domain == "acme.com"
        assert "acme.com" not in identity.alternate_domains
        assert "acme.co" in identity.alternate_domains

    def test_finalize_majority_country(self) -> None:
        identity = CompanyIdentity(
            canonical_name="Acme",
            country_codes=["US", "US", "GB"],
        )
        identity.finalize()
        # country_codes are deduplicated in finalize, so US appears once and
        # GB once; both tie on frequency and the lexicographic tiebreak wins.
        assert identity.country_code == "GB"


# ---------------------------------------------------------------------------
# CompanyIdentityBuilder
# ---------------------------------------------------------------------------

class TestCompanyIdentityBuilder:
    def test_single_record(self) -> None:
        builder = CompanyIdentityBuilder()
        record = _make_record("r1", "Acme Corp")
        identity = builder.build([record])
        assert identity.canonical_name == "Acme Corp"
        assert identity.record_ids == ["r1"]
        assert identity.merge_decision == "singleton"

    def test_multiple_records_merge(self) -> None:
        builder = CompanyIdentityBuilder()
        r1 = _make_record("r1", "Acme Corp", domain="acme.com")
        r2 = _make_record("r2", "Acme Inc", domain="acme.com")
        identity = builder.build([r1, r2], merge_decision="automatic_merge")
        assert identity.merge_decision == "automatic_merge"
        assert "r1" in identity.record_ids
        assert "r2" in identity.record_ids

    def test_canonical_name_most_frequent(self) -> None:
        builder = CompanyIdentityBuilder()
        r1 = _make_record("r1", "Acme Corp")
        r2 = _make_record("r2", "Acme Corp")
        r3 = _make_record("r3", "Acme Inc")
        identity = builder.build([r1, r2, r3])
        assert identity.canonical_name == "Acme Corp"

    def test_canonical_name_shortest_tiebreak(self) -> None:
        builder = CompanyIdentityBuilder()
        r1 = _make_record("r1", "Alpha Beta Corp")
        r2 = _make_record("r2", "Alpha Beta")
        identity = builder.build([r1, r2])
        # Both appear once; shortest wins.
        assert identity.canonical_name == "Alpha Beta"

    def test_canonical_domain_from_members(self) -> None:
        builder = CompanyIdentityBuilder()
        r1 = _make_record("r1", "Acme", domain="acme.com")
        r2 = _make_record("r2", "Acme")
        r3 = _make_record("r3", "Acme")
        identity = builder.build([r1, r2, r3])
        assert identity.canonical_domain == "acme.com"

    def test_canonical_domain_tiebreak_shortest(self) -> None:
        builder = CompanyIdentityBuilder()
        r1 = _make_record("r1", "Acme", domain="acme.com")
        r2 = _make_record("r2", "Acme", domain="acme.co")
        identity = builder.build([r1, r2])
        # alternate_domains is deduplicated, so both count once; shortest wins.
        assert identity.canonical_domain == "acme.co"

    def test_empty_raises(self) -> None:
        builder = CompanyIdentityBuilder()
        with pytest.raises(ValueError, match="empty"):
            builder.build([])

    def test_sorted_record_ids(self) -> None:
        builder = CompanyIdentityBuilder()
        r1 = _make_record("z1", "Acme")
        r2 = _make_record("a1", "Acme")
        identity = builder.build([r1, r2])
        assert identity.record_ids == ["a1", "z1"]

    def test_metadata_aliases_captured(self) -> None:
        r1 = _make_record("r1", "Acme Corp", metadata={"former_name": "Old Acme"})
        builder = CompanyIdentityBuilder()
        identity = builder.build([r1])
        assert "Old Acme" in identity.aliases

    def test_metadata_former_name_alias(self) -> None:
        r1 = _make_record("r1", "Acme", metadata={"former_name": "Old Acme Ltd"})
        builder = CompanyIdentityBuilder()
        identity = builder.build([r1])
        assert "Old Acme Ltd" in identity.aliases


# ---------------------------------------------------------------------------
# Merge correctness
# ---------------------------------------------------------------------------

class TestMergeCorrectness:
    def test_never_discards_conflicting_names(self) -> None:
        r1 = _make_record("r1", "Acme Corp")
        r2 = _make_record("r2", "Acme Industries")
        builder = CompanyIdentityBuilder()
        identity = builder.build([r1, r2])
        # Both names should appear somewhere.
        all_names = set(identity.record_names.values()) | set(identity.aliases)
        assert "Acme Corp" in all_names
        assert "Acme Industries" in all_names

    def test_never_discards_conflicting_domains(self) -> None:
        r1 = _make_record("r1", "Acme", domain="acme.com")
        r2 = _make_record("r2", "Acme", domain="acme.co")
        builder = CompanyIdentityBuilder()
        identity = builder.build([r1, r2])
        assert "acme.com" in identity.alternate_domains or identity.canonical_domain == "acme.com"
        assert "acme.co" in identity.alternate_domains or identity.canonical_domain == "acme.co"

    def test_preserves_all_sources(self) -> None:
        r1 = _make_record("r1", "Acme", source="sec_edgar")
        r2 = _make_record("r2", "Acme", source="companies_house")
        builder = CompanyIdentityBuilder()
        identity = builder.build([r1, r2])
        assert "sec_edgar" in identity.sources
        assert "companies_house" in identity.sources

    def test_preserves_all_identifiers(self) -> None:
        r1 = _make_record("r1", "Acme", metadata={"sec_cik": "111"})
        r2 = _make_record("r2", "Acme", metadata={"company_number": "222"})
        builder = CompanyIdentityBuilder()
        identity = builder.build([r1, r2])
        assert "111" in identity.identifiers.get("sec_cik", [])
        assert "222" in identity.identifiers.get("company_number", [])

    def test_identifier_sources_track_provenance(self) -> None:
        r1 = _make_record("r1", "Acme", metadata={"sec_cik": "111"})
        r2 = _make_record("r2", "Acme", metadata={"sec_cik": "111"})
        builder = CompanyIdentityBuilder()
        identity = builder.build([r1, r2])
        sources = identity.identifier_sources.get("sec_cik:111", [])
        assert "r1" in sources
        assert "r2" in sources
