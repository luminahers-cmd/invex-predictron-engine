"""Tests for company profile enrichment (Project E1)."""

from __future__ import annotations

from predictron_engine.dataset.enrichment import (
    EnrichmentReport,
    EnrichmentService,
    employee_range,
    enrich_record,
    extract_domain,
    merge_profiles,
    normalize_country_code,
    profile_from_metadata,
)
from predictron_engine.dataset.models import CompanyProfile
from tests.dataset.conftest import make_record


class TestExtractDomain:
    def test_full_url(self) -> None:
        assert extract_domain("https://www.Acme.com/path?q=1") == "acme.com"

    def test_scheme_less(self) -> None:
        assert extract_domain("acme.com") == "acme.com"

    def test_www_stripped(self) -> None:
        assert extract_domain("https://www.example.co.uk") == "example.co.uk"

    def test_empty(self) -> None:
        assert extract_domain("") is None
        assert extract_domain(None) is None


class TestNormalizeCountryCode:
    def test_two_letter_passthrough(self) -> None:
        assert normalize_country_code("US") == "US"
        assert normalize_country_code("de") == "DE"

    def test_uk_maps_to_gb(self) -> None:
        assert normalize_country_code("UK") == "GB"
        assert normalize_country_code("uk") == "GB"

    def test_country_name(self) -> None:
        assert normalize_country_code("United States") == "US"
        assert normalize_country_code("england") == "GB"
        assert normalize_country_code("Germany") == "DE"

    def test_three_letter(self) -> None:
        assert normalize_country_code("USA") == "US"
        assert normalize_country_code("GBR") == "GB"

    def test_unknown(self) -> None:
        assert normalize_country_code("") is None
        assert normalize_country_code(None) is None


class TestEmployeeRange:
    def test_bands(self) -> None:
        assert employee_range(5) == "1-10"
        assert employee_range(25) == "11-50"
        assert employee_range(150) == "51-200"
        assert employee_range(12000) == "10001+"

    def test_none(self) -> None:
        assert employee_range(None) is None
        assert employee_range(-1) is None


class TestProfileFromMetadata:
    def test_full_metadata(self) -> None:
        meta = {
            "industry_category": "AI Infrastructure",
            "country": "United States",
            "state_of_incorporation": "Delaware",
            "city": "San Francisco",
            "description": "Builds AI infra",
            "founded_year": "2019",
            "employee_count": "45",
            "company_status": "active",
            "legal_name": "Acme Systems, Inc.",
        }
        profile = profile_from_metadata(meta, website="https://www.acme.com")
        assert profile.domain == "acme.com"
        assert profile.industries == ["ai infrastructure"]
        assert profile.country_code == "US"
        assert profile.city == "San Francisco"
        assert "Delaware" in (profile.region or "")
        assert profile.founded_year == 2019
        assert profile.employee_count == 45
        assert profile.employee_range == "11-50"
        assert profile.status == "active"
        assert profile.description == "Builds AI infra"

    def test_domain_from_website(self) -> None:
        profile = profile_from_metadata({}, website="https://beta.com")
        assert profile.domain == "beta.com"

    def test_empty_metadata_yields_empty_profile(self) -> None:
        profile = profile_from_metadata({}, website="")
        assert profile.is_empty()

    def test_unknown_country_not_normalized(self) -> None:
        profile = profile_from_metadata({"country": "Atlantis"})
        assert profile.country_code is None


class TestMergeProfiles:
    def test_only_fills_empty_fields(self) -> None:
        base = CompanyProfile(domain="acme.com", country_code="US")
        incoming = CompanyProfile(
            domain="www.acme.com",
            industries=["software"],
            country_code="DE",
        )
        merged = merge_profiles(base, incoming)
        # base domain/country preserved
        assert merged.domain == "acme.com"
        assert merged.country_code == "US"
        # industries unioned
        assert merged.industries == ["software"]

    def test_industries_union_deduped(self) -> None:
        base = CompanyProfile(industries=["software"])
        incoming = CompanyProfile(industries=["software", "SaaS"])
        merged = merge_profiles(base, incoming)
        assert merged.industries == ["software", "saas"]

    def test_inputs_not_mutated(self) -> None:
        base = CompanyProfile(domain="a.com")
        incoming = CompanyProfile(domain="b.com", industries=["x"])
        _ = merge_profiles(base, incoming)
        assert base.domain == "a.com"
        assert incoming.domain == "b.com"


class TestEnrichRecord:
    def test_metadata_promoted_to_profile(self) -> None:
        record = make_record(
            startup_name="Acme",
            website="https://acme.com",
        ).model_copy(
            update={"analysis_metadata": {"sector": "Software", "country": "US"}}
        )
        enriched, fields_added = enrich_record(record)
        assert enriched.profile.domain == "acme.com"
        assert enriched.profile.industries == ["software"]
        assert enriched.profile.country_code == "US"
        assert "domain" in fields_added
        # original untouched
        assert record.profile.is_empty()

    def test_no_change_when_profile_complete(self) -> None:
        record = make_record(
            website="https://acme.com"
        ).model_copy(
            update={"profile": CompanyProfile(domain="acme.com")}
        )
        enriched, fields_added = enrich_record(record)
        assert fields_added == []
        assert enriched is record

    def test_existing_profile_not_overwritten(self) -> None:
        record = make_record(
            website="https://acme.com"
        ).model_copy(
            update={
                "profile": CompanyProfile(domain="canonical.example.com"),
                "analysis_metadata": {"sector": "Software"},
            }
        )
        enriched, _ = enrich_record(record)
        # Domain preserved; industry added
        assert enriched.profile.domain == "canonical.example.com"
        assert enriched.profile.industries == ["software"]


class TestEnrichmentService:
    def test_enrich_in_memory(self) -> None:
        records = [
            make_record(record_id="a").model_copy(
                update={"analysis_metadata": {"country": "US"}}
            ),
            make_record(record_id="b", website="https://beta.com"),
        ]
        service = EnrichmentService()
        enriched, report = service.enrich(records)
        assert report.records_processed == 2
        assert report.records_updated == 2
        assert enriched[0].profile.country_code == "US"
        assert enriched[1].profile.domain == "beta.com"

    def test_enrich_report_fields_populated(self) -> None:
        records = [
            make_record(record_id="a").model_copy(
                update={"analysis_metadata": {"country": "US", "sector": "AI"}}
            )
        ]
        service = EnrichmentService()
        _, report = service.enrich(records)
        assert report.fields_populated["country_code"] == 1
        assert report.fields_populated["industries"] == 1

    def test_enrich_store_new_and_existing(self, dataset_store) -> None:
        existing = make_record(
            record_id="existing",
            website="https://ex.com",
        ).model_copy(
            update={"analysis_metadata": {"country": "Canada"}}
        )
        dataset_store.save_record(existing)
        service = EnrichmentService(dataset_store)
        report = service.enrich_store()
        assert report.records_processed == 1
        assert report.records_updated == 1

        loaded = dataset_store.load_record("existing")
        assert loaded is not None
        assert loaded.profile.country_code == "CA"
        assert loaded.profile.domain == "ex.com"

    def test_enrich_store_persists_update_not_new(self, dataset_store) -> None:
        rec = make_record(record_id="r", website="https://r.com")
        dataset_store.save_record(rec)
        before = dataset_store.count_records()
        service = EnrichmentService(dataset_store)
        service.enrich_store()
        assert dataset_store.count_records() == before


class TestEnrichmentReport:
    def test_to_dict_roundtrip(self) -> None:
        report = EnrichmentReport(
            records_processed=2,
            records_updated=1,
            fields_populated={"domain": 1},
        )
        d = report.to_dict()
        assert d["records_processed"] == 2
        assert d["enrichment_rate"] == 0.5

    def test_zero_division_rate(self) -> None:
        report = EnrichmentReport(records_processed=0)
        assert report.to_dict()["enrichment_rate"] == 0.0


def test_founded_year_from_date() -> None:
    meta = {"incorporation_date": "2015-06-01"}
    profile = profile_from_metadata(meta)
    assert profile.founded_year == 2015
    assert profile.founded_date is not None


def test_founded_date_parsed() -> None:
    meta = {"founding_date": "2018-04-12"}
    profile = profile_from_metadata(meta)
    assert profile.founded_date is not None
    assert profile.founded_date.year == 2018
