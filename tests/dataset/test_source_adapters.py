"""Tests for source adapters (Part C)."""

from __future__ import annotations

import csv
import json

from predictron_engine.dataset.imports import ImportSourceRegistry
from predictron_engine.dataset.sources import (
    GovRegistrySource,
    SecEdgarSource,
    YcOssSource,
)


def _write_json(tmp_path, name: str, data) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _write_csv(tmp_path, name: str, headers, rows, delimiter=",") -> str:
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


class TestSecEdgarSource:
    def test_reads_edgar_json(self, tmp_path) -> None:
        data = [
            {
                "cik": "0001234567",
                "company_name": "Example Corp",
                "website_url": "https://example.com",
                "state_of_incorporation": "DE",
                "sic_code": "7372",
                "sic_description": "Prepackaged Software",
                "latest_filing_date": "2024-01-15",
            }
        ]
        path = _write_json(tmp_path, "edgar.json", data)
        source = SecEdgarSource()
        records = source.read(path)
        assert len(records) == 1
        assert records[0].startup_name == "Example Corp"
        assert records[0].website == "https://example.com"
        assert records[0].metadata["sec_cik"] == "0001234567"
        assert records[0].metadata["sector"] == "Prepackaged Software"

    def test_source_name(self) -> None:
        assert SecEdgarSource().source_name == "sec_edgar"


class TestYcOssSource:
    def test_reads_yc_oss_csv(self, tmp_path) -> None:
        headers = ["company", "website", "description", "region", "industry"]
        rows = [
            {
                "company": "Stripe",
                "website": "stripe.com",
                "description": "Payments",
                "region": "San Francisco",
                "industry": "FinTech",
            }
        ]
        path = _write_csv(tmp_path, "yc.csv", headers, rows)
        source = YcOssSource()
        records = source.read(path)
        assert len(records) == 1
        assert records[0].startup_name == "Stripe"
        assert records[0].website == "https://stripe.com"
        assert records[0].metadata["sector"] == "FinTech"

    def test_source_name(self) -> None:
        assert YcOssSource().source_name == "yc_oss"


class TestCompaniesHouse:
    def test_reads_companies_house_csv(self, tmp_path) -> None:
        headers = [
            "company_number",
            "company_name",
            "company_status",
            "country_of_origin",
            "incorporation_date",
            "sic_code_1",
        ]
        rows = [
            {
                "company_number": "01234567",
                "company_name": "Acme Ltd",
                "company_status": "Active",
                "country_of_origin": "United Kingdom",
                "incorporation_date": "2020-06-01",
                "sic_code_1": "62020",
            }
        ]
        path = _write_csv(tmp_path, "ch.csv", headers, rows)
        registry = GovRegistrySource()
        records = registry.read(path)
        assert len(records) == 1
        assert records[0].startup_name == "Acme Ltd"
        assert records[0].metadata["company_number"] == "01234567"
        assert records[0].metadata["country"] == "United Kingdom"

    def test_detects_us_state_format(self, tmp_path) -> None:
        headers = ["company_name", "website", "jurisdiction", "status"]
        rows = [
            {
                "company_name": "Beta LLC",
                "website": "beta.com",
                "jurisdiction": "Texas",
                "status": "Active",
            }
        ]
        path = _write_csv(tmp_path, "state.csv", headers, rows)
        registry = GovRegistrySource()
        records = registry.read(path)
        assert len(records) == 1
        assert records[0].startup_name == "Beta LLC"
        assert records[0].website == "https://beta.com"
        assert records[0].metadata["country_code"] == "Texas"


class TestRegistry:
    def test_default_registry_registers_all(self) -> None:
        registry = ImportSourceRegistry.default()
        names = registry.list_sources()
        assert "json_file" in names
        assert "csv_file" in names
        assert "sec_edgar" in names
        assert "yc_oss" in names
        assert "gov_registries" in names

    def test_can_retrieve_sources(self) -> None:
        registry = ImportSourceRegistry.default()
        assert registry.get("sec_edgar") is not None
        assert registry.get("yc_oss") is not None
        assert registry.get("gov_registries") is not None
