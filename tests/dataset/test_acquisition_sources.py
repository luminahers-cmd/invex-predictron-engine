"""Tests for acquisition source connectors."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from predictron_engine.dataset.acquisition.sources.base import (
    FetchResult,
    SourceDescriptor,
)
from predictron_engine.dataset.acquisition.sources.companies_house_connector import (
    CompaniesHouseConnector,
)
from predictron_engine.dataset.acquisition.sources.csv_export_connector import (
    CsvExportConnector,
)
from predictron_engine.dataset.acquisition.sources.registry import (
    SourceConnectorRegistry,
)
from predictron_engine.dataset.acquisition.sources.sec_edgar_connector import (
    SecEdgarConnector,
)
from predictron_engine.dataset.acquisition.sources.yc_oss_connector import (
    YcOssConnector,
)


@pytest.fixture()
def edgar_json(tmp_path: Path) -> Path:
    data = [
        {
            "cik": "0001234567",
            "company_name": "Test Corp",
            "website_url": "https://test.com",
            "state_of_incorporation": "DE",
            "sic_code": "7372",
            "sic_description": "Software",
            "latest_filing_date": "2024-01-15",
            "form_type": "10-K",
        }
    ]
    p = tmp_path / "edgar.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def yc_csv(tmp_path: Path) -> Path:
    p = tmp_path / "yc.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["company", "website", "description"])
        writer.writeheader()
        writer.writerow({
            "company": "Acme Corp",
            "website": "https://acme.com",
            "description": "Widgets",
        })
    return p


@pytest.fixture()
def ch_csv(tmp_path: Path) -> Path:
    p = tmp_path / "companies_house.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "company_number", "company_name", "company_status",
            "country_of_origin", "incorporation_date",
        ])
        writer.writeheader()
        writer.writerow({
            "company_number": "12345678",
            "company_name": "UK Startup Ltd",
            "company_status": "active",
            "country_of_origin": "GB",
            "incorporation_date": "2020-01-15",
        })
    return p


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "startups.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Website"])
        writer.writeheader()
        writer.writerow({"Name": "DataBot", "Website": "https://databot.io"})
    return p


class TestSecEdgarConnector:
    def test_descriptor(self) -> None:
        conn = SecEdgarConnector()
        d = conn.descriptor
        assert d.name == "sec_edgar"
        assert d.reliability == "high"

    def test_discover_local(self, edgar_json: Path) -> None:
        conn = SecEdgarConnector()
        files = conn.discover({"file_path": str(edgar_json)})
        assert len(files) == 1
        assert files[0] == str(edgar_json)

    def test_discover_missing(self) -> None:
        conn = SecEdgarConnector()
        assert conn.discover({"file_path": "/nonexistent.json"}) == []

    def test_fetch(self, edgar_json: Path, tmp_path: Path) -> None:
        conn = SecEdgarConnector()
        dest = tmp_path / "fetched"
        result = conn.fetch(str(edgar_json), str(dest))
        assert isinstance(result, FetchResult)
        assert Path(result.file_path).exists()
        assert len(result.file_hash) == 64

    def test_normalize(self, edgar_json: Path) -> None:
        conn = SecEdgarConnector()
        records = conn.normalize(str(edgar_json))
        assert len(records) == 1
        assert records[0].startup_name == "Test Corp"
        assert records[0].website == "https://test.com"
        assert records[0].source == "sec_edgar"

    def test_checkpoint(self, edgar_json: Path) -> None:
        conn = SecEdgarConnector()
        h = conn.checkpoint(str(edgar_json))
        assert len(h) == 64


class TestCompaniesHouseConnector:
    def test_descriptor(self) -> None:
        conn = CompaniesHouseConnector()
        d = conn.descriptor
        assert d.name == "companies_house"
        assert d.reliability == "high"

    def test_normalize(self, ch_csv: Path) -> None:
        conn = CompaniesHouseConnector()
        records = conn.normalize(str(ch_csv))
        assert len(records) == 1
        assert records[0].startup_name == "UK Startup Ltd"
        assert records[0].source == "companies_house"


class TestYcOssConnector:
    def test_descriptor(self) -> None:
        conn = YcOssConnector()
        d = conn.descriptor
        assert d.name == "yc_oss"

    def test_normalize(self, yc_csv: Path) -> None:
        conn = YcOssConnector()
        records = conn.normalize(str(yc_csv))
        assert len(records) == 1
        assert records[0].startup_name == "Acme Corp"
        assert records[0].website == "https://acme.com"
        assert records[0].source == "yc_oss"


class TestCsvExportConnector:
    def test_descriptor(self) -> None:
        conn = CsvExportConnector()
        d = conn.descriptor
        assert d.name == "csv_export"

    def test_discover_directory(self, tmp_path: Path) -> None:
        conn = CsvExportConnector()
        for name in ["a.csv", "b.csv", "c.txt"]:
            (tmp_path / name).write_text("dummy")
        files = conn.discover({"directory": str(tmp_path)})
        assert len(files) == 2

    def test_normalize(self, csv_file: Path) -> None:
        conn = CsvExportConnector()
        records = conn.normalize(str(csv_file))
        assert len(records) == 1
        assert records[0].startup_name == "DataBot"


class TestSourceConnectorRegistry:
    def test_default_has_all(self) -> None:
        reg = SourceConnectorRegistry.default()
        names = reg.list_sources()
        assert "sec_edgar" in names
        assert "companies_house" in names
        assert "yc_oss" in names
        assert "csv_export" in names

    def test_get_connector(self) -> None:
        reg = SourceConnectorRegistry.default()
        conn = reg.get("yc_oss")
        assert conn is not None
        assert conn.descriptor.name == "yc_oss"

    def test_list_descriptors(self) -> None:
        reg = SourceConnectorRegistry.default()
        descs = reg.list_descriptors()
        assert len(descs) >= 4
        names = {d["name"] for d in descs}
        assert "sec_edgar" in names
