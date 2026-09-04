"""Tests for the CSV import adapter (Part A)."""

from __future__ import annotations

import csv

from predictron_engine.dataset.csv_import import CsvFileSource


def _write_csv(tmp_path, name: str, headers, rows) -> str:
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


class TestCsvFileSource:
    def test_basic_import(self, tmp_path) -> None:
        headers = ["startup_name", "website"]
        rows = [
            {"startup_name": "Acme", "website": "https://acme.example.com"},
            {"startup_name": "Beta", "website": "https://beta.example.com"},
        ]
        path = _write_csv(tmp_path, "data.csv", headers, rows)
        source = CsvFileSource()
        records = source.read(path)
        assert len(records) == 2
        assert records[0].startup_name == "Acme"
        assert records[0].website == "https://acme.example.com"
        assert records[0].source == "csv_file"

    def test_missing_file_returns_empty(self, tmp_path) -> None:
        source = CsvFileSource()
        assert source.read(str(tmp_path / "nope.csv")) == []

    def test_flexible_column_mapping(self, tmp_path) -> None:
        headers = ["Name", "URL"]
        rows = [{"Name": "Acme Corp", "URL": "https://acme.example.com"}]
        path = _write_csv(tmp_path, "alt.csv", headers, rows)
        source = CsvFileSource()
        records = source.read(path)
        assert len(records) == 1
        assert records[0].startup_name == "Acme Corp"
        assert records[0].website == "https://acme.example.com"

    def test_prediction_and_outcome_columns(self, tmp_path) -> None:
        headers = [
            "startup_name",
            "website",
            "confidence",
            "decision",
            "composite_score",
            "outcome_status",
            "total_funding_usd",
        ]
        rows = [
            {
                "startup_name": "Foo",
                "website": "https://foo.example.com",
                "confidence": "0.8",
                "decision": "invest",
                "composite_score": "72.0",
                "outcome_status": "fully_verified",
                "total_funding_usd": "5000000",
            }
        ]
        path = _write_csv(tmp_path, "full.csv", headers, rows)
        source = CsvFileSource()
        records = source.read(path)
        assert records[0].prediction_data["confidence"] == "0.8"
        assert records[0].prediction_data["decision"] == "invest"
        assert records[0].outcome_data["status"] == "fully_verified"
        assert records[0].outcome_data["total_funding_usd"] == "5000000"

    def test_validation(self) -> None:
        from predictron_engine.dataset.imports import RawImportRecord

        source = CsvFileSource()
        valid = RawImportRecord(startup_name="X", website="https://x.com")
        assert source.validate(valid) == []
        invalid = RawImportRecord(startup_name="", website="")
        assert "startup_name is required" in source.validate(invalid)
        assert "website is required" in source.validate(invalid)
