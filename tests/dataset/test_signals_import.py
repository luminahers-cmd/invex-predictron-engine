"""Tests for predictron_engine.dataset.signals.importers — Project E4."""

from __future__ import annotations

import json

from predictron_engine.dataset.signals.importers import (
    SignalImportReport,
    import_signals_from_json,
    parse_signal_record,
)
from predictron_engine.dataset.signals.model import SignalType


def _write(payload, tmp_path, filename: str = "signals.json") -> str:
    path = tmp_path / filename
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _valid_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "company_id": "company:a",
        "signal_type": "founder_change",
        "timestamp": "2024-03-01T00:00:00+00:00",
        "source": "news",
        "provenance": "imported:json",
    }
    record.update(overrides)
    return record


VALID_RECORD = _valid_record()


class TestImportSignalsFromJson:
    def test_happy_path(self, tmp_path) -> None:
        signals, report = import_signals_from_json(
            _write([VALID_RECORD], tmp_path)
        )
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.FOUNDER_CHANGE
        assert report.imported == 1
        assert report.rejected == 0
        assert report.requested == 1

    def test_rejects_missing_file(self, tmp_path) -> None:
        signals, report = import_signals_from_json(tmp_path / "nope.json")
        assert signals == []
        assert report.validation_errors == [
            {"index": None, "error": f"file not found: {tmp_path / 'nope.json'}"}
        ]

    def test_rejects_non_list_payload(self, tmp_path) -> None:
        signals, report = import_signals_from_json(_write({"not": "a list"}, tmp_path))
        assert signals == []
        assert "JSON array" in report.validation_errors[0]["error"]

    def test_rejects_non_object_record(self, tmp_path) -> None:
        signals, report = import_signals_from_json(_write([42], tmp_path))
        assert signals == []
        assert report.rejected == 1
        assert report.validation_errors[0]["error"] == "record is not an object"

    def test_rejects_invalid_signal(self, tmp_path) -> None:
        bad = _valid_record(timestamp="not-a-date")
        signals, report = import_signals_from_json(_write([bad], tmp_path))
        assert signals == []
        assert report.rejected == 1

    def test_rejects_missing_company(self, tmp_path) -> None:
        bad = _valid_record(company_id="", startup_name="", website="")
        signals, report = import_signals_from_json(_write([bad], tmp_path))
        assert signals == []
        assert report.rejected == 1
        assert "no resolvable company" in report.validation_errors[0]["error"]

    def test_report_counts_and_companies(self, tmp_path) -> None:
        payload = [
            _valid_record(company_id="company:a"),
            _valid_record(company_id="company:b"),
            _valid_record(timestamp="bad"),
        ]
        signals, report = import_signals_from_json(_write(payload, tmp_path))
        assert report.imported == 2
        assert report.rejected == 1
        assert report.companies == ["company:a", "company:b"]

    def test_recomputes_signal_id(self, tmp_path) -> None:
        record = dict(VALID_RECORD)
        record["signal_id"] = "fake"
        signals, _ = import_signals_from_json(_write([record], tmp_path))
        assert signals[0].signal_id != "fake"
        assert len(signals[0].signal_id) == 64


class TestCompanyResolution:
    def test_precedence_company_id_first(self) -> None:
        record = _valid_record(
            company_id="ramp",
            record_id="rec-1",
            startup_name="Ramp",
            website="https://ramp.example.com",
        )
        signal = parse_signal_record(record)
        assert signal.company_id == "ramp"

    def test_company_map_lookup_wins(self) -> None:
        record = _valid_record(startup_name="Ramp", website="")
        signal = parse_signal_record(
            record, company_map={"Ramp": "company:x"}
        )
        assert signal.company_id == "company:x"

    def test_company_map_scans_all_keys(self) -> None:
        record = _valid_record(
            company_id="", record_id="rec-9", startup_name="", website="https://w.example.com"
        )
        signal = parse_signal_record(
            record, company_map={"https://w.example.com": "company:y"}
        )
        assert signal.company_id == "company:y"

    def test_unresolved_falls_back_to_name(self) -> None:
        record = _valid_record(company_id="", startup_name="Ramp")
        signal = parse_signal_record(record)
        assert signal.company_id == "Ramp"

    def test_unresolved_website_fallback(self) -> None:
        record = _valid_record(company_id="", startup_name="", website="https://w.example.com")
        signal = parse_signal_record(record)
        assert signal.company_id == "https://w.example.com"

    def test_fully_empty(self) -> None:
        signal = parse_signal_record(
            {"signal_type": "ipo", "timestamp": "2024-01-01T00:00:00+00:00"}
        )
        assert signal.company_id == ""


class TestParseSignalRecord:
    def test_preserves_metadata_and_evidence(self) -> None:
        record = _valid_record(
            metadata={"round_type": "series_b"},
            evidence={"reference": "https://news.example.com", "description": "d"},
        )
        signal = parse_signal_record(record)
        assert signal.metadata["round_type"] == "series_b"
        assert signal.evidence.reference == "https://news.example.com"

    def test_unknown_fields_ignored(self) -> None:
        record = dict(_valid_record(extra="junk", number=42))
        signal = parse_signal_record(record)
        assert signal.company_id == "company:a"

    def test_default_provenance(self) -> None:
        record = _valid_record()
        del record["provenance"]
        signal = parse_signal_record(record)
        assert signal.provenance == "manual"


class TestSignalImportReport:
    def test_to_dict(self) -> None:
        report = SignalImportReport(
            requested=3, imported=2, rejected=1,
            companies=["company:a"], per_type={"ipo": 2},
        )
        d = report.to_dict()
        assert d["requested"] == 3
        assert d["imported"] == 2
        assert d["rejected"] == 1
        assert d["companies"] == ["company:a"]
        assert d["per_type"] == {"ipo": 2}

    def test_defaults(self) -> None:
        report = SignalImportReport()
        assert report.to_dict()["requested"] == 0
