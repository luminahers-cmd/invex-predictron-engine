"""Tests for predictron_engine.dataset.cli_signal — Project E4 CLI subcommands."""

from __future__ import annotations

import json

import pytest

from predictron_engine.dataset import cli
from predictron_engine.dataset.outcomes import OutcomeRecord, StartupOutcome
from predictron_engine.dataset.store import DatasetStore
from tests.dataset.conftest import make_record


def _prepare_store(tmp_path, *, funding_dates=None):
    """Build a dataset store with one record+outcome, then build signal timelines."""
    store = DatasetStore(tmp_path / "ds")
    store.initialize()
    record = make_record(startup_name="Acme", website="https://acme.example.com")
    store.save_record(record)
    funding_rounds = None
    if funding_dates:
        from tests.dataset.signals_helpers import make_funding_event
        funding_rounds = [
            make_funding_event(date=d, amount=1_000_000.0) for d in funding_dates
        ]
    outcome = OutcomeRecord(
        record_id=record.record_id,
        outcome=StartupOutcome(funding_rounds=funding_rounds or []),
    )
    store.save_outcome(outcome)
    # Build signal timelines
    from predictron_engine.dataset.signals.integration import SignalDatasetManager
    manager = SignalDatasetManager(store)
    manager.build()
    return store, record.record_id


class TestSignalImport:
    def test_import_valid_file(self, tmp_path, capsys) -> None:
        store, rec_id = _prepare_store(tmp_path)
        sig_file = tmp_path / "signals.json"
        payload = [
            {
                "company_id": f"company:{rec_id}",
                "signal_type": "founder_change",
                "timestamp": "2024-03-01T00:00:00+00:00",
                "source": "news",
            }
        ]
        sig_file.write_text(json.dumps(payload), encoding="utf-8")
        cli.main(["signal-import", str(sig_file), "--dataset", str(tmp_path / "ds")])
        out = json.loads(capsys.readouterr().out)
        assert out["imported"] == 1
        assert out["rejected"] == 0

    def test_import_invalid_signal(self, tmp_path, capsys) -> None:
        store, _ = _prepare_store(tmp_path)
        sig_file = tmp_path / "bad.json"
        sig_file.write_text(json.dumps([{"bad": "record"}]), encoding="utf-8")
        cli.main(["signal-import", str(sig_file), "--dataset", str(tmp_path / "ds")])
        out = json.loads(capsys.readouterr().out)
        assert out["rejected"] >= 1

    def test_import_empty_file(self, tmp_path, capsys) -> None:
        store, _ = _prepare_store(tmp_path)
        sig_file = tmp_path / "empty.json"
        sig_file.write_text("[]", encoding="utf-8")
        cli.main(["signal-import", str(sig_file), "--dataset", str(tmp_path / "ds")])
        out = json.loads(capsys.readouterr().out)
        assert out["imported"] == 0


class TestSignalReport:
    def test_report_structure(self, tmp_path, capsys) -> None:
        _prepare_store(tmp_path, funding_dates=["2024-01-15T00:00:00+00:00"])
        rc = cli.main(["signal-report", "--dataset", str(tmp_path / "ds")])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["report_type"] == "company_signals_report"
        assert "statistics" in out
        assert "distribution" in out

    def test_report_with_output_file(self, tmp_path) -> None:
        _prepare_store(tmp_path)
        output = tmp_path / "report.json"
        rc = cli.main(["signal-report", "--dataset", str(tmp_path / "ds"), "--output", str(output)])
        assert rc == 0
        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["report_type"] == "company_signals_report"


class TestTimeline:
    def test_timeline_overview(self, tmp_path, capsys) -> None:
        _prepare_store(tmp_path, funding_dates=["2024-01-15T00:00:00+00:00"])
        rc = cli.main(["timeline", "--dataset", str(tmp_path / "ds")])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert "company_count" in out
        assert out["company_count"] >= 1

    def test_timeline_company_flag(self, tmp_path, capsys) -> None:
        _prepare_store(tmp_path, funding_dates=["2024-01-15T00:00:00+00:00"])
        store = DatasetStore(tmp_path / "ds")
        store.initialize()
        companies = store.list_signal_company_ids()
        assert len(companies) >= 1
        company_id = companies[0]
        rc = cli.main(["timeline", "--dataset", str(tmp_path / "ds"), "--company", company_id])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["company_id"] == company_id

    def test_timeline_missing_company(self, tmp_path, capsys) -> None:
        _prepare_store(tmp_path)
        rc = cli.main(["timeline", "--dataset", str(tmp_path / "ds"), "--company", "company:nope"])
        assert rc == 1
        assert "no signals" in capsys.readouterr().err


class TestTrendReport:
    def test_overview(self, tmp_path, capsys) -> None:
        _prepare_store(tmp_path, funding_dates=["2024-01-15T00:00:00+00:00"])
        rc = cli.main(["trend-report", "--dataset", str(tmp_path / "ds")])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert isinstance(out, dict)

    def test_company_flag(self, tmp_path, capsys) -> None:
        _prepare_store(tmp_path, funding_dates=["2024-01-15T00:00:00+00:00"])
        store = DatasetStore(tmp_path / "ds")
        store.initialize()
        companies = store.list_signal_company_ids()
        company_id = companies[0]
        rc = cli.main(["trend-report", "--dataset", str(tmp_path / "ds"), "--company", company_id])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert "funding_velocity" in out["trends"]

    def test_missing_company_returns_error(self, tmp_path, capsys) -> None:
        _prepare_store(tmp_path)
        args = ["trend-report", "--dataset", str(tmp_path / "ds"), "--company", "company:nope"]
        rc = cli.main(args)
        assert rc == 1
        assert "error" in capsys.readouterr().err.lower()

    def test_output_file(self, tmp_path) -> None:
        _prepare_store(tmp_path)
        output = tmp_path / "trends.json"
        rc = cli.main(["trend-report", "--dataset", str(tmp_path / "ds"), "--output", str(output)])
        assert rc == 0
        assert output.exists()


class TestSignalValidate:
    def test_valid_exits_zero(self, tmp_path, capsys) -> None:
        _prepare_store(tmp_path, funding_dates=["2024-01-15T00:00:00+00:00"])
        rc = cli.main(["signal-validate", "--dataset", str(tmp_path / "ds")])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["is_valid"] is True

    def test_empty_exits_zero(self, tmp_path, capsys) -> None:
        store = DatasetStore(tmp_path / "ds")
        store.initialize()
        rc = cli.main(["signal-validate", "--dataset", str(tmp_path / "ds")])
        assert rc == 0


class TestParserWiring:
    def test_all_subcommands_present(self) -> None:
        parser = cli.build_parser()
        names = parser._subparsers._group_actions[0].choices  # type: ignore[union-attr]
        commands = ("signal-import", "signal-report", "timeline", "trend-report", "signal-validate")
        for cmd in commands:
            assert cmd in names, cmd

    def test_main_with_empty_args(self) -> None:
        with pytest.raises(SystemExit):
            cli.main([])
