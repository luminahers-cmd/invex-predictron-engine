"""Tests for the dataset CLI (Part A)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

from predictron_engine.dataset import cli
from predictron_engine.dataset.models import DecisionLabel
from predictron_engine.dataset.store import DatasetStore
from tests.dataset.conftest import make_record


def _import_json(tmp_path, filename: str, data) -> str:
    path = tmp_path / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestCliImport:
    def test_import_creates_records(self, tmp_path) -> None:
        data = [
            {
                "startup_name": "Foo",
                "website": "https://foo.example.com",
                "prediction": {
                    "decision": "invest",
                    "confidence": 0.8,
                    "composite_score": 70.0,
                },
            }
        ]
        data_file = _import_json(tmp_path, "data.json", data)
        dataset = tmp_path / "ds"
        rc = cli.main(
            ["import", data_file, "--dataset", str(dataset)]
        )
        assert rc == 0
        store = DatasetStore(dataset)
        store.initialize()
        assert store.count_records() == 1

    def test_import_rejects_invalid(self, tmp_path, capsys) -> None:
        data = [
            {"startup_name": "", "website": ""},
            {
                "startup_name": "Valid",
                "website": "https://valid.example.com",
                "prediction": {
                    "decision": "pass",
                    "confidence": 0.5,
                    "composite_score": 40.0,
                },
            },
        ]
        data_file = _import_json(tmp_path, "data.json", data)
        dataset = tmp_path / "ds"
        rc = cli.main(["import", data_file, "--dataset", str(dataset)])
        assert rc == 0
        store = DatasetStore(dataset)
        store.initialize()
        assert store.count_records() == 1
        err = capsys.readouterr().err
        assert "record 0 rejected" in err


class TestCliVerify:
    def test_verify_valid_store(self, tmp_path) -> None:
        dataset = tmp_path / "ds"
        store = DatasetStore(dataset)
        store.initialize()
        record = make_record()
        store.save_record(record)
        rc = cli.main(["verify", "--dataset", str(dataset)])
        assert rc == 0

    def test_verify_invalid_store(self, tmp_path) -> None:
        dataset = tmp_path / "ds"
        store = DatasetStore(dataset)
        store.initialize()
        bad = make_record().model_copy(
            update={"analysis_date": datetime.now(UTC) + __import__("datetime").timedelta(days=30)}
        )
        store.save_record(bad)
        rc = cli.main(["verify", "--dataset", str(dataset)])
        assert rc == 1


class TestCliStatsExport:
    def test_stats_output(self, tmp_path, capsys) -> None:
        dataset = tmp_path / "ds"
        store = DatasetStore(dataset)
        store.initialize()
        store.save_record(make_record())
        rc = cli.main(["stats", "--dataset", str(dataset)])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["dataset_summary"]["record_count"] == 1

    def test_export_writes_file(self, tmp_path) -> None:
        dataset = tmp_path / "ds"
        store = DatasetStore(dataset)
        store.initialize()
        store.save_record(make_record())
        out_path = tmp_path / "out" / "report.json"
        rc = cli.main(
            ["export", "--dataset", str(dataset), "--output", str(out_path)]
        )
        assert rc == 0
        assert out_path.exists()
        parsed = json.loads(out_path.read_text(encoding="utf-8"))
        assert parsed["dataset_summary"]["record_count"] == 1


class TestCliEvaluate:
    def test_evaluate(self, tmp_path) -> None:
        dataset = tmp_path / "ds"
        store = DatasetStore(dataset)
        store.initialize()
        record = make_record(decision=DecisionLabel.INVEST)
        store.save_record(record)
        from predictron_engine.dataset.outcomes import OutcomeRecord, OutcomeStatus, StartupOutcome

        outcome = OutcomeRecord(
            record_id=record.record_id,
            outcome=StartupOutcome(
                acquisition="Acq",
                status=OutcomeStatus.FULLY_VERIFIED,
            ),
        )
        outcome.outcome.acquisition_price_usd = 1_000_000
        store.save_outcome(outcome)

        rc = cli.main(["evaluate", "--dataset", str(dataset)])
        assert rc == 0
        assert store.count_evaluations() == 1


class TestCliAnalyze:
    def test_analyze_record_not_found(self, tmp_path, capsys) -> None:
        dataset = tmp_path / "ds"
        store = DatasetStore(dataset)
        store.initialize()
        rc = cli.main(
            [
                "analyze",
                "--dataset",
                str(dataset),
                "--record-id",
                "nonexistent",
            ]
        )
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    def test_analyze_runs(self, tmp_path, monkeypatch) -> None:
        dataset = tmp_path / "ds"
        store = DatasetStore(dataset)
        store.initialize()
        record = make_record()
        store.save_record(record)

        metadata = SimpleNamespace(
            engine_version="0.12.1",
            processing_time_ms=10.0,
            pipeline_stages_completed=["normalize"],
        )
        decision = SimpleNamespace(
            category=SimpleNamespace(value="invest"), composite_score=70.0
        )
        report = SimpleNamespace(
            analysis_metadata=metadata,
            investment_decision=decision,
            overall_confidence=0.8,
            overall_score=70.0,
            scores=[SimpleNamespace(dimension="market", score=70.0)],
            recommendations=[],
            investment_readiness=SimpleNamespace(readiness_score=75.0),
        )

        fake_engine = type(
            "FakeEngine",
            (),
            {"analyze": lambda self, request, evidence_bundle=None: report},
        )()

        monkeypatch.setattr(cli, "_build_engine", lambda: fake_engine)
        rc = cli.main(["analyze", "--dataset", str(dataset)])
        assert rc == 0
        assert store.count_runs() == 1


class TestCliParser:
    def test_no_command_raises(self) -> None:
        import pytest

        with pytest.raises(SystemExit):
            cli.main([])

    def test_build_parser_has_commands(self) -> None:
        import argparse

        parser = cli.build_parser()
        sub = None
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                sub = action
                break
        assert sub is not None
        for cmd in ("import", "analyze", "evaluate", "verify", "stats", "export"):
            assert cmd in sub.choices
