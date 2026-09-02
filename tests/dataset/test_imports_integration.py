"""End-to-end integration tests for the dataset pipeline."""

from __future__ import annotations

import json

from predictron_engine.dataset.cli import main as cli_main
from predictron_engine.dataset.imports import (
    ImportPipeline,
    ImportSourceRegistry,
)
from predictron_engine.dataset.models import DecisionLabel
from predictron_engine.dataset.outcomes import OutcomeStatus
from predictron_engine.dataset.store import DatasetStore


def _write_json(tmp_path, name: str, data) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestImportPipelineIntegration:
    def test_import_then_evaluate_then_verify(self, tmp_path) -> None:
        """Full workflow: import -> evaluate -> verify -> stats."""
        data = [
            {
                "startup_name": "Wonky",
                "website": "https://wonky.example.com",
                "analysis_date": "2022-01-01T00:00:00Z",
                "prediction": {
                    "decision": "invest",
                    "confidence": 0.8,
                    "composite_score": 72.0,
                },
                "outcome": {
                    "acquisition": "Corp",
                    "status": "fully_verified",
                },
                "metadata": {"recorded_at": "2023-01-01T00:00:00Z"},
            }
        ]
        data_file = _write_json(tmp_path, "data.json", data)
        dataset = tmp_path / "ds"

        # import
        assert cli_main(["import", data_file, "--dataset", str(dataset)]) == 0
        store = DatasetStore(dataset)
        store.initialize()
        assert store.count_records() == 1
        assert store.count_outcomes() == 1

        record = store.load_record(store.list_records()[0])
        outcome = store.find_outcome_by_record(record.record_id)
        assert outcome is not None
        # imported outcome verdict should be derived
        assert outcome.derive_verdict().value == "success"

        # evaluate
        assert cli_main(["evaluate", "--dataset", str(dataset)]) == 0
        assert store.count_evaluations() == 1

        ev = store.find_evaluation_by_record(record.record_id)
        assert ev is not None
        assert ev.record_id == record.record_id
        # prediction in evaluation must be the imported prediction
        assert ev.prediction.composite_score == 72.0

        # verify - should be valid (all timestamps are in the past)
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_main(["verify", "--dataset", str(dataset)])
        assert rc == 0

        # stats
        with contextlib.redirect_stdout(buf):
            assert cli_main(["stats", "--dataset", str(dataset)]) == 0

    def test_import_pipeline_reuses_registry(self, tmp_path) -> None:
        data = [
            {
                "startup_name": "Foo",
                "website": "https://foo.example.com",
                "prediction": {
                    "decision": "watch",
                    "confidence": 0.5,
                    "composite_score": 50.0,
                },
            }
        ]
        data_file = _write_json(tmp_path, "data.json", data)
        registry = ImportSourceRegistry.default()
        source = registry.get("json_file")
        assert source is not None
        pipeline = ImportPipeline(source)
        result = pipeline.run(data_file)
        assert result.records_imported == 1
        assert result.imported_records[0].prediction.decision == DecisionLabel.WATCH


class TestOutcomeStatusPreserved:
    def test_import_preserves_outcome_status(self, tmp_path) -> None:
        """The pipeline must reuse V1-style outcome statuses, never fabricate."""
        data = [
            {
                "startup_name": "Partial",
                "website": "https://partial.example.com",
                "prediction": {
                    "decision": "pass",
                    "confidence": 0.5,
                    "composite_score": 40.0,
                },
                "outcome": {
                    "shutdown": True,
                    "status": "partially_verified",
                },
            }
        ]
        data_file = _write_json(tmp_path, "data.json", data)
        dataset = tmp_path / "ds"
        assert cli_main(["import", data_file, "--dataset", str(dataset)]) == 0
        store = DatasetStore(dataset)
        store.initialize()
        outcome = store.find_outcome_by_record(store.list_records()[0])
        assert outcome is not None
        assert outcome.outcome.status == OutcomeStatus.PARTIALLY_VERIFIED
