"""Tests for dataset validation (Part F)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from predictron_engine.dataset.evaluation import PredictionEvaluation
from predictron_engine.dataset.store import DatasetStore
from predictron_engine.dataset.validation import validate_dataset
from tests.dataset.conftest import make_outcome, make_record


def _now() -> datetime:
    return datetime.now(UTC)


class TestValidateDataset:
    def test_empty_store_is_valid(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        report = validate_dataset(store)
        assert report.is_valid
        assert report.issue_count == 0

    def test_valid_store(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record()
        store.save_record(record)
        outcome = make_outcome(record.record_id, shutdown=False)
        store.save_outcome(outcome)
        ev = PredictionEvaluation.from_records(record, outcome)
        store.save_evaluation(ev)
        report = validate_dataset(store)
        assert report.is_valid

    def test_missing_prediction(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        # Write a record JSON that omits the required prediction field,
        # simulating a schema violation / missing prediction on disk.
        import json

        bad = {
            "record_id": "bad-id-1",
            "startup_name": "NoPred",
            "website": "https://nopred.example.com",
            "analysis_date": datetime.now(UTC).isoformat(),
            "engine_version": "0.12.1",
        }
        (store._records_dir / "bad-id-1.json").write_text(
            json.dumps(bad), encoding="utf-8"
        )
        report = validate_dataset(store)
        kinds = report.by_kind()
        assert kinds.get("unreadable_record", 0) == 1

    def test_future_analysis_date(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record().model_copy(
            update={
                "analysis_date": datetime.now(UTC) + timedelta(days=365)
            }
        )
        store.save_record(record)
        report = validate_dataset(store)
        kinds = report.by_kind()
        assert kinds.get("future_analysis_date", 0) == 1

    def test_invalid_chronology(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        analysis_date = _now() - timedelta(days=10)
        # recorded_at before analysis_date -> invalid
        bad_recorded = _now() - timedelta(days=20)
        record = make_record().model_copy(
            update={
                "analysis_date": analysis_date,
                "analysis_metadata": {"recorded_at": bad_recorded.isoformat()},
            }
        )
        store.save_record(record)
        report = validate_dataset(store)
        kinds = report.by_kind()
        assert kinds.get("invalid_chronology", 0) == 1

    def test_orphan_outcome(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        outcome = make_outcome("nonexistent-record", shutdown=False)
        store.save_outcome(outcome)
        report = validate_dataset(store)
        kinds = report.by_kind()
        assert kinds.get("orphan_outcome", 0) == 1

    def test_future_verification_date(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record()
        store.save_record(record)
        outcome = make_outcome(record.record_id, shutdown=False)
        outcome.outcome.latest_verification_date = _now() + timedelta(days=30)
        store.save_outcome(outcome)
        report = validate_dataset(store)
        kinds = report.by_kind()
        assert kinds.get("future_verification_date", 0) == 1

    def test_orphan_evaluation(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record(record_id="orphan-record")
        # don't save the record, but save an evaluation referencing it
        outcome = make_outcome(record.record_id, shutdown=True)
        ev = PredictionEvaluation.from_records(record, outcome)
        store.save_evaluation(ev)
        report = validate_dataset(store)
        kinds = report.by_kind()
        assert kinds.get("orphan_evaluation", 0) == 1

    def test_validation_never_repairs(self, tmp_path) -> None:
        """Validation must not mutate stored data."""
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record().model_copy(
            update={"analysis_date": _now() + timedelta(days=10)}
        )
        store.save_record(record)
        before = store.load_record(record.record_id)
        validate_dataset(store)
        after = store.load_record(record.record_id)
        assert before == after

    def test_report_dict(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        report = validate_dataset(store)
        d = report.to_dict()
        assert d["is_valid"] is True
        assert d["issue_count"] == 0
        assert d["issues"] == []
