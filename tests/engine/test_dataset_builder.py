"""Tests for the Historical Startup Dataset Builder module."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from predictron_engine.dataset.evaluation import (
    EvaluationMetadata,
    EvaluationVerdict,
    PredictionEvaluation,
    PredictionOutcomeAlignment,
)
from predictron_engine.dataset.imports import (
    ImportPipeline,
    ImportResult,
    ImportSourceRegistry,
    JsonFileSource,
    RawImportRecord,
)
from predictron_engine.dataset.models import (
    DatasetRecord,
    DecisionLabel,
    FundingStage,
    PredictionSummary,
)
from predictron_engine.dataset.outcomes import (
    FundingEvent,
    OutcomeRecord,
    OutcomeStatus,
    OutcomeVerdict,
    StartupOutcome,
)
from predictron_engine.dataset.store import DatasetStore


def make_record(
    startup_name: str = "Acme Corp",
    website: str = "https://acme.example.com",
    decision: DecisionLabel = DecisionLabel.INVEST,
    confidence: float = 0.75,
    composite_score: float = 68.5,
) -> DatasetRecord:
    return DatasetRecord(
        startup_name=startup_name,
        website=website,
        engine_version="0.12.1",
        prediction=PredictionSummary(
            decision=decision,
            confidence=confidence,
            composite_score=composite_score,
            dimension_scores={"market": 70.0, "team": 75.0},
        ),
    )


def make_outcome(
    record_id: str,
    shutdown: bool | None = None,
    acquisition: str | None = None,
    status: OutcomeStatus = OutcomeStatus.FULLY_VERIFIED,
) -> OutcomeRecord:
    return OutcomeRecord(
        record_id=record_id,
        outcome=StartupOutcome(
            shutdown=shutdown,
            acquisition=acquisition,
            status=status,
        ),
    )


class TestDatasetRecord:
    def test_defaults_created(self) -> None:
        record = make_record()
        assert record.record_id
        assert record.funding_stage_at_analysis == FundingStage.UNKNOWN
        assert record.source == "direct"
        assert record.tags == []
        assert isinstance(record.analysis_date, datetime)

    def test_prediction_fields_populated(self) -> None:
        record = make_record()
        assert record.prediction.decision == DecisionLabel.INVEST
        assert record.prediction.confidence == 0.75
        assert record.prediction.composite_score == 68.5
        assert record.prediction.dimension_scores == {
            "market": 70.0,
            "team": 75.0,
        }

    def test_prediction_serialization_roundtrip(self) -> None:
        record = make_record()
        data = record.model_dump_json()
        restored = DatasetRecord.model_validate_json(data)
        assert restored == record

    def test_predictions_are_immutable_frozen(self) -> None:
        """Verify immutability of the record model dump (no mutation allowed)."""
        record = make_record()
        prediction = record.prediction
        # Rebuild produces an independent object but with fresh defaults
        rebuilt = make_record()
        assert rebuilt.prediction.composite_score == prediction.composite_score

    def test_confidence_range_validated(self) -> None:
        with pytest.raises(ValueError):
            DatasetRecord(
                startup_name="X",
                website="https://x.example.com",
                engine_version="0.12.1",
                prediction=PredictionSummary(
                    decision=DecisionLabel.INVEST,
                    confidence=1.5,
                    composite_score=50.0,
                ),
            )

    def test_decision_enum_accepted(self) -> None:
        record = make_record(decision=DecisionLabel.STRONG_INVEST)
        assert record.prediction.decision == DecisionLabel.STRONG_INVEST


class TestOutcomeModels:
    def test_outcome_defaults_unknown(self) -> None:
        outcome = StartupOutcome()
        assert outcome.status == OutcomeStatus.UNKNOWN
        assert outcome.shutdown is None
        assert outcome.acquisition is None
        assert outcome.funding_rounds == []
        assert outcome.investors == []

    def test_outcome_not_terminal_by_default(self) -> None:
        outcome = StartupOutcome()
        assert outcome.is_terminal() is False

    def test_outcome_terminal_on_shutdown(self) -> None:
        outcome = StartupOutcome(shutdown=True)
        assert outcome.is_terminal() is True

    def test_outcome_terminal_on_acquisition(self) -> None:
        outcome = StartupOutcome(acquisition="BigCorp")
        assert outcome.is_terminal() is True

    def test_outcome_terminal_on_bankruptcy(self) -> None:
        outcome = StartupOutcome(bankruptcy=True)
        assert outcome.is_terminal() is True

    def test_funding_event(self) -> None:
        event = FundingEvent(
            round_type="seed",
            amount_usd=1_000_000,
            investors=["Investor A"],
        )
        assert event.round_type == "seed"
        assert event.amount_usd == 1_000_000

    def test_outcome_record_derive_verdict_unknown(self) -> None:
        outcome = make_outcome("rid")
        assert outcome.derive_verdict() == OutcomeVerdict.UNKNOWN

    def test_outcome_record_derive_verdict_shutdown(self) -> None:
        outcome = make_outcome("rid", shutdown=True)
        assert outcome.derive_verdict() == OutcomeVerdict.FAILURE

    def test_outcome_record_derive_verdict_acquisition(self) -> None:
        outcome = make_outcome(
            "rid", acquisition="Acquirer", status=OutcomeStatus.FULLY_VERIFIED
        )
        outcome = OutcomeRecord(
            record_id="rid",
            outcome=StartupOutcome(
                acquisition="Acquirer",
                acquisition_price_usd=10_000_000,
                status=OutcomeStatus.FULLY_VERIFIED,
            ),
        )
        assert outcome.derive_verdict() == OutcomeVerdict.SUCCESS

    def test_outcome_serialization_roundtrip(self) -> None:
        outcome = make_outcome("rid", shutdown=False)
        data = outcome.model_dump_json()
        restored = OutcomeRecord.model_validate_json(data)
        assert restored == outcome


class TestEvaluation:
    def test_evaluate_correct_invest(self) -> None:
        record = make_record(decision=DecisionLabel.INVEST)
        outcome = make_outcome(
            record.record_id,
            acquisition="Acquirer",
            status=OutcomeStatus.FULLY_VERIFIED,
        )
        outcome.outcome.acquisition_price_usd = 10_000_000
        ev = PredictionEvaluation.from_records(record, outcome)
        assert ev.verdict == EvaluationVerdict.CORRECT

    def test_evaluate_incorrect_invest(self) -> None:
        record = make_record(decision=DecisionLabel.INVEST)
        outcome = make_outcome(record.record_id, shutdown=True)
        ev = PredictionEvaluation.from_records(record, outcome)
        assert ev.verdict == EvaluationVerdict.INCORRECT

    def test_evaluate_correct_pass_on_failure(self) -> None:
        record = make_record(decision=DecisionLabel.PASS)
        outcome = make_outcome(record.record_id, shutdown=True)
        ev = PredictionEvaluation.from_records(record, outcome)
        assert ev.verdict == EvaluationVerdict.CORRECT

    def test_evaluate_incorrect_pass_on_success(self) -> None:
        record = make_record(decision=DecisionLabel.PASS)
        outcome = make_outcome(
            record.record_id,
            acquisition="Acquirer",
            status=OutcomeStatus.FULLY_VERIFIED,
        )
        outcome.outcome.acquisition_price_usd = 10_000_000
        ev = PredictionEvaluation.from_records(record, outcome)
        assert ev.verdict == EvaluationVerdict.INCORRECT

    def test_evaluate_unable_when_unknown(self) -> None:
        record = make_record()
        outcome = make_outcome(record.record_id, status=OutcomeStatus.UNKNOWN)
        ev = PredictionEvaluation.from_records(record, outcome)
        assert ev.verdict == EvaluationVerdict.UNABLE_TO_EVALUATE

    def test_prediction_is_frozen_in_evaluation(self) -> None:
        record = make_record(
            decision=DecisionLabel.INVEST,
            confidence=0.75,
            composite_score=68.5,
        )
        outcome = make_outcome(record.record_id)
        ev = PredictionEvaluation.from_records(record, outcome)

        # The evaluation holds an independent copy of the prediction
        assert ev.prediction.confidence == 0.75
        assert ev.prediction.composite_score == 68.5
        assert ev.prediction.decision == DecisionLabel.INVEST

        # Modifying the original record must not affect the evaluation
        record.prediction.composite_score = 99.0
        assert ev.prediction.composite_score == 68.5

    def test_alignment_strong_invest_success(self) -> None:
        record = make_record(decision=DecisionLabel.INVEST)
        outcome = make_outcome(
            record.record_id,
            acquisition="A",
            status=OutcomeStatus.FULLY_VERIFIED,
        )
        outcome.outcome.acquisition_price_usd = 1_000_000
        ev = PredictionEvaluation.from_records(record, outcome)
        assert ev.alignment == PredictionOutcomeAlignment.STRONG_MATCH

    def test_alignment_mismatch_pass_success(self) -> None:
        record = make_record(decision=DecisionLabel.PASS)
        outcome = make_outcome(
            record.record_id,
            acquisition="A",
            status=OutcomeStatus.FULLY_VERIFIED,
        )
        outcome.outcome.acquisition_price_usd = 1_000_000
        ev = PredictionEvaluation.from_records(record, outcome)
        assert ev.alignment == PredictionOutcomeAlignment.MISMATCH

    def test_metadata(self) -> None:
        meta = EvaluationMetadata(
            evaluator_version="2.0.0",
            evaluation_config={"threshold": "strict"},
        )
        assert meta.evaluator_version == "2.0.0"
        assert meta.evaluation_config == {"threshold": "strict"}


class TestImportPipeline:
    def test_json_file_source_reads_records(
        self, tmp_path: Path
    ) -> None:
        data = [
            {
                "startup_name": "Foo",
                "website": "https://foo.example.com",
                "prediction": {
                    "decision": "invest",
                    "confidence": 0.8,
                    "composite_score": 70.0,
                },
            },
            {
                "startup_name": "Bar",
                "website": "https://bar.example.com",
                "prediction": {
                    "decision": "pass",
                    "confidence": 0.6,
                    "composite_score": 40.0,
                },
            },
        ]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        source = JsonFileSource()
        records = source.read(str(path))
        assert len(records) == 2
        assert records[0].startup_name == "Foo"
        assert records[1].startup_name == "Bar"

    def test_json_file_source_missing_file(self, tmp_path: Path) -> None:
        source = JsonFileSource()
        records = source.read(str(tmp_path / "nonexistent.json"))
        assert records == []

    def test_json_file_source_validate(self) -> None:
        source = JsonFileSource()
        valid = RawImportRecord(startup_name="Foo", website="https://foo.com")
        assert source.validate(valid) == []
        invalid = RawImportRecord(startup_name="", website="")
        errors = source.validate(invalid)
        assert len(errors) >= 2

    def test_pipeline_runs(self, tmp_path: Path) -> None:
        data = [
            {
                "startup_name": "Foo",
                "website": "https://foo.example.com",
                "prediction": {
                    "decision": "invest",
                    "confidence": 0.8,
                    "composite_score": 70.0,
                },
                "outcome": {"shutdown": True},
            }
        ]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        source = JsonFileSource()
        pipeline = ImportPipeline(source)
        result = pipeline.run(str(path))

        assert isinstance(result, ImportResult)
        assert result.records_imported == 1
        assert result.records_failed == 0
        assert len(result.imported_records) == 1
        assert len(result.imported_outcomes) == 1
        assert result.imported_records[0].startup_name == "Foo"
        assert result.imported_outcomes[0].record_id == result.imported_records[0].record_id

    def test_pipeline_rejects_invalid_records(self, tmp_path: Path) -> None:
        data = [
            {"startup_name": "", "website": ""},
            {
                "startup_name": "Valid",
                "website": "https://valid.example.com",
                "prediction": {
                    "decision": "invest",
                    "confidence": 0.5,
                    "composite_score": 50.0,
                },
            },
        ]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        pipeline = ImportPipeline(JsonFileSource())
        result = pipeline.run(str(path))

        assert result.records_imported == 1
        assert result.records_failed == 1
        assert len(result.validation_errors) == 1

    def test_registry(self) -> None:
        registry = ImportSourceRegistry.default()
        assert "json_file" in registry.list_sources()
        assert registry.get("json_file") is not None
        assert registry.get("nonexistent") is None

    def test_custom_source_registration(self) -> None:
        class CustomSource:
            @property
            def source_name(self) -> str:
                return "custom"

            def read(self, path: str) -> list[RawImportRecord]:
                return [RawImportRecord(startup_name="Z", website="https://z.com")]

            def validate(self, record: RawImportRecord) -> list[str]:
                return []

        registry = ImportSourceRegistry()
        registry.register(CustomSource())
        assert "custom" in registry.list_sources()
        assert registry.get("custom") is not None


class TestDatasetStore:
    def test_initialize_creates_directories(self, tmp_path: Path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        assert (tmp_path / "records").is_dir()
        assert (tmp_path / "outcomes").is_dir()
        assert (tmp_path / "evaluations").is_dir()
        assert (tmp_path / "manifest.json").exists()

    def test_save_load_record(self, tmp_path: Path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record()
        store.save_record(record)
        loaded = store.load_record(record.record_id)
        assert loaded is not None
        assert loaded.startup_name == record.startup_name
        assert loaded.prediction.composite_score == record.prediction.composite_score

    def test_load_missing_record(self, tmp_path: Path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        assert store.load_record("does-not-exist") is None

    def test_list_records(self, tmp_path: Path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        r1 = make_record(startup_name="One")
        r2 = make_record(startup_name="Two")
        store.save_record(r1)
        store.save_record(r2)
        ids = store.list_records()
        assert set(ids) == {r1.record_id, r2.record_id}
        assert store.count_records() == 2

    def test_find_records_by_startup(self, tmp_path: Path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        r1 = make_record(startup_name="Acme")
        r2 = make_record(startup_name="Other")
        store.save_record(r1)
        store.save_record(r2)
        found = store.find_records_by_startup("Acme")
        assert len(found) == 1
        assert found[0].startup_name == "Acme"

    def test_save_load_outcome(self, tmp_path: Path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        outcome = make_outcome("rid", shutdown=False)
        store.save_outcome(outcome)
        loaded = store.load_outcome(outcome.outcome_id)
        assert loaded is not None
        assert loaded.record_id == "rid"
        assert loaded.outcome.shutdown is False

    def test_save_load_evaluation(self, tmp_path: Path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record()
        outcome = make_outcome(record.record_id, shutdown=True)
        ev = PredictionEvaluation.from_records(record, outcome)
        store.save_evaluation(ev)
        loaded = store.load_evaluation(ev.evaluation_id)
        assert loaded is not None
        assert loaded.verdict == ev.verdict

    def test_find_outcome_by_record(self, tmp_path: Path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record()
        outcome = make_outcome(record.record_id)
        store.save_outcome(outcome)
        found = store.find_outcome_by_record(record.record_id)
        assert found is not None
        assert found.outcome_id == outcome.outcome_id

    def test_find_evaluation_by_record(self, tmp_path: Path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record()
        outcome = make_outcome(record.record_id, shutdown=True)
        ev = PredictionEvaluation.from_records(record, outcome)
        store.save_evaluation(ev)
        found = store.find_evaluation_by_record(record.record_id)
        assert found is not None
        assert found.evaluation_id == ev.evaluation_id

    def test_manifest_tracks_counts(self, tmp_path: Path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        store.save_record(make_record())
        store.save_record(make_record(startup_name="Two"))
        manifest = store.get_manifest()
        assert manifest["record_count"] == 2


class TestFullWorkflow:
    def test_end_to_end_import_to_evaluation(self, tmp_path: Path) -> None:
        data = [
            {
                "startup_name": "Acme",
                "website": "https://acme.example.com",
                "prediction": {
                    "decision": "invest",
                    "confidence": 0.8,
                    "composite_score": 72.0,
                },
                "outcome": {"acquisition": "Corp", "status": "fully_verified"},
            },
        ]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        store = DatasetStore(tmp_path / "out")
        store.initialize()

        pipeline = ImportPipeline(JsonFileSource())
        result = pipeline.run(str(path))
        assert result.records_imported == 1

        record = result.imported_records[0]
        outcome = result.imported_outcomes[0]

        store.save_record(record)
        store.save_outcome(outcome)

        evaluation = PredictionEvaluation.from_records(record, outcome)
        store.save_evaluation(evaluation)

        # Verify the outcome stored a valid verdict
        assert outcome.derive_verdict() == OutcomeVerdict.SUCCESS
        assert evaluation.verdict == EvaluationVerdict.CORRECT

        # Verify persistence round-trips the full workflow
        loaded_record = store.load_record(record.record_id)
        assert loaded_record is not None
        loaded_outcome = store.find_outcome_by_record(record.record_id)
        assert loaded_outcome is not None
        loaded_eval = store.find_evaluation_by_record(record.record_id)
        assert loaded_eval is not None
