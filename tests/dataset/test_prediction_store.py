"""Tests for the append-only frozen-prediction store (Phase 4).

Covers save/load round trips, duplicate handling, idempotent re-freeze,
store integrity verification (tamper detection), and the deterministic
prediction identity/hash helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from predictron_engine.dataset.models import DecisionLabel, PredictionSummary
from predictron_engine.dataset.prediction import TimeScopedPrediction
from predictron_engine.dataset.prediction_store import (
    DEFAULT_PREDICTION_ROOT,
    PredictionStore,
    PredictionStoreError,
    prediction_hash,
    stable_prediction_id,
)

ANALYSIS = datetime(2026, 1, 1, tzinfo=UTC)


def _prediction(company_id: str = "cmp-1") -> TimeScopedPrediction:
    return TimeScopedPrediction(
        prediction_id=stable_prediction_id(
            company_id=company_id, analysis_timestamp=ANALYSIS
        ),
        company_id=company_id,
        company_name="Example",
        analysis_timestamp=ANALYSIS,
        evaluation_horizon_days=180,
        engine_version="9.9.9",
        evidence_reference="b2b_saas",
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
        prediction=PredictionSummary(
            decision=DecisionLabel.INVEST,
            confidence=0.75,
            composite_score=70.0,
            dimension_scores={"market": 80.0},
            investment_readiness_score=60.0,
            recommendation_count=3,
        ),
    )


class TestIdentityHelpers:
    def test_stable_prediction_id_is_deterministic(self) -> None:
        first = stable_prediction_id(company_id="cmp-1", analysis_timestamp=ANALYSIS)
        second = stable_prediction_id(company_id="cmp-1", analysis_timestamp=ANALYSIS)
        assert first == second
        assert len(first) == 64

    def test_stable_prediction_id_distinguishes_company_and_time(self) -> None:
        other_time = datetime(2026, 2, 1, tzinfo=UTC)
        other_company = stable_prediction_id(
            company_id="cmp-2", analysis_timestamp=ANALYSIS
        )
        assert stable_prediction_id(
            company_id="cmp-1", analysis_timestamp=other_time
        ) != stable_prediction_id(
            company_id="cmp-1", analysis_timestamp=ANALYSIS
        )
        assert other_company != stable_prediction_id(
            company_id="cmp-1", analysis_timestamp=ANALYSIS
        )

    def test_stable_prediction_id_rejects_naive_timestamp(self) -> None:
        with pytest.raises(PredictionStoreError):
            stable_prediction_id(company_id="cmp-1", analysis_timestamp=datetime(2026, 1, 1))

    def test_prediction_hash_is_deterministic(self) -> None:
        first = prediction_hash(_prediction())
        second = prediction_hash(_prediction())
        assert first == second
        assert len(first) == 64

    def test_prediction_hash_detects_content_change(self) -> None:
        original = _prediction()
        tampered = original.model_copy(
            update={
                "prediction": original.prediction.model_copy(update={"confidence": 0.99})
            }
        )
        assert prediction_hash(original) != prediction_hash(tampered)


class TestPredictionStore:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        store = PredictionStore(tmp_path)
        prediction = _prediction()
        path, persisted = store.save_if_absent(prediction)
        assert persisted is True
        assert path.exists()
        loaded = store.load(prediction.prediction_id)
        assert loaded == prediction

    def test_duplicate_save_if_absent_is_idempotent(self, tmp_path: Path) -> None:
        store = PredictionStore(tmp_path)
        prediction = _prediction()
        first_path, first_created = store.save_if_absent(prediction)
        second_path, second_created = store.save_if_absent(prediction)
        assert first_created is True
        assert second_created is False
        assert first_path == second_path
        assert store.count == 1
        assert store.load(prediction.prediction_id) == prediction

    def test_strict_save_rejects_duplicate(self, tmp_path: Path) -> None:
        store = PredictionStore(tmp_path)
        prediction = _prediction()
        store.save(prediction)
        with pytest.raises(PredictionStoreError, match="already exists"):
            store.save(prediction)

    def test_has_prediction_and_count(self, tmp_path: Path) -> None:
        store = PredictionStore(tmp_path)
        assert store.count == 0
        assert store.has_prediction("nope") is False
        store.save_if_absent(_prediction("cmp-1"))
        store.save_if_absent(_prediction("cmp-2"))
        assert store.count == 2
        assert store.has_prediction(
            stable_prediction_id(company_id="cmp-1", analysis_timestamp=ANALYSIS)
        )

    def test_list_ids_and_summaries_sorted(self, tmp_path: Path) -> None:
        store = PredictionStore(tmp_path)
        store.save_if_absent(_prediction("cmp-2"))
        store.save_if_absent(_prediction("cmp-1"))
        ids = store.list_prediction_ids()
        assert ids == sorted(ids)
        assert len(ids) == 2

        summaries = store.list_summaries()
        assert {s.company_id for s in summaries} == {"cmp-1", "cmp-2"}
        assert [s.prediction_id for s in summaries] == sorted(
            s.prediction_id for s in summaries
        )
        for entry in summaries:
            assert entry.prediction_id
            assert entry.prediction_hash
            assert entry.analysis_timestamp.tzinfo is not None
            assert len(entry.prediction_hash) == 64

    def test_verify_integrity_returns_ok_map(self, tmp_path: Path) -> None:
        store = PredictionStore(tmp_path)
        store.save_if_absent(_prediction())
        assert store.verify_integrity() == {
            stable_prediction_id(company_id="cmp-1", analysis_timestamp=ANALYSIS): "ok"
        }

    def test_verify_integrity_detects_tampering(self, tmp_path: Path) -> None:
        store = PredictionStore(tmp_path)
        prediction = _prediction()
        path, _ = store.save_if_absent(prediction)
        text = path.read_text(encoding="utf-8")
        text = text.replace('"confidence": 0.75', '"confidence": 0.99')
        path.write_text(text, encoding="utf-8")
        with pytest.raises(PredictionStoreError, match="integrity"):
            store.verify_integrity()
        with pytest.raises(PredictionStoreError, match="integrity"):
            store.load(prediction.prediction_id)

    def test_missing_document_load_raises(self, tmp_path: Path) -> None:
        store = PredictionStore(tmp_path)
        with pytest.raises(PredictionStoreError):
            store.load("missing-id")

    def test_default_root_is_repo_data_dir(self) -> None:
        root = Path(DEFAULT_PREDICTION_ROOT)
        assert root.name == "frozen_predictions"
        assert root.parent.name == "data"


class TestDocumentValidation:
    def test_prediction_id_used_as_filename_must_be_safe(self, tmp_path: Path) -> None:
        store = PredictionStore(tmp_path)
        with pytest.raises(PredictionStoreError, match="filename-safe"):
            store.prediction_path("unsafe/id")

    def test_load_of_foreign_id_not_found(self, tmp_path: Path) -> None:
        store = PredictionStore(tmp_path)
        store.save(_prediction())
        other = stable_prediction_id(company_id="cmp-9", analysis_timestamp=ANALYSIS)
        with pytest.raises(PredictionStoreError, match="not found"):
            store.load(other)
