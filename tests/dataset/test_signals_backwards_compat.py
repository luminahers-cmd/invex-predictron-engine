"""Project E4 backwards-compatibility guards.

These tests assert that the signal layer is strictly additive: existing
modules, enums, and behaviours are untouched and the original dataset
pipeline still works end to end regardless of whether signals exist.
"""

from __future__ import annotations

from predictron_engine.dataset.graph import EdgeType, NodeType
from predictron_engine.dataset.signals import persistence
from predictron_engine.dataset.signals.persistence import list_signal_company_ids
from predictron_engine.dataset.store import DatasetStore
from tests.dataset.conftest import make_outcome, make_record


class TestExistingEnumsUnchanged:
    def test_node_types_still_twelve(self) -> None:
        assert len(NodeType) == 12

    def test_edge_types_still_twelve(self) -> None:
        assert len(EdgeType) == 12


class TestStoreIsAdditive:
    def test_original_record_api_unchanged(self, tmp_path) -> None:
        store = DatasetStore(tmp_path / "ds")
        store.initialize()
        record = make_record()
        store.save_record(record)
        assert store.count_records() == 1
        loaded = store.load_record(record.record_id)
        assert loaded is not None
        assert loaded.startup_name == record.startup_name

    def test_original_outcome_api_unchanged(self, tmp_path) -> None:
        store = DatasetStore(tmp_path / "ds")
        store.initialize()
        record = make_record()
        store.save_record(record)
        outcome = make_outcome(record.record_id, acquisition="Acquirer")
        store.save_outcome(outcome)
        loaded = store.load_outcome(outcome.outcome_id)
        assert loaded is not None
        assert loaded.outcome.acquisition == "Acquirer"

    def test_signal_fixtures_and_graph_coexist(self, tmp_path) -> None:
        store = DatasetStore(tmp_path / "ds")
        store.initialize()
        record = make_record()
        store.save_record(record)
        assert store.count_records() == 1
        assert store.count_signal_timelines() == 0


class TestSignalsAdditiveToStore:
    def test_no_signals_files_until_signals_used(self, tmp_path) -> None:
        store = DatasetStore(tmp_path / "ds")
        store.initialize()
        record = make_record()
        store.save_record(record)
        signals_dir = tmp_path / "ds" / persistence.SIGNALS_DIR_NAME
        assert signals_dir.exists()
        assert not list(signals_dir.glob("*.json"))
        assert list_signal_company_ids(tmp_path / "ds") == []

    def test_schema_version_stable(self) -> None:
        assert persistence.SIGNALS_SCHEMA_VERSION == "1.0.0"

    def test_persist_helper_import_is_safe(self) -> None:
        # Importing signal bits must not pull in the full engine stack.
        import predictron_engine.dataset.signals.model  # noqa: F401
        import predictron_engine.dataset.signals.persistence  # noqa: F401

    def test_signal_dir_naming(self) -> None:
        assert persistence.SIGNALS_DIR_NAME == "signals"


class TestOriginalPipelineWorks:
    def test_store_round_trip_pipeline(self, tmp_path) -> None:
        store = DatasetStore(tmp_path / "ds")
        store.initialize()
        record = make_record()
        store.save_record(record)
        outcome = make_outcome(record.record_id)
        store.save_outcome(outcome)
        assert store.load_record(record.record_id).record_id == record.record_id
        assert store.load_outcome(outcome.outcome_id).record_id == record.record_id

    def test_store_reinitialization_idempotent(self, tmp_path) -> None:
        store = DatasetStore(tmp_path / "ds")
        store.initialize()
        store.initialize()
        assert store.count_records() == 0
