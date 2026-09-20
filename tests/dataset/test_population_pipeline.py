"""Tests for the population orchestrator (Project V4)."""

from __future__ import annotations

from pathlib import Path

from predictron_engine.dataset.acquisition.sources.csv_export_connector import (
    CsvExportConnector,
)
from predictron_engine.dataset.acquisition.state import (
    AcquisitionStateManager,
    CheckpointData,
)
from predictron_engine.dataset.models import (
    DatasetRecord,
    DecisionLabel,
    PredictionSummary,
)
from predictron_engine.dataset.population import (
    PopulateOptions,
    PopulationOrchestrator,
)
from predictron_engine.dataset.population_config import (
    PopulationConfig,
    SourceConfig,
)
from predictron_engine.dataset.store import DatasetStore
from tests.dataset.conftest import make_csv


class TestIterateSources:
    def test_iterates_over_multiple_configured_sources(
        self, dataset_store: DatasetStore, tmp_path: Path
    ) -> None:
        """Files from multiple sources are all processed."""
        f1 = make_csv(tmp_path, "s1.csv", [("Alpha", "https://alpha.com")])
        f2 = make_csv(tmp_path, "s2.csv", [("Beta", "https://beta.com")])
        config = PopulationConfig(
            sources=[
                SourceConfig(name="csv_export", file_paths=[str(f1)]),
                SourceConfig(name="csv_export", file_paths=[str(f2)]),
            ]
        )
        orch = PopulationOrchestrator(dataset_store, config=config)
        report = orch.populate()
        assert report.counts["imported"] == 2
        assert dataset_store.count_records() == 2

    def test_selects_single_source(self, dataset_store: DatasetStore, tmp_path: Path) -> None:
        f1 = make_csv(tmp_path, "s1.csv", [("Alpha", "https://alpha.com")])
        f2 = make_csv(tmp_path, "s2.csv", [("Beta", "https://beta.com")])
        config = PopulationConfig(
            sources=[
                SourceConfig(name="csv_export", file_paths=[str(f1)]),
                SourceConfig(name="csv_export", file_paths=[str(f2)]),
            ]
        )
        orch = PopulationOrchestrator(dataset_store, config=config)
        # --source is not in config; unknown source -> no matching connector
        report = orch.populate(PopulateOptions(source_names=["nonexistent"]))
        assert report.counts["imported"] == 0

    def test_unknown_source_returns_error(
        self, dataset_store: DatasetStore
    ) -> None:
        config = PopulationConfig(
            sources=[SourceConfig(name="does_not_exist")]
        )
        orch = PopulationOrchestrator(dataset_store, config=config)
        report = orch.populate()
        assert report.counts["failed"] == 0
        assert report.counts["imported"] == 0


class TestImportBehaviour:
    def test_imports_all_records(
        self, dataset_store: DatasetStore, csv_export_config: PopulationConfig
    ) -> None:
        orch = PopulationOrchestrator(dataset_store, config=csv_export_config)
        report = orch.populate()
        assert report.counts["imported"] == 3
        assert dataset_store.count_records() == 3

    def test_records_provenance(
        self, dataset_store: DatasetStore, csv_export_config: PopulationConfig
    ) -> None:
        orch = PopulationOrchestrator(dataset_store, config=csv_export_config)
        orch.populate()
        record = dataset_store.load_record(dataset_store.list_records()[0])
        assert record is not None
        assert record.source == "csv_export"

    def test_limit(
        self, dataset_store: DatasetStore, csv_export_config: PopulationConfig
    ) -> None:
        orch = PopulationOrchestrator(dataset_store, config=csv_export_config)
        report = orch.populate(PopulateOptions(limit=1))
        assert report.counts["imported"] == 1
        assert dataset_store.count_records() == 1

    def test_dry_run_imports_nothing(
        self, dataset_store: DatasetStore, csv_export_config: PopulationConfig
    ) -> None:
        orch = PopulationOrchestrator(dataset_store, config=csv_export_config)
        report = orch.populate(PopulateOptions(dry_run=True))
        assert report.dry_run is True
        assert report.counts["imported"] == 0
        assert dataset_store.count_records() == 0


class TestDeduplication:
    def test_counts_pre_existing_duplicates(
        self, dataset_store: DatasetStore, tmp_path: Path
    ) -> None:
        """A record matching an existing store record counts as a duplicate."""
        # Pre-populate the store with one company.
        f0 = make_csv(tmp_path, "seed.csv", [("Acme Corp", "https://acme.com")])
        seed_cfg = PopulationConfig(
            sources=[SourceConfig(name="csv_export", file_paths=[str(f0)])]
        )
        PopulationOrchestrator(dataset_store, config=seed_cfg).populate()
        assert dataset_store.count_records() == 1

        # A new source repeats Acme and adds a new company.
        f1 = make_csv(tmp_path, "more.csv", [
            ("Acme Corp", "https://acme.com"),
            ("Brand New", "https://brandnew.com"),
        ])
        cfg = PopulationConfig(
            sources=[SourceConfig(name="csv_export", file_paths=[str(f1)])]
        )
        report = PopulationOrchestrator(dataset_store, config=cfg).populate()
        assert report.counts["duplicates"] == 1
        # Parity with the acquisition framework: the normal (non-resume)
        # path re-imports a duplicate rather than skipping it, so two
        # records are imported even though one is a duplicate.
        assert report.counts["imported"] == 2
        assert dataset_store.count_records() == 3


class TestReportWriting:
    def test_report_path_written(
        self, dataset_store: DatasetStore, csv_export_config: PopulationConfig,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "reports" / "run.json"
        orch = PopulationOrchestrator(dataset_store, config=csv_export_config)
        orch.populate(PopulateOptions(report_path=str(out)))
        assert out.exists()
        import json

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["report_type"] == "population_report"
        assert data["counts"]["imported"] == 3


class TestResume:
    def _seed_interrupted(
        self,
        dataset_store: DatasetStore,
        file_path: Path,
        imported_name: str,
        pending_name: str,
    ) -> None:
        """Simulate an interrupted run: one record saved, batch still pending."""
        dataset_store.save_record(
            DatasetRecord(
                startup_name=imported_name,
                website=f"https://{imported_name.lower()}.com",
                engine_version="0.12.1",
                prediction=PredictionSummary(
                    decision=DecisionLabel.WATCH,
                    confidence=0.5,
                    composite_score=50.0,
                ),
            )
        )
        connector = CsvExportConnector()
        state = AcquisitionStateManager(dataset_store._root)
        state.initialize()
        state.save_checkpoint(
            CheckpointData(
                batch_id="batch-pending",
                source_name="csv_export",
                file_path=str(file_path),
                file_hash=connector.checkpoint(str(file_path)),
                total_records=2,
                processed_records=1,
                imported_records=1,
                failed_records=0,
                skipped_records=0,
            )
        )

    def test_resume_imports_only_missing_records(
        self, dataset_store: DatasetStore, tmp_path: Path
    ) -> None:
        """Resume re-imports only records not already in the store."""
        f = make_csv(tmp_path, "data.csv", [
            ("Seeded", "https://seeded.com"),
            ("New", "https://new.com"),
        ])
        self._seed_interrupted(dataset_store, f, "Seeded", "New")
        assert dataset_store.count_records() == 1

        cfg = PopulationConfig(
            sources=[SourceConfig(name="csv_export", file_paths=[str(f)])]
        )
        report = PopulationOrchestrator(dataset_store, config=cfg).populate(
            PopulateOptions(resume=True)
        )
        assert report.counts["imported"] == 1
        assert report.counts["skipped"] >= 1  # Seeded counted as duplicate
        assert dataset_store.count_records() == 2
        assert set(dataset_store.find_distinct_startups()) == {"Seeded", "New"}

    def test_resume_clears_pending_checkpoint(
        self, dataset_store: DatasetStore, tmp_path: Path
    ) -> None:
        f = make_csv(tmp_path, "data.csv", [
            ("Seeded", "https://seeded.com"),
            ("New", "https://new.com"),
        ])
        self._seed_interrupted(dataset_store, f, "Seeded", "New")
        cfg = PopulationConfig(
            sources=[SourceConfig(name="csv_export", file_paths=[str(f)])]
        )
        PopulationOrchestrator(dataset_store, config=cfg).populate(
            PopulateOptions(resume=True)
        )
        state = AcquisitionStateManager(dataset_store._root)
        state.initialize()
        assert state.list_checkpoints() == []

    def test_resume_with_no_pending_work_imports_nothing(
        self, dataset_store: DatasetStore, tmp_path: Path
    ) -> None:
        """A fully-imported file is skipped on resume."""
        f = make_csv(tmp_path, "data.csv", [
            ("A", "https://a.com"),
            ("B", "https://b.com"),
        ])
        cfg = PopulationConfig(
            sources=[SourceConfig(name="csv_export", file_paths=[str(f)])]
        )
        orch = PopulationOrchestrator(dataset_store, config=cfg)
        orch.populate()
        assert dataset_store.count_records() == 2
        report = orch.populate(PopulateOptions(resume=True))
        assert report.counts["imported"] == 0
        assert dataset_store.count_records() == 2
