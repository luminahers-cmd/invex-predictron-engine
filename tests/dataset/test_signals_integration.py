"""Tests for predictron_engine.dataset.signals.integration — Project E4."""

from __future__ import annotations

import json

import pytest

from predictron_engine.dataset.outcomes import OutcomeRecord, StartupOutcome
from predictron_engine.dataset.signals.importers import SignalImportReport
from predictron_engine.dataset.signals.integration import (
    SignalDatasetManager,
    SignalImportOutcome,
)
from predictron_engine.dataset.store import DatasetStore
from tests.dataset.conftest import make_record
from tests.dataset.signals_helpers import dt, make_funding_event, make_signal

AS_OF = dt("2024-12-31T00:00:00+00:00")


def make_store(tmp_path) -> tuple[DatasetStore, str]:
    store = DatasetStore(tmp_path / "ds")
    store.initialize()
    record = make_record(startup_name="Acme", website="https://acme.example.com")
    store.save_record(record)
    outcome = OutcomeRecord(
        record_id=record.record_id,
        outcome=StartupOutcome(
            funding_rounds=[
                make_funding_event(
                    date="2024-01-15T00:00:00+00:00",
                    amount=1_000_000.0,
                    investors=["Alpha", "Beta"],
                )
            ]
        ),
    )
    store.save_outcome(outcome)
    return store, record.record_id


class TestIdentityResolution:
    def test_company_id_for_record(self, tmp_path) -> None:
        store, rec_id = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        company = manager.company_id_for(rec_id)
        assert company is not None
        assert company.startswith("company:")

    def test_company_id_for_name(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        assert manager.company_id_for("Acme") is not None

    def test_company_id_for_website(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        assert manager.company_id_for("https://acme.example.com") is not None

    def test_company_id_missing(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        assert manager.company_id_for("no-such-company") is None

    def test_company_map(self, tmp_path) -> None:
        store, rec_id = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        mapping = manager.company_map()
        assert mapping[rec_id].startswith("company:")

    def test_distinct_startup_count(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        assert manager.distinct_startup_count() == 1


class TestBuild:
    def test_build_creates_timelines(self, tmp_path) -> None:
        store, rec_id = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        result = manager.build()
        assert result.report.timeline_count == 1
        assert result.report.signal_count >= 2  # funding_round + investor signals
        assert store.count_signal_timelines() == 1

    def test_build_identity_aligned_with_graph(self, tmp_path) -> None:
        store, rec_id = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        result = manager.build()
        company = next(iter(result.timelines))
        from predictron_engine.dataset.entity_resolution import EntityResolver
        from predictron_engine.dataset.graph.builder import company_node_id

        records = [r for r in manager._load_records()]
        identities = EntityResolver().resolve(records).identities
        assert company in {company_node_id(i) for i in identities}


class TestImportFile:
    def test_import_and_persist(self, tmp_path) -> None:
        store, rec_id = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        company = manager.company_id_for(rec_id)
        sig_file = tmp_path / "signals.json"
        sig_file.write_text(
            json.dumps([
                {
                    "company_id": company,
                    "signal_type": "founder_change",
                    "timestamp": "2024-05-01T00:00:00+00:00",
                    "source": "news",
                }
            ]),
            encoding="utf-8",
        )
        outcome = manager.import_file(sig_file)
        assert isinstance(outcome, SignalImportOutcome)
        assert outcome.report.imported == 1
        timeline = manager.timeline(company)
        assert timeline is not None
        assert timeline.signal_count == 1

    def test_import_outcome_to_dict(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        _manager = SignalDatasetManager(store)
        report = SignalImportReport()
        outcome = SignalImportOutcome(report=report)
        assert outcome.to_dict()["requested"] == 0


class TestReads:
    def test_timeline_none_when_absent(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        assert manager.timeline("company:absent") is None

    def test_all_timelines_sorted(self, tmp_path) -> None:
        store, rec_id = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        manager.build()
        names = [t.company_id for t in manager.all_timelines()]
        assert names == sorted(names)

    def test_aggregate_missing_raises(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        with pytest.raises(KeyError, match="no signals stored"):
            manager.aggregate("company:absent")

    def test_aggregate_bundle(self, tmp_path) -> None:
        store, rec_id = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        manager.build()
        company = manager.company_id_for(rec_id)
        result = manager.aggregate(company, as_of=AS_OF)
        assert "momentum" in result
        assert "recent_activity" in result

    def test_trends(self, tmp_path) -> None:
        store, rec_id = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        manager.build()
        company = manager.company_id_for(rec_id)
        result = manager.trends(company, as_of=AS_OF)
        assert "funding_velocity" in result
        assert "stagnation" in result


class TestValidationStatistics:
    def test_validate_empty(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        assert manager.validate().is_valid

    def test_validate_after_build(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        manager.build()
        assert manager.validate().is_valid

    def test_statistics_coverage(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        manager.build()
        stats = manager.statistics(as_of=AS_OF)
        assert stats.companies_with_signals == 1
        assert stats.funding_events >= 1
        assert stats.total_amount_usd == pytest.approx(1_000_000.0, rel=1e-4)


class TestReport:
    def test_report_document(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        manager.build()
        doc = manager.report(as_of=AS_OF)
        assert doc["report_type"] == "company_signals_report"
        assert "coverage" in doc


class TestDerivedViews:
    def test_profile_highlights(self, tmp_path) -> None:
        store, rec_id = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        manager.build()
        company = manager.company_id_for(rec_id)
        highlights = manager.profile_highlights(company, as_of=AS_OF)
        assert isinstance(highlights, list)
        assert all(isinstance(h, str) for h in highlights)
        assert len(highlights) <= 5

    def test_profile_highlights_empty_company(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        assert manager.profile_highlights("company:absent") == []

    def test_graph_overview(self, tmp_path) -> None:
        store, rec_id = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        manager.build()
        company = manager.company_id_for(rec_id)
        overview = manager.graph_overview()
        assert company in overview
        assert overview[company]["signal_count"] >= 1
        assert "types" in overview[company]


class TestStorePersist:
    def test_save_load_round_trip_via_manager(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        manager = SignalDatasetManager(store)
        manager.build()
        companies = store.list_signal_company_ids()
        for company in companies:
            tl = manager.timeline(company)
            assert tl is not None
            assert tl.company_id == company

    def test_save_signal_merges(self, tmp_path) -> None:
        store, _ = make_store(tmp_path)
        store.initialize()
        s1 = make_signal(company_id="company:c", at="2024-01-01T00:00:00+00:00")
        s2 = make_signal(company_id="company:c", at="2024-06-01T00:00:00+00:00")
        store.save_signal(s1)
        store.save_signal(s2)
        tl = store.load_timeline("company:c")
        assert tl is not None
        assert tl.signal_count == 2
        assert store.count_signal_timelines() == 1
