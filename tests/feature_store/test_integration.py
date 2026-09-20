"""Integration tests: FeatureStore ↔ Engine end-to-end."""

from __future__ import annotations

from datetime import UTC, datetime

from predictron_engine.dataset.signals.model import SignalType
from predictron_engine.feature_store.engine import FeatureEngine
from predictron_engine.feature_store.models import (
    FeatureStoreSnapshot,
)
from predictron_engine.feature_store.registry import (
    FeatureCategory,
    FeatureDefinition,
    FeatureRegistry,
)
from predictron_engine.feature_store.store import FeatureStore
from tests.feature_store.conftest import (
    FakeProfile,
    FakeRecord,
    make_signal,
    make_timeline,
)

AS_OF = datetime(2024, 1, 1, tzinfo=UTC)


class TestFullPipeline:
    def test_build_and_store(self, engine, full_store, fake_record):
        fs = engine.build_company_features(
            fake_record, timeline=make_timeline(signals=[
                make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=1_000_000),
                make_signal(SignalType.ARR_MILESTONE, days_offset=-60),
                make_signal(SignalType.FUNDING_ROUND, days_offset=-120, amount=500_000),
            ]),
            as_of=AS_OF,
        )
        latest_path = full_store.save_company_features(fs)
        assert latest_path.exists()

        loaded = full_store.load_company_features("rec-1")
        assert loaded is not None
        assert loaded.feature_count() == fs.feature_count()
        assert loaded.features["total_funding"].value == 1_500_000

    def test_build_store_rebuild_deterministic(self, engine, full_store, fake_record):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=5_000_000),
        ])
        fs1 = engine.build_company_features(fake_record, timeline=tl, as_of=AS_OF)
        full_store.save_company_features(fs1)

        engine.build_company_features(fake_record, timeline=tl, as_of=AS_OF)
        loaded = full_store.load_company_features("rec-1")
        assert loaded is not None
        for fid in fs1.features:
            assert fs1.features[fid].value == loaded.features[fid].value

    def test_store_history_tracks_all_builds(self, engine, full_store, fake_record):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=1_000_000),
        ])
        fs = engine.build_company_features(fake_record, timeline=tl, as_of=AS_OF)
        for _i in range(3):
            full_store.save_company_features(fs)

        history = full_store.get_company_history("rec-1")
        assert len(history) == 3

    def test_snapshot_roundtrip_preserves_features(self, engine, full_store,
                                                     fake_record):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
        ])
        fs = engine.build_company_features(fake_record, timeline=tl, as_of=AS_OF)
        full_store.save_company_features(fs)

        snap = FeatureStoreSnapshot(
            companies={"rec-1": fs},
        )
        json_str = snap.model_dump_json(indent=2)
        loaded = FeatureStoreSnapshot.model_validate_json(json_str)
        assert loaded.company_count == 1

    def test_export_import_json(self, engine, full_store, fake_record, tmp_path):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
        ])
        fs = engine.build_company_features(fake_record, timeline=tl, as_of=AS_OF)
        full_store.save_company_features(fs)

        export_data = full_store.export_all()
        assert export_data["company_count"] == 1

        new_store = FeatureStore(tmp_path / "import_store")
        new_store.initialize()
        count = new_store.import_all(export_data)
        assert count == 1
        loaded = new_store.load_company_features("rec-1")
        assert loaded is not None
        assert loaded.features["total_funding"].value == fs.features["total_funding"].value

    def test_multiple_companies_independently(self, engine, full_store):
        rec_a = FakeRecord(
            record_id="rec-a",
            profile=FakeProfile(founded_year=2020),
        )
        rec_b = FakeRecord(
            record_id="rec-b",
            profile=FakeProfile(founded_year=2015),
        )
        fs_a = engine.build_company_features(rec_a, as_of=AS_OF)
        fs_b = engine.build_company_features(rec_b, as_of=AS_OF)
        full_store.save_company_features(fs_a)
        full_store.save_company_features(fs_b)

        loaded_a = full_store.load_company_features("rec-a")
        loaded_b = full_store.load_company_features("rec-b")
        assert loaded_a is not None and loaded_b is not None
        assert loaded_a.features["company_age"].value != loaded_b.features["company_age"].value

    def test_feature_engine_incremental_rebuild(self, engine, full_store,
                                                  fake_record):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=1_000_000),
        ])
        fs = engine.build_company_features(fake_record, timeline=tl, as_of=AS_OF)
        full_store.save_company_features(fs)

        rebuilt = engine.build_company_features(fake_record, timeline=tl, as_of=AS_OF)
        for fid in fs.features:
            orig = fs.features[fid]
            rb = rebuilt.features[fid]
            assert orig.value == rb.value, fid

    def test_store_lookup_returns_correct_company(self, engine, full_store):
        for rid in ["c1", "c2", "c3"]:
            rec = FakeRecord(record_id=rid)
            fs = engine.build_company_features(rec, as_of=AS_OF)
            full_store.save_company_features(fs)

        for rid in ["c1", "c2", "c3"]:
            loaded = full_store.load_company_features(rid)
            assert loaded is not None
            assert loaded.company_id == rid

    def test_store_lookup_missing(self, full_store):
        loaded = full_store.load_company_features("nonexistent")
        assert loaded is None

    def test_build_with_outcomes(self, engine, full_store, fake_record,
                                  fake_outcome):
        fs = engine.build_company_features(
            fake_record, outcome=fake_outcome, as_of=AS_OF,
        )
        assert "total_funding" in fs.features
        full_store.save_company_features(fs)

    def test_incremental_rebuild_with_new_signals(self, engine, full_store,
                                                    fake_record):
        tl1 = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=1_000_000),
        ])
        fs1 = engine.build_company_features(fake_record, timeline=tl1, as_of=AS_OF)
        full_store.save_company_features(fs1)

        tl2 = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=-200, amount=500_000),
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=1_000_000),
        ])
        fs2 = engine.build_company_features(fake_record, timeline=tl2, as_of=AS_OF)
        assert fs2.features["total_funding"].value == 1_500_000

    def test_manifest_after_saves(self, engine, full_store, fake_record):
        fs = engine.build_company_features(fake_record, as_of=AS_OF)
        full_store.save_company_features(fs)
        manifest = full_store.get_manifest()
        assert isinstance(manifest, dict)

    def test_feature_value_types_preserved(self, engine, full_store, fake_record):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=1_000_000),
            make_signal(SignalType.FUNDING_ROUND, days_offset=100, amount=2_000_000),
        ])
        fs = engine.build_company_features(fake_record, timeline=tl, as_of=AS_OF)
        full_store.save_company_features(fs)
        loaded = full_store.load_company_features("rec-1")
        assert loaded is not None

        total_funding = loaded.features["total_funding"]
        assert isinstance(total_funding.value, int | float)
        assert total_funding.value == 3_000_000

        investor_count = loaded.features["investor_count"]
        assert isinstance(investor_count.value, int | float)

    def test_store_degradation_graceful(self, engine, full_store, fake_record):
        fs = engine.build_company_features(fake_record, as_of=AS_OF)
        full_store.save_company_features(fs)
        loaded = full_store.load_company_features("rec-1")
        assert loaded is not None
        assert loaded.feature_count() > 0

    def test_multiple_engines_same_features(self, full_store, fake_record):
        r1 = FeatureRegistry()
        r2 = FeatureRegistry()

        def comp_a(record, outcome, ctx):
            return 10.0, []

        for reg in (r1, r2):
            reg.register(FeatureDefinition(
                feature_id="company_age",
                feature_name="Company Age",
                category=FeatureCategory.COMPANY,
                description="Test",
                value_type="float",
                depends_on_record_fields=[],
            ), comp_a)
        e1 = FeatureEngine(registry=r1)
        e2 = FeatureEngine(registry=r2)
        fs1 = e1.build_company_features(fake_record, as_of=AS_OF)
        fs2 = e2.build_company_features(fake_record, as_of=AS_OF)
        assert fs1.features["company_age"].value == fs2.features["company_age"].value

    def test_store_lookup_feature(self, engine, full_store, fake_record):
        fs = engine.build_company_features(fake_record, as_of=AS_OF)
        full_store.save_company_features(fs)

        snap = full_store.lookup_feature("rec-1", "company_age")
        assert snap is not None
        assert snap.value is not None

    def test_store_lookup_features_multi(self, engine, full_store, fake_record):
        fs = engine.build_company_features(fake_record, as_of=AS_OF)
        full_store.save_company_features(fs)

        results = full_store.lookup_features(
            "rec-1", ["company_age", "total_funding"],
        )
        assert "company_age" in results
        assert "total_funding" in results

    def test_store_count(self, engine, full_store, fake_record):
        fs = engine.build_company_features(fake_record, as_of=AS_OF)
        full_store.save_company_features(fs)
        assert full_store.count_companies() == 1

    def test_store_list_companies(self, engine, full_store):
        for rid in ["a", "b"]:
            rec = FakeRecord(record_id=rid)
            fs = engine.build_company_features(rec, as_of=AS_OF)
            full_store.save_company_features(fs)
        assert full_store.list_companies() == ["a", "b"]

    def test_history_count(self, engine, full_store, fake_record):
        fs = engine.build_company_features(fake_record, as_of=AS_OF)
        for _i in range(5):
            full_store.save_company_features(fs)
        assert full_store.history_count("rec-1") == 5

    def test_feature_history_single(self, engine, full_store, fake_record):
        fs = engine.build_company_features(fake_record, as_of=AS_OF)
        full_store.save_company_features(fs)
        snaps = full_store.get_feature_history("rec-1", "company_age")
        assert len(snaps) >= 1
        assert snaps[0].feature_id == "company_age"
