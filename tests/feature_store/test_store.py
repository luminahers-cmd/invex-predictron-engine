"""Tests for the feature store persistence layer."""

from __future__ import annotations

from datetime import UTC, datetime

from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    FeatureCategory,
    FeatureSnapshot,
    FeatureStatus,
    FeatureStoreSnapshot,
)
from predictron_engine.feature_store.store import FeatureStore


def make_feature_set(company_id: str = "c1") -> CompanyFeatureSet:
    snap = FeatureSnapshot(
        company_id=company_id,
        feature_id="f1",
        feature_name="F1",
        category=FeatureCategory.COMPANY,
        value=1.5,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    return CompanyFeatureSet(
        company_id=company_id,
        features={"f1": snap},
        built_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


class TestFeatureStoreInit:
    def test_initialize_creates_dirs(self, tmp_path):
        store = FeatureStore(tmp_path / "fs")
        store.initialize()
        assert (tmp_path / "fs" / "companies").is_dir()
        assert (tmp_path / "fs" / "snapshots").is_dir()
        assert (tmp_path / "fs" / "history").is_dir()

    def test_manifest_created(self, tmp_path):
        store = FeatureStore(tmp_path / "fs")
        store.initialize()
        manifest = store.get_manifest()
        assert manifest["version"] == "1.0.0"

    def test_initialize_idempotent(self, tmp_path):
        store = FeatureStore(tmp_path / "fs")
        store.initialize()
        store.initialize()
        assert store.list_companies() == []


class TestFeatureStorePersistence:
    def test_save_and_load(self, feature_store):
        fs = make_feature_set()
        feature_store.save_company_features(fs)
        loaded = feature_store.load_company_features("c1")
        assert loaded is not None
        assert loaded.company_id == "c1"
        assert loaded.get_feature("f1").value == 1.5

    def test_load_missing(self, feature_store):
        assert feature_store.load_company_features("nope") is None

    def test_list_companies_sorted(self, feature_store):
        feature_store.save_company_features(make_feature_set("zeta"))
        feature_store.save_company_features(make_feature_set("alpha"))
        assert feature_store.list_companies() == ["alpha", "zeta"]

    def test_count_companies(self, feature_store):
        assert feature_store.count_companies() == 0
        feature_store.save_company_features(make_feature_set("c1"))
        assert feature_store.count_companies() == 1
        feature_store.save_company_features(make_feature_set("c2"))
        assert feature_store.count_companies() == 2

    def test_roundtrip_after_reload(self, feature_store, tmp_path):
        feature_store.save_company_features(make_feature_set("c1"))
        store2 = FeatureStore(tmp_path / "features")
        loaded = store2.load_company_features("c1")
        assert loaded is not None


class TestFeatureStoreHistory:
    def test_history_preserved(self, feature_store):
        fs1 = make_feature_set("c1")
        feature_store.save_company_features(fs1)

        snap2 = FeatureSnapshot(
            company_id="c1", feature_id="f1", feature_name="F1",
            category=FeatureCategory.COMPANY, value=2.5,
            computed_at=datetime(2024, 2, 1, tzinfo=UTC),
        )
        fs2 = CompanyFeatureSet(
            company_id="c1", features={"f1": snap2},
            built_at=datetime(2024, 2, 1, tzinfo=UTC),
        )
        feature_store.save_company_features(fs2)

        history = feature_store.get_company_history("c1")
        assert len(history) == 2
        assert history[0].get_feature("f1").value == 1.5
        assert history[1].get_feature("f1").value == 2.5

    def test_history_append_only(self, feature_store):
        fs = make_feature_set("c1")
        feature_store.save_company_features(fs)
        feature_store.save_company_features(fs)
        feature_store.save_company_features(fs)
        assert feature_store.history_count("c1") == 3

    def test_get_feature_history(self, feature_store):
        fs1 = make_feature_set("c1")
        feature_store.save_company_features(fs1)
        snap2 = FeatureSnapshot(
            company_id="c1", feature_id="f1", feature_name="F1",
            category=FeatureCategory.COMPANY, value=3.0,
            computed_at=datetime(2024, 3, 1, tzinfo=UTC),
        )
        fs2 = CompanyFeatureSet(
            company_id="c1", features={"f1": snap2},
            built_at=datetime(2024, 3, 1, tzinfo=UTC),
        )
        feature_store.save_company_features(fs2)

        history = feature_store.get_feature_history("c1", "f1")
        assert len(history) == 2
        assert [h.value for h in history] == [1.5, 3.0]

    def test_feature_history_missing_feature(self, feature_store):
        fs = make_feature_set("c1")
        feature_store.save_company_features(fs)
        history = feature_store.get_feature_history("c1", "nope")
        assert history == []

    def test_history_unknown_company(self, feature_store):
        assert feature_store.get_company_history("nope") == []
        assert feature_store.history_count("nope") == 0


class TestFeatureStoreSnapshots:
    def test_save_and_load_snapshot(self, feature_store):
        fs = make_feature_set("c1")
        snap = FeatureStoreSnapshot(companies={"c1": fs})
        path = feature_store.save_snapshot(snap)
        assert path.exists()
        loaded = feature_store.load_snapshot(snap.snapshot_id)
        assert loaded is not None
        assert loaded.company_count == 1

    def test_load_missing_snapshot(self, feature_store):
        assert feature_store.load_snapshot("nope") is None

    def test_list_snapshots(self, feature_store):
        snap1 = FeatureStoreSnapshot()
        feature_store.save_snapshot(snap1)
        snap2 = FeatureStoreSnapshot()
        feature_store.save_snapshot(snap2)
        assert len(feature_store.list_snapshots()) == 2

    def test_count_snapshots(self, feature_store):
        assert feature_store.count_snapshots() == 0
        feature_store.save_snapshot(FeatureStoreSnapshot())
        assert feature_store.count_snapshots() == 1


class TestFeatureStoreLookup:
    def test_lookup_feature(self, feature_store):
        feature_store.save_company_features(make_feature_set("c1"))
        snap = feature_store.lookup_feature("c1", "f1")
        assert snap is not None
        assert snap.value == 1.5

    def test_lookup_missing_feature(self, feature_store):
        feature_store.save_company_features(make_feature_set("c1"))
        assert feature_store.lookup_feature("c1", "missing") is None

    def test_lookup_unknown_company(self, feature_store):
        assert feature_store.lookup_feature("nope", "f1") is None

    def test_lookup_features(self, feature_store):
        feature_store.save_company_features(make_feature_set("c1"))
        result = feature_store.lookup_features("c1", ["f1", "missing"])
        assert result["f1"] is not None
        assert result["missing"] is None

    def test_lookup_features_unknown_company(self, feature_store):
        result = feature_store.lookup_features("nope", ["f1"])
        assert result == {"f1": None}


class TestFeatureStoreIncremental:
    def test_update_company_features(self, feature_store):
        fs1 = make_feature_set("c1")
        feature_store.save_company_features(fs1)
        snap2 = FeatureSnapshot(
            company_id="c1", feature_id="f1", feature_name="F1",
            category=FeatureCategory.COMPANY, value=10.0,
            computed_at=datetime(2024, 4, 1, tzinfo=UTC),
        )
        fs2 = CompanyFeatureSet(
            company_id="c1", features={"f1": snap2},
            built_at=datetime(2024, 4, 1, tzinfo=UTC),
        )
        feature_store.update_company_features(fs2)
        loaded = feature_store.load_company_features("c1")
        assert loaded.get_feature("f1").value == 10.0
        assert feature_store.history_count("c1") == 2


class TestFeatureStoreExportImport:
    def test_export_empty(self, feature_store):
        data = feature_store.export_all()
        assert data["version"] == "1.0.0"
        assert data["company_count"] == 0
        assert data["companies"] == {}

    def test_export_with_companies(self, feature_store):
        feature_store.save_company_features(make_feature_set("c1"))
        data = feature_store.export_all()
        assert data["company_count"] == 1
        assert "c1" in data["companies"]

    def test_import_all(self, feature_store):
        feature_store.save_company_features(make_feature_set("c1"))
        exported = feature_store.export_all()

        store2 = FeatureStore(feature_store._root / "imported")
        store2.initialize()
        count = store2.import_all(exported)
        assert count == 1
        loaded = store2.load_company_features("c1")
        assert loaded is not None
        assert loaded.get_feature("f1").value == 1.5

    def test_import_invalid_data(self, feature_store):
        count = feature_store.import_all({"companies": "invalid"})
        assert count == 0

    def test_import_roundtrip_identical(self, feature_store):
        fs = make_feature_set("c1")
        feature_store.save_company_features(fs)
        data = feature_store.export_all()
        store2 = FeatureStore(feature_store._root / "copy")
        store2.initialize()
        store2.import_all(data)
        original = feature_store.load_company_features("c1")
        restored = store2.load_company_features("c1")
        assert original.model_dump_json() == restored.model_dump_json()


class TestFeatureStoreManifest:
    def test_manifest_counts(self, feature_store):
        feature_store.save_company_features(make_feature_set("c1"))
        feature_store.save_company_features(make_feature_set("c2"))
        manifest = feature_store.get_manifest()
        assert manifest["company_count"] == 0

    def test_manifest_missing(self, tmp_path):
        store = FeatureStore(tmp_path / "no-init")
        assert store.get_manifest() == {}


class TestFeatureStoreStatusTracking:
    def test_failed_features_persisted(self, feature_store):
        failed = FeatureSnapshot(
            company_id="c1", feature_id="f_bad", feature_name="Bad",
            category=FeatureCategory.COMPANY,
            status=FeatureStatus.COMPUTATION_ERROR,
            error_message="it broke",
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        ok = FeatureSnapshot(
            company_id="c1", feature_id="f_ok", feature_name="Ok",
            category=FeatureCategory.COMPANY, value=1,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        fs = CompanyFeatureSet(
            company_id="c1",
            features={"f_bad": failed, "f_ok": ok},
            built_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        feature_store.save_company_features(fs)
        loaded = feature_store.load_company_features("c1")
        assert loaded.get_feature("f_bad").status == \
            FeatureStatus.COMPUTATION_ERROR
        assert loaded.get_feature("f_bad").error_message == "it broke"
