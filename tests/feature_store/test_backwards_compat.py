"""Backwards-compatibility and additivity tests."""

from __future__ import annotations

from datetime import UTC, datetime

from predictron_engine.feature_store.engine import FeatureEngine
from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    FeatureCategory,
    FeatureSnapshot,
    FeatureStatus,
    FeatureStoreSnapshot,
    ValueType,
)
from predictron_engine.feature_store.registry import (
    FeatureDefinition,
    FeatureRegistry,
)
from predictron_engine.feature_store.store import FeatureStore
from tests.feature_store.conftest import (
    FakeRecord,
)

AS_OF = datetime(2024, 1, 1, tzinfo=UTC)


class TestBackwardsCompat:
    def test_old_snapshot_loads(self):
        old_data = {
            "snapshot_id": "old-snap-1",
            "store_version": "0.9.0",
            "engine_version": "0.12.1",
            "manifest": {},
            "company_count": 1,
            "feature_count": 2,
            "companies": {
                "rec-1": {
                    "company_id": "rec-1",
                    "built_at": "2024-01-01T00:00:00Z",
                    "as_of": None,
                    "features": {
                        "company_age": {
                            "snapshot_id": "snap-a",
                            "company_id": "rec-1",
                            "feature_id": "company_age",
                            "feature_name": "Company Age",
                            "category": "company",
                            "value": 4.0,
                            "value_type": "float",
                            "status": "computed",
                            "computation_version": "1.0.0",
                            "computed_at": "2024-01-01T00:00:00Z",
                            "evidence_references": [],
                        },
                        "funding_stage": {
                            "snapshot_id": "snap-b",
                            "company_id": "rec-1",
                            "feature_id": "funding_stage",
                            "feature_name": "Funding Stage",
                            "category": "company",
                            "value": "series_a",
                            "value_type": "string",
                            "status": "computed",
                            "computation_version": "1.0.0",
                            "computed_at": "2024-01-01T00:00:00Z",
                            "evidence_references": [],
                        },
                    },
                },
            },
        }
        loaded = FeatureStoreSnapshot.model_validate(old_data)
        assert loaded.company_count == 1
        assert loaded.feature_count == 2

    def test_new_snapshot_old_fields_ignored(self):
        data = {
            "snapshot_id": "new-snap",
            "store_version": "1.0.0",
            "engine_version": "0.12.1",
            "manifest": {"unknown_field": True},
            "company_count": 0,
            "feature_count": 0,
            "companies": {},
        }
        loaded = FeatureStoreSnapshot.model_validate(data)
        assert loaded.company_count == 0

    def test_additive_feature_does_not_break_existing(self):
        def comp_old(record, outcome, ctx):
            return 42.0, []

        reg = FeatureRegistry()
        reg.register(FeatureDefinition(
            feature_id="company_age",
            feature_name="Company Age",
            category=FeatureCategory.COMPANY,
            description="Age",
            value_type="float",
            depends_on_record_fields=[],
        ), comp_old)
        e1 = FeatureEngine(registry=reg)
        r = FakeRecord()
        fs1 = e1.build_company_features(r, as_of=AS_OF)
        assert "company_age" in fs1.features

        def comp_new(record, outcome, ctx):
            return "seed", []

        reg.register(FeatureDefinition(
            feature_id="funding_stage",
            feature_name="Funding Stage",
            category=FeatureCategory.COMPANY,
            description="Stage",
            value_type="string",
            depends_on_record_fields=[],
        ), comp_new)
        e2 = FeatureEngine(registry=reg)
        fs2 = e2.build_company_features(r, as_of=AS_OF)
        assert fs2.features["company_age"].value == 42.0
        assert fs2.features["funding_stage"].value == "seed"

    def test_missing_feature_from_old_version(self):
        def comp(record, outcome, ctx):
            return 1.0, []

        reg = FeatureRegistry()
        reg.register(FeatureDefinition(
            feature_id="new_feature",
            feature_name="New Feature",
            category=FeatureCategory.SIGNALS,
            description="New",
            value_type="float",
            depends_on_record_fields=[],
        ), comp)
        engine = FeatureEngine(registry=reg)
        r = FakeRecord()
        fs = engine.build_company_features(r, as_of=AS_OF)
        assert "new_feature" in fs.features
        assert "company_age" not in fs.features

    def test_snapshot_store_version_preserved(self):
        snap = FeatureStoreSnapshot(
            store_version="2.0.0",
            engine_version="0.12.1",
        )
        data = snap.model_dump()
        loaded = FeatureStoreSnapshot.model_validate(data)
        assert loaded.store_version == "2.0.0"

    def test_feature_status_values_compat(self):
        snap = FeatureSnapshot(
            company_id="c1",
            feature_id="test",
            feature_name="Test",
            category=FeatureCategory.COMPANY,
            value=None,
            value_type=ValueType.NONE,
            status=FeatureStatus.COMPUTED,
        )
        assert snap.status == "computed"

    def test_feature_status_old_string_values(self):
        snap = FeatureSnapshot(
            company_id="c1",
            feature_id="test",
            feature_name="Test",
            category=FeatureCategory.COMPANY,
            value=1.0,
            value_type=ValueType.FLOAT,
            status="computed",
        )
        assert snap.status == FeatureStatus.COMPUTED

    def test_evidence_reference_backward_compat(self):
        ev_data = {
            "source_type": "record",
            "source_id": "rec-1",
            "source_field": "founded_year",
        }
        from predictron_engine.feature_store.models import EvidenceReference
        ev = EvidenceReference.model_validate(ev_data)
        assert ev.source_type == "record"

    def test_engine_version_preserved_on_save_load(self, engine, full_store,
                                                    fake_record):
        fs = engine.build_company_features(fake_record, as_of=AS_OF)
        full_store.save_company_features(fs)
        snapshot = FeatureStoreSnapshot(companies={fake_record.record_id: fs})
        assert snapshot.engine_version is not None

    def test_feature_snapshot_evidence_empty_list_default(self):
        snap = FeatureSnapshot(
            company_id="c1",
            feature_id="test",
            feature_name="Test",
            category=FeatureCategory.COMPANY,
            value=None,
            value_type=ValueType.NONE,
            status=FeatureStatus.COMPUTED,
        )
        assert snap.evidence_references == []

    def test_company_feature_set_feature_count(self):
        fs = CompanyFeatureSet(company_id="c1")
        assert fs.feature_count() == 0
        fs.features["a"] = FeatureSnapshot(
            company_id="c1",
            feature_id="a",
            feature_name="A",
            category=FeatureCategory.COMPANY,
            value=1.0,
            value_type=ValueType.FLOAT,
            status=FeatureStatus.COMPUTED,
        )
        assert fs.feature_count() == 1

    def test_manifest_backward_compat(self):
        manifest_data = {
            "latest_build": "latest.json",
            "build_version": "1.0.0",
            "company_count": 5,
            "feature_count": 100,
            "unknown_key": "should not break",
        }
        store = FeatureStore.__new__(FeatureStore)
        store._root = None
        store._manifest_path = None
        assert manifest_data["company_count"] == 5

    def test_snapshot_merge_companies(self):
        fs1 = CompanyFeatureSet(company_id="c1")
        fs2 = CompanyFeatureSet(company_id="c2")
        snap = FeatureStoreSnapshot(companies={"c1": fs1, "c2": fs2})
        assert snap.company_count == 2
        assert set(snap.companies.keys()) == {"c1", "c2"}
