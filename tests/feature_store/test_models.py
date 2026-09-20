"""Tests for feature store models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    EvidenceReference,
    FeatureCategory,
    FeatureDefinition,
    FeatureSnapshot,
    FeatureStatus,
    FeatureStoreSnapshot,
    ValueType,
)


class TestFeatureCategory:
    def test_all_categories(self):
        categories = {c.value for c in FeatureCategory}
        assert categories == {
            "company",
            "growth",
            "founder",
            "funding",
            "knowledge_graph",
            "signals",
            "benchmark",
        }

    def test_category_ordering(self):
        values = [c.value for c in FeatureCategory]
        assert values == [
            "company",
            "growth",
            "founder",
            "funding",
            "knowledge_graph",
            "signals",
            "benchmark",
        ]

    def test_from_value(self):
        assert FeatureCategory("company") == FeatureCategory.COMPANY
        assert FeatureCategory("funding") == FeatureCategory.FUNDING

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            FeatureCategory("nope")


class TestValueType:
    def test_all_types(self):
        types = {t.value for t in ValueType}
        assert types == {
            "float", "int", "bool", "string", "list", "dict", "none",
        }

    def test_from_value(self):
        assert ValueType("float") == ValueType.FLOAT
        assert ValueType("int") == ValueType.INT


class TestFeatureDefinition:
    def test_defaults(self):
        defn = FeatureDefinition(
            feature_id="f1", feature_name="F1", category=FeatureCategory.COMPANY,
            description="desc", value_type=ValueType.FLOAT,
        )
        assert defn.computation_version == "1.0.0"
        assert defn.dependencies == []
        assert defn.source_fields == []
        assert defn.min_value is None
        assert defn.max_value is None
        assert defn.tags == []

    def test_full_definition(self):
        defn = FeatureDefinition(
            feature_id="f1",
            feature_name="F1",
            category=FeatureCategory.FUNDING,
            description="desc",
            value_type=ValueType.INT,
            computation_version="2.0.0",
            dependencies=["dep1"],
            source_fields=["field1"],
            min_value=0,
            max_value=100,
            tags=["tag1"],
        )
        assert defn.computation_version == "2.0.0"
        assert defn.dependencies == ["dep1"]
        assert defn.min_value == 0
        assert defn.max_value == 100
        assert defn.tags == ["tag1"]

    def test_serialization_roundtrip(self):
        defn = FeatureDefinition(
            feature_id="f1", feature_name="F1", category=FeatureCategory.COMPANY,
            description="desc", value_type=ValueType.FLOAT,
            dependencies=["a", "b"],
        )
        data = defn.model_dump()
        loaded = FeatureDefinition.model_validate(data)
        assert loaded == defn


class TestEvidenceReference:
    def test_defaults(self):
        ev = EvidenceReference(source_type="record", source_id="id-1")
        assert ev.source_field == ""
        assert ev.confidence == 1.0

    def test_full(self):
        ev = EvidenceReference(
            source_type="timeline",
            source_id="id-2",
            source_field="funding",
            confidence=0.8,
        )
        assert ev.source_type == "timeline"
        assert ev.confidence == 0.8

    def test_confidence_validation(self):
        with pytest.raises(ValueError):
            EvidenceReference(
                source_type="record", source_id="id", confidence=1.5,
            )

    def test_serialization_roundtrip(self):
        ev = EvidenceReference(
            source_type="record", source_id="id", source_field="f", confidence=0.5,
        )
        data = ev.model_dump()
        loaded = EvidenceReference.model_validate(data)
        assert loaded == ev


class TestFeatureSnapshot:
    def test_defaults(self):
        snap = FeatureSnapshot(
            company_id="c1", feature_id="f1", feature_name="F1",
            category=FeatureCategory.COMPANY,
        )
        assert snap.value is None
        assert snap.value_type == ValueType.NONE
        assert snap.status == FeatureStatus.COMPUTED
        assert snap.computation_version == "1.0.0"
        assert snap.dependencies_resolved == []
        assert snap.error_message is None
        assert snap.snapshot_id != ""

    def test_snapshot_id_unique(self):
        snaps = [
            FeatureSnapshot(
                company_id="c", feature_id="f", feature_name="F",
                category=FeatureCategory.COMPANY,
            )
            for _ in range(10)
        ]
        ids = {s.snapshot_id for s in snaps}
        assert len(ids) == 10

    def test_with_value(self):
        snap = FeatureSnapshot(
            company_id="c1", feature_id="f1", feature_name="F1",
            category=FeatureCategory.FUNDING, value=5, value_type=ValueType.INT,
        )
        assert snap.value == 5

    def test_to_dict(self):
        snap = FeatureSnapshot(
            company_id="c1", feature_id="f1", feature_name="F1",
            category=FeatureCategory.COMPANY, value=1.5,
            value_type=ValueType.FLOAT,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        data = snap.to_dict()
        assert data["company_id"] == "c1"
        assert data["feature_id"] == "f1"
        assert data["category"] == "company"
        assert data["value"] == 1.5
        assert data["value_type"] == "float"
        assert data["status"] == "computed"
        assert data["computed_at"] == "2024-01-01T00:00:00+00:00"

    def test_serialization_roundtrip(self):
        snap = FeatureSnapshot(
            company_id="c1", feature_id="f1", feature_name="F1",
            category=FeatureCategory.COMPANY, value=3.14,
            value_type=ValueType.FLOAT,
            dependencies_resolved=["dep-a"],
        )
        data = snap.model_dump()
        loaded = FeatureSnapshot.model_validate(data)
        assert loaded == snap


class TestCompanyFeatureSet:
    def test_empty(self):
        fs = CompanyFeatureSet(company_id="c1")
        assert fs.features == {}
        assert fs.feature_count() == 0
        assert fs.computed_features() == []
        assert fs.failed_features() == []

    def test_get_feature(self):
        snap = FeatureSnapshot(
            company_id="c1", feature_id="f1", feature_name="F1",
            category=FeatureCategory.COMPANY,
        )
        fs = CompanyFeatureSet(company_id="c1", features={"f1": snap})
        assert fs.get_feature("f1") == snap
        assert fs.get_feature("nope") is None

    def test_has_feature(self):
        snap = FeatureSnapshot(
            company_id="c1", feature_id="f1", feature_name="F1",
            category=FeatureCategory.COMPANY,
        )
        fs = CompanyFeatureSet(company_id="c1", features={"f1": snap})
        assert fs.has_feature("f1")
        assert not fs.has_feature("missing")

    def test_computed_features_only(self):
        computed = FeatureSnapshot(
            company_id="c1", feature_id="f1", feature_name="F1",
            category=FeatureCategory.COMPANY, value=1,
        )
        failed = FeatureSnapshot(
            company_id="c1", feature_id="f2", feature_name="F2",
            category=FeatureCategory.COMPANY,
            status=FeatureStatus.COMPUTATION_ERROR,
        )
        fs = CompanyFeatureSet(
            company_id="c1", features={"f1": computed, "f2": failed},
        )
        assert len(fs.computed_features()) == 1
        assert len(fs.failed_features()) == 1

    def test_serialization_roundtrip(self):
        snap = FeatureSnapshot(
            company_id="c1", feature_id="f1", feature_name="F1",
            category=FeatureCategory.FUNDING, value=10,
            value_type=ValueType.INT,
        )
        fs = CompanyFeatureSet(company_id="c1", features={"f1": snap})
        data = fs.model_dump()
        loaded = CompanyFeatureSet.model_validate(data)
        assert loaded.company_id == "c1"
        assert loaded.get_feature("f1").value == 10


class TestFeatureStoreSnapshot:
    def test_defaults(self):
        snap = FeatureStoreSnapshot()
        assert snap.store_version == "1.0.0"
        assert snap.engine_version == "0.12.1"
        assert snap.feature_version == "1.0.0"
        assert snap.company_count == 0
        assert snap.feature_count == 0
        assert snap.companies == {}

    def test_list_companies(self):
        a = CompanyFeatureSet(company_id="zebra")
        b = CompanyFeatureSet(company_id="apple")
        snap = FeatureStoreSnapshot(companies={"zebra": a, "apple": b})
        assert snap.list_companies() == ["apple", "zebra"]

    def test_get_company_features(self):
        a = CompanyFeatureSet(company_id="c1")
        snap = FeatureStoreSnapshot(companies={"c1": a})
        assert snap.get_company_features("c1") == a
        assert snap.get_company_features("missing") is None

    def test_serialization_roundtrip(self):
        fs = CompanyFeatureSet(company_id="c1")
        snap = FeatureStoreSnapshot(companies={"c1": fs})
        data = snap.model_dump()
        loaded = FeatureStoreSnapshot.model_validate(data)
        assert loaded.company_count == 1
        assert loaded.get_company_features("c1") is not None

    def test_counts_synced_from_companies(self):
        a = CompanyFeatureSet(company_id="c1")
        b = CompanyFeatureSet(company_id="c2")
        snap = FeatureStoreSnapshot(companies={"c1": a, "c2": b})
        assert snap.company_count == 2
