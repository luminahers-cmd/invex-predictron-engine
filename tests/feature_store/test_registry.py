"""Tests for the feature registry."""

from __future__ import annotations

from typing import Any

import pytest

from predictron_engine.feature_store.models import (
    FeatureCategory,
    FeatureDefinition,
    ValueType,
)
from predictron_engine.feature_store.registry import FeatureRegistry


def make_defn(
    fid: str,
    category: FeatureCategory = FeatureCategory.COMPANY,
    deps: list[str] | None = None,
    tags: list[str] | None = None,
) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=fid,
        feature_name=fid,
        category=category,
        description=f"description for {fid}",
        value_type=ValueType.FLOAT,
        computation_version="1.0.0",
        dependencies=deps or [],
        tags=tags or ["test"],
    )


def noop_computor(
    record: Any, deps: dict[str, Any], ctx: dict[str, Any]
) -> tuple[Any, list[dict[str, Any]]]:
    return None, []


class TestRegistryBasics:
    def test_empty_registry(self):
        reg = FeatureRegistry()
        assert reg.count() == 0
        assert reg.list_all() == []
        assert reg.list_ids() == []

    def test_register(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1"), noop_computor)
        assert reg.count() == 1
        assert reg.has("f1")
        assert reg.get("f1") is not None
        assert reg.get_computor("f1") is not None

    def test_register_duplicate_raises(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1"), noop_computor)
        with pytest.raises(ValueError):
            reg.register(make_defn("f1"), noop_computor)

    def test_register_all(self):
        reg = FeatureRegistry()
        reg.register_all([
            (make_defn("f1"), noop_computor),
            (make_defn("f2"), noop_computor),
        ])
        assert reg.count() == 2

    def test_get_missing(self):
        reg = FeatureRegistry()
        assert reg.get("nope") is None
        assert reg.get_computor("nope") is None

    def test_list_ids_sorted(self):
        reg = FeatureRegistry()
        for fid in ["b", "a", "c"]:
            reg.register(make_defn(fid), noop_computor)
        assert reg.list_ids() == ["a", "b", "c"]

    def test_missing_features(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1"), noop_computor)
        assert reg.missing_features(["f1", "f2", "f3"]) == ["f2", "f3"]
        assert reg.missing_features([]) == []
        assert reg.missing_features(["f1"]) == []


class TestRegistryCategories:
    def test_list_by_category(self):
        reg = FeatureRegistry()
        reg.register(make_defn("c1", FeatureCategory.COMPANY), noop_computor)
        reg.register(make_defn("g1", FeatureCategory.GROWTH), noop_computor)
        reg.register(make_defn("c2", FeatureCategory.COMPANY), noop_computor)

        company = reg.list_by_category(FeatureCategory.COMPANY)
        assert [d.feature_id for d in company] == ["c1", "c2"]
        growth = reg.list_by_category(FeatureCategory.GROWTH)
        assert [d.feature_id for d in growth] == ["g1"]
        assert reg.list_by_category(FeatureCategory.FOUNDER) == []

    def test_list_by_tag(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1", tags=["alpha", "shared"]), noop_computor)
        reg.register(make_defn("f2", tags=["beta", "shared"]), noop_computor)
        shared = reg.list_by_tag("shared")
        assert [d.feature_id for d in shared] == ["f1", "f2"]
        alpha = reg.list_by_tag("alpha")
        assert [d.feature_id for d in alpha] == ["f1"]
        assert reg.list_by_tag("missing") == []


class TestRegistryDependencies:
    def test_dependency_graph(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1", deps=["f2", "f3"]), noop_computor)
        reg.register(make_defn("f2", deps=["f3"]), noop_computor)
        reg.register(make_defn("f3"), noop_computor)

        graph = reg.dependency_graph()
        assert graph["f1"] == ["f2", "f3"]
        assert graph["f2"] == ["f3"]
        assert graph["f3"] == []

    def test_all_dependencies(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1", deps=["f2"]), noop_computor)
        reg.register(make_defn("f2", deps=["f3"]), noop_computor)
        reg.register(make_defn("f3"), noop_computor)

        deps = reg.all_dependencies("f1")
        assert deps[0] == "f3"
        assert deps[1] == "f2"
        assert deps[2] == "f1"

    def test_all_dependencies_single(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1"), noop_computor)
        assert reg.all_dependencies("f1") == ["f1"]

    def test_all_dependencies_shared(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1", deps=["f2", "f3"]), noop_computor)
        reg.register(make_defn("f2", deps=["f3"]), noop_computor)
        reg.register(make_defn("f3"), noop_computor)
        deps = reg.all_dependencies("f1")
        assert deps.count("f3") == 1

    def test_topological_order_simple(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1"), noop_computor)
        reg.register(make_defn("f2"), noop_computor)
        assert reg.topological_order() == ["f1", "f2"]

    def test_topological_order_dependencies(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1", deps=["f2"]), noop_computor)
        reg.register(make_defn("f2"), noop_computor)
        order = reg.topological_order()
        assert order.index("f2") < order.index("f1")

    def test_topological_order_chain(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f3", deps=["f2"]), noop_computor)
        reg.register(make_defn("f2", deps=["f1"]), noop_computor)
        reg.register(make_defn("f1"), noop_computor)
        order = reg.topological_order()
        assert order == ["f1", "f2", "f3"]

    def test_topological_order_diamond(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f4", deps=["f2", "f3"]), noop_computor)
        reg.register(make_defn("f2", deps=["f1"]), noop_computor)
        reg.register(make_defn("f3", deps=["f1"]), noop_computor)
        reg.register(make_defn("f1"), noop_computor)
        order = reg.topological_order()
        assert order.index("f1") < order.index("f2")
        assert order.index("f1") < order.index("f3")
        assert order.index("f2") < order.index("f4")
        assert order.index("f3") < order.index("f4")

    def test_topological_order_deterministic(self):
        reg = FeatureRegistry()
        reg.register(make_defn("b", deps=["a"]), noop_computor)
        reg.register(make_defn("a"), noop_computor)
        reg.register(make_defn("c", deps=["a"]), noop_computor)
        first = reg.topological_order()
        second = reg.topological_order()
        assert first == second

    def test_validate_dependencies_ok(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1", deps=["f2"]), noop_computor)
        reg.register(make_defn("f2"), noop_computor)
        assert reg.validate_dependencies() == []

    def test_validate_dependencies_broken(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1", deps=["missing"]), noop_computor)
        errors = reg.validate_dependencies()
        assert len(errors) == 1
        assert "missing" in errors[0]

    def test_validate_dependencies_multiple_broken(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1", deps=["a", "b"]), noop_computor)
        errors = reg.validate_dependencies()
        assert len(errors) == 2

    def test_validate_dependencies_mixed(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1", deps=["f2", "missing"]), noop_computor)
        reg.register(make_defn("f2"), noop_computor)
        errors = reg.validate_dependencies()
        assert len(errors) == 1

    def test_to_dict(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1", FeatureCategory.FUNDING, deps=["f2"]),
                     noop_computor)
        reg.register(make_defn("f2"), noop_computor)
        data = reg.to_dict()
        assert data["feature_count"] == 2
        assert "funding" in data["categories"]
        assert data["features"]["f1"]["category"] == "funding"
        assert data["features"]["f1"]["dependencies"] == ["f2"]

    def test_to_dict_deterministic(self):
        reg = FeatureRegistry()
        reg.register(make_defn("f1", deps=["f2"]), noop_computor)
        reg.register(make_defn("f2"), noop_computor)
        first = reg.to_dict()
        second = reg.to_dict()
        assert first == second


class TestRegistryAllFeatures:
    @pytest.fixture(scope="class")
    def all_registry(self):
        from predictron_engine.feature_store.features import ALL_FEATURES
        reg = FeatureRegistry()
        reg.register_all(ALL_FEATURES)
        return reg

    def test_all_features_count(self, all_registry):
        assert all_registry.count() == 29

    def test_every_category_covered(self, all_registry):
        categories = {
            d.category for d in all_registry.list_all()
        }
        assert categories == {c for c in FeatureCategory}

    def test_benchmark_dependencies_resolved(self, all_registry):
        errors = all_registry.validate_dependencies()
        assert errors == []

    def test_benchmark_similarity_deps(self, all_registry):
        defn = all_registry.get("benchmark_similarity")
        assert defn is not None
        assert set(defn.dependencies) == {
            "company_age", "funding_stage", "industry",
        }

    def test_topological_order_no_cycles(self, all_registry):
        order = all_registry.topological_order()
        assert len(order) == all_registry.count()
        assert len(set(order)) == len(order)

    def test_specific_feature_exists(self, all_registry):
        for fid in [
            "company_age", "funding_stage", "employee_band",
            "operating_country", "industry",
            "funding_velocity", "hiring_velocity",
            "milestone_frequency", "growth_consistency", "momentum_score",
            "founder_count", "repeat_founder_indicator",
            "founder_change_count",
            "total_funding", "funding_round_count", "average_round_size",
            "investor_count", "funding_recency",
        ]:
            assert all_registry.has(fid), f"missing {fid}"

    def test_computors_registered(self, all_registry):
        for fid in all_registry.list_ids():
            assert all_registry.get_computor(fid) is not None

    def test_category_counts(self, all_registry):
        counts = {
            FeatureCategory.COMPANY: 5,
            FeatureCategory.GROWTH: 5,
            FeatureCategory.FOUNDER: 3,
            FeatureCategory.FUNDING: 5,
            FeatureCategory.KNOWLEDGE_GRAPH: 4,
            FeatureCategory.SIGNALS: 4,
            FeatureCategory.BENCHMARK: 3,
        }
        for category, expected in counts.items():
            assert len(all_registry.list_by_category(category)) == expected, \
                f"{category} expected {expected}"

    def test_total_is_sum_of_categories(self, all_registry):
        total = sum(
            len(all_registry.list_by_category(c))
            for c in FeatureCategory
        )
        assert total == all_registry.count()
