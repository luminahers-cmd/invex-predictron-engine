"""Tests for DecisionFeatureEngine — normalized consumption of Feature Store outputs."""

from __future__ import annotations

import pytest

from predictron_engine.decision.feature_engine import DecisionFeatureEngine
from predictron_engine.feature_store.models import (
    FeatureCategory,
    FeatureStatus,
)

from .conftest import build_feature_set, make_snapshot

REMOTE_FEATURES = [
    ("total_funding", 50_000_000.0, FeatureCategory.FUNDING, 50_000_000.0),
    ("average_round_size", 10_000_000.0, FeatureCategory.FUNDING, 30_000_000.0),
    ("funding_recency", 180.0, FeatureCategory.FUNDING, 30_000_000.0),
    ("funding_velocity", 0.8, FeatureCategory.GROWTH, 50_000_000.0),
    ("hiring_velocity", 1.5, FeatureCategory.GROWTH, 50_000_000.0),
    ("milestone_frequency", 2.0, FeatureCategory.GROWTH, 50_000_000.0),
]


class TestNormalizeFeature:
    def test_none_value(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.COMPANY, None, evidence=[])
        assert engine.normalize_feature(snap) == 0.0

    def test_bool_true(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.COMPANY, True, evidence=[])
        assert engine.normalize_feature(snap) == 1.0

    def test_bool_false(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.COMPANY, False, evidence=[])
        assert engine.normalize_feature(snap) == -1.0

    @pytest.mark.parametrize("feature_id,value,category,center_flag", [
        (fid, value, cat, _)
        for (fid, value, cat, _) in REMOTE_FEATURES
    ])
    def test_known_range_normalization(
        self,
        feature_id: str,
        value: float,
        category: FeatureCategory,
        center_flag: float,
    ) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot(feature_id, feature_id, category, value)
        result = engine.normalize_feature(snap)
        assert -1.0 <= result <= 1.0

    def test_value_in_zero_one_range(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.BENCHMARK, 0.5)
        result = engine.normalize_feature(snap)
        assert -1.0 <= result <= 1.0

    def test_list_empty(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.GROWTH, [], evidence=[])
        assert engine.normalize_feature(snap) == -0.2

    def test_list_populated(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.GROWTH, ["a", "b"], evidence=[])
        assert engine.normalize_feature(snap) >= 0.0

    def test_string_positive(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.COMPANY, "strong", evidence=[])
        assert engine.normalize_feature(snap) > 0.0

    def test_string_negative(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.COMPANY, "weak", evidence=[])
        assert engine.normalize_feature(snap) < 0.0

    def test_string_neutral(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.COMPANY, "USA", evidence=[])
        assert engine.normalize_feature(snap) == 0.0

    def test_failed_feature_normalizes_zero(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot(
            "f", "F", FeatureCategory.GROWTH, None,
            status=FeatureStatus.COMPUTATION_ERROR, evidence=[],
        )
        assert engine.normalize_feature(snap) == 0.0

    @pytest.mark.parametrize("value", [0.0, 1.0, 0.25, 0.75, 0.5])
    def test_in_unit_interval(self, value: float) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.SIGNALS, value)
        assert -1.0 <= engine.normalize_feature(snap) <= 1.0


class TestComputeFeatureWeight:
    def test_computed_feature_gets_category_weight(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("total_funding", "Total", FeatureCategory.FUNDING, 1.0)
        weight = engine.compute_feature_weight(snap)
        assert 0.0 < weight <= 1.0

    def test_failed_feature_zero_weight(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot(
            "f", "F", FeatureCategory.GROWTH, None,
            status=FeatureStatus.COMPUTATION_ERROR, evidence=[],
        )
        assert engine.compute_feature_weight(snap) == 0.0

    def test_evidence_boost(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.GROWTH, 0.5)
        weight = engine.compute_feature_weight(snap)
        assert weight <= 1.0

    @pytest.mark.parametrize("category", list(FeatureCategory))
    def test_all_categories_have_weight(self, category: FeatureCategory) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", category, 0.5)
        weight = engine.compute_feature_weight(snap)
        assert 0.0 <= weight <= 1.0

    def test_override_weights(self) -> None:
        engine = DecisionFeatureEngine({
            FeatureCategory.FUNDING: 0.5,
            FeatureCategory.GROWTH: 0.2,
        })
        snap = make_snapshot("total_funding", "Total", FeatureCategory.FUNDING, 1.0)
        assert engine.compute_feature_weight(snap) >= 0.5


class TestComputeContribution:
    def test_returns_contribution_with_provenance(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("funding_velocity", "Funding Velocity", FeatureCategory.GROWTH, 0.8)
        contribution = engine.compute_contribution(snap)
        assert contribution.feature_id == "funding_velocity"
        assert contribution.provenance["snapshot_id"] == snap.snapshot_id

    def test_evidence_references_preserved(self) -> None:
        engine = DecisionFeatureEngine()
        from predictron_engine.feature_store.models import EvidenceReference
        snap = make_snapshot(
            "f", "F", FeatureCategory.GROWTH, 0.5,
            evidence=[
                EvidenceReference(
                    source_type="timeline", source_id="rec-1",
                    source_field="signals",
                ),
            ],
        )
        contribution = engine.compute_contribution(snap)
        assert len(contribution.evidence_references) == 1
        assert contribution.evidence_references[0]["source_type"] == "timeline"

    def test_human_explanation_present(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.GROWTH, 0.5)
        contribution = engine.compute_contribution(snap)
        assert contribution.human_explanation

    def test_supporting_evidence_present(self) -> None:
        engine = DecisionFeatureEngine()
        snap = make_snapshot("f", "F", FeatureCategory.GROWTH, 0.5)
        contribution = engine.compute_contribution(snap)
        assert contribution.supporting_evidence


class TestComputeAllContributions:
    def test_all_features_contributions(self) -> None:
        engine = DecisionFeatureEngine()
        feature_set = build_feature_set()
        contributions = engine.compute_all_contributions(feature_set)
        assert len(contributions) == feature_set.feature_count()

    def test_deterministic_order(self) -> None:
        engine = DecisionFeatureEngine()
        feature_set = build_feature_set()
        c1 = engine.compute_all_contributions(feature_set)
        c2 = engine.compute_all_contributions(feature_set)
        assert [c.feature_id for c in c1] == [c.feature_id for c in c2]


class TestClassifyContributions:
    def test_returns_all_buckets(self) -> None:
        engine = DecisionFeatureEngine()
        feature_set = build_feature_set()
        contributions = engine.compute_all_contributions(feature_set)
        classified = engine.classify_contributions(contributions)
        assert set(classified.keys()) == {
            "positive", "negative", "neutral", "confidence",
        }

    def test_buckets_are_sorted(self) -> None:
        engine = DecisionFeatureEngine()
        feature_set = build_feature_set()
        contributions = engine.compute_all_contributions(feature_set)
        classified = engine.classify_contributions(contributions)
        for bucket in classified.values():
            magnitudes = [abs(c.computed_contribution) for c in bucket]
            assert magnitudes == sorted(magnitudes, reverse=True)


class TestProvenance:
    def test_get_feature_provenance(self) -> None:
        engine = DecisionFeatureEngine()
        feature_set = build_feature_set()
        provenance = engine.get_feature_provenance(feature_set)
        assert len(provenance) == feature_set.feature_count()
        for p in provenance:
            assert "feature_id" in p
            assert "evidence_references" in p

    def test_get_all_evidence_references(self) -> None:
        engine = DecisionFeatureEngine()
        feature_set = build_feature_set()
        refs = engine.get_all_evidence_references(feature_set)
        assert len(refs) >= 1
        assert all("source_type" in r for r in refs)
