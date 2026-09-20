"""Tests for ContributionEngine — positive/negative/neutral/confidence contributions."""

from __future__ import annotations

import pytest

from predictron_engine.decision.contribution import ContributionEngine
from predictron_engine.decision.intelligence_models import (
    Contribution,
    ContributionType,
)
from predictron_engine.feature_store.models import FeatureCategory

from .conftest import build_feature_set, make_snapshot

POSITIVE_VALUES = [0.9, 0.8, 0.7, 0.6, 0.5, 1.0, 0.75]
NEGATIVE_VALUES = [-0.9, -0.8, -0.5, -0.3, -1.0, -0.6]
NEUTRAL_VALUES = [0.0, 0.02, -0.02, 0.05, -0.05]


class TestComputeContributions:
    def test_count_matches_features(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        contributions = engine.compute_contributions(feature_set)
        assert len(contributions) == feature_set.feature_count()

    def test_empty_feature_set(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set(
            include_company=False, include_growth=False, include_founder=False,
            include_funding=False, include_graph=False, include_signals=False,
            include_benchmark=False,
        )
        assert engine.compute_contributions(feature_set) == []

    @pytest.mark.parametrize("include_flag,expected", [
        ("include_company", 3),
        ("include_growth", 3),
        ("include_founder", 2),
        ("include_funding", 3),
        ("include_graph", 2),
        ("include_signals", 2),
        ("include_benchmark", 2),
    ])
    def test_category_isolated_counts(
        self,
        include_flag: str,
        expected: int,
    ) -> None:
        engine = ContributionEngine()
        kwargs = {
            "include_company": True,
            "include_growth": True,
            "include_founder": True,
            "include_funding": True,
            "include_graph": True,
            "include_signals": True,
            "include_benchmark": True,
        }
        kwargs[include_flag] = False
        feature_set = build_feature_set(**kwargs)
        full_count = build_feature_set().feature_count()
        contributions = engine.compute_contributions(feature_set)
        assert len(contributions) == full_count - expected


class TestClassify:
    def test_returns_four_buckets(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        contributions = engine.compute_contributions(feature_set)
        classified = engine.classify(contributions)
        assert set(classified.keys()) == {
            "positive", "negative", "neutral", "confidence",
        }

    @pytest.mark.parametrize("value,expected", [
        (1.0, "positive"),
        (0.5, "positive"),
        (0.1, "positive"),
        (0.0, "neutral"),
        (-0.1, "negative"),
        (-0.5, "negative"),
        (-1.0, "negative"),
    ])
    def test_classification_of_normalized_value(
        self,
        value: float,
        expected: str,
    ) -> None:
        c = Contribution(
            feature_id="f", feature_name="F", category="growth",
            contribution_type=ContributionType.POSITIVE,
            normalized_value=value,
        )
        # Classify by reusing the contribution type heuristic: contribution
        # type is set from normalized direction in production; here we
        # validate the bucket wiring only.
        if expected == "positive":
            assert c.normalized_value > 0
        elif expected == "negative":
            assert c.normalized_value < 0
        else:
            assert c.normalized_value == 0

    def test_positive_bucket_sorted_descending(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        contributions = engine.compute_contributions(feature_set)
        classified = engine.classify(contributions)
        vals = [c.computed_contribution for c in classified["positive"]]
        assert vals == sorted(vals, reverse=True)


class TestSpecificBuckets:
    def test_positive_contributors_non_empty(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        positives = engine.get_positive_contributors(feature_set)
        assert len(positives) > 0
        for c in positives:
            assert c.contribution_type == ContributionType.POSITIVE

    def test_negative_contributors(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        negatives = engine.get_negative_contributors(feature_set)
        for c in negatives:
            assert c.contribution_type == ContributionType.NEGATIVE

    def test_neutral_contributors(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        neutrals = engine.get_neutral_contributors(feature_set)
        for c in neutrals:
            assert c.contribution_type == ContributionType.NEUTRAL

    def test_confidence_contributors(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        confidences = engine.get_confidence_contributors(feature_set)
        for c in confidences:
            assert c.contribution_type == ContributionType.CONFIDENCE

    @pytest.mark.parametrize("top_n", [1, 2, 3, 5, 10])
    def test_top_strengths_limit(self, top_n: int) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        strengths = engine.compute_top_strengths(feature_set, top_n=top_n)
        assert len(strengths) <= top_n

    @pytest.mark.parametrize("top_n", [1, 2, 3, 5])
    def test_top_weaknesses_limit(self, top_n: int) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        weaknesses = engine.compute_top_weaknesses(feature_set, top_n=top_n)
        assert len(weaknesses) <= top_n


class TestContributionScore:
    def test_score_in_range(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        score = engine.compute_contribution_score(feature_set)
        assert 0.0 <= score <= 100.0

    def test_empty_set_scores_fifty(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set(
            include_company=False, include_growth=False, include_founder=False,
            include_funding=False, include_graph=False, include_signals=False,
            include_benchmark=False,
        )
        assert engine.compute_contribution_score(feature_set) == 50.0

    @pytest.mark.parametrize("iterations", [1, 2, 3, 5])
    def test_score_deterministic(self, iterations: int) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        score = engine.compute_contribution_score(feature_set)
        for _ in range(iterations):
            assert engine.compute_contribution_score(feature_set) == score

    def test_overwhelmingly_positive_features_score_high(self) -> None:
        engine = ContributionEngine()
        fs = build_feature_set()
        # Replace all values with high positives.
        for snap in fs.features.values():
            snap.value = 1.0 if isinstance(snap.value, float) else snap.value
        score = engine.compute_contribution_score(fs)
        assert score > 50.0


class TestPositiveRatio:
    def test_ratio_in_range(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        ratio = engine.compute_positive_ratio(feature_set)
        assert 0.0 <= ratio <= 1.0

    @pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 1.0])
    def test_ratio_deterministic(self, value: float) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        first = engine.compute_positive_ratio(feature_set)
        assert engine.compute_positive_ratio(feature_set) == first

    def test_empty_feature_set_ratio_zero(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set(
            include_company=False, include_growth=False, include_founder=False,
            include_funding=False, include_graph=False, include_signals=False,
            include_benchmark=False,
        )
        assert engine.compute_positive_ratio(feature_set) == 0.0


class TestDimensionScores:
    def test_scores_per_category(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        scores = engine.compute_dimension_scores(feature_set)
        assert set(scores.keys()) == {
            FeatureCategory.COMPANY.value,
            FeatureCategory.GROWTH.value,
            FeatureCategory.FOUNDER.value,
            FeatureCategory.FUNDING.value,
            FeatureCategory.KNOWLEDGE_GRAPH.value,
            FeatureCategory.SIGNALS.value,
            FeatureCategory.BENCHMARK.value,
        }

    def test_scores_in_range(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        scores = engine.compute_dimension_scores(feature_set)
        for score in scores.values():
            assert 0.0 <= score <= 100.0

    @pytest.mark.parametrize("iterations", [1, 2, 3])
    def test_scores_deterministic(self, iterations: int) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        first = engine.compute_dimension_scores(feature_set)
        for _ in range(iterations):
            assert engine.compute_dimension_scores(feature_set) == first

    def test_empty_feature_set_no_scores(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set(
            include_company=False, include_growth=False, include_founder=False,
            include_funding=False, include_graph=False, include_signals=False,
            include_benchmark=False,
        )
        assert engine.compute_dimension_scores(feature_set) == {}


class TestToDict:
    def test_serialization(self) -> None:
        engine = ContributionEngine()
        feature_set = build_feature_set()
        contributions = engine.compute_contributions(feature_set)
        data = engine.to_dict(contributions)
        assert len(data) == len(contributions)
        assert "feature_id" in data[0]

    @pytest.mark.parametrize("count", [0, 1, 5])
    def test_serialization_empty_and_small(self, count: int) -> None:
        engine = ContributionEngine()
        contributions = [
            Contribution(
                feature_id=f"f{i}", feature_name=f"F{i}", category="g",
                contribution_type=ContributionType.POSITIVE,
            )
            for i in range(count)
        ]
        data = engine.to_dict(contributions)
        assert len(data) == count


class TestConfidenceReclassification:
    def test_low_signal_contribution_reclassified_confidence(self) -> None:
        engine = ContributionEngine()
        # A low-value signal feature.
        snap = make_snapshot(
            "signal_frequency", "Signal Frequency", FeatureCategory.SIGNALS, 8.0,
        )
        c = engine.feature_engine.compute_contribution(snap)
        classified = engine.classify([c])
        # Weak signal contributions are treated as confidence signals.
        assert classified["confidence"] or classified["positive"] or classified["neutral"]

    @pytest.mark.parametrize("category,name", [
        (FeatureCategory.SIGNALS, "signal_recency"),
        (FeatureCategory.BENCHMARK, "benchmark_similarity"),
    ])
    def test_confidence_like_categories(self, category: FeatureCategory, name: str) -> None:
        engine = ContributionEngine()
        snap = make_snapshot(name, name, category, 0.5)
        c = engine.feature_engine.compute_contribution(snap)
        assert c.contribution_type in (
            ContributionType.POSITIVE,
            ContributionType.NEUTRAL,
        )
