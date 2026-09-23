"""Unit tests for the Phase 7 learning engine builders.

All analytics are pure and deterministic: identical inputs must produce
identical snapshots (same ids, same hashes, same ordering).  The fixtures
and pinned values mirror ``benchmarks/learning_eval`` so the numbers asserted
here are the same hand-derived literals validated by the benchmark suite.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from predictron_engine.dataset.evaluation import EvaluationVerdict
from predictron_engine.learning.engine import (
    build_learning_snapshot,
    compute_calibration,
)
from predictron_engine.learning.models import (
    LEARNING_DIMENSIONS,
    LearningObservationCategory,
    LearningPeriodKind,
    LearningSample,
)

ANCHOR = date(2026, 9, 21)
NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
EPSILON = 1e-6


def _sample(
    evaluation_id: str,
    confidence: float,
    verdict: EvaluationVerdict,
    actual_positive: bool | None,
    resolved: dict[str, str] | None = None,
) -> LearningSample:
    return LearningSample(
        evaluation_id=evaluation_id,
        confidence=confidence,
        verdict=verdict,
        actual_positive=actual_positive,
        resolved=resolved
        or {
            "sector": "ai",
            "stage": "seed",
            "country": "us",
            "technology": "ml",
            "business_model": "saas",
            "founder": "team",
        },
    )


def _empty() -> list[LearningSample]:
    return []


def _perfect() -> list[LearningSample]:
    samples = []
    for index in range(5):
        samples.append(_sample(f"p-pos-{index}", 1.0, EvaluationVerdict.CORRECT, True))
        samples.append(_sample(f"p-neg-{index}", 0.0, EvaluationVerdict.CORRECT, False))
    return samples


def _mixed_bias() -> list[LearningSample]:
    samples = []
    for index in range(5):
        samples.append(_sample(f"m-pos-{index}", 0.9, EvaluationVerdict.CORRECT, True))
        samples.append(_sample(f"m-neg-{index}", 0.6, EvaluationVerdict.CORRECT, False))
    return samples


def _imperfect_balanced() -> list[LearningSample]:
    samples = []
    for index in range(2):
        samples.append(_sample(f"i-tp-{index}", 0.7, EvaluationVerdict.CORRECT, True))
    for index in range(2):
        samples.append(_sample(f"i-tn-{index}", 0.3, EvaluationVerdict.CORRECT, False))
    for index in range(3):
        samples.append(_sample(f"i-fp-{index}", 0.8, EvaluationVerdict.INCORRECT, False))
    for index in range(3):
        samples.append(_sample(f"i-fn-{index}", 0.4, EvaluationVerdict.INCORRECT, True))
    return samples


def _overconfident() -> list[LearningSample]:
    samples = []
    for index in range(5):
        samples.append(_sample(f"o-fn-{index}", 0.9, EvaluationVerdict.INCORRECT, True))
    for index in range(5):
        samples.append(_sample(f"o-tn-{index}", 0.2, EvaluationVerdict.CORRECT, False))
    return samples


def build(samples: list[LearningSample], *, anchor: date = ANCHOR, scope: str = "repository"):
    return build_learning_snapshot(
        samples,
        anchor_date=anchor,
        period_kind=LearningPeriodKind.DAILY,
        scope=scope,
        engine_version="7.0.0",
        recorded_at=NOW,
        as_of=NOW,
    )


class TestDeterminism:
    def test_identical_inputs_identical_snapshot(self) -> None:
        first = build(_mixed_bias())
        second = build(_mixed_bias())
        assert first.snapshot_id == second.snapshot_id
        assert first.content_hash == second.content_hash
        assert first.verify()
        assert first.model_dump(mode="json") == second.model_dump(mode="json")

    def test_different_scope_changes_id_not_hash(self) -> None:
        repo = build(_perfect(), scope="repository")
        user = build(_perfect(), scope="user:u1")
        assert repo.snapshot_id != user.snapshot_id
        # scope is part of the analytic payload, so the hash differs too
        assert repo.content_hash != user.content_hash

    def test_recorded_at_not_part_of_content_hash(self) -> None:
        a = build(_perfect(), anchor=ANCHOR)
        b = build_learning_snapshot(
            _perfect(),
            anchor_date=ANCHOR,
            period_kind=LearningPeriodKind.DAILY,
            scope="repository",
            engine_version="7.0.0",
            recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
            as_of=NOW,
        )
        assert a.content_hash == b.content_hash
        assert a.verify() and b.verify()


class TestConfusionAndDigest:
    def test_perfect_confusion(self) -> None:
        snap = build(_perfect())
        digest = snap.digest
        assert digest.scoreable == 10
        assert digest.true_positive == 5
        assert digest.true_negative == 5
        assert digest.false_positive == 0
        assert digest.false_negative == 0
        assert abs((digest.accuracy or 0.0) - 1.0) < EPSILON
        assert abs((digest.precision or 0.0) - 1.0) < EPSILON
        assert digest.recall is None or abs((digest.recall or 0.0) - 1.0) < EPSILON
        assert digest.verdict_counts == {"correct": 10}

    def test_imperfect_confusion(self) -> None:
        snap = build(_imperfect_balanced())
        digest = snap.digest
        assert digest.true_positive == 2
        assert digest.true_negative == 2
        assert digest.false_positive == 3
        assert digest.false_negative == 3
        assert abs((digest.accuracy or 0.0) - 0.4) < EPSILON
        assert abs((digest.precision or 0.0) - 0.4) < EPSILON
        assert abs((digest.recall or 0.0) - 0.4) < EPSILON
        assert abs((digest.false_positive_rate or 0.0) - 0.6) < EPSILON
        assert abs((digest.false_negative_rate or 0.0) - 0.6) < EPSILON

    def test_overconfident_precision_none(self) -> None:
        snap = build(_overconfident())
        assert snap.digest.precision is None  # no true positives
        assert abs((snap.digest.false_negative_rate or 0.0) - 1.0) < EPSILON
        assert snap.digest.false_positive_rate == 0.0

    def test_counters_match_fixtures(self) -> None:
        snap = build(_perfect())
        assert snap.counts == {"evaluations": 10, "samples": 10, "scoreable": 10}
        assert snap.metrics["accuracy"] == 1.0
        assert snap.metrics["ece"] == 0.0

    def test_empty_snapshot(self) -> None:
        snap = build(_empty())
        assert snap.counts == {"evaluations": 0, "samples": 0, "scoreable": 0}
        assert snap.digest.accuracy is None
        assert snap.verify()
        assert snap.observations == []
        assert snap.recommendations == []
        assert snap.metrics["confidence.mean"] is None


class TestCalibration:
    def test_calibration_empty(self) -> None:
        cal = compute_calibration(_empty())
        assert cal.expected_calibration_error == 0.0
        assert cal.overconfidence_detected is False
        assert cal.overconfident_bins == 0
        assert cal.total_samples == 0
        assert cal.bins == []

    def test_calibration_perfect_zero_ece(self) -> None:
        cal = compute_calibration(_perfect())
        assert abs(cal.expected_calibration_error - 0.0) < EPSILON
        assert cal.overconfidence_detected is False
        assert cal.overconfident_bins == 0

    def test_calibration_mixed_bias(self) -> None:
        cal = compute_calibration(_mixed_bias())
        # pos bin: conf 0.9 vs acc 1.0 -> gap 0.1 * (5/10)
        # neg bin (bin index 3 = conf 0.6): conf 0.6 vs acc 0.0 -> gap 0.6 * (5/10)
        assert abs(cal.expected_calibration_error - 0.35) < EPSILON
        assert cal.overconfidence_detected is True
        assert cal.overconfident_bins == 1
        assert cal.total_samples == 10

    def test_calibration_imperfect_two_overconfident_bins(self) -> None:
        cal = compute_calibration(_imperfect_balanced())
        assert abs(cal.expected_calibration_error - 0.54) < EPSILON
        assert cal.overconfident_bins == 2

    def test_calibration_overconfident(self) -> None:
        cal = compute_calibration(_overconfident())
        assert abs(cal.expected_calibration_error - 0.15) < EPSILON
        assert cal.overconfidence_detected is True
        assert cal.overconfident_bins == 1


class TestConfidence:
    def test_perfect_confidence(self) -> None:
        snap = build(_perfect())
        confidence = snap.confidence
        assert confidence.count == 10
        assert abs((confidence.mean or 0.0) - 0.5) < EPSILON
        assert confidence.bias == 0.0
        assert confidence.calibrated is True

    def test_mixed_bias_confidence(self) -> None:
        snap = build(_mixed_bias())
        assert abs((snap.confidence.mean or 0.0) - 0.75) < EPSILON
        assert abs(snap.confidence.bias - 0.25) < EPSILON

    def test_imperfect_confidence(self) -> None:
        snap = build(_imperfect_balanced())
        assert abs((snap.confidence.mean or 0.0) - 0.56) < EPSILON
        assert abs(snap.confidence.bias - 0.06) < EPSILON

    def test_overconfident_confidence(self) -> None:
        snap = build(_overconfident())
        assert abs((snap.confidence.mean or 0.0) - 0.55) < EPSILON
        assert abs(snap.confidence.bias - 0.05) < EPSILON


class TestPatterns:
    def test_pattern_ordering_follows_learning_dimensions(self) -> None:
        snap = build(_mixed_bias())
        assert [p.dimension.value for p in snap.patterns] == list(LEARNING_DIMENSIONS)

    def test_single_bucket_per_dimension(self) -> None:
        snap = build(_overconfident())
        # All samples share one attribute value per dimension -> 6 patterns.
        assert len(snap.patterns) == 6
        assert [p.dimension.value for p in snap.patterns] == list(LEARNING_DIMENSIONS)

    def test_mixed_bias_pattern_confusion(self) -> None:
        snap = build(_mixed_bias())
        pattern = next(p for p in snap.patterns if p.dimension.value == "sector")
        assert pattern.value == "ai"
        assert pattern.scoreable == 10
        assert pattern.true_positive == 5
        assert pattern.true_negative == 5
        assert pattern.false_positive == 0
        assert pattern.false_negative == 0
        assert abs((pattern.accuracy or 0.0) - 1.0) < EPSILON
        assert abs((pattern.confidence_bias or 0.0) - 0.25) < EPSILON

    def test_overconfident_patterns(self) -> None:
        snap = build(_overconfident())
        for pattern in snap.patterns:
            assert abs((pattern.false_negative_rate or 0.0) - 1.0) < EPSILON
            assert pattern.recommendation  # canned guidance always present


class TestObservations:
    def test_perfect_one_observation_per_dimension(self) -> None:
        snap = build(_perfect())
        assert len(snap.observations) == 6
        assert {o.category for o in snap.observations} == {
            LearningObservationCategory.STRONG_ACCURACY
        }

    def test_mixed_bias_observations(self) -> None:
        snap = build(_mixed_bias())
        assert len(snap.observations) == 12
        categories = {o.category for o in snap.observations}
        assert LearningObservationCategory.STRONG_ACCURACY in categories
        assert LearningObservationCategory.OVERCONFIDENCE in categories

    def test_imperfect_observations(self) -> None:
        snap = build(_imperfect_balanced())
        assert len(snap.observations) == 12
        categories = {o.category for o in snap.observations}
        assert LearningObservationCategory.HIGH_FALSE_POSITIVE_RATE in categories
        assert LearningObservationCategory.OVERCONFIDENCE in categories

    def test_overconfident_observations(self) -> None:
        snap = build(_overconfident())
        assert len(snap.observations) == 12
        categories = {o.category for o in snap.observations}
        assert LearningObservationCategory.HIGH_FALSE_NEGATIVE_RATE in categories
        assert LearningObservationCategory.OVERCONFIDENCE in categories

    def test_empty_observations(self) -> None:
        snap = build(_empty())
        assert snap.observations == []


class TestRecommendations:
    def test_perfect_recommendation(self) -> None:
        snap = build(_perfect())
        assert [r.kind for r in snap.recommendations] == [
            "maintain_confidence_weighting"
        ]

    def test_mixed_bias_recommendations(self) -> None:
        snap = build(_mixed_bias())
        kinds = [r.kind for r in snap.recommendations]
        assert "recalibrate_confidence" in kinds
        assert "maintain_confidence_weighting" in kinds  # accuracy 1.0 >= 0.6

    def test_imperfect_recommendation(self) -> None:
        snap = build(_imperfect_balanced())
        # ECE 0.54 > 0.10 -> recalibrate; accuracy 0.4 < 0.6 -> no maintain
        assert [r.kind for r in snap.recommendations] == ["recalibrate_confidence"]

    def test_empty_recommendations(self) -> None:
        snap = build(_empty())
        assert snap.recommendations == []


class TestKnowledgeAndDistributions:
    def test_knowledge_has_all_dimensions_sorted(self) -> None:
        snap = build(_mixed_bias())
        assert set(snap.knowledge) == set(LEARNING_DIMENSIONS)
        for entries in snap.knowledge.values():
            keys = [(e.value, e.sample_size) for e in entries]
            assert keys == sorted(keys)

    def test_distributions_are_sorted_count_maps(self) -> None:
        snap = build(_mixed_bias())
        assert snap.distributions["sector"] == {"ai": 10}
        assert snap.distributions["stage"] == {"seed": 10}

    def test_snapshot_contract(self) -> None:
        snap = build(_perfect())
        assert snap.scope == "repository"
        assert snap.period_kind == LearningPeriodKind.DAILY
        assert snap.anchor_date == ANCHOR
        assert snap.engine_version == "7.0.0"
        assert snap.snapshot_id
        assert len(snap.snapshot_id) == 64
        assert len(snap.content_hash) == 64
        assert snap.meta["schema_version"] == snap.schema_version

    def test_unknown_dimension_values_resolve(self) -> None:
        samples = [
            LearningSample(
                evaluation_id="u1",
                confidence=0.5,
                verdict=EvaluationVerdict.CORRECT,
                actual_positive=True,
                resolved={},  # nothing resolved -> "unknown" buckets
            )
        ]
        snap = build(samples)
        assert snap.distributions["sector"] == {"unknown": 1}
        pattern = snap.patterns[0]
        assert pattern.value == "unknown"
