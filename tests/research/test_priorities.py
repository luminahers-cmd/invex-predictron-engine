"""Tests for the deterministic priority engine."""

from __future__ import annotations

from predictron_engine.research import (
    ResearchPriority,
    ResearchTopic,
    compute_priority_score,
    evidence_missing_factor,
    horizon_freshness_scale,
    priority_from_score,
    source_availability_factor,
)
from predictron_engine.research.models import EvidenceStatus


def _topic(**overrides: object) -> ResearchTopic:
    base: dict[str, object] = {
        "topic_id": "market",
        "name": "Market",
        "description": "Assess market size.",
        "importance": 0.80,
        "prediction_impact": 0.80,
        "freshness_requirement_days": 180,
        "freshness_sensitivity": 0.60,
        "source_categories": ("web_search",),
    }
    base.update(overrides)
    return ResearchTopic(**base)  # type: ignore[arg-type]


def _score(
    topic: ResearchTopic,
    status: EvidenceStatus = EvidenceStatus.NONE,
    *,
    dependency_factor: float = 0.0,
    prediction_horizon_days: int = 365,
    website_present: bool = False,
) -> float:
    return compute_priority_score(
        topic,
        status,
        dependency_factor=dependency_factor,
        prediction_horizon_days=prediction_horizon_days,
        website_present=website_present,
    )


class TestFactors:
    """Named deterministic factors."""

    def test_missing_evidence_factors(self) -> None:
        assert evidence_missing_factor(EvidenceStatus.NONE) == 1.0
        assert evidence_missing_factor(EvidenceStatus.PARTIAL) == 0.5
        assert evidence_missing_factor(EvidenceStatus.COMPLETE) == 0.0

    def test_horizon_scale_clamped(self) -> None:
        assert horizon_freshness_scale(365) == 1.0
        assert horizon_freshness_scale(10000) == 1.5
        assert horizon_freshness_scale(0) == 0.5

    def test_source_availability_factor(self) -> None:
        assert source_availability_factor("market", website_present=False) == 0.0
        assert source_availability_factor("product", website_present=True) == 1.0
        assert source_availability_factor("market", website_present=True) == 0.5


class TestScoreBehaviour:
    """The score is deterministic and monotonic per factor."""

    def test_deterministic_and_bounded(self) -> None:
        topic = _topic()
        for _ in range(5):
            score = _score(topic)
            assert 0.0 <= score <= 100.0
            assert round(score, 4) == score

    def test_incomplete_evidence_scores_ordered(self) -> None:
        topic = _topic()
        scores = [
            _score(topic, EvidenceStatus.NONE),
            _score(topic, EvidenceStatus.PARTIAL),
            _score(topic, EvidenceStatus.COMPLETE),
        ]
        assert scores[0] > scores[1] > scores[2]

    def test_factors_monotonically_raise_score(self) -> None:
        topic = _topic()
        base = _score(topic)
        assert _score(_topic(importance=0.9)) > base
        assert _score(_topic(prediction_impact=0.9)) > base
        assert _score(topic, dependency_factor=1.0) > base
        assert _score(_topic(topic_id="product"), website_present=True) > base

    def test_freshness_counts_only_when_missing(self) -> None:
        volatile = _topic(freshness_sensitivity=0.9)
        stable = _topic(freshness_sensitivity=0.2)
        assert _score(volatile, EvidenceStatus.COMPLETE) == _score(
            stable, EvidenceStatus.COMPLETE
        )
        assert _score(volatile, EvidenceStatus.NONE) > _score(
            stable, EvidenceStatus.NONE
        )

    def test_horizon_scales_freshness_weight(self) -> None:
        topic = _topic(freshness_sensitivity=0.9)
        assert _score(topic, prediction_horizon_days=730) > _score(
            topic, prediction_horizon_days=30
        )


class TestPriorityBands:
    """Score-to-band mapping is deterministic."""

    def test_band_thresholds(self) -> None:
        assert priority_from_score(90.0) is ResearchPriority.CRITICAL
        assert priority_from_score(74.99) is ResearchPriority.HIGH
        assert priority_from_score(54.99) is ResearchPriority.MEDIUM
        assert priority_from_score(34.99) is ResearchPriority.LOW
