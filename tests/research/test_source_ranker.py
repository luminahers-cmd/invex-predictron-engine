"""Tests for the deterministic Source Ranker.

Covers the weight set, each ranking factor in isolation (ceteris
paribus), topic-fit coverage, tie-breaking, freshness context,
determinism, and score component correctness.
"""

from __future__ import annotations

import pytest

from predictron_engine.research import (
    RANKING_WEIGHTS,
    ResearchSource,
    SourceCategory,
    compute_source_score,
    freshness_priority_scale,
    rank_sources,
)
from predictron_engine.research.source_ranker import (
    COST_WEIGHT,
    COVERAGE_WEIGHT,
    FRESHNESS_WEIGHT,
    STRUCTURED_WEIGHT,
    TRUST_WEIGHT,
)

TOPIC = "technology"


def _source(
    identifier: str, **overrides: object
) -> ResearchSource:
    base: dict[str, object] = {
        "identifier": identifier,
        "display_name": identifier.title(),
        "source_category": SourceCategory.GITHUB,
        "trust_score": 0.5,
        "freshness_score": 0.5,
        "coverage_score": 0.5,
        "relative_cost": 0.5,
        "supports_structured_data": False,
        "preferred_topics": (TOPIC,),
    }
    base.update(overrides)
    return ResearchSource(**base)  # type: ignore[arg-type]


def _ids(ranked: object) -> tuple[str, ...]:
    return tuple(entry.source.identifier for entry in ranked)


def _totals(ranked: object) -> list[float]:
    return [entry.score.total for entry in ranked]


class TestWeights:
    """The fixed weight set is normalized and exposed."""

    def test_weights_sum_to_one(self) -> None:
        assert sum(RANKING_WEIGHTS.values()) == pytest.approx(1.0)

    def test_positive_weights(self) -> None:
        assert all(weight > 0 for weight in RANKING_WEIGHTS.values())

    def test_rank_weights_reconcile(self) -> None:
        assert RANKING_WEIGHTS["trust"] == TRUST_WEIGHT
        assert RANKING_WEIGHTS["coverage"] == COVERAGE_WEIGHT
        assert RANKING_WEIGHTS["freshness"] == FRESHNESS_WEIGHT
        assert RANKING_WEIGHTS["cost"] == COST_WEIGHT
        assert RANKING_WEIGHTS["structured_data"] == STRUCTURED_WEIGHT


class TestFreshnessScale:
    """Freshness emphasis shrinks as the requirement lengthens."""

    def test_monotonic_non_increasing(self) -> None:
        short = freshness_priority_scale(30)
        nominal = freshness_priority_scale(180)
        long = freshness_priority_scale(365)
        extended = freshness_priority_scale(730)
        assert short >= nominal >= long >= extended
        assert short <= 1.0

    def test_unbounded_horizon(self) -> None:
        assert freshness_priority_scale(1000) == freshness_priority_scale(365 * 2)

    def test_invalid_requirement_rejected(self) -> None:
        for days in (0, -1):
            with pytest.raises(ValueError):
                freshness_priority_scale(days)


class TestRankingFactors:
    """Each factor orders candidates deterministically when isolated."""

    def assert_ranks(self, higher: ResearchSource, lower: ResearchSource) -> None:
        ranked = rank_sources((lower, higher), TOPIC)
        high_order = _ids(ranked).index(higher.identifier)
        low_order = _ids(ranked).index(lower.identifier)
        assert high_order < low_order

    @pytest.mark.parametrize(
        ("higher", "lower"),
        [
            ({"trust_score": 0.9}, {"trust_score": 0.2}),
            ({"coverage_score": 0.9}, {"coverage_score": 0.2}),
            ({"freshness_score": 0.9}, {"freshness_score": 0.2}),
            ({"relative_cost": 0.0}, {"relative_cost": 0.9}),
            (
                {"supports_structured_data": True},
                {"supports_structured_data": False},
            ),
            (
                {
                    "coverage_score": 0.8,
                    "preferred_topics": (TOPIC,),
                },
                {
                    "coverage_score": 0.8,
                    "preferred_topics": ("market",),
                },
            ),
        ],
    )
    def test_factor_orders_candidates(
        self,
        higher: dict[str, object],
        lower: dict[str, object],
    ) -> None:
        self.assert_ranks(
            _source("factor_high", **higher),  # type: ignore[arg-type]
            _source("factor_low", **lower),  # type: ignore[arg-type]
        )

    def test_freshness_requirement_shifts_rankings(self) -> None:
        fresh = _source("fr_fresh", freshness_score=0.9)
        stale = _source("fr_stale", freshness_score=0.2)
        short = rank_sources((stale, fresh), TOPIC, freshness_requirement_days=30)
        long = rank_sources((stale, fresh), TOPIC, freshness_requirement_days=365)
        gap_short = short[0].score.total - short[1].score.total
        gap_long = long[0].score.total - long[1].score.total
        assert gap_short > gap_long


class TestScoring:
    """compute_source_score produces consistent normalized components."""

    def test_components_match_source_metadata(self) -> None:
        source = _source(
            "score_me",
            trust_score=0.8,
            freshness_score=0.7,
            coverage_score=0.6,
            relative_cost=0.4,
            supports_structured_data=True,
        )
        score = compute_source_score(source, TOPIC)
        assert score.trust == 0.8
        assert score.coverage == pytest.approx(0.6)  # full topic fit
        assert score.topic_fit == 1.0
        assert score.cost == pytest.approx(0.6)
        assert score.structured_data == 1.0

    def test_topic_fit_halves_coverage_for_unpreferred(self) -> None:
        source = _source(
            "no_fit",
            coverage_score=0.6,
            preferred_topics=("market",),
        )
        score = compute_source_score(source, TOPIC)
        assert score.topic_fit == 0.0
        assert score.coverage == pytest.approx(0.3)

    def test_total_matches_weighted_formula(self) -> None:
        source = _source(
            "formula",
            trust_score=0.8,
            freshness_score=0.9,
            coverage_score=0.7,
            relative_cost=0.4,
            supports_structured_data=True,
        )
        # freshness_requirement_days=30 -> freshness scale of 1.0.
        score = compute_source_score(source, TOPIC, freshness_requirement_days=30)
        expected = round(
            source.trust_score * TRUST_WEIGHT
            + source.coverage_score * COVERAGE_WEIGHT
            + source.freshness_score * FRESHNESS_WEIGHT
            + (1.0 - source.relative_cost) * COST_WEIGHT
            + STRUCTURED_WEIGHT,
            6,
        )
        assert score.total == pytest.approx(expected)

    def test_total_within_unit_range(self) -> None:
        perfect = _source(
            "perfect",
            trust_score=1.0,
            coverage_score=1.0,
            freshness_score=1.0,
            relative_cost=0.0,
            supports_structured_data=True,
        )
        score = compute_source_score(
            perfect, TOPIC, freshness_requirement_days=30
        )
        assert 0.0 <= score.total <= 1.0

    def test_invalid_requirement_rejected(self) -> None:
        source = _source("invalid_fd")
        with pytest.raises(ValueError):
            compute_source_score(
                source, TOPIC, freshness_requirement_days=0
            )
        with pytest.raises(ValueError):
            rank_sources(
                (source,),
                TOPIC,
                freshness_requirement_days=-3,
            )


class TestRankingBehaviour:
    """Rank output is ranked, sequential, and deterministic."""

    def test_ranks_sequential_and_scores_non_increasing(self) -> None:
        ranked = rank_sources(
            (
                _source("a", trust_score=0.9),
                _source("b", trust_score=0.8),
                _source("c", trust_score=0.7),
            ),
            TOPIC,
        )
        assert [entry.rank for entry in ranked] == [1, 2, 3]
        totals = _totals(ranked)
        assert totals == sorted(totals, reverse=True)

    def test_deterministic_across_calls(self) -> None:
        sources = (
            _source("a", trust_score=0.9),
            _source("b", trust_score=0.5),
            _source("c", trust_score=0.8),
        )
        first = rank_sources(sources, TOPIC)
        second = rank_sources(sources, TOPIC)
        assert first == second
        assert _ids(first) == _ids(second)

    def test_input_order_irrelevant_for_distinct_scores(self) -> None:
        sources = (
            _source("a", trust_score=0.9),
            _source("b", trust_score=0.8),
        )
        forward = _ids(rank_sources(sources, TOPIC))
        reverse = _ids(rank_sources(tuple(reversed(sources)), TOPIC))
        assert forward == reverse == ("a", "b")

    def test_tie_broken_by_registry_order(self) -> None:
        tied = (
            _source("tie_b"),
            _source("tie_a"),
        )
        order = {"tie_a": 0, "tie_b": 1}
        ranked = rank_sources(tied, TOPIC, registry_order=order)
        assert _ids(ranked) == ("tie_a", "tie_b")

    def test_registry_order_overrides_input_position(self) -> None:
        tied = (_source("z_first_in_input"), _source("a_last_in_input"))
        order = {"z_first_in_input": 1, "a_last_in_input": 0}
        ranked = rank_sources(
            tied,
            TOPIC,
            registry_order=order,
        )
        assert _ids(ranked) == ("a_last_in_input", "z_first_in_input")

    def test_unknown_order_key_falls_back_to_input_position(self) -> None:
        ranked = rank_sources(
            (_source("k_unknown"), _source("k_other")),
            TOPIC,
            registry_order={"k_other": 0},
        )
        # Identical scores; k_unknown has no order entry so uses position 0.
        assert _ids(ranked) == ("k_unknown", "k_other")
