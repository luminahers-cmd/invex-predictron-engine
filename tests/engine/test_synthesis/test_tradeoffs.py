"""Tests for deterministic trade-off analysis (Sprint 6C)."""

from predictron_engine.synthesis.tradeoffs import analyze_trade_offs
from tests.engine.test_synthesis.conftest import (
    make_assessment,
    make_citation,
    make_observation,
    make_readiness,
    make_relationship,
    make_score,
)


class TestGeneration:
    def test_polarized_dimension_produces_trade_off(self):
        scores = [make_score("market_opportunity", 75.0)]
        observations = [
            make_observation("market_opportunity", statement="Large market"),
        ]
        weak_assessment = make_assessment("market_opportunity", confidence=0.2)
        result = analyze_trade_offs(scores, [weak_assessment], observations)
        assert len(result) == 1
        trade_off = result[0]
        assert trade_off.dimension == "market_opportunity"
        assert "75.0/100" in trade_off.strength
        assert trade_off.net_assessment == "strength_dominant"

    def test_weak_dimension_produces_concern_side(self):
        scores = [make_score("product_strength", 35.0)]
        observations = [
            make_observation("product_strength", statement="Thin product"),
        ]
        readiness = make_readiness(
            key_strengths=["Product: strong roadmap signals"],
        )
        result = analyze_trade_offs(scores, [], observations, readiness)
        assert len(result) == 1
        trade_off = result[0]
        assert "35.0/100" in trade_off.concern
        assert trade_off.net_assessment == "concern_dominant"
        assert trade_off.strength_score is None

    def test_mid_score_without_extra_concerns_produces_nothing(self):
        scores = [make_score("team_execution", 52.0)]
        observations = [make_observation("team_execution")]
        assert analyze_trade_offs(scores, [], observations) == []

    def test_low_confidence_assessment_creates_concern_side(self):
        scores = [make_score("market_opportunity", 70.0)]
        weak_assessment = make_assessment("market_opportunity", confidence=0.2)
        result = analyze_trade_offs(scores, [weak_assessment], [])
        assert len(result) == 1
        assert "confidence is only 0.20" in result[0].concern

    def test_conflicting_relationship_creates_concern_side(self):
        scores = [make_score("traction_signals", 65.0)]
        readiness = make_readiness(
            key_strengths=["Traction: strong growth"],
            signal_relationships=[
                make_relationship(
                    "traction_signals",
                    "business_model_viability",
                    "conflicting",
                )
            ],
        )
        result = analyze_trade_offs(scores, [], [], readiness)
        assert len(result) == 1
        assert "conflicting cross-signal" in result[0].concern

    def test_no_trade_off_when_only_one_side_exists(self):
        strong_scores = [make_score("market_opportunity", 80.0)]
        assert analyze_trade_offs(strong_scores, [], []) == []


class TestPropagation:
    def test_confidence_is_min_of_contributors(self):
        observation = make_observation(
            "market_opportunity",
            confidence=0.55,
            citations=[make_citation(claim="Market claim")],
        )
        assessment = make_assessment("market_opportunity", confidence=0.2)
        scores = [make_score("market_opportunity", 72.0)]
        result = analyze_trade_offs(scores, [assessment], [observation])
        assert result[0].confidence == 0.2

    def test_citations_carried_from_supporting_observations(self):
        citation = make_citation(claim="Traction claim")
        observation = make_observation(
            "traction_signals",
            confidence=0.9,
            citations=[citation],
        )
        scores = [make_score("traction_signals", 30.0)]
        readiness = make_readiness(
            key_strengths=["Traction: ARR growing quickly"],
        )
        result = analyze_trade_offs(scores, [], [observation], readiness)
        claims = [c.claim for c in result[0].supporting_citations]
        assert claims == ["Traction claim"]

    def test_supporting_evidence_statements_are_verbatim(self):
        observation = make_observation(
            "product_strength", statement="Roadmap is detailed."
        )
        scores = [make_score("product_strength", 42.0)]
        readiness = make_readiness(
            key_strengths=["Product: loved by early users"],
        )
        result = analyze_trade_offs(scores, [], [observation], readiness)
        assert result[0].supporting_evidence == ["Roadmap is detailed."]


class TestOrderingAndCapping:
    def test_sorted_by_polarization_descending(self):
        scores = [
            make_score("market_opportunity", 90.0),
            make_score("product_strength", 20.0),
        ]
        readiness = make_readiness(
            key_strengths=["Market strength high", "Product strength high"],
            key_concerns=["Market concerns exist", "Product concerns exist"],
        )
        result = analyze_trade_offs(scores, [], [], readiness)
        assert len(result) == 2
        polarizations = []
        for trade_off in result:
            value = trade_off.concern_score or trade_off.strength_score or 0.0
            polarizations.append(abs(value - 50.0))
        assert polarizations == sorted(polarizations, reverse=True)

    def test_capped_at_five(self):
        scores = [make_score(f"dimension_{index}", index * 10 + 5) for index in range(8)]
        readiness = make_readiness(
            key_concerns=[f"Dimension {index} concern" for index in range(8)],
        )
        result = analyze_trade_offs(scores, [], [], readiness)
        assert len(result) <= 5

    def test_output_is_deterministic(self):
        scores = [
            make_score("market_opportunity", 75.0),
            make_score("product_strength", 40.0),
        ]
        assessments = [
            make_assessment("product_strength", confidence=0.2),
        ]
        one = analyze_trade_offs(scores, assessments, [])
        two = analyze_trade_offs(scores, assessments, [])
        assert [t.model_dump() for t in one] == [t.model_dump() for t in two]
