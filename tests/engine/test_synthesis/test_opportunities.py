"""Tests for the unified opportunity register (Sprint 6C)."""

from predictron_engine.synthesis.opportunities import aggregate_opportunities
from tests.engine.test_synthesis.conftest import (
    make_assessment,
    make_citation,
    make_features,
    make_observation,
    make_readiness,
    make_relationship,
    make_score,
)


class TestScoreOpportunities:
    def test_high_score_creates_opportunity(self):
        scores = [make_score("market_opportunity", 85.0)]
        result = aggregate_opportunities(make_features(), scores, [], [], None)
        assert any(
            item.label == "strong market opportunity" for item in result
        )

    def test_score_bands_map_to_impact(self):
        strong = make_score("market_opportunity", 82.0)
        notable = make_score("product_strength", 66.0)
        weak = make_score("team_execution", 40.0)
        result = aggregate_opportunities(
            make_features(), [strong, notable, weak], [], [], None
        )
        by_label = {item.label: item for item in result}
        assert by_label["strong market opportunity"].impact.value == "high"
        assert by_label["strong product strength"].impact.value == "moderate"
        assert "strong team execution" not in by_label

    def test_low_scores_produce_nothing(self):
        scores = [
            make_score("market_opportunity", 30.0),
            make_score("product_strength", 45.0),
        ]
        assert aggregate_opportunities(make_features(), scores, [], [], None) == []


class TestReadinessAndRelationships:
    def test_readiness_strengths_captured(self):
        readiness = make_readiness(
            key_strengths=["Traction: ARR of $15M exceeds $10M threshold"]
        )
        result = aggregate_opportunities(make_features(), [], [], [], readiness)
        assert any(
            "readiness:key_strength" in item.sources for item in result
        )

    def test_reinforcing_relationships_captured(self):
        readiness = make_readiness(
            signal_relationships=[
                make_relationship("market_opportunity", "traction_signals"),
                make_relationship("market_opportunity", "team_execution"),
                make_relationship("market_opportunity", "product_strength"),
            ]
        )
        result = aggregate_opportunities(make_features(), [], [], [], readiness)
        reinforcing = next(
            item
            for item in result
            if item.label == "market opportunity reinforcing signals"
        )
        assert reinforcing.impact.value == "high"
        assert len(reinforcing.statements) >= 3


class TestQuantitativeAndEvidence:
    def test_quantitative_thresholds_create_opportunities(self):
        features = make_features(
            arr_usd=12_000_000.0,
            nrr_pct=115.0,
            growth_rate_pct=60.0,
            ltv_cac_ratio=5.5,
        )
        result = aggregate_opportunities(features, [], [], [], None)
        labels = [item.label for item in result]
        assert "strong annual recurring revenue" in labels
        assert "expansion-grade net revenue retention" in labels
        assert "high growth rate" in labels
        assert "strong unit economics" in labels

    def test_below_threshold_metrics_ignored(self):
        features = make_features(
            arr_usd=1_000_000.0,
            nrr_pct=95.0,
            growth_rate_pct=10.0,
            ltv_cac_ratio=2.0,
        )
        assert aggregate_opportunities(features, [], [], [], None) == []

    def test_strong_evidence_observations_included(self):
        observation = make_observation(
            "market_opportunity",
            trust_score=0.9,
            confidence=0.8,
            citations=[make_citation(claim="Strong market source")],
            provenance_document_ids=["doc-7"],
        )
        result = aggregate_opportunities(
            make_features(), [], [], [observation], None
        )
        strong = next(
            item
            for item in result
            if item.label.startswith("strongly evidenced")
        )
        assert strong.statements == [observation.statement]
        assert any(c.claim == "Strong market source" for c in strong.citations)
        assert "doc-7" in strong.provenance_document_ids

    def test_weak_evidence_observations_excluded(self):
        observation = make_observation(
            "market_opportunity",
            trust_score=0.3,
            confidence=0.4,
        )
        result = aggregate_opportunities(
            make_features(), [], [], [observation], None
        )
        assert not any(
            item.label.startswith("strongly evidenced") for item in result
        )


class TestDeduplicationRankingAndCaps:
    def test_duplicate_labels_merge(self):
        scores = [make_score("market_opportunity", 85.0)]
        assessment = make_assessment("market_opportunity", confidence=0.9)
        one = aggregate_opportunities(
            make_features(), scores, [assessment], [], None
        )
        two = aggregate_opportunities(
            make_features(), scores, [assessment], [], None
        )
        assert len(one) == len(two)
        labels = [item.label for item in one]
        assert len(labels) == len(set(labels))

    def test_ranking_by_impact_then_dimension(self):
        from predictron_engine.synthesis.models import severity_order

        scores = [
            make_score("market_opportunity", 85.0),
            make_score("product_strength", 70.0),
        ]
        readiness = make_readiness(
            signal_relationships=[
                make_relationship("team_execution", "founder_quality"),
            ]
        )
        result = aggregate_opportunities(
            make_features(), scores, [], [], readiness
        )
        orders = [severity_order(item.impact) for item in result]
        assert orders == sorted(orders)

    def test_confidence_propagates_from_assessment(self):
        scores = [make_score("market_opportunity", 80.0)]
        assessments = [make_assessment("market_opportunity", confidence=0.55)]
        result = aggregate_opportunities(
            make_features(), scores, assessments, [], None
        )
        entry = next(
            item for item in result if item.label == "strong market opportunity"
        )
        assert entry.confidence == 0.55

    def test_cap_at_eight_and_deterministic(self):
        observations = [
            make_observation(
                f"dimension_{index}",
                statement=f"Evidence {index}",
                trust_score=0.9,
                confidence=0.9,
            )
            for index in range(12)
        ]
        one = aggregate_opportunities(make_features(), [], [], observations, None)
        two = aggregate_opportunities(make_features(), [], [], observations, None)
        assert len(one) <= 8
        assert [o.model_dump() for o in one] == [o.model_dump() for o in two]
