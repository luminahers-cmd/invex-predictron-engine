"""Tests for the DefaultDecisionEngine (Sprint 13).

Covers:
  - Decision category classification
  - Conviction level classification
  - Rationale generation
  - Deterministic behavior
  - Edge cases (empty inputs, boundary values)
  - Factor computation
  - Margin calculation
  - Integration with pipeline
"""

from __future__ import annotations

import pytest

from predictron_engine.decision.decision_engine import (
    _CONVICTION_THRESHOLDS,
    _DECISION_THRESHOLDS,
    _READINESS_WEIGHT,
    DefaultDecisionEngine,
)
from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.knowledge.concepts import AnalysisDimension
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    ConfidenceAssessment,
    ConvictionLevel,
    DecisionCategory,
    DimensionAssessment,
    InvestmentDecision,
    Observation,
    ScoreResult,
    SignalRelationship,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scores(**overrides: float) -> list[ScoreResult]:
    """Create a list of ScoreResult for all dimensions with given overrides."""
    defaults = {
        AnalysisDimension.MARKET_OPPORTUNITY.value: 60.0,
        AnalysisDimension.PRODUCT_STRENGTH.value: 55.0,
        AnalysisDimension.FOUNDER_QUALITY.value: 58.0,
        AnalysisDimension.TRACTION_SIGNALS.value: 50.0,
        AnalysisDimension.BUSINESS_MODEL_VIABILITY.value: 52.0,
        AnalysisDimension.COMPETITIVE_POSITION.value: 48.0,
        AnalysisDimension.TEAM_EXECUTION.value: 53.0,
    }
    defaults.update(overrides)
    return [
        ScoreResult(dimension=dim, score=score, rationale=f"Score for {dim}")
        for dim, score in defaults.items()
    ]


def _make_confidence(
    avg: float = 0.5,
) -> list[ConfidenceAssessment]:
    """Create ConfidenceAssessment for all dimensions with given avg."""
    dims = [d.value for d in AnalysisDimension]
    return [
        ConfidenceAssessment(
            dimension=d, confidence=avg, factors=[], data_completeness=0.5,
        )
        for d in dims
    ]


def _make_features(**overrides) -> ExtractedFeatures:
    """Create ExtractedFeatures with reasonable defaults and overrides."""
    defaults = dict(
        industry="enterprise_saas",
        business_model="saas",
        geography="north_america",
        has_revenue=True,
        has_pitch_deck=True,
        founder_profile_count=2,
        data_completeness=0.75,
        description_length=300,
    )
    defaults.update(overrides)
    return ExtractedFeatures(**defaults)


def _make_observations(
    count: int = 3, avg_confidence: float = 0.6,
) -> list[Observation]:
    """Create Observations across multiple dimensions."""
    dims = [d.value for d in AnalysisDimension]
    observations: list[Observation] = []
    for i in range(count):
        observations.append(
            Observation(
                dimension=dims[i % len(dims)],
                category="market_context" if i % 2 == 0 else "team_assessment",
                statement=f"Observation {i}",
                evidence=[f"evidence_{i}"],
                confidence=avg_confidence,
                importance=0.6,
                source_rule="TestRule",
            )
        )
    return observations


def _make_assessments() -> list[DimensionAssessment]:
    """Create DimensionAssessment for all dimensions."""
    dims = [d.value for d in AnalysisDimension]
    return [
        DimensionAssessment(
            dimension=d,
            summary=f"Summary for {d}",
            rationale=f"Rationale for {d}",
            confidence=0.65,
            score=58.0,
        )
        for d in dims
    ]


def _make_relationships(
    reinforcing: int = 2, conflicting: int = 1,
) -> list[SignalRelationship]:
    """Create SignalRelationships."""
    rels: list[SignalRelationship] = []
    for i in range(reinforcing):
        rels.append(
            SignalRelationship(
                source_dimension="market_opportunity",
                target_dimension="product_strength",
                relationship_type="reinforcing",
                description=f"Reinforcing {i}",
                confidence=0.7,
            )
        )
    for i in range(conflicting):
        rels.append(
            SignalRelationship(
                source_dimension="founder_quality",
                target_dimension="traction_signals",
                relationship_type="conflicting",
                description=f"Conflict {i}",
                confidence=0.6,
            )
        )
    return rels


# ---------------------------------------------------------------------------
# Tests: Decision category classification
# ---------------------------------------------------------------------------


class TestDecisionClassification:
    """Tests for decision category threshold mapping."""

    def test_classify_decision_direct(self) -> None:
        """Test _classify_decision directly with exact thresholds."""
        engine = DefaultDecisionEngine()
        assert engine._classify_decision(75.0) == DecisionCategory.STRONG_INVEST
        assert engine._classify_decision(60.0) == DecisionCategory.INVEST
        assert engine._classify_decision(45.0) == DecisionCategory.WATCH
        assert engine._classify_decision(30.0) == DecisionCategory.INVESTIGATE_FURTHER
        assert engine._classify_decision(0.0) == DecisionCategory.PASS
        assert engine._classify_decision(99.0) == DecisionCategory.STRONG_INVEST
        assert engine._classify_decision(29.9) == DecisionCategory.PASS

    def test_high_composite_yields_strong_invest(self) -> None:
        engine = DefaultDecisionEngine()
        scores = _make_scores(
            **{d.value: 90.0 for d in AnalysisDimension}
        )
        confidence = _make_confidence(avg=0.9)
        features = _make_features(data_completeness=0.95)
        observations = _make_observations(count=5, avg_confidence=0.85)
        assessments = _make_assessments()
        for a in assessments:
            a.score = 90.0
        relationships = _make_relationships(reinforcing=4, conflicting=0)

        decision = engine.decide(
            features, observations, scores, confidence,
            assessments, relationships,
        )

        assert decision.composite_score >= 75.0
        assert decision.category == DecisionCategory.STRONG_INVEST

    def test_mid_composite_yields_invest_or_watch(self) -> None:
        engine = DefaultDecisionEngine()
        scores = _make_scores(
            **{d.value: 65.0 for d in AnalysisDimension}
        )
        confidence = _make_confidence(avg=0.65)
        features = _make_features(data_completeness=0.8)
        observations = _make_observations(count=5, avg_confidence=0.65)
        assessments = _make_assessments()
        for a in assessments:
            a.score = 65.0
        relationships = _make_relationships(reinforcing=3, conflicting=0)

        decision = engine.decide(
            features, observations, scores, confidence,
            assessments, relationships,
        )

        assert decision.category in (
            DecisionCategory.INVEST,
            DecisionCategory.WATCH,
        )

    def test_low_composite_yields_investigate_or_pass(self) -> None:
        engine = DefaultDecisionEngine()
        scores = _make_scores(
            **{d.value: 30.0 for d in AnalysisDimension}
        )
        confidence = _make_confidence(avg=0.25)
        features = _make_features(data_completeness=0.2)
        observations = _make_observations(count=1, avg_confidence=0.3)

        decision = engine.decide(features, observations, scores, confidence)

        assert decision.category in (
            DecisionCategory.INVESTIGATE_FURTHER,
            DecisionCategory.PASS,
        )

    def test_very_low_composite_yields_pass(self) -> None:
        engine = DefaultDecisionEngine()
        scores = _make_scores(
            **{d.value: 15.0 for d in AnalysisDimension}
        )
        confidence = _make_confidence(avg=0.15)
        features = _make_features(data_completeness=0.1)

        decision = engine.decide(features, [], scores, confidence)

        assert decision.category == DecisionCategory.PASS

    def test_all_threshold_boundaries(self) -> None:
        """Test that _classify_decision correctly maps each boundary."""
        engine = DefaultDecisionEngine()
        for threshold, expected_cat in _DECISION_THRESHOLDS:
            result = engine._classify_decision(threshold)
            assert result == expected_cat, (
                f"Threshold {threshold} should map to {expected_cat.value}"
            )


# ---------------------------------------------------------------------------
# Tests: Conviction classification
# ---------------------------------------------------------------------------


class TestConvictionClassification:
    """Tests for conviction level mapping."""

    def test_very_high_conviction(self) -> None:
        engine = DefaultDecisionEngine()
        scores = _make_scores(
            **{d.value: 90.0 for d in AnalysisDimension}
        )
        confidence = _make_confidence(avg=0.95)
        features = _make_features(data_completeness=0.95)
        relationships = _make_relationships(reinforcing=5, conflicting=0)

        decision = engine.decide(
            features, _make_observations(count=8, avg_confidence=0.9),
            scores, confidence,
            _make_assessments(), relationships,
        )

        assert decision.conviction == ConvictionLevel.VERY_HIGH

    def test_very_low_conviction(self) -> None:
        engine = DefaultDecisionEngine()
        scores = _make_scores(
            **{d.value: 10.0 for d in AnalysisDimension}
        )
        confidence = _make_confidence(avg=0.1)
        features = _make_features(data_completeness=0.1)

        decision = engine.decide(features, [], scores, confidence)

        assert decision.conviction == ConvictionLevel.VERY_LOW

    def test_conviction_independent_of_decision(self) -> None:
        """Conviction can differ from decision category."""
        engine = DefaultDecisionEngine()
        # High scores but low confidence and low data completeness
        scores = _make_scores(
            **{d.value: 70.0 for d in AnalysisDimension}
        )
        confidence = _make_confidence(avg=0.2)
        features = _make_features(data_completeness=0.2)

        decision = engine.decide(features, [], scores, confidence)

        # High scores -> decent decision category, but low confidence
        # and data completeness -> low conviction
        assert decision.conviction in (
            ConvictionLevel.LOW,
            ConvictionLevel.VERY_LOW,
            ConvictionLevel.MODERATE,
        )

    def test_conviction_thresholds_are_valid(self) -> None:
        """Verify threshold ordering."""
        thresholds = [t for t, _ in _CONVICTION_THRESHOLDS]
        assert thresholds == sorted(thresholds, reverse=True)


# ---------------------------------------------------------------------------
# Tests: Deterministic behavior
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify identical inputs produce identical outputs."""

    def test_same_inputs_same_decision(self) -> None:
        engine = DefaultDecisionEngine()
        features = _make_features()
        observations = _make_observations()
        scores = _make_scores()
        confidence = _make_confidence()
        assessments = _make_assessments()
        relationships = _make_relationships()

        d1 = engine.decide(
            features, observations, scores, confidence,
            assessments, relationships,
        )
        d2 = engine.decide(
            features, observations, scores, confidence,
            assessments, relationships,
        )

        assert d1.category == d2.category
        assert d1.conviction == d2.conviction
        assert d1.composite_score == d2.composite_score
        assert d1.rationale.why_not_higher == d2.rationale.why_not_higher

    def test_different_inputs_different_decision(self) -> None:
        engine = DefaultDecisionEngine()
        features = _make_features()

        high_scores = _make_scores(**{d.value: 90.0 for d in AnalysisDimension})
        low_scores = _make_scores(**{d.value: 20.0 for d in AnalysisDimension})
        confidence = _make_confidence(avg=0.7)

        d_high = engine.decide(features, _make_observations(), high_scores, confidence)
        d_low = engine.decide(features, _make_observations(), low_scores, confidence)

        assert d_high.category != d_low.category


# ---------------------------------------------------------------------------
# Tests: Composite score and factors
# ---------------------------------------------------------------------------


class TestCompositeScore:
    """Tests for composite score computation."""

    def test_composite_score_in_range(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(),
            _make_observations(),
            _make_scores(),
            _make_confidence(),
        )

        assert 0.0 <= decision.composite_score <= 100.0

    def test_decision_factors_populated(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(),
            _make_observations(),
            _make_scores(),
            _make_confidence(),
        )

        assert "score_contribution" in decision.decision_factors
        assert "readiness_contribution" in decision.decision_factors
        assert "confidence_contribution" in decision.decision_factors
        assert "evidence_quality_contribution" in decision.decision_factors
        assert "cross_signal_contribution" in decision.decision_factors
        assert "data_quality_modifier" in decision.decision_factors
        assert "risk_modifier" in decision.decision_factors
        assert "conviction_composite" in decision.decision_factors

    def test_data_quality_modifier_range(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(data_completeness=0.5),
            _make_observations(),
            _make_scores(),
            _make_confidence(),
        )

        assert 0.5 <= decision.data_quality_modifier <= 1.0

    def test_risk_modifier_range(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(),
            _make_observations(),
            _make_scores(),
            _make_confidence(),
        )

        assert 0.5 <= decision.risk_modifier <= 1.0

    def test_low_data_completeness_reduces_composite(self) -> None:
        engine = DefaultDecisionEngine()
        observations = _make_observations()
        scores = _make_scores()
        confidence = _make_confidence()

        high_dc = engine.decide(
            _make_features(data_completeness=0.9),
            observations, scores, confidence,
        )
        low_dc = engine.decide(
            _make_features(data_completeness=0.1),
            observations, scores, confidence,
        )

        assert high_dc.composite_score >= low_dc.composite_score


# ---------------------------------------------------------------------------
# Tests: Margin to next category
# ---------------------------------------------------------------------------


class TestMarginCalculation:
    """Tests for margin_to_next_category."""

    def test_margin_is_non_negative(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(),
            _make_observations(),
            _make_scores(),
            _make_confidence(),
        )

        assert decision.margin_to_next_category >= 0.0

    def test_margin_at_strong_invest(self) -> None:
        """At the highest category, margin is distance to 100."""
        engine = DefaultDecisionEngine()
        scores = _make_scores(**{d.value: 95.0 for d in AnalysisDimension})
        features = _make_features(data_completeness=0.95)
        confidence = _make_confidence(avg=0.9)
        observations = _make_observations(count=5, avg_confidence=0.8)

        decision = engine.decide(features, observations, scores, confidence)

        if decision.category == DecisionCategory.STRONG_INVEST:
            assert decision.next_category_threshold == 100.0
            assert decision.margin_to_next_category >= 0.0


# ---------------------------------------------------------------------------
# Tests: Rationale
# ---------------------------------------------------------------------------


class TestRationale:
    """Tests for decision rationale generation."""

    def test_rationale_has_all_fields(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(),
            _make_observations(),
            _make_scores(),
            _make_confidence(),
        )

        assert isinstance(decision.rationale.primary_reasons_for, list)
        assert isinstance(decision.rationale.primary_reasons_against, list)
        assert isinstance(decision.rationale.highest_impact_positive, list)
        assert isinstance(decision.rationale.highest_impact_negative, list)
        assert isinstance(decision.rationale.missing_information, list)
        assert isinstance(decision.rationale.confidence_explanation, str)
        assert isinstance(decision.rationale.why_not_higher, str)
        assert isinstance(decision.rationale.key_evidence_summary, str)
        assert isinstance(
            decision.rationale.information_that_could_change_decision, list
        )

    def test_rationale_why_not_higher_at_strong_invest(self) -> None:
        engine = DefaultDecisionEngine()
        scores = _make_scores(**{d.value: 90.0 for d in AnalysisDimension})
        features = _make_features(data_completeness=0.95)
        confidence = _make_confidence(avg=0.9)
        observations = _make_observations(count=5, avg_confidence=0.9)

        decision = engine.decide(features, observations, scores, confidence)

        if decision.category == DecisionCategory.STRONG_INVEST:
            assert "highest" in decision.rationale.why_not_higher.lower()

    def test_rationale_identifies_missing_info(self) -> None:
        engine = DefaultDecisionEngine()
        features = _make_features(
            industry=None,
            business_model=None,
            funding_stage=None,
            founder_profile_count=0,
            has_pitch_deck=False,
        )

        decision = engine.decide(
            features, _make_observations(), _make_scores(), _make_confidence(),
        )

        assert len(decision.rationale.missing_information) > 0

    def test_high_data_completeness_confidence_explanation(self) -> None:
        engine = DefaultDecisionEngine()
        confidence = _make_confidence(avg=0.8)
        features = _make_features(data_completeness=0.8)

        decision = engine.decide(
            features, _make_observations(), _make_scores(), confidence,
        )

        assert "high confidence" in decision.rationale.confidence_explanation.lower()

    def test_low_data_completeness_confidence_explanation(self) -> None:
        engine = DefaultDecisionEngine()
        confidence = _make_confidence(avg=0.2)
        features = _make_features(data_completeness=0.15)

        decision = engine.decide(
            features, _make_observations(count=0), _make_scores(), confidence,
        )

        assert "low" in decision.rationale.confidence_explanation.lower()

    def test_positive_signals_from_high_scores(self) -> None:
        engine = DefaultDecisionEngine()
        scores = _make_scores(
            **{d.value: 80.0 for d in AnalysisDimension}
        )
        features = _make_features()
        confidence = _make_confidence(avg=0.7)

        decision = engine.decide(
            features, _make_observations(), scores, confidence,
        )

        assert len(decision.rationale.highest_impact_positive) > 0

    def test_negative_signals_from_low_scores(self) -> None:
        engine = DefaultDecisionEngine()
        scores = _make_scores(
            **{d.value: 30.0 for d in AnalysisDimension}
        )
        features = _make_features()
        confidence = _make_confidence(avg=0.3)

        decision = engine.decide(
            features, _make_observations(), scores, confidence,
        )

        assert len(decision.rationale.highest_impact_negative) > 0


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_scores(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(), _make_observations(), [], _make_confidence(),
        )

        assert isinstance(decision, InvestmentDecision)
        assert 0.0 <= decision.composite_score <= 100.0

    def test_empty_confidence(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(), _make_observations(), _make_scores(), [],
        )

        assert isinstance(decision, InvestmentDecision)
        assert decision.rationale.confidence_explanation != ""

    def test_empty_observations(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(), [], _make_scores(), _make_confidence(),
        )

        assert isinstance(decision, InvestmentDecision)

    def test_empty_assessments(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(),
            _make_observations(),
            _make_scores(),
            _make_confidence(),
            assessments=None,
        )

        assert isinstance(decision, InvestmentDecision)

    def test_empty_relationships(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(),
            _make_observations(),
            _make_scores(),
            _make_confidence(),
            signal_relationships=None,
        )

        assert isinstance(decision, InvestmentDecision)

    def test_all_zeros(self) -> None:
        engine = DefaultDecisionEngine()
        features = ExtractedFeatures()
        scores = _make_scores(**{d.value: 0.0 for d in AnalysisDimension})
        confidence = _make_confidence(avg=0.0)

        decision = engine.decide(features, [], scores, confidence)

        assert decision.category == DecisionCategory.PASS
        assert decision.conviction == ConvictionLevel.VERY_LOW

    def test_all_100s(self) -> None:
        engine = DefaultDecisionEngine()
        features = _make_features(data_completeness=1.0)
        scores = _make_scores(**{d.value: 100.0 for d in AnalysisDimension})
        confidence = _make_confidence(avg=1.0)
        observations = _make_observations(count=10, avg_confidence=1.0)
        assessments = _make_assessments()
        relationships = _make_relationships(reinforcing=5, conflicting=0)

        decision = engine.decide(
            features, observations, scores, confidence,
            assessments, relationships,
        )

        assert decision.category == DecisionCategory.STRONG_INVEST
        assert decision.composite_score >= 75.0

    def test_conflicting_relationships_reduce_cross_signal(self) -> None:
        engine = DefaultDecisionEngine()
        features = _make_features()
        scores = _make_scores()
        confidence = _make_confidence()
        observations = _make_observations()

        all_reinforcing = _make_relationships(reinforcing=5, conflicting=0)
        all_conflicting = _make_relationships(reinforcing=0, conflicting=5)

        d_reinforcing = engine.decide(
            features, observations, scores, confidence,
            _make_assessments(), all_reinforcing,
        )
        d_conflicting = engine.decide(
            features, observations, scores, confidence,
            _make_assessments(), all_conflicting,
        )

        assert (
            d_reinforcing.decision_factors["cross_signal_contribution"]
            >= d_conflicting.decision_factors["cross_signal_contribution"]
        )

    def test_many_risk_signals_reduce_composite(self) -> None:
        engine = DefaultDecisionEngine()
        scores = _make_scores()
        confidence = _make_confidence()
        observations = _make_observations()

        features_clean = _make_features()
        features_risky = _make_features(
            market_risk=["risk1", "risk2", "risk3", "risk4", "risk5"],
            founder_risk=["risk1", "risk2", "risk3", "risk4", "risk5"],
            execution_risk=["risk1", "risk2", "risk3"],
            product_risk=["risk1", "risk2"],
        )

        d_clean = engine.decide(
            features_clean, observations, scores, confidence,
        )
        d_risky = engine.decide(
            features_risky, observations, scores, confidence,
        )

        assert d_clean.risk_modifier >= d_risky.risk_modifier
        assert d_clean.composite_score >= d_risky.composite_score


# ---------------------------------------------------------------------------
# Tests: Factor computation
# ---------------------------------------------------------------------------


class TestFactorComputation:
    """Tests for individual factor computation methods."""

    def test_score_factor_averages(self) -> None:
        engine = DefaultDecisionEngine()
        scores = [
            ScoreResult(dimension="a", score=60.0),
            ScoreResult(dimension="b", score=80.0),
        ]
        assert engine._compute_score_factor(scores) == 70.0

    def test_score_factor_empty(self) -> None:
        """Empty input yields no evidence (0), not a fabricated neutral 50.

        H3/H2 — the decision's score factor is the same canonical mean used
        by the report ``overall_score``, which also reports 0.0 for empty.
        """
        engine = DefaultDecisionEngine()
        assert engine._compute_score_factor([]) == 0.0

    def test_confidence_factor_averages(self) -> None:
        engine = DefaultDecisionEngine()
        confidence = [
            ConfidenceAssessment(dimension="a", confidence=0.6),
            ConfidenceAssessment(dimension="b", confidence=0.8),
        ]
        assert engine._compute_confidence_factor(confidence) == 0.7

    def test_confidence_factor_empty(self) -> None:
        engine = DefaultDecisionEngine()
        assert engine._compute_confidence_factor([]) == 0.0

    def test_evidence_quality_empty(self) -> None:
        engine = DefaultDecisionEngine()
        assert engine._compute_evidence_quality_factor([]) == 0.0

    def test_evidence_quality_high(self) -> None:
        engine = DefaultDecisionEngine()
        observations = [
            Observation(
                dimension="d", category="cat_a", statement="s",
                confidence=0.9, importance=0.9, source_rule="r",
            ),
            Observation(
                dimension="d", category="cat_b", statement="s",
                confidence=0.85, importance=0.8, source_rule="r",
            ),
            Observation(
                dimension="d", category="cat_c", statement="s",
                confidence=0.8, importance=0.75, source_rule="r",
            ),
        ]
        quality = engine._compute_evidence_quality_factor(observations)
        assert quality > 0.5

    def test_cross_signal_neutral_when_empty(self) -> None:
        engine = DefaultDecisionEngine()
        assert engine._compute_cross_signal_factor([]) == 0.5

    def test_cross_signal_all_reinforcing(self) -> None:
        engine = DefaultDecisionEngine()
        rels = [
            SignalRelationship(
                source_dimension="a", target_dimension="b",
                relationship_type="reinforcing", description="r",
            ),
            SignalRelationship(
                source_dimension="c", target_dimension="d",
                relationship_type="reinforcing", description="r",
            ),
        ]
        assert engine._compute_cross_signal_factor(rels) == 1.0

    def test_cross_signal_all_conflicting(self) -> None:
        engine = DefaultDecisionEngine()
        rels = [
            SignalRelationship(
                source_dimension="a", target_dimension="b",
                relationship_type="conflicting", description="r",
            ),
        ]
        assert engine._compute_cross_signal_factor(rels) == 0.0

    def test_risk_modifier_no_risk(self) -> None:
        engine = DefaultDecisionEngine()
        features = _make_features()
        assert engine._compute_risk_modifier(features) == 1.0

    def test_risk_modifier_many_risks(self) -> None:
        engine = DefaultDecisionEngine()
        features = _make_features(
            market_risk=["r1", "r2", "r3", "r4", "r5"],
            founder_risk=["r1", "r2", "r3", "r4", "r5"],
            execution_risk=["r1", "r2", "r3", "r4", "r5"],
            product_risk=["r1", "r2", "r3", "r4", "r5"],
        )
        modifier = engine._compute_risk_modifier(features)
        assert modifier < 1.0
        assert modifier >= 0.5

    def test_data_quality_modifier_high(self) -> None:
        engine = DefaultDecisionEngine()
        features = _make_features(data_completeness=0.9)
        assert engine._compute_data_quality_modifier(features) == 1.0

    def test_data_quality_modifier_low(self) -> None:
        engine = DefaultDecisionEngine()
        features = _make_features(data_completeness=0.1)
        modifier = engine._compute_data_quality_modifier(features)
        assert 0.5 <= modifier < 1.0


# ---------------------------------------------------------------------------
# Tests: InvestmentDecision model validation
# ---------------------------------------------------------------------------


class TestInvestmentDecisionModel:
    """Tests for the InvestmentDecision Pydantic model."""

    def test_model_creation(self) -> None:
        decision = InvestmentDecision(
            category=DecisionCategory.INVEST,
            conviction=ConvictionLevel.MODERATE,
            composite_score=62.5,
        )

        assert decision.category == DecisionCategory.INVEST
        assert decision.conviction == ConvictionLevel.MODERATE
        assert decision.composite_score == 62.5

    def test_model_score_bounds(self) -> None:
        with pytest.raises(Exception):
            InvestmentDecision(
                category=DecisionCategory.PASS,
                conviction=ConvictionLevel.VERY_LOW,
                composite_score=-1.0,
            )

        with pytest.raises(Exception):
            InvestmentDecision(
                category=DecisionCategory.PASS,
                conviction=ConvictionLevel.VERY_LOW,
                composite_score=101.0,
            )

    def test_model_defaults(self) -> None:
        decision = InvestmentDecision(
            category=DecisionCategory.WATCH,
            conviction=ConvictionLevel.MODERATE,
            composite_score=50.0,
        )

        assert decision.rationale is not None
        assert decision.decision_factors == {}
        assert decision.data_quality_modifier == 1.0
        assert decision.risk_modifier == 1.0


# ---------------------------------------------------------------------------
# Tests: DecisionCategory and ConvictionLevel enums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for enum definitions."""

    def test_decision_category_values(self) -> None:
        assert DecisionCategory.STRONG_INVEST.value == "strong_invest"
        assert DecisionCategory.INVEST.value == "invest"
        assert DecisionCategory.WATCH.value == "watch"
        assert DecisionCategory.INVESTIGATE_FURTHER.value == "investigate_further"
        assert DecisionCategory.PASS.value == "pass"

    def test_conviction_level_values(self) -> None:
        assert ConvictionLevel.VERY_HIGH.value == "very_high"
        assert ConvictionLevel.HIGH.value == "high"
        assert ConvictionLevel.MODERATE.value == "moderate"
        assert ConvictionLevel.LOW.value == "low"
        assert ConvictionLevel.VERY_LOW.value == "very_low"

    def test_decision_category_ordering(self) -> None:
        """Categories from strongest to weakest."""
        categories = [
            DecisionCategory.STRONG_INVEST,
            DecisionCategory.INVEST,
            DecisionCategory.WATCH,
            DecisionCategory.INVESTIGATE_FURTHER,
            DecisionCategory.PASS,
        ]
        assert len(categories) == 5

    def test_conviction_level_ordering(self) -> None:
        """Levels from highest to lowest."""
        levels = [
            ConvictionLevel.VERY_HIGH,
            ConvictionLevel.HIGH,
            ConvictionLevel.MODERATE,
            ConvictionLevel.LOW,
            ConvictionLevel.VERY_LOW,
        ]
        assert len(levels) == 5


# ---------------------------------------------------------------------------
# Tests: Sprint P8A — readiness & evidence-quality wiring (C1, M2)
# ---------------------------------------------------------------------------


def _make_trusted_bundle() -> EvidenceBundle:
    """A bundle with highly trusted, diverse, quality documents (M2)."""
    from datetime import UTC, datetime

    from predictron_engine.evidence.models import (
        DocumentMetadata,
        DocumentStatus,
        EvidenceDocument,
        EvidenceSource,
        PageType,
    )
    from predictron_engine.evidence.provenance import TrustScore

    fetched_at = datetime(2024, 1, 1, tzinfo=UTC)

    def _doc(idx: int, provider: str, trust: float) -> EvidenceDocument:
        return EvidenceDocument(
            id=f"doc-{idx}",
            original_url=f"https://example.com/{idx}",
            url=f"https://example.com/{idx}",
            page_type=PageType.HOMEPAGE,
            status=DocumentStatus.SUCCESS,
            fetched_at=fetched_at,
            response_time_ms=10,
            http_status=200,
            metadata=DocumentMetadata(
                source_provider=provider,
                trust_score=TrustScore(overall=trust),
                authority_score=trust,
                quality_score=0.9,
            ),
        )

    return EvidenceBundle(
        startup_name="TestCo",
        documents=[
            _doc(1, "website", 0.9),
            _doc(2, "search", 0.85),
            _doc(3, "crunchbase", 0.95),
        ],
        sources=[
            EvidenceSource(
                original_url="https://example.com/1",
                page_type=PageType.HOMEPAGE,
                fetched_at=fetched_at,
                success=True,
            )
        ],
    )


class TestReadinessWiring:
    """C1 — the decision must consume the canonical readiness score."""

    def test_readiness_score_used_when_provided(self) -> None:
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(),
            _make_observations(),
            _make_scores(),
            _make_confidence(),
            readiness_score=85.0,
        )
        expected = 85.0 * _READINESS_WEIGHT
        assert decision.decision_factors["readiness_contribution"] == round(
            expected, 4
        )

    def test_readiness_defaults_to_assessment_average(self) -> None:
        """Backward compatible: without a readiness score, per-assessment
        scores (where populated) drive the readiness factor."""
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(),
            _make_observations(),
            _make_scores(),
            _make_confidence(),
            _make_assessments(),
        )
        # _make_assessments() sets every score to 58.0
        expected = 58.0 * _READINESS_WEIGHT
        assert decision.decision_factors["readiness_contribution"] == round(
            expected, 4
        )

    def test_readiness_bounds(self) -> None:
        """Scoring mathematics are unchanged — only the source is wired."""
        engine = DefaultDecisionEngine()
        assert engine._resolve_readiness_factor(120.0, []) == 100.0
        assert engine._resolve_readiness_factor(-5.0, []) == 0.0
        assert engine._resolve_readiness_factor(None, []) == 50.0


class TestEvidenceQualityWiring:
    """M2 — genuine evidence quality comes from bundle provenance."""

    def test_evidence_quality_uses_bundle_trust(self) -> None:
        engine = DefaultDecisionEngine()
        bundle = _make_trusted_bundle()
        quality = engine._compute_evidence_quality_factor(
            [], evidence_bundle=bundle,
        )
        # Bundle is trusted -> genuine quality well above a raw empty proxy.
        assert quality >= 0.8

    def test_evidence_quality_falls_back_without_bundle(self) -> None:
        """Direct callers without a bundle retain the observation proxy."""
        engine = DefaultDecisionEngine()
        observations = [
            Observation(
                dimension="d", category="cat_a", statement="s",
                confidence=0.9, importance=0.9, source_rule="r",
            ),
        ]
        assert engine._compute_evidence_quality_factor(observations) > 0.5

    def test_decide_accepts_evidence_bundle(self) -> None:
        """The pipeline-level bundle is threaded into the decision without
        changing thresholds or public category semantics."""
        engine = DefaultDecisionEngine()
        decision = engine.decide(
            _make_features(),
            _make_observations(),
            _make_scores(),
            _make_confidence(),
            evidence_bundle=_make_trusted_bundle(),
        )
        factors = decision.decision_factors
        assert "evidence_quality_contribution" in factors
        assert 0.0 <= decision.composite_score <= 100.0
