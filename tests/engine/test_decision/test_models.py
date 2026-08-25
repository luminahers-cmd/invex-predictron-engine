"""Tests for decision calibration value objects."""

import pytest
from pydantic import ValidationError

from predictron_engine.decision.models import (
    CalibrationSummary,
    ConfidenceBreakdown,
    ConfidenceFactor,
    ConfidenceLevel,
    DecisionConfidence,
    UncertaintyBreakdown,
)


def make_factor(name: str = "evidence_trust") -> ConfidenceFactor:
    return ConfidenceFactor(name=name, value=0.5, weight=0.25, contribution=0.125)


class TestConfidenceLevel:
    def test_values(self):
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.VERY_LOW.value == "very_low"

    def test_all_five_bands_exist(self):
        assert len(ConfidenceLevel) == 5


class TestConfidenceFactor:
    def test_construction(self):
        factor = make_factor()
        assert factor.value == 0.5
        assert factor.weight == 0.25
        assert factor.contribution == 0.125

    @pytest.mark.parametrize("field", ["value", "weight", "contribution"])
    def test_bounds_enforced(self, field):
        data = {
            "name": "x",
            "value": 0.5,
            "weight": 0.5,
            "contribution": 0.25,
        }
        data[field] = 1.5
        with pytest.raises(ValidationError):
            ConfidenceFactor(**data)


class TestBreakdowns:
    def test_confidence_breakdown_value_of(self):
        breakdown = ConfidenceBreakdown(
            factors=[make_factor("evidence_trust"), make_factor("diversity")],
            composite=0.5,
        )
        assert breakdown.value_of("evidence_trust") == 0.5
        assert breakdown.value_of("missing") is None

    def test_uncertainty_breakdown_value_of(self):
        breakdown = UncertaintyBreakdown(factors=[], composite=0.4)
        assert breakdown.value_of("anything") is None

    def test_composite_bounds(self):
        with pytest.raises(ValidationError):
            ConfidenceBreakdown(factors=[], composite=2.0)


class TestDecisionConfidence:
    def test_recommended_action_property(self):
        dc = DecisionConfidence(
            confidence=0.9,
            level=ConfidenceLevel.VERY_HIGH,
            uncertainty_score=0.1,
            breakdown=ConfidenceBreakdown(factors=[], composite=0.9),
            uncertainty_breakdown=UncertaintyBreakdown(factors=[], composite=0.1),
        )
        assert dc.recommended_action == "Proceed with standard due diligence"

    @pytest.mark.parametrize(
        ("level", "action_fragment"),
        [
            (ConfidenceLevel.HIGH, "due diligence"),
            (ConfidenceLevel.LOW, "Manual review"),
            (ConfidenceLevel.VERY_LOW, "Do not act"),
        ],
    )
    def test_action_varies_by_level(self, level, action_fragment):
        dc = DecisionConfidence(
            confidence=0.5,
            level=level,
            uncertainty_score=0.5,
            breakdown=ConfidenceBreakdown(factors=[], composite=0.5),
            uncertainty_breakdown=UncertaintyBreakdown(factors=[], composite=0.5),
        )
        assert action_fragment in dc.recommended_action


class TestCalibrationSummary:
    def test_digest_fields(self):
        summary = CalibrationSummary(
            confidence=0.7,
            level=ConfidenceLevel.HIGH,
            uncertainty_score=0.3,
            recommended_action="Proceed",
            strong_factor_count=3,
            weak_factor_count=1,
            top_supporting_factors=["a", "b", "c"],
            top_weakening_factors=["d"],
            observation_count=10,
            assessed_dimension_count=6,
            evidence_document_count=4,
        )
        assert summary.confidence == 0.7
        assert summary.level == ConfidenceLevel.HIGH
        assert summary.strong_factor_count == 3
        assert summary.observation_count == 10

    def test_counts_reject_negatives(self):
        with pytest.raises(ValidationError):
            CalibrationSummary(
                confidence=0.5,
                level=ConfidenceLevel.MEDIUM,
                uncertainty_score=0.5,
                recommended_action="x",
                observation_count=-1,
            )
