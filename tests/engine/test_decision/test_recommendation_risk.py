"""Tests for recommendation risk exposure (Sprint 6B)."""

import pytest

from predictron_engine.decision.calibration import (
    apply_recommendation_risk,
    compute_decision_confidence,
)
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Recommendation
from tests.engine.test_decision.conftest import (
    make_assessment,
    make_bundle,
    make_document,
    make_observation,
)


def make_recommendation(confidence: float) -> Recommendation:
    return Recommendation(
        category="due_diligence",
        action="Verify traction metrics",
        confidence=confidence,
    )


def strong_calibration():
    return compute_decision_confidence(
        bundle=make_bundle(make_document(trust=0.9)),
        observations=[make_observation()],
        assessments=[make_assessment()],
        features=ExtractedFeatures(data_completeness=0.95),
    )


class TestApplyRecommendationRisk:
    def test_all_fields_populated(self):
        annotated = apply_recommendation_risk(
            [make_recommendation(0.5)], strong_calibration(),
        )
        rec = annotated[0]
        assert rec.expected_confidence is not None
        assert 0.0 <= rec.expected_confidence <= 1.0
        assert rec.expected_uncertainty is not None
        assert 0.0 <= rec.expected_uncertainty <= 1.0
        assert rec.recommended_action

    def test_inputs_are_not_mutated(self):
        recs = [make_recommendation(0.5)]
        apply_recommendation_risk(recs, strong_calibration())
        assert recs[0].expected_confidence is None
        assert recs[0].recommended_action == ""

    def test_deterministic_outputs(self):
        recs = [make_recommendation(0.4), make_recommendation(0.8)]
        first = apply_recommendation_risk(recs, strong_calibration())
        second = apply_recommendation_risk(recs, strong_calibration())
        assert [r.model_dump() for r in first] == [
            r.model_dump() for r in second
        ]

    def test_fully_supported_rec_matches_decision_confidence(self):
        dc = strong_calibration()
        annotated = apply_recommendation_risk(
            [make_recommendation(confidence=1.0)], dc,
        )
        assert annotated[0].expected_confidence == pytest.approx(dc.confidence)
        assert annotated[0].expected_uncertainty == pytest.approx(
            dc.uncertainty_score,
        )

    def test_unsupported_rec_is_penalized(self):
        dc = strong_calibration()
        supported, unsupported = apply_recommendation_risk(
            [
                make_recommendation(confidence=1.0),
                make_recommendation(confidence=0.0),
            ],
            dc,
        )
        assert unsupported.expected_confidence < supported.expected_confidence
        assert unsupported.expected_uncertainty > supported.expected_uncertainty

    def test_empty_list_round_trips(self):
        assert apply_recommendation_risk([], strong_calibration()) == []

    def test_action_reflects_confidence_level(self):
        dc = strong_calibration()
        annotated = apply_recommendation_risk(
            [make_recommendation(0.5)], dc,
        )
        assert annotated[0].recommended_action == dc.recommended_action
