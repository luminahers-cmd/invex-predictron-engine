"""Tests for deterministic uncertainty estimation."""

import pytest

from predictron_engine.decision.calibration import compute_decision_confidence
from predictron_engine.models.extracted_features import ExtractedFeatures
from tests.engine.test_decision.conftest import (
    make_assessment,
    make_bundle,
    make_document,
    make_observation,
)


def driver_value(dc, name: str) -> float:
    value = dc.uncertainty_breakdown.value_of(name)
    assert value is not None
    return value


class TestMissingEvidence:
    def test_all_dimensions_covered(self):
        observations = [make_observation("market"), make_observation("team")]
        assessments = [
            make_assessment("market"),
            make_assessment("team"),
        ]
        dc = compute_decision_confidence(
            bundle=None,
            observations=observations,
            assessments=assessments,
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "missing_evidence") == pytest.approx(0.0)

    def test_half_the_dimensions_missing(self):
        observations = [make_observation("market")]
        assessments = [
            make_assessment("market"),
            make_assessment("team"),
        ]
        dc = compute_decision_confidence(
            bundle=None,
            observations=observations,
            assessments=assessments,
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "missing_evidence") == pytest.approx(0.5)

    def test_nothing_known_at_all_is_maximally_missing(self):
        dc = compute_decision_confidence(
            bundle=None,
            observations=[],
            assessments=[],
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "missing_evidence") == 1.0


class TestConflictingEvidence:
    def test_signal_conflict_category_counts(self):
        observations = [
            make_observation(category="signal_conflict", conflict_count=0),
            make_observation(),
            make_observation(),
        ]
        dc = compute_decision_confidence(
            bundle=None,
            observations=observations,
            assessments=[],
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "conflicting_evidence") == pytest.approx(
            1 / 3, abs=1e-4,
        )

    def test_metadata_conflict_counts_add(self):
        observations = [
            make_observation(conflict_count=2),
            make_observation(conflict_count=1),
            make_observation(),
        ]
        dc = compute_decision_confidence(
            bundle=None,
            observations=observations,
            assessments=[],
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "conflicting_evidence") == 1.0  # capped

    def test_no_conflicts(self):
        dc = compute_decision_confidence(
            bundle=None,
            observations=[make_observation()],
            assessments=[],
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "conflicting_evidence") == 0.0


class TestLowTrust:
    def test_inverse_of_trust(self):
        dc = compute_decision_confidence(
            bundle=make_bundle(make_document(trust=0.3)),
            observations=[make_observation()],
            assessments=[],
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "low_trust") == pytest.approx(0.7)

    def test_no_bundle_means_full_distrust(self):
        dc = compute_decision_confidence(
            bundle=None,
            observations=[make_observation()],
            assessments=[],
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "low_trust") == 1.0


class TestLowCoverage:
    def test_dense_observations_fully_cover(self):
        observations = [
            make_observation(d)
            for d in ("market", "team")
            for _ in range(3)  # 3 obs per dimension saturates coverage
        ]
        dc = compute_decision_confidence(
            bundle=None,
            observations=observations,
            assessments=list(map(make_assessment, ("market", "team"))),
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "low_coverage") == pytest.approx(0.0)

    def test_sparse_observations_leave_coverage_gaps(self):
        dc = compute_decision_confidence(
            bundle=None,
            observations=[make_observation()],
            assessments=[],
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "low_coverage") > 0.5


class TestEvaluatorDisagreement:
    def test_unanimous_evaluators_have_zero_disagreement(self):
        dc = compute_decision_confidence(
            bundle=None,
            observations=[make_observation()],
            assessments=[
                make_assessment("a", confidence=0.8),
                make_assessment("b", confidence=0.8),
            ],
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "evaluator_disagreement") == 0.0

    def test_maximal_spread_is_full_disagreement(self):
        dc = compute_decision_confidence(
            bundle=None,
            observations=[make_observation()],
            assessments=[
                make_assessment("a", confidence=1.0),
                make_assessment("b", confidence=0.0),
            ],
            features=ExtractedFeatures(),
        )
        assert driver_value(dc, "evaluator_disagreement") == 1.0


class TestUncertaintyScore:
    def test_strong_inputs_are_less_uncertain_than_weak(self, strong_inputs, weak_inputs):
        high = compute_decision_confidence(**strong_inputs)
        low = compute_decision_confidence(**weak_inputs)
        assert high.uncertainty_score < low.uncertainty_score

    def test_range_bounds(self, weak_inputs):
        dc = compute_decision_confidence(**weak_inputs)
        assert 0.0 <= dc.uncertainty_score <= 1.0

    def test_known_empty_scenario_value(self, weak_inputs):
        """Deterministic baseline: missing .25 + distrust .20 + no coverage
        .15 = 0.60 (conflicts and evaluator disagreement are absent)."""
        dc = compute_decision_confidence(**weak_inputs)
        assert dc.uncertainty_score == pytest.approx(0.6)
