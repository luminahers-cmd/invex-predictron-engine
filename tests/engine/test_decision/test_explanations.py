"""Tests for decision confidence end-to-end and structured explanations."""

import pytest

from predictron_engine.decision.calibration import (
    build_calibration_summary,
    compute_decision_confidence,
)
from predictron_engine.decision.models import ConfidenceLevel
from predictron_engine.models.extracted_features import ExtractedFeatures
from tests.engine.test_decision.conftest import (
    make_assessment,
    make_observation,
)


class TestDeterminism:
    def test_identical_inputs_identical_outputs(self, strong_inputs):
        a = compute_decision_confidence(**strong_inputs)
        b = compute_decision_confidence(**strong_inputs)
        assert a.model_dump() == b.model_dump()

    def test_output_is_frozen_snapshot_of_inputs(self, strong_inputs):
        """Mutating input lists afterwards must not affect the result."""
        inputs = dict(strong_inputs)
        dc = compute_decision_confidence(**inputs)
        inputs["observations"].clear()
        assert compute_decision_confidence(**strong_inputs).model_dump() != (
            dc.model_dump()
        )


class TestConfidenceOrdering:
    def test_strong_beats_weak(self, strong_inputs, weak_inputs):
        high = compute_decision_confidence(**strong_inputs)
        low = compute_decision_confidence(**weak_inputs)
        assert high.confidence > low.confidence
        assert high.level in (ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH)
        assert low.level in (ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW)

    def test_empty_pipeline_has_zero_confidence(self, weak_inputs):
        dc = compute_decision_confidence(**weak_inputs)
        # Only feature completeness (0.05 * 0.10 weight) contributes.
        assert dc.confidence == pytest.approx(0.005)
        assert dc.level == ConfidenceLevel.VERY_LOW

    def test_conflicts_raise_uncertainty_not_confidence(self, strong_inputs):
        """Conflicting evidence is an uncertainty driver by design; the
        confidence composite only combines the six specified inputs."""
        conflicted = {
            **strong_inputs,
            "observations": [
                make_observation(
                    "market",
                    category="signal_conflict",
                    conflict_count=1,
                ),
                *strong_inputs["observations"],
            ],
        }
        base = compute_decision_confidence(**strong_inputs)
        with_conflict = compute_decision_confidence(**conflicted)
        assert with_conflict.uncertainty_score > base.uncertainty_score
        assert any(
            "conflicting evidence signal(s) detected" in s
            for s in with_conflict.weakening_factors
        )
        assert (
            "no major contradictions detected"
            not in " | ".join(with_conflict.supporting_factors)
        )


class TestRanges:
    def test_all_values_in_unit_range(self, weak_inputs, strong_inputs):
        for inputs in (weak_inputs, strong_inputs):
            dc = compute_decision_confidence(**inputs)
            assert 0.0 <= dc.confidence <= 1.0
            assert 0.0 <= dc.uncertainty_score <= 1.0
            for factor in dc.breakdown.factors:
                assert 0.0 <= factor.value <= 1.0
            for factor in dc.uncertainty_breakdown.factors:
                assert 0.0 <= factor.value <= 1.0


class TestSupportingExplanations:
    def test_high_quality_scenario_lists_strengths(self, strong_inputs):
        dc = compute_decision_confidence(**strong_inputs)
        joined = " | ".join(dc.supporting_factors)
        assert "no major contradictions detected" in joined
        # Three distinct providers -> diversity statement present.
        assert any("distinct sources" in s for s in dc.supporting_factors)

    def test_official_support_statement(self, strong_inputs):
        dc = compute_decision_confidence(**strong_inputs)
        assert any(
            "official website supports claims" in s
            for s in dc.supporting_factors
        )

    def test_no_supporting_statements_for_empty_inputs(self, weak_inputs):
        dc = compute_decision_confidence(**weak_inputs)
        assert dc.supporting_factors == []


class TestWeakeningExplanations:
    def test_sprint_example_low_confidence_reasons(self):
        """founder missing + technology weak + conflicting claims."""
        features = ExtractedFeatures(data_completeness=0.2)
        observations = [
            make_observation("market", category="signal_conflict"),
            make_observation("market"),
        ]
        dc = compute_decision_confidence(
            bundle=None,
            observations=observations,
            assessments=[],
            features=features,
        )
        joined = " | ".join(dc.weakening_factors)
        assert "founder information missing" in joined
        assert "technology evidence weak" in joined
        assert "conflicting evidence signal(s) detected" in joined

    def test_missing_dimension_evidence_reported(self):
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
        assert any(
            "evidence missing for 1 assessed dimension" in s
            for s in dc.weakening_factors
        )

    def test_no_observations_statement(self, weak_inputs):
        dc = compute_decision_confidence(**weak_inputs)
        assert "no reasoning observations available" in dc.weakening_factors

    def test_explanation_lists_are_capped(self):
        many_observations = [
            make_observation(f"dim{i}", category="signal_conflict")
            for i in range(20)
        ]
        dc = compute_decision_confidence(
            bundle=None,
            observations=many_observations,
            assessments=[],
            features=ExtractedFeatures(),
        )
        assert len(dc.weakening_factors) <= 8


class TestCalibrationSummary:
    def test_summary_digests_decision_confidence(self, strong_inputs):
        dc = compute_decision_confidence(**strong_inputs)
        summary = build_calibration_summary(
            dc,
            observation_count=len(strong_inputs["observations"]),
            assessed_dimension_count=len(strong_inputs["assessments"]),
            evidence_document_count=3,
        )
        assert summary.confidence == dc.confidence
        assert summary.level == dc.level
        assert summary.uncertainty_score == dc.uncertainty_score
        assert summary.recommended_action == dc.recommended_action
        assert summary.observation_count == 3
        assert summary.assessed_dimension_count == 3
        assert summary.evidence_document_count == 3
        assert summary.top_supporting_factors == dc.supporting_factors[:3]

    def test_summary_is_deterministic(self, strong_inputs):
        dc = compute_decision_confidence(**strong_inputs)
        kwargs = dict(
            observation_count=3,
            assessed_dimension_count=3,
            evidence_document_count=3,
        )
        assert build_calibration_summary(dc, **kwargs).model_dump() == (
            build_calibration_summary(dc, **kwargs).model_dump()
        )
