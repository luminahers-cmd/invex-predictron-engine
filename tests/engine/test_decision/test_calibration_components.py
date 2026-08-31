"""Tests for the deterministic confidence component functions."""

import pytest

from predictron_engine.decision.calibration import (
    action_for_confidence_level,
    classify_confidence_level,
    compute_decision_confidence,
    compute_evaluator_agreement,
    compute_evidence_agreement,
    compute_evidence_confidence,
    compute_evidence_diversity,
    compute_evidence_trust,
    compute_reasoning_confidence,
)
from predictron_engine.decision.models import ConfidenceLevel
from predictron_engine.evidence.models import IntelligenceSummary
from predictron_engine.evidence.provenance import TrustSummary
from predictron_engine.models.extracted_features import ExtractedFeatures
from tests.engine.test_decision.conftest import (
    make_assessment,
    make_bundle,
    make_document,
    make_observation,
    make_source,
)


class TestClassifyConfidenceLevel:
    def test_boundaries(self):
        assert classify_confidence_level(0.80) == ConfidenceLevel.VERY_HIGH
        assert classify_confidence_level(0.65) == ConfidenceLevel.HIGH
        assert classify_confidence_level(0.45) == ConfidenceLevel.MEDIUM
        assert classify_confidence_level(0.25) == ConfidenceLevel.LOW
        assert classify_confidence_level(0.00) == ConfidenceLevel.VERY_LOW

    def test_just_below_thresholds(self):
        assert classify_confidence_level(0.7999) == ConfidenceLevel.HIGH
        assert classify_confidence_level(0.6499) == ConfidenceLevel.MEDIUM
        assert classify_confidence_level(0.4499) == ConfidenceLevel.LOW
        assert classify_confidence_level(0.2499) == ConfidenceLevel.VERY_LOW

    def test_out_of_range_values_are_clamped(self):
        assert classify_confidence_level(5.0) == ConfidenceLevel.VERY_HIGH
        assert classify_confidence_level(-1.0) == ConfidenceLevel.VERY_LOW


class TestActionForConfidenceLevel:
    def test_mapping_is_total_and_deterministic(self):
        for level in ConfidenceLevel:
            action = action_for_confidence_level(level)
            assert isinstance(action, str) and action
            assert action == action_for_confidence_level(level)


class TestComputeEvidenceTrust:
    def test_prefers_trust_summary(self):
        summary = TrustSummary(
            average_trust_score=0.75, total_documents=4,
        )
        bundle = make_bundle(
            make_document("doc-1", trust=0.1),
            trust_summary=summary,
        )
        # Summary is reused as-is; per-document scores are ignored.
        assert compute_evidence_trust(bundle) == pytest.approx(0.75)

    def test_falls_back_to_document_scores(self):
        bundle = make_bundle(
            make_document("doc-1", trust=0.8),
            make_document("doc-2", trust=0.6),
        )
        assert compute_evidence_trust(bundle) == pytest.approx(0.7)

    def test_no_trust_information_returns_zero(self):
        assert compute_evidence_trust(None) == 0.0
        assert compute_evidence_trust(make_bundle()) == 0.0


class TestComputeEvidenceConfidence:
    def test_blends_intelligence_and_retrieval(self):
        bundle = make_bundle(
            sources=[make_source(True), make_source(False)],
            intelligence=IntelligenceSummary(
                average_quality=1.0,
                classification_confidence=1.0,
            ),
        )
        # 1.0*0.50 + 1.0*0.20 + 0.5*0.30 = 0.85
        assert compute_evidence_confidence(bundle) == pytest.approx(0.85)

    def test_retrieval_only_without_intelligence(self):
        bundle = make_bundle(
            sources=[make_source(True), make_source(True), make_source(False)],
        )
        assert compute_evidence_confidence(bundle) == pytest.approx(
            2.0 / 3.0, abs=1e-4,
        )

    def test_empty_inputs(self):
        assert compute_evidence_confidence(None) == 0.0
        assert compute_evidence_confidence(make_bundle()) == 0.0


class TestObservationAggregates:
    def test_agreement_is_mean_of_metadata(self):
        observations = [
            make_observation(agreement=1.0),
            make_observation(agreement=0.5),
        ]
        assert compute_evidence_agreement(observations) == pytest.approx(0.75)

    def test_agreement_empty(self):
        assert compute_evidence_agreement([]) == 0.0

    def test_reasoning_confidence_mean(self):
        observations = [
            make_observation(confidence=0.8),
            make_observation(confidence=0.6),
        ]
        assert compute_reasoning_confidence(observations) == pytest.approx(0.7)

    def test_reasoning_confidence_empty(self):
        assert compute_reasoning_confidence([]) == 0.0


class TestEvaluatorAgreement:
    def test_unanimous_assessments_keep_their_mean(self):
        assessments = [make_assessment(confidence=c) for c in (0.8, 0.8, 0.8)]
        assert compute_evaluator_agreement(assessments) == pytest.approx(0.8)

    def test_dispersal_penalizes_disagreement(self):
        unanimous = [make_assessment(confidence=0.8)] * 3
        split = [make_assessment(confidence=c) for c in (1.0, 1.0, 0.0, 0.0)]
        assert (
            compute_evaluator_agreement(split)
            < compute_evaluator_agreement(unanimous)
        )

    def test_single_assessment_has_no_disagreement(self):
        single = [make_assessment(confidence=0.7)]
        assert compute_evaluator_agreement(single) == pytest.approx(0.7)

    def test_empty(self):
        assert compute_evaluator_agreement([]) == 0.0


class TestEvidenceDiversity:
    def test_saturates_at_three_providers(self):
        bundle = make_bundle(
            make_document("d1", provider="a"),
            make_document("d2", provider="b"),
            make_document("d3", provider="c"),
            make_document("d4", provider="c"),
        )
        assert compute_evidence_diversity(bundle) == 1.0

    def test_two_providers(self):
        bundle = make_bundle(
            make_document("d1", provider="a"),
            make_document("d2", provider="b"),
        )
        assert compute_evidence_diversity(bundle) == pytest.approx(
            2.0 / 3.0, abs=1e-4,
        )

    def test_empty(self):
        assert compute_evidence_diversity(None) == 0.0
        assert compute_evidence_diversity(make_bundle()) == 0.0


class TestBreakdownWeights:
    def test_confidence_weights_sum_to_one(self, strong_inputs):
        dc = compute_decision_confidence(**strong_inputs)
        total = sum(f.weight for f in dc.breakdown.factors)
        assert abs(total - 1.0) < 1e-9

    def test_uncertainty_weights_sum_to_one(self, strong_inputs):
        dc = compute_decision_confidence(**strong_inputs)
        total = sum(f.weight for f in dc.uncertainty_breakdown.factors)
        assert abs(total - 1.0) < 1e-9

    def test_expected_factor_names_present(self, strong_inputs):
        dc = compute_decision_confidence(**strong_inputs)
        names = {f.name for f in dc.breakdown.factors}
        assert names == {
            "evidence_trust",
            "evidence_confidence",
            "evidence_agreement",
            "reasoning_confidence",
            "evaluator_agreement",
            "feature_completeness",
            "evidence_diversity",
        }

    def test_expected_driver_names_present(self, strong_inputs):
        dc = compute_decision_confidence(**strong_inputs)
        names = {f.name for f in dc.uncertainty_breakdown.factors}
        assert names == {
            "missing_evidence",
            "conflicting_evidence",
            "low_trust",
            "low_coverage",
            "evaluator_disagreement",
        }


class TestCompositeConsistency:
    def test_composite_equals_weighted_contributions(self, strong_inputs):
        dc = compute_decision_confidence(**strong_inputs)
        expected = round(sum(f.contribution for f in dc.breakdown.factors), 4)
        assert dc.breakdown.composite == pytest.approx(expected)

    def test_uncertainty_composite_matches_contributions(self, strong_inputs):
        dc = compute_decision_confidence(**strong_inputs)
        expected = round(
            sum(f.contribution for f in dc.uncertainty_breakdown.factors), 4,
        )
        assert dc.uncertainty_breakdown.composite == pytest.approx(expected)


def test_entry_point_smoke_all_components():
    """The public entry point exercises every component end to end."""
    bundle = make_bundle(
        make_document(),
        sources=[make_source(True)],
        intelligence=IntelligenceSummary(average_quality=0.5),
    )
    dc = compute_decision_confidence(
        bundle=bundle,
        observations=[make_observation()],
        assessments=[make_assessment()],
        features=ExtractedFeatures(data_completeness=1.0),
    )
    assert dc.breakdown.composite >= 0.0
    assert dc.uncertainty_breakdown.composite >= 0.0


class TestContradictionGraphCalibration:
    """Sprint P8D — calibration reuses the contradiction graph."""

    def _conflicting_observations(self):
        """Tensioned categories plus divergent confidence/importance that
        the contradiction graph classifies as a conflict, but which carry
        no per-observation conflict tally."""
        from predictron_engine.models.report import Observation

        return [
            Observation(
                dimension="market",
                category="strength",
                statement="Market is large.",
                confidence=0.9,
                importance=0.9,
            ),
            Observation(
                dimension="market",
                category="risk",
                statement="Market is saturated.",
                confidence=0.2,
                importance=0.1,
            ),
        ]

    def test_graph_raises_conflicting_evidence_uncertainty(self):
        from predictron_engine.reasoning.contradiction_graph import (
            build_contradiction_graph,
        )

        observations = self._conflicting_observations()
        graph = build_contradiction_graph(observations)
        assert graph.conflicting_count >= 1

        base = dict(
            bundle=None,
            observations=observations,
            assessments=[],
            features=ExtractedFeatures(data_completeness=0.9),
        )
        without = compute_decision_confidence(**base)
        with_graph = compute_decision_confidence(
            **base, contradiction_graph=graph
        )

        def driver(dc, name):
            return dc.uncertainty_breakdown.value_of(name)

        assert (
            driver(with_graph, "conflicting_evidence")
            > driver(without, "conflicting_evidence")
        )

    def test_graph_adds_dominant_conflict_weakening_factor(self):
        from predictron_engine.reasoning.contradiction_graph import (
            build_contradiction_graph,
        )

        observations = self._conflicting_observations()
        graph = build_contradiction_graph(observations)
        base = dict(
            bundle=None,
            observations=observations,
            assessments=[],
            features=ExtractedFeatures(data_completeness=0.9),
        )
        without = compute_decision_confidence(**base)
        with_graph = compute_decision_confidence(
            **base, contradiction_graph=graph
        )

        assert any(
            "dominant contradiction" in f for f in with_graph.weakening_factors
        )
        assert not any(
            "dominant contradiction" in f for f in without.weakening_factors
        )
