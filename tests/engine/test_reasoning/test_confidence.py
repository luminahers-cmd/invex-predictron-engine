"""Tests for Sprint 6A deterministic reasoning confidence."""

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.models.report import EvidenceCitation
from predictron_engine.reasoning.confidence import (
    WEIGHTS,
    compute_context_confidence,
    compute_reasoning_confidence,
    compute_reasoning_confidence_breakdown,
)
from predictron_engine.reasoning.context import ReasoningContext
from tests.engine.test_reasoning.conftest import make_document


class TestComputeReasoningConfidence:
    def test_all_max_factors_yield_one(self):
        assert (
            compute_reasoning_confidence(
                extraction_confidence=1.0,
                evidence_trust=1.0,
                evidence_agreement=1.0,
                citation_diversity=1.0,
                feature_completeness=1.0,
            )
            == 1.0
        )

    def test_all_zero_factors_yield_zero(self):
        assert (
            compute_reasoning_confidence(
                extraction_confidence=0.0,
                evidence_trust=0.0,
                evidence_agreement=0.0,
                citation_diversity=0.0,
                feature_completeness=0.0,
            )
            == 0.0
        )

    def test_weighted_combination_is_exact(self):
        result = compute_reasoning_confidence(
            extraction_confidence=0.8,
            evidence_trust=0.6,
            evidence_agreement=0.4,
            citation_diversity=1.0,
            feature_completeness=0.5,
        )
        expected = round(
            0.8 * WEIGHTS["extraction_confidence"]
            + 0.6 * WEIGHTS["evidence_trust"]
            + 0.4 * WEIGHTS["evidence_agreement"]
            + 1.0 * WEIGHTS["citation_diversity"]
            + 0.5 * WEIGHTS["feature_completeness"],
            4,
        )
        assert result == expected

    def test_weights_sum_to_one(self):
        assert round(sum(WEIGHTS.values()), 6) == 1.0

    def test_values_above_one_are_clamped(self):
        assert (
            compute_reasoning_confidence(
                extraction_confidence=5.0,
                evidence_trust=-1.0,
                evidence_agreement=2.0,
                citation_diversity=1.0,
                feature_completeness=1.0,
            )
            == compute_reasoning_confidence(
                extraction_confidence=1.0,
                evidence_trust=0.0,
                evidence_agreement=1.0,
                citation_diversity=1.0,
                feature_completeness=1.0,
            )
        )

    def test_output_always_in_unit_range(self):
        for trust in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = compute_reasoning_confidence(
                extraction_confidence=trust,
                evidence_trust=trust,
                evidence_agreement=trust,
                citation_diversity=trust,
                feature_completeness=trust,
            )
            assert 0.0 <= value <= 1.0

    def test_deterministic_outputs(self):
        kwargs = dict(
            extraction_confidence=0.7,
            evidence_trust=0.55,
            evidence_agreement=0.4,
            citation_diversity=0.66,
            feature_completeness=0.8,
        )
        assert compute_reasoning_confidence(**kwargs) == (
            compute_reasoning_confidence(**kwargs)
        )

    def test_higher_trust_increases_confidence(self):
        base = dict(
            extraction_confidence=0.5,
            evidence_agreement=0.5,
            citation_diversity=0.5,
            feature_completeness=0.5,
        )
        low = compute_reasoning_confidence(evidence_trust=0.1, **base)
        high = compute_reasoning_confidence(evidence_trust=0.9, **base)
        assert high > low


class TestConfidenceBreakdown:
    def test_breakdown_matches_scalar_version(self):
        kwargs = dict(
            extraction_confidence=0.9,
            evidence_trust=0.7,
            evidence_agreement=0.5,
            citation_diversity=0.33,
            feature_completeness=0.6,
        )
        breakdown = compute_reasoning_confidence_breakdown(**kwargs)
        assert breakdown.overall == compute_reasoning_confidence(**kwargs)
        assert breakdown.extraction_confidence == 0.9
        assert breakdown.evidence_trust == 0.7

    def test_breakdown_exposes_every_factor(self):
        breakdown = compute_reasoning_confidence_breakdown(
            extraction_confidence=1.0,
            evidence_trust=1.0,
            evidence_agreement=1.0,
            citation_diversity=1.0,
            feature_completeness=1.0,
        )
        assert breakdown.citation_diversity == 1.0
        assert breakdown.feature_completeness == 1.0
        assert breakdown.evidence_agreement == 1.0


class TestContextConfidence:
    """Factors derived from a ReasoningContext are deterministic."""

    def _context(self, features, items, docs):
        from predictron_engine.evidence.models import EvidenceBundle

        bundle = EvidenceBundle(startup_name="x", documents=docs)
        return ReasoningContext(features, items, bundle)

    def test_factors_derive_from_context_state(self, rich_features):
        item_a = EvidenceItem(
            domain="industry",
            category="market_size",
            statement="Big market.",
            source="src-a",
            citations=[EvidenceCitation(
                claim="Big market.",
                domain="industry",
                category="market_size",
                source_document_ids=["d1"],
            )],
        )
        item_b = EvidenceItem(
            domain="industry",
            category="market_size",
            statement="Big market again.",
            source="src-b",
            citations=[EvidenceCitation(
                claim="Big market again.",
                domain="industry",
                category="market_size",
                source_document_ids=["d2"],
            )],
        )
        docs = [
            make_document("d1", trust=0.8),
            make_document("d2", url="https://example.com/d2", trust=0.6),
        ]
        ctx = self._context(rich_features, [item_a, item_b], docs)
        breakdown = compute_context_confidence(ctx)

        assert breakdown.evidence_trust == round((0.8 + 0.6) / 2, 4)
        # Both items share (domain, category) with two distinct documents.
        assert breakdown.evidence_agreement == 1.0
        # Two distinct sources saturate at 3 -> 2/3.
        assert breakdown.citation_diversity == round(2 / 3, 4)
        assert breakdown.feature_completeness == rich_features.data_completeness
        assert 0.0 <= breakdown.overall <= 1.0

    def test_empty_context_yields_zero_factors_except_fallbacks(
        self, minimal_features
    ):
        ctx = ReasoningContext(minimal_features, [], None)
        breakdown = compute_context_confidence(ctx)
        assert breakdown.evidence_trust == 0.0
        assert breakdown.extraction_confidence == 0.0
        assert breakdown.overall >= 0.0

    def test_deterministic_outputs_for_same_context(
        self, rich_features, rich_evidence
    ):
        ctx = ReasoningContext(rich_features, rich_evidence, None)
        a = compute_context_confidence(ctx)
        b = compute_context_confidence(ctx)
        assert a == b
