"""Sprint 6A backward-compatibility and integration tests.

Guarantees verified here:

* Legacy rules implementing only ``evaluate(features, evidence)``
  continue to work unchanged through the composite.
* New context-aware rules receive a ReasoningContext.
* ``DefaultReasoningEngine.reason`` keeps its original signature.
* The full default rule set produces deterministic output across runs.
* Every observation produced by a reasoning pass carries the new
  evidence-backed fields populated deterministically.
* Provenance is never lost through a reasoning pass.
"""

import json

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.models.report import EvidenceCitation, Observation
from predictron_engine.reasoning.composite import CompositeReasoner
from predictron_engine.reasoning.context import ReasoningContext
from predictron_engine.reasoning.reasoning_engine import DefaultReasoningEngine
from predictron_engine.reasoning.rules import DEFAULT_RULES
from predictron_engine.reasoning.traceability import (
    resolve_provenance_chain,
    verify_traceability,
)
from tests.engine.test_reasoning.conftest import make_document


class LegacyOnlyRule:
    """Rule implementing ONLY the pre-6A protocol."""

    def evaluate(self, features, evidence):
        return [
            Observation(
                dimension="legacy",
                category="legacy_category",
                statement="Legacy observation.",
                confidence=0.5,
                source_rule="LegacyOnlyRule",
            )
        ]


class ContextAwareRule:
    """Rule implementing the Sprint 6A context protocol."""

    name = "context_aware"

    def evaluate_context(self, context) -> list[Observation]:
        assert isinstance(context, ReasoningContext)
        return [
            Observation(
                dimension="ctx",
                category="ctx_category",
                statement=(
                    f"Domains available: "
                    f"{len(context._items_by_domain)}"
                ),
                confidence=0.9,
                source_rule="ContextAwareRule",
            )
        ]


class TestLegacyRuleCompatibility:
    def test_legacy_rule_still_invoked(self, rich_features):
        reasoner = CompositeReasoner([LegacyOnlyRule()])
        result = reasoner.reason(rich_features, [])
        assert len(result) == 1
        assert result[0].source_rule == "LegacyOnlyRule"
        assert result[0].dimension == "legacy"

    def test_legacy_rule_receives_features_and_evidence(
        self, rich_features, rich_evidence
    ):
        captured = {}

        class CaptureRule:
            def evaluate(self, features, evidence):
                captured["features"] = features
                captured["evidence"] = evidence
                return []

        CompositeReasoner([CaptureRule()]).reason(
            rich_features, rich_evidence
        )
        assert captured["features"] is rich_features
        assert captured["evidence"] == rich_evidence

    def test_mixed_legacy_and_context_rules(self, rich_features):
        reasoner = CompositeReasoner([LegacyOnlyRule(), ContextAwareRule()])
        result = reasoner.reason(rich_features, [])
        assert {o.source_rule for o in result} == {
            "LegacyOnlyRule",
            "ContextAwareRule",
        }

    def test_default_reasoning_engine_signature_unchanged(
        self, sample_features, sample_evidence
    ):
        engine = DefaultReasoningEngine()
        result = engine.reason(sample_features, sample_evidence)
        assert isinstance(result, list)
        # Legacy keyword-free positional call style:
        result2 = engine.reason(sample_features)
        assert isinstance(result2, list)

    def test_observation_model_accepts_legacy_construction(self):
        obs = Observation(
            dimension="d",
            category="c",
            statement="s",
            evidence=["e"],
            confidence=0.5,
            importance=0.5,
            source_rule="R",
        )
        assert obs.trust_score == 0.0
        assert obs.provenance_document_ids == []
        assert obs.evidence_agreement_ratio == 0.0
        assert obs.evidence_conflict_count == 0


class TestDeterministicOutputs:
    def test_full_rule_set_deterministic_across_runs(
        self, rich_features, rich_evidence
    ):
        engine = DefaultReasoningEngine()
        first = engine.reason(rich_features, rich_evidence)
        second = engine.reason(rich_features, rich_evidence)
        assert [o.model_dump() for o in first] == [
            o.model_dump() for o in second
        ]

    def test_composite_deterministic_with_bundle(self, rich_features):
        from predictron_engine.evidence.models import EvidenceBundle

        bundle = EvidenceBundle(
            startup_name="x", documents=[make_document("doc-1", trust=0.8)]
        )
        reasoner = CompositeReasoner(list(DEFAULT_RULES))
        a = reasoner.reason(rich_features, [], bundle)
        b = reasoner.reason(rich_features, [], bundle)
        assert [o.model_dump() for o in a] == [o.model_dump() for o in b]


class TestEvidenceBackedPopulation:
    def test_every_observation_carries_new_fields(
        self, rich_features, rich_evidence
    ):
        engine = DefaultReasoningEngine()
        result = engine.reason(rich_features, rich_evidence)
        assert result, "expected observations from default rules"
        for obs in result:
            assert 0.0 <= obs.trust_score <= 1.0
            assert 0.0 <= obs.evidence_agreement_ratio <= 1.0
            assert obs.evidence_conflict_count >= 0
            assert isinstance(obs.provenance_document_ids, list)

    def test_agreement_ratio_populated_for_corroborated_evidence(
        self, rich_features
    ):
        corroborated = [
            EvidenceItem(
                domain="industry",
                category="market_size",
                statement="Big market.",
                source="src-a",
            ),
            EvidenceItem(
                domain="industry",
                category="market_size",
                statement="Bigger market.",
                source="src-b",
            ),
        ]
        reasoner = CompositeReasoner([MarketRefRule()])
        result = reasoner.reason(rich_features, corroborated)
        assert result[0].evidence_agreement_ratio == 1.0

    def test_conflicts_populated_for_contradictory_evidence(
        self, rich_features
    ):
        conflicting = [
            EvidenceItem(
                domain="business_model",
                category="revenue",
                statement="The company has recurring revenue streams.",
                source="src-a",
            ),
            EvidenceItem(
                domain="business_model",
                category="revenue",
                statement="The company has transactional revenue streams.",
                source="src-b",
            ),
        ]
        reasoner = CompositeReasoner([BusinessModelRefRule()])
        result = reasoner.reason(rich_features, conflicting)
        assert result[0].evidence_conflict_count >= 1


class MarketRefRule:
    name = "market_ref"

    def evaluate(self, features, evidence):
        refs = [
            f"evidence:{i.domain}/{i.category}: {i.statement}"
            for i in evidence
        ]
        return [
            Observation(
                dimension="market_opportunity",
                category="market_context",
                statement="Market observation.",
                evidence=refs,
                confidence=0.7,
                source_rule="MarketRefRule",
            )
        ]


class BusinessModelRefRule:
    name = "bm_ref"

    def evaluate(self, features, evidence):
        refs = [
            f"evidence:{i.domain}/{i.category}: {i.statement}"
            for i in evidence
        ]
        return [
            Observation(
                dimension="business_model_viability",
                category="business_model_assessment",
                statement="Business model observation.",
                evidence=refs,
                confidence=0.7,
                source_rule="BusinessModelRefRule",
            )
        ]


class TestProvenancePreservationThroughPipeline:
    def test_cited_document_ids_survive_reasoning_pass(self, rich_features):
        doc = make_document(
            "doc-1", url="https://example.com/about", provider="website"
        )
        citation = EvidenceCitation(
            claim="Big market.",
            domain="industry",
            category="market_size",
            source_document_ids=["doc-1"],
            best_trust_score=0.9,
        )
        item = EvidenceItem(
            domain="industry",
            category="market_size",
            statement="Big market.",
            source="src",
            provenance_record=json.dumps({"document_id": "doc-1"}),
            citations=[citation],
        )

        reasoner = CompositeReasoner([MarketRefRule()])
        result = reasoner.reason(rich_features, [item])

        obs = result[0]
        assert "doc-1" in obs.provenance_document_ids
        assert verify_traceability(obs, [doc]) is True
        chain = resolve_provenance_chain(obs, [doc])
        assert chain[0].original_url == "https://example.com/about"
        assert chain[0].provider == "website"

    def test_trust_score_flows_to_observation(self, rich_features):
        citation = EvidenceCitation(
            claim="Big market.",
            domain="industry",
            category="market_size",
            best_trust_score=0.77,
        )
        item = EvidenceItem(
            domain="industry",
            category="market_size",
            statement="Big market.",
            source="src",
            citations=[citation],
        )
        reasoner = CompositeReasoner([MarketRefRule()])
        result = reasoner.reason(rich_features, [item])
        assert result[0].trust_score == 0.77


class TestEngineDiagnosticsIntegration:
    def test_default_engine_exposes_diagnostics(
        self, rich_features, rich_evidence
    ):
        engine = DefaultReasoningEngine()
        observations, diagnostics = engine.reason_with_diagnostics(
            rich_features, rich_evidence
        )
        assert len(observations) > 0
        assert len(diagnostics) == len(DEFAULT_RULES)
        assert all(d.rule_name for d in diagnostics)

    def test_last_consistency_report_available(
        self, rich_features, rich_evidence
    ):
        engine = DefaultReasoningEngine()
        engine.reason(rich_features, rich_evidence)
        report = engine.last_consistency
        assert report is not None
        assert hasattr(report, "contradictions")
        assert hasattr(report, "missing_evidence_domains")
