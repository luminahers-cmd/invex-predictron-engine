"""Tests for Sprint 6A per-rule diagnostics."""

from predictron_engine.models.report import Observation
from predictron_engine.reasoning.composite import CompositeReasoner
from predictron_engine.reasoning.diagnostics import build_rule_diagnostic


class RefRule:
    """Rule that references one evidence item and one feature."""

    name = "ref_rule"

    def evaluate(self, features, evidence):
        obs = Observation(
            dimension="market_opportunity",
            category="market_context",
            statement="Ref observation.",
            confidence=0.8,
            source_rule="RefRule",
        )
        if evidence:
            item = evidence[0]
            obs = obs.model_copy(
                update={
                    "evidence": [
                        f"evidence:{item.domain}/{item.category}:"
                        f" {item.statement}"
                    ]
                }
            )
        return [obs]


class SilentRule:
    name = "silent_rule"

    def evaluate(self, features, evidence):
        return []


def _item(domain="industry", category="market_size", statement="Big market."):
    from predictron_engine.evidence.evidence_models import EvidenceItem

    return EvidenceItem(
        domain=domain,
        category=category,
        statement=statement,
        source="src",
    )


class TestCompositeDiagnosticsCollection:
    def test_diagnostics_collected_per_rule(self, rich_features):
        reasoner = CompositeReasoner([RefRule(), SilentRule()])
        reasoner.reason(rich_features, [_item()])
        diags = reasoner.last_diagnostics
        assert [d.rule_name for d in diags] == ["ref_rule", "silent_rule"]

    def test_evidence_examined_counts_pool(self, rich_features):
        pool = [_item(), _item(category="regulatory")]
        reasoner = CompositeReasoner([RefRule()])
        reasoner.reason(rich_features, pool)
        assert reasoner.last_diagnostics[0].evidence_examined == 2

    def test_evidence_selected_counts_referenced_items(self, rich_features):
        reasoner = CompositeReasoner([RefRule()])
        reasoner.reason(rich_features, [_item()])
        diag = reasoner.last_diagnostics[0]
        assert diag.evidence_selected == 1

    def test_confidence_is_mean_of_observations(self, rich_features):
        reasoner = CompositeReasoner([RefRule()])
        reasoner.reason(rich_features, [])
        assert reasoner.last_diagnostics[0].confidence == 0.8

    def test_silent_rule_has_zero_confidence_and_selection(
        self, rich_features
    ):
        reasoner = CompositeReasoner([SilentRule()])
        reasoner.reason(rich_features, [_item()])
        diag = reasoner.last_diagnostics[0]
        assert diag.confidence == 0.0
        assert diag.evidence_selected == 0

    def test_missing_evidence_exposed_to_rules(self):
        from predictron_engine.models.extracted_features import (
            ExtractedFeatures,
        )

        features = ExtractedFeatures(industry="fintech")
        reasoner = CompositeReasoner([SilentRule()])
        reasoner.reason(features, [])
        diag = reasoner.last_diagnostics[0]
        assert "industry" in diag.missing_evidence
        assert diag.missing_evidence_count >= 1

    def test_duration_recorded_non_negative(self, rich_features):
        reasoner = CompositeReasoner([RefRule()])
        reasoner.reason(rich_features, [])
        assert reasoner.last_diagnostics[0].duration_ms >= 0.0

    def test_reason_with_diagnostics_returns_both(self, rich_features):
        reasoner = CompositeReasoner([RefRule()])
        observations, diags = reasoner.reason_with_diagnostics(
            rich_features, []
        )
        assert len(observations) == 1
        assert len(diags) == 1


class TestDiagnosticFactory:
    def test_factory_computes_fields_directly(self):
        obs_a = Observation(
            dimension="d",
            category="c",
            statement="a",
            confidence=0.4,
            trust_score=0.2,
            evidence_conflict_count=1,
            source_rule="X",
        )
        obs_b = Observation(
            dimension="d",
            category="c",
            statement="b",
            confidence=0.8,
            trust_score=0.6,
            evidence_conflict_count=3,
            source_rule="X",
        )
        diag = build_rule_diagnostic(
            rule_name="X",
            evidence_examined=5,
            observations=[obs_a, obs_b],
            duration_ms=1.5,
            missing_evidence=["industry"],
        )
        assert diag.confidence == round((0.4 + 0.8) / 2, 4)
        assert diag.trust == round((0.2 + 0.6) / 2, 4)
        assert diag.conflicts == 4
        assert diag.evidence_examined == 5
        assert diag.missing_evidence == ["industry"]
        assert diag.duration_ms == 1.5

    def test_empty_observations_yield_zero_aggregates(self):
        diag = build_rule_diagnostic(
            rule_name="X",
            evidence_examined=0,
            observations=[],
            duration_ms=0.0,
        )
        assert diag.confidence == 0.0
        assert diag.trust == 0.0
        assert diag.conflicts == 0
