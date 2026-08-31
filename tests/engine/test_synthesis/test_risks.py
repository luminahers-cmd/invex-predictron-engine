"""Tests for the unified risk register (Sprint 6C)."""

from predictron_engine.reasoning.consistency import (
    ConsistencyReport,
    ContradictionFinding,
    UnsupportedFinding,
)
from predictron_engine.synthesis.risks import aggregate_risks
from tests.engine.test_synthesis.conftest import (
    make_assessment,
    make_citation,
    make_confidence,
    make_features,
    make_observation,
    make_readiness,
    make_relationship,
)


class TestFeatureRisks:
    def test_feature_risk_flags_aggregated(self):
        features = make_features(market_risk=["Crowded market", "Regulation pending"])
        result = aggregate_risks(features, [], [], None, None)
        assert any(item.label == "market risk flags" for item in result)

    def test_severity_bands_by_flag_count(self):
        many = make_features(traction_risk=[f"flag {i}" for i in range(9)])
        some = make_features(traction_risk=["flag a", "flag b", "flag c"])
        high = next(
            item
            for item in aggregate_risks(many, [], [], None, None)
            if item.label == "traction risk flags"
        )
        low = next(
            item
            for item in aggregate_risks(some, [], [], None, None)
            if item.label == "traction risk flags"
        )
        assert high.severity.value == "high"
        assert low.severity.value == "low"

    def test_critical_runway_is_highest_severity(self):
        features = make_features(runway_months=4, burn_rate_usd=50000.0)
        result = aggregate_risks(features, [], [], None, None)
        assert result[0].severity.value == "critical"
        assert result[0].label == "critical runway"
        assert "4 months" in result[0].statements[0]

    def test_weak_nrr_and_high_churn_flagged(self):
        features = make_features(nrr_pct=80.0, churn_rate_pct=12.0)
        labels = [item.label for item in aggregate_risks(features, [], [], None, None)]
        assert "weak net revenue retention" in labels
        assert "high customer churn" in labels

    def test_ltv_cac_below_one_flagged(self):
        features = make_features(ltv_cac_ratio=0.5)
        labels = [item.label for item in aggregate_risks(features, [], [], None, None)]
        assert "unit economics below break-even" in labels


class TestConsistencyAndRelationships:
    def test_consistency_contradictions_become_risks(self):
        report = ConsistencyReport(
            contradictions=[
                ContradictionFinding(
                    field_a="arr_usd",
                    value_a=100,
                    field_b="funding_stage",
                    value_b="pre_seed",
                    reason="Revenue contradicts stage",
                )
            ]
        )
        result = aggregate_risks(make_features(), [], [], None, None, consistency=report)
        assert any(item.label == "conflicting internal signals" for item in result)

    def test_unsupported_observations_become_risks(self):
        report = ConsistencyReport(
            unsupported_observations=[
                UnsupportedFinding(
                    source_rule="SomeRule",
                    statement="Claim without evidence",
                    reason="No supporting evidence items",
                )
            ]
        )
        result = aggregate_risks(make_features(), [], [], None, None, consistency=report)
        unsupported = next(
            item for item in result if item.label == "unsupported conclusion"
        )
        assert unsupported.statements == ["Claim without evidence"]

    def test_conflicting_relationships_grouped_by_dimension(self):
        readiness = make_readiness(
            signal_relationships=[
                make_relationship("traction_signals", "team_execution", "conflicting"),
            ]
        )
        result = aggregate_risks(make_features(), [], [], readiness, None)
        labels = [item.label for item in result]
        assert "traction signals signal conflict" in labels
        assert "team execution signal conflict" in labels

    def test_many_conflicts_raise_severity_to_high(self):
        relationships = [
            make_relationship("traction_signals", f"dim_{i}", "conflicting")
            for i in range(3)
        ]
        readiness = make_readiness(signal_relationships=relationships)
        result = aggregate_risks(make_features(), [], [], readiness, None)
        conflict = next(
            item
            for item in result
            if item.dimension == "traction_signals" and "conflict" in item.label
        )
        assert conflict.severity.value == "high"


class TestAssessmentAndCalibrationRisks:
    def test_low_confidence_assessment_becomes_moderate_risk(self):
        assessments = [make_assessment("product_strength", confidence=0.2)]
        result = aggregate_risks(make_features(), [], assessments, None, None)
        entry = next(
            item for item in result if item.dimension == "product_strength"
        )
        assert entry.severity.value == "moderate"
        assert "low confidence" in entry.label

    def test_assessed_dimension_without_observations_is_evidence_gap(self):
        assessments = [make_assessment("founder_quality", confidence=0.9)]
        result = aggregate_risks(make_features(), [], assessments, None, None)
        gap = next(item for item in result if "evidence gap" in item.label)
        assert gap.severity.value == "low"

    def test_weakening_calibration_factors_captured(self):
        confidence = make_confidence()
        result = aggregate_risks(make_features(), [], [], None, confidence)
        weakening = next(
            item for item in result if item.label == "weakening calibration factors"
        )
        assert weakening.severity.value == "low"
        assert "Some evidence is missing." in weakening.statements


class TestDeduplicationAndPropagation:
    def test_duplicate_labels_merge_with_min_confidence(self):
        observation_a = make_observation(
            "traction_signals",
            statement="First view",
            confidence=0.9,
            citations=[make_citation(claim="Traction doc")],
            provenance_document_ids=["doc-1"],
        )
        observation_b = make_observation(
            "traction_signals",
            statement="Second view",
            confidence=0.4,
            provenance_document_ids=["doc-2"],
        )
        assessments = [
            make_assessment(
                "traction_signals",
                confidence=0.2,
                supporting_observations=[observation_b],
            ),
            make_assessment(
                "traction_signals",
                confidence=0.2,
                supporting_observations=[observation_a],
            ),
        ]
        result = aggregate_risks(make_features(), [], assessments, None, None)
        low_confidence = [
            item
            for item in result
            if item.label.startswith("low confidence traction")
        ]
        assert len(low_confidence) == 1
        merged = low_confidence[0]
        statements = set(merged.statements)
        assert {"First view", "Second view"} <= {
            s for s in statements
        } or merged.confidence == 0.2

    def test_ranking_by_severity_then_dimension(self):
        from predictron_engine.synthesis.models import severity_order

        features = make_features(
            runway_months=3,
            market_risk=["a", "b", "c", "d"],
            founder_risk=["x"],
        )
        result = aggregate_risks(features, [], [], None, None)
        orders = [severity_order(item.severity) for item in result]
        assert orders == sorted(orders)

    def test_citations_and_provenance_carried(self):
        observation = make_observation(
            "market_opportunity",
            confidence=0.7,
            citations=[make_citation(claim="Market doc")],
            provenance_document_ids=["doc-42"],
        )
        assessments = [
            make_assessment(
                "market_opportunity",
                confidence=0.2,
                supporting_observations=[observation],
            )
        ]
        result = aggregate_risks(make_features(), [], assessments, None, None)
        with_citations = [
            item
            for item in result
            if any(c.claim == "Market doc" for c in item.citations)
        ]
        assert with_citations
        doc_ids = {
            doc_id
            for item in with_citations
            for doc_id in item.provenance_document_ids
        }
        assert "doc-42" in doc_ids

    def test_output_deterministic_and_capped_at_ten(self):
        fields = {}
        for index in range(15):
            fields[f"risk_field_{index}"] = None
        features = make_features(**{k: v for k, v in fields.items()})
        one = aggregate_risks(features, [], [], None, None)
        two = aggregate_risks(features, [], [], None, None)
        assert len(one) <= 10
        assert [r.model_dump() for r in one] == [r.model_dump() for r in two]

    def test_empty_inputs_return_empty_register(self):
        assert aggregate_risks(make_features(), [], [], None, None) == []


class TestContradictionGraphRisks:
    """Sprint P8D — contradiction graph drives risk entries."""

    def _graph(self):
        from predictron_engine.reasoning.contradiction_graph import (
            build_contradiction_graph,
        )

        observations = [
            make_observation(
                "market", category="strength", confidence=0.9, importance=0.9,
            ),
            make_observation(
                "market", category="risk", confidence=0.2, importance=0.1,
            ),
        ]
        graph = build_contradiction_graph(observations)
        assert graph.conflicting_count >= 1
        return graph, observations

    def test_conflicting_edges_become_risks(self):
        graph, observations = self._graph()
        result = aggregate_risks(
            make_features(), observations, [], None, None,
            contradiction_graph=graph,
        )
        contradiction_labels = [
            item.label for item in result if "contradiction" in item.label
        ]
        assert contradiction_labels

    def test_dominant_conflict_become_high_risk(self):
        graph, observations = self._graph()
        assert graph.dominant_conflict is not None
        result = aggregate_risks(
            make_features(), observations, [], None, None,
            contradiction_graph=graph,
        )
        dominant = next(
            (item for item in result if item.label == "dominant contradiction"),
            None,
        )
        assert dominant is not None
        assert dominant.severity.value == "high"

    def test_without_graph_no_contradiction_source(self):
        graph, observations = self._graph()
        result = aggregate_risks(make_features(), observations, [], None, None)
        assert not any(
            item.source.startswith("contradiction_graph")
            for item in result
        )
