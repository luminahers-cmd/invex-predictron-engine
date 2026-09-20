"""Tests for DecisionTraceEngine — deterministic reasoning graphs."""

from __future__ import annotations

import pytest

from predictron_engine.decision.feature_engine import DecisionFeatureEngine
from predictron_engine.decision.intelligence_models import (
    DecisionVerdict,
    TraceNodeType,
)
from predictron_engine.decision.trace import DecisionTraceEngine

from .conftest import build_feature_set

VERDICT_SCORES = [
    (80.0, DecisionVerdict.STRONG_INVEST),
    (75.0, DecisionVerdict.STRONG_INVEST),
    (70.0, DecisionVerdict.INVEST),
    (60.0, DecisionVerdict.INVEST),
    (55.0, DecisionVerdict.WATCH),
    (45.0, DecisionVerdict.WATCH),
    (40.0, DecisionVerdict.INVESTIGATE_FURTHER),
    (30.0, DecisionVerdict.INVESTIGATE_FURTHER),
    (20.0, DecisionVerdict.PASS),
    (0.0, DecisionVerdict.PASS),
]


class TestBuildTrace:
    def test_trace_has_company_id(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set(company_id="rec-9")
        trace = engine.build_trace(feature_set)
        assert trace.company_id == "rec-9"

    def test_trace_has_nodes(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        assert len(trace.nodes) > 0

    def test_trace_has_all_node_types(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        node_types = {n.node_type for n in trace.nodes}
        assert TraceNodeType.RAW_FEATURE in node_types
        assert TraceNodeType.NORMALIZED_FEATURE in node_types
        assert TraceNodeType.RULE in node_types
        assert TraceNodeType.INTERMEDIATE_SCORE in node_types
        assert TraceNodeType.DIMENSION_SCORE in node_types
        assert TraceNodeType.OVERALL_SCORE in node_types
        assert TraceNodeType.DECISION in node_types

    def test_trace_has_edges(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        assert len(trace.edges) > 0

    def test_every_edge_endpoint_exists(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        node_ids = {n.node_id for n in trace.nodes}
        for src, dst in trace.edges:
            assert src in node_ids
            assert dst in node_ids

    def test_no_cycles_in_edges(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        # Simple DAG validation via visited set traversal (partial check).
        visited: set[str] = set()
        stack: set[str] = set()

        def visit(node_id: str) -> None:
            assert node_id not in stack, "cycle detected"
            if node_id in visited:
                return
            stack.add(node_id)
            children = [dst for src, dst in trace.edges if src == node_id]
            for child in children:
                visit(child)
            stack.remove(node_id)
            visited.add(node_id)

        for node in trace.nodes:
            visit(node.node_id)

    def test_overall_score_in_range(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        assert 0.0 <= trace.overall_score <= 100.0

    def test_decision_node_present(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        decisions = trace.get_nodes_by_type(TraceNodeType.DECISION)
        assert len(decisions) == 1


class TestClassifyVerdict:
    @pytest.mark.parametrize("score,verdict", VERDICT_SCORES)
    def test_score_maps_to_verdict(
        self,
        score: float,
        verdict: DecisionVerdict,
    ) -> None:
        engine = DecisionTraceEngine()
        assert engine.classify_verdict(score) == verdict

    @pytest.mark.parametrize("score", [-10.0, 150.0, 100.0])
    def test_out_of_range_scores(self, score: float) -> None:
        engine = DecisionTraceEngine()
        verdict = engine.classify_verdict(score)
        assert verdict in DecisionVerdict

    def test_custom_thresholds(self) -> None:
        custom = [
            (90.0, DecisionVerdict.STRONG_INVEST),
            (50.0, DecisionVerdict.INVEST),
            (10.0, DecisionVerdict.WATCH),
            (0.0, DecisionVerdict.PASS),
        ]
        engine = DecisionTraceEngine(verdict_thresholds=custom)
        assert engine.classify_verdict(95.0) == DecisionVerdict.STRONG_INVEST
        assert engine.classify_verdict(70.0) == DecisionVerdict.INVEST
        assert engine.classify_verdict(20.0) == DecisionVerdict.WATCH
        assert engine.classify_verdict(5.0) == DecisionVerdict.PASS


class TestNodeStructure:
    def test_raw_nodes_have_feature_ids(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        raw = trace.get_nodes_by_type(TraceNodeType.RAW_FEATURE)
        for node in raw:
            assert node.feature_id is not None

    def test_normalized_nodes_have_values(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        norm = trace.get_nodes_by_type(TraceNodeType.NORMALIZED_FEATURE)
        for node in norm:
            assert node.value is not None

    def test_rule_nodes_have_rules(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        rules = trace.get_nodes_by_type(TraceNodeType.RULE)
        for node in rules:
            assert node.rule

    def test_intermediate_nodes_trace_to_features(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        inter = trace.get_nodes_by_type(TraceNodeType.INTERMEDIATE_SCORE)
        for node in inter:
            assert node.feature_id is not None

    def test_dimension_nodes_are_scored(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        dims = trace.get_nodes_by_type(TraceNodeType.DIMENSION_SCORE)
        for node in dims:
            assert 0.0 <= float(node.value) <= 100.0

    def test_overall_node_single(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        overall = trace.get_nodes_by_type(TraceNodeType.OVERALL_SCORE)
        assert len(overall) == 1


class TestTraceDeterminism:
    def test_custom_thresholds_trace(self) -> None:
        custom = [(90.0, DecisionVerdict.STRONG_INVEST), (0.0, DecisionVerdict.PASS)]
        engine = DecisionTraceEngine(verdict_thresholds=custom)
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        assert trace.verdict in (DecisionVerdict.STRONG_INVEST, DecisionVerdict.PASS)

    @pytest.mark.parametrize("iterations", [1, 2, 3, 5])
    def test_trace_deterministic_across_runs(self, iterations: int) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        first = engine.build_trace(feature_set)
        for _ in range(iterations):
            second = engine.build_trace(feature_set)
            assert second.overall_score == first.overall_score
            assert second.verdict == first.verdict
            assert len(second.nodes) == len(first.nodes)
            assert len(second.edges) == len(first.edges)

    def test_trace_with_contributions(self) -> None:
        engine = DecisionTraceEngine()
        feature_engine = DecisionFeatureEngine()
        feature_set = build_feature_set()
        contributions = feature_engine.compute_all_contributions(feature_set)
        trace = engine.build_trace(feature_set, contributions)
        assert trace.company_id == feature_set.company_id

    def test_trace_with_confidence(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set, confidence=0.85)
        assert trace.confidence == 0.85

    def test_trace_to_dict(self) -> None:
        engine = DecisionTraceEngine()
        feature_set = build_feature_set()
        trace = engine.build_trace(feature_set)
        data = engine.trace_to_dict(trace)
        assert data["company_id"] == feature_set.company_id
        assert len(data["nodes"]) > 0
