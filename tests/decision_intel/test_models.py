"""Tests for Decision Intelligence value models."""

from __future__ import annotations

import pytest

from predictron_engine.decision.intelligence_models import (
    CalibrationAdjustment,
    Contribution,
    ContributionType,
    DecisionIntelligenceReport,
    DecisionTrace,
    DecisionVerdict,
    Explanation,
    TraceNode,
    TraceNodeType,
)


class TestContributionType:
    @pytest.mark.parametrize("member,expected", [
        ("POSITIVE", "positive"),
        ("NEGATIVE", "negative"),
        ("NEUTRAL", "neutral"),
        ("CONFIDENCE", "confidence"),
    ])
    def test_enum_values(self, member: str, expected: str) -> None:
        assert ContributionType[member].value == expected

    @pytest.mark.parametrize("value", [
        "positive", "negative", "neutral", "confidence",
    ])
    def test_enum_lookup(self, value: str) -> None:
        assert ContributionType(value).value == value


class TestDecisionVerdict:
    @pytest.mark.parametrize("member,expected", [
        ("STRONG_INVEST", "strong_invest"),
        ("INVEST", "invest"),
        ("WATCH", "watch"),
        ("INVESTIGATE_FURTHER", "investigate_further"),
        ("PASS", "pass"),
    ])
    def test_enum_values(self, member: str, expected: str) -> None:
        assert DecisionVerdict[member].value == expected

    @pytest.mark.parametrize("value", [
        "strong_invest", "invest", "watch", "investigate_further", "pass",
    ])
    def test_enum_lookup(self, value: str) -> None:
        assert DecisionVerdict(value).value == value


class TestTraceNodeType:
    @pytest.mark.parametrize("member,expected", [
        ("RAW_FEATURE", "raw_feature"),
        ("NORMALIZED_FEATURE", "normalized_feature"),
        ("RULE", "rule"),
        ("INTERMEDIATE_SCORE", "intermediate_score"),
        ("DIMENSION_SCORE", "dimension_score"),
        ("OVERALL_SCORE", "overall_score"),
        ("DECISION", "decision"),
    ])
    def test_enum_values(self, member: str, expected: str) -> None:
        assert TraceNodeType[member].value == expected


class TestTraceNode:
    def test_basic_construction(self) -> None:
        node = TraceNode(node_type=TraceNodeType.RAW_FEATURE, label="Raw")
        assert node.node_type == TraceNodeType.RAW_FEATURE
        assert node.value is None

    def test_node_id_generated(self) -> None:
        node1 = TraceNode(node_type=TraceNodeType.RULE, label="R1")
        node2 = TraceNode(node_type=TraceNodeType.RULE, label="R2")
        assert node1.node_id != node2.node_id

    @pytest.mark.parametrize("value", [1.0, "pass", True, None, 42, [1, 2]])
    def test_value_roundtrip(self, value: object) -> None:
        node = TraceNode(node_type=TraceNodeType.DECISION, label="D", value=value)
        assert node.value == value

    def test_to_dict_contains_all_fields(self) -> None:
        node = TraceNode(
            node_type=TraceNodeType.NORMALIZED_FEATURE,
            feature_id="funding_velocity",
            label="Normalized",
            value=0.5,
            rule="normalize_feature(funding_velocity)",
        )
        data = node.to_dict()
        assert data["node_type"] == "normalized_feature"
        assert data["feature_id"] == "funding_velocity"
        assert data["label"] == "Normalized"
        assert data["value"] == 0.5
        assert "timestamp" in data


class TestContribution:
    def test_basic_construction(self) -> None:
        c = Contribution(
            feature_id="funding_velocity",
            feature_name="Funding Velocity",
            category="growth",
            contribution_type=ContributionType.POSITIVE,
        )
        assert c.computed_contribution == 0.0
        assert c.weight == 0.0

    @pytest.mark.parametrize("ctype", list(ContributionType))
    def test_contribution_type_roundtrip(self, ctype: ContributionType) -> None:
        c = Contribution(
            feature_id="f", feature_name="F", category="c",
            contribution_type=ctype,
        )
        assert c.contribution_type == ctype

    def test_to_dict(self) -> None:
        c = Contribution(
            feature_id="f",
            feature_name="F",
            category="growth",
            contribution_type=ContributionType.POSITIVE,
            raw_value=1.0,
            normalized_value=0.5,
            weight=0.2,
            computed_contribution=0.1,
            evidence_references=[{"source_type": "timeline"}],
        )
        data = c.to_dict()
        assert data["feature_id"] == "f"
        assert data["contribution_type"] == "positive"
        assert data["normalized_value"] == 0.5
        assert data["computed_contribution"] == 0.1

    @pytest.mark.parametrize("normalized", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_normalized_range(self, normalized: float) -> None:
        c = Contribution(
            feature_id="f", feature_name="F", category="c",
            contribution_type=ContributionType.POSITIVE,
            normalized_value=normalized,
        )
        assert -1.0 <= c.normalized_value <= 1.0


class TestDecisionTrace:
    def test_construction(self) -> None:
        trace = DecisionTrace(company_id="rec-1")
        assert trace.company_id == "rec-1"
        assert trace.nodes == []

    def test_default_verdict(self) -> None:
        trace = DecisionTrace(company_id="rec-1")
        assert trace.verdict == DecisionVerdict.PASS

    def test_get_nodes_by_type(self) -> None:
        trace = DecisionTrace(company_id="rec-1")
        trace.nodes.append(TraceNode(node_type=TraceNodeType.RAW_FEATURE, label="A"))
        trace.nodes.append(TraceNode(node_type=TraceNodeType.DECISION, label="B"))
        raw = trace.get_nodes_by_type(TraceNodeType.RAW_FEATURE)
        assert len(raw) == 1
        assert raw[0].label == "A"

    def test_get_node_by_id(self) -> None:
        trace = DecisionTrace(company_id="rec-1")
        node = TraceNode(node_type=TraceNodeType.RULE, label="R")
        trace.nodes.append(node)
        assert trace.get_node_by_id(node.node_id) is node

    def test_get_node_by_id_missing(self) -> None:
        trace = DecisionTrace(company_id="rec-1")
        assert trace.get_node_by_id("nope") is None

    def test_to_dict(self) -> None:
        trace = DecisionTrace(
            company_id="rec-1",
            overall_score=72.5,
            verdict=DecisionVerdict.INVEST,
            confidence=0.8,
        )
        data = trace.to_dict()
        assert data["company_id"] == "rec-1"
        assert data["overall_score"] == 72.5
        assert data["verdict"] == "invest"
        assert data["confidence"] == 0.8


class TestExplanation:
    def test_construction(self) -> None:
        exp = Explanation(company_id="rec-1")
        assert exp.strengths == []
        assert exp.headline == ""

    def test_to_dict(self) -> None:
        exp = Explanation(
            company_id="rec-1",
            headline="Strong opportunity",
            strengths=["Funding velocity"],
            weaknesses=["Low churn"],
        )
        data = exp.to_dict()
        assert data["headline"] == "Strong opportunity"
        assert data["strengths"] == ["Funding velocity"]


class TestCalibrationAdjustment:
    def test_construction(self) -> None:
        cal = CalibrationAdjustment()
        assert cal.benchmark_case_id == ""
        assert cal.expected_confidence == 0.0
        assert cal.actual_confidence == 0.0

    def test_to_dict(self) -> None:
        cal = CalibrationAdjustment(
            benchmark_case_id="case-1",
            expected_confidence=0.8,
            actual_confidence=0.6,
            calibration_delta=0.2,
            adjustment_applied=0.1,
        )
        data = cal.to_dict()
        assert data["benchmark_case_id"] == "case-1"
        assert data["calibration_delta"] == 0.2


class TestDecisionIntelligenceReport:
    def test_construction(self) -> None:
        report = DecisionIntelligenceReport(company_id="rec-1")
        assert report.engine_version == "0.13.0"
        assert report.contributions == []

    def test_to_dict_empty(self) -> None:
        report = DecisionIntelligenceReport(company_id="rec-1")
        data = report.to_dict()
        assert data["company_id"] == "rec-1"
        assert data["contributions"] == []

    def test_to_dict_with_content(self) -> None:
        report = DecisionIntelligenceReport(company_id="rec-1")
        report.contributions.append(Contribution(
            feature_id="f", feature_name="F", category="growth",
            contribution_type=ContributionType.POSITIVE,
        ))
        report.trace = DecisionTrace(company_id="rec-1")
        report.explanation = Explanation(company_id="rec-1")
        data = report.to_dict()
        assert len(data["contributions"]) == 1
        assert data["trace"] is not None
        assert data["explanation"] is not None
