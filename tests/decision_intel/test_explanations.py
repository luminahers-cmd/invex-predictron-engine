"""Tests for ExplainabilityEngine — deterministic, evidence-referencing explanations."""

from __future__ import annotations

import pytest

from predictron_engine.decision.explainability import ExplainabilityEngine
from predictron_engine.decision.intelligence_models import DecisionVerdict

from .conftest import build_feature_set


class TestBuildExplanation:
    def test_explanation_has_company_id(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set(company_id="rec-42")
        explanation = engine.build_explanation(feature_set)
        assert explanation.company_id == "rec-42"

    def test_empty_feature_set_produces_explanation(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set(
            include_company=False, include_growth=False, include_founder=False,
            include_funding=False, include_graph=False, include_signals=False,
            include_benchmark=False,
        )
        explanation = engine.build_explanation(feature_set)
        assert explanation.strengths == []

    def test_headline_for_verdict(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(
            feature_set, verdict=DecisionVerdict.INVEST,
        )
        assert "Investment" in explanation.headline

    @pytest.mark.parametrize("verdict,keyword", [
        (DecisionVerdict.STRONG_INVEST, "Strong"),
        (DecisionVerdict.INVEST, "Investment"),
        (DecisionVerdict.WATCH, "Watchlist"),
        (DecisionVerdict.INVESTIGATE_FURTHER, "further investigation"),
        (DecisionVerdict.PASS, "Pass"),
    ])
    def test_headline_keywords(
        self,
        verdict: DecisionVerdict,
        keyword: str,
    ) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(feature_set, verdict=verdict)
        assert keyword.lower() in explanation.headline.lower()

    def test_strengths_reference_actual_features(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(feature_set)
        for strength in explanation.strengths:
            assert len(strength) > 0

    def test_evidence_references_present(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(feature_set)
        assert len(explanation.evidence_references) > 0

    def test_supporting_contributions_present(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(feature_set)
        assert len(explanation.supporting_contributions) == feature_set.feature_count()

    def test_recommendation_built(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(feature_set)
        assert explanation.recommendation

    def test_custom_recommendation_used(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(
            feature_set,
            recommendation_text="Custom rec",
        )
        assert explanation.recommendation == "Custom rec"

    def test_full_explanation_contains_strengths(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(feature_set)
        text = explanation.full_explanation
        assert "Investment recommendation driven by:" in text

    @pytest.mark.parametrize("iterations", [1, 2, 3])
    def test_deterministic_explanations(self, iterations: int) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        first = engine.build_explanation(feature_set)
        for _ in range(iterations):
            second = engine.build_explanation(feature_set)
            assert second.full_explanation == first.full_explanation
            assert second.strengths == first.strengths
            assert second.weaknesses == first.weaknesses


class TestFormatting:
    def test_format_strengths_empty(self) -> None:
        engine = ExplainabilityEngine()
        assert engine.format_strengths([]) == ""

    def test_format_strengths_with_prefix(self) -> None:
        engine = ExplainabilityEngine()
        text = engine.format_strengths(["Funding velocity", "Hiring growth"])
        assert text.startswith("Investment recommendation driven by:")
        assert "Funding velocity" in text
        assert "Hiring growth" in text

    def test_format_weaknesses_empty(self) -> None:
        engine = ExplainabilityEngine()
        assert engine.format_weaknesses([]) == ""

    def test_format_weaknesses_with_prefix(self) -> None:
        engine = ExplainabilityEngine()
        text = engine.format_weaknesses(["Low market momentum"])
        assert "Constrained by:" in text
        assert "Low market momentum" in text


class TestRecommendation:
    @pytest.mark.parametrize("verdict,term", [
        (DecisionVerdict.STRONG_INVEST, "Proceed"),
        (DecisionVerdict.INVEST, "Proceed"),
        (DecisionVerdict.WATCH, "Monitor"),
        (DecisionVerdict.INVESTIGATE_FURTHER, "investigation"),
        (DecisionVerdict.PASS, "invest"),
    ])
    def test_recommendation_terms(
        self,
        verdict: DecisionVerdict,
        term: str,
    ) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(feature_set, verdict=verdict)
        assert term.lower() in explanation.recommendation.lower()

    def test_pass_recommendation_negative(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(
            feature_set, verdict=DecisionVerdict.PASS,
        )
        assert "Do not invest" in explanation.recommendation


class TestEvidenceSummary:
    def test_evidence_summary_mentions_counts(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(feature_set)
        assert "evidence reference" in explanation.evidence_summary.lower()

    def test_no_contributions_evidence_summary(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set(
            include_company=False, include_growth=False, include_founder=False,
            include_funding=False, include_graph=False, include_signals=False,
            include_benchmark=False,
        )
        explanation = engine.build_explanation(feature_set)
        assert "No contributions" in explanation.evidence_summary

    def test_metadata_verdict(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(
            feature_set, verdict=DecisionVerdict.WATCH,
        )
        assert explanation.metadata["verdict"] == DecisionVerdict.WATCH.value


class TestToDict:
    def test_explanation_serialization(self) -> None:
        engine = ExplainabilityEngine()
        feature_set = build_feature_set()
        explanation = engine.build_explanation(feature_set)
        data = explanation.to_dict()
        assert data["company_id"] == feature_set.company_id
        assert "strengths" in data
        assert "evidence_references" in data
