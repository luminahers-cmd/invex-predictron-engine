"""Tests for the deterministic executive summary (Sprint 6C)."""

from predictron_engine.models.report import DecisionSynthesis
from predictron_engine.synthesis.summary import build_executive_summary
from tests.engine.test_synthesis.conftest import (
    make_confidence,
    make_decision,
    make_readiness,
    make_recommendation,
)


class TestSummaryContent:
    def test_summary_contains_decision_conviction_and_calibration(self):
        synthesis = DecisionSynthesis(
            overall_confidence=0.55,
            uncertainty_score=0.4,
            confidence_level="medium",
            recommended_action="Proceed with targeted diligence.",
        )
        decision = make_decision(composite=63.0)
        summary, points = build_executive_summary(decision, None, synthesis)
        assert "invest" in summary
        assert "63.0/100" in summary
        assert "55%" in summary
        assert "40%" in summary
        assert "Proceed with targeted diligence." in summary
        assert points[0].startswith("Investment decision: invest")

    def test_summary_lists_top_items(self):
        from predictron_engine.synthesis.engine import DecisionSynthesisEngine
        from tests.engine.test_synthesis.conftest import (
            make_features,
        )

        engine = DecisionSynthesisEngine()
        confidence = make_confidence()
        draft = engine.synthesize(
            features=make_features(runway_months=3),
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[make_recommendation("Request audited financials.")],
            readiness=None,
            decision=make_decision(),
            decision_confidence=confidence,
            calibration_summary=None,
        )
        summary, points = build_executive_summary(
            make_decision(), make_readiness(gaps=["No churn data"]), draft
        )
        assert "Principal risks" in summary
        assert any(p.startswith("Risk:") for p in points)
        assert any(p.startswith("Recommendation #1:") for p in points)
        assert any(p.startswith("Next action:") for p in points)

    def test_key_points_include_next_action(self):
        synthesis = DecisionSynthesis(recommended_action="Collect more evidence.")
        _, points = build_executive_summary(make_decision(), None, synthesis)
        assert "Next action: Collect more evidence." in points


class TestFallbackAndDeterminism:
    def test_missing_decision_returns_fallback_text(self):
        summary, points = build_executive_summary(
            None, None, DecisionSynthesis()
        )
        assert "no investment decision" in summary
        assert points == []

    def test_output_is_deterministic(self):
        synthesis = DecisionSynthesis(
            overall_confidence=0.5,
            uncertainty_score=0.5,
            confidence_level="medium",
            recommended_action="Act.",
        )
        one = build_executive_summary(make_decision(), None, synthesis)
        two = build_executive_summary(make_decision(), None, synthesis)
        assert one == two
