"""Tests for deterministic scenario generation (Sprint 6C)."""

from predictron_engine.synthesis.scenarios import generate_scenarios
from tests.engine.test_synthesis.conftest import (
    make_confidence,
    make_decision,
    make_readiness,
)


class TestImproveScenarios:
    def test_data_quality_recovery_scenario(self):
        decision = make_decision(
            composite=63.0, data_quality_modifier=0.9, margin=0.0
        )
        scenarios = generate_scenarios(decision, None, None)
        improve = [s for s in scenarios if s.direction == "improve"]
        assert any(s.title == "Improve data completeness" for s in improve)
        dq_scenario = next(
            s for s in improve if s.title == "Improve data completeness"
        )
        assert dq_scenario.projected_impact is not None
        assert dq_scenario.projected_impact > 0.0

    def test_margin_scenario_present_when_margin_positive(self):
        decision = make_decision(composite=63.0, margin=7.0, next_threshold=70.0)
        scenarios = generate_scenarios(decision, None, None)
        margin_scenarios = [
            s
            for s in scenarios
            if s.direction == "improve" and "next category threshold" in s.condition
        ]
        assert len(margin_scenarios) == 1
        assert "7.0 points" in margin_scenarios[0].condition

    def test_no_margin_scenario_when_already_top(self):
        from predictron_engine.models.report import DecisionCategory

        decision = make_decision(
            composite=95.0,
            margin=0.0,
            next_threshold=0.0,
            category=DecisionCategory.STRONG_INVEST,
        )
        scenarios = generate_scenarios(decision, None, None)
        assert all("threshold of" not in s.condition for s in scenarios)

    def test_conflict_resolution_scenario_uses_existing_counts(self):
        readiness = make_readiness(conflicting_count=2)
        decision = make_decision(composite=60.0, risk_modifier=0.9)
        scenarios = generate_scenarios(decision, None, readiness)
        resolve = [
            s
            for s in scenarios
            if s.direction == "improve" and s.title == "Resolve conflicting signals"
        ]
        assert len(resolve) == 1
        assert "2 conflicting" in resolve[0].condition


class TestWorsenAndInformation:
    def test_worsen_scenario_always_present_with_decision(self):
        scenarios = generate_scenarios(make_decision(), None, None)
        worsen = [s for s in scenarios if s.direction == "worsen"]
        assert len(worsen) == 1
        assert worsen[0].condition

    def test_worsen_plausibility_reflects_uncertainty(self):
        confidence = make_confidence(uncertainty=0.8)
        scenarios = generate_scenarios(make_decision(), confidence, None)
        worsen = next(s for s in scenarios if s.direction == "worsen")
        assert abs(worsen.plausibility - 0.8) < 1e-9

    def test_information_scenarios_from_rationale_and_gaps(self):
        from predictron_engine.models.report import DecisionRationale

        decision = make_decision()
        object.__setattr__(
            decision,
            "rationale",
            DecisionRationale(
                information_that_could_change_decision=[
                    "Audited financials for the last two years",
                ]
            ),
        )
        readiness = make_readiness(gaps=["No churn data", "Single founder"])
        scenarios = generate_scenarios(decision, None, readiness)
        information = [s for s in scenarios if s.direction == "information"]
        conditions = {s.condition for s in information}
        assert "Audited financials for the last two years" in conditions
        assert "No churn data" in conditions
        assert "Single founder" in conditions

    def test_information_scenarios_deduplicate(self):
        from predictron_engine.models.report import DecisionRationale

        decision = make_decision()
        object.__setattr__(
            decision,
            "rationale",
            DecisionRationale(
                information_that_could_change_decision=["Same info"]
            ),
        )
        readiness = make_readiness(gaps=["same info"])
        scenarios = generate_scenarios(decision, None, readiness)
        information = [s for s in scenarios if s.direction == "information"]
        assert len(information) == 1


class TestDeterminismAndBounds:
    def test_no_decision_returns_empty(self):
        assert generate_scenarios(None, None, None) == []

    def test_direction_ordering_improve_first_information_last(self):
        from predictron_engine.models.report import DecisionRationale

        decision = make_decision(composite=55.0, data_quality_modifier=0.8)
        object.__setattr__(
            decision,
            "rationale",
            DecisionRationale(
                information_that_could_change_decision=["More data"]
            ),
        )
        readiness = make_readiness(conflicting_count=1, gaps=["Gap"])
        scenarios = generate_scenarios(decision, make_confidence(), readiness)
        directions = [s.direction for s in scenarios]
        first_info = directions.index("information")
        assert set(directions[:first_info]) <= {"improve", "worsen"}

    def test_plausibility_bounded_in_unit_interval(self):
        decision = make_decision(composite=55.0, data_quality_modifier=0.5)
        readiness = make_readiness(
            conflicting_count=5, gaps=["a", "b", "c", "d"]
        )
        for scenario in generate_scenarios(decision, make_confidence(), readiness):
            assert 0.0 <= scenario.plausibility <= 1.0

    def test_output_is_deterministic(self):
        decision = make_decision(composite=58.0, data_quality_modifier=0.85)
        readiness = make_readiness(conflicting_count=1, gaps=["Gap one"])
        one = generate_scenarios(decision, make_confidence(), readiness)
        two = generate_scenarios(decision, make_confidence(), readiness)
        assert [s.model_dump() for s in one] == [s.model_dump() for s in two]
