"""Tests for Sprint 8 — Adaptive Reasoning Budget.

Validates that the adaptive budget module:
  - Produces deterministic budget allocations
  - Reduces computation for straightforward cases
  - Increases computation for uncertain/complex cases
  - Preserves determinism across runs
"""

from __future__ import annotations

import pytest

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.reasoning.adaptive_budget import (
    BudgetReport,
    ReasoningBudget,
    compute_reasoning_budget,
    should_skip_rule,
)


class TestReasoningBudget:
    """Tests for ReasoningBudget data class."""

    def test_default_budget_is_full(self) -> None:
        budget = ReasoningBudget()
        assert budget.budget_fraction == 1.0
        assert budget.skipped_rules == ()
        assert budget.savings_fraction == 0.0

    def test_reduced_budget_has_savings(self) -> None:
        budget = ReasoningBudget(
            budget_fraction=0.7,
            max_rules=11,
            skipped_rules=("QuantitativeCrossSignalRule",),
        )
        assert budget.savings_fraction == pytest.approx(0.3, abs=0.01)
        assert budget.rules_executed == 10

    def test_budget_report_adherence(self) -> None:
        budget = ReasoningBudget(budget_fraction=0.7)
        report = BudgetReport(
            budget=budget,
            total_rules=11,
            rules_executed=8,
            rules_skipped=3,
        )
        assert report.actual_savings_fraction == pytest.approx(3 / 11, abs=0.01)
        assert report.budget_adherence > 0.5


class TestComputeReasoningBudget:
    """Tests for compute_reasoning_budget function."""

    def test_high_completeness_reduces_budget(self) -> None:
        features = ExtractedFeatures(
            industry="SaaS",
            business_model="subscription",
            funding_stage="series_a",
            has_revenue=True,
            data_completeness=0.90,
            founder_profile_count=3,
            technology_stack=["Python", "React"],
        )
        evidence = [
            EvidenceItem(domain="industry", category="market", statement="Evidence 1", source="s1"),
            EvidenceItem(domain="industry", category="market", statement="Evidence 2", source="s2"),
            EvidenceItem(domain="industry", category="market", statement="Evidence 3", source="s3"),
            EvidenceItem(
                domain="business_model", category="model",
                statement="Evidence 4", source="s4",
            ),
        ]
        budget = compute_reasoning_budget(features, evidence)
        assert budget.budget_fraction < 1.0
        assert budget.savings_fraction > 0.0
        assert len(budget.skipped_rules) > 0

    def test_low_completeness_uses_full_budget(self) -> None:
        features = ExtractedFeatures(
            data_completeness=0.20,
            founder_profile_count=0,
        )
        evidence = [
            EvidenceItem(domain="test", category="test", statement="Evidence 1", source="s1"),
        ]
        budget = compute_reasoning_budget(features, evidence)
        assert budget.budget_fraction >= 0.85

    def test_no_evidence_uses_elevated_budget(self) -> None:
        features = ExtractedFeatures(
            industry="SaaS",
            data_completeness=0.50,
        )
        evidence: list[EvidenceItem] = []
        budget = compute_reasoning_budget(features, evidence)
        assert budget.budget_fraction >= 0.70

    def test_budget_is_deterministic(self) -> None:
        features = ExtractedFeatures(
            industry="SaaS",
            business_model="subscription",
            data_completeness=0.75,
            founder_profile_count=2,
        )
        evidence = [
            EvidenceItem(domain="industry", category="m", statement="S", source="s"),
        ]
        b1 = compute_reasoning_budget(features, evidence)
        b2 = compute_reasoning_budget(features, evidence)
        assert b1.budget_fraction == b2.budget_fraction
        assert b1.skipped_rules == b2.skipped_rules
        assert b1.confidence_estimate == b2.confidence_estimate
        assert b1.impact_score == b2.impact_score

    def test_budget_rationale_is_nonempty(self) -> None:
        features = ExtractedFeatures(data_completeness=0.60)
        budget = compute_reasoning_budget(features, [])
        assert budget.rationale != ""


class TestShouldSkipRule:
    """Tests for should_skip_rule function."""

    def test_rule_in_skip_list_returns_true(self) -> None:
        budget = ReasoningBudget(
            skipped_rules=("RuleA", "RuleB"),
        )
        assert should_skip_rule("RuleA", budget) is True
        assert should_skip_rule("RuleB", budget) is True

    def test_rule_not_in_skip_list_returns_false(self) -> None:
        budget = ReasoningBudget(
            skipped_rules=("RuleA",),
        )
        assert should_skip_rule("RuleB", budget) is False

    def test_empty_skip_list_skips_nothing(self) -> None:
        budget = ReasoningBudget(skipped_rules=())
        assert should_skip_rule("AnyRule", budget) is False


class TestBudgetReport:
    """Tests for BudgetReport metrics."""

    def test_report_serialization(self) -> None:
        budget = ReasoningBudget(budget_fraction=0.70)
        report = BudgetReport(
            budget=budget,
            total_rules=11,
            rules_executed=8,
            rules_skipped=3,
        )
        d = report.to_dict()
        assert "budget_fraction" in d
        assert "savings_fraction" in d
        assert "actual_savings_fraction" in d
        assert "budget_adherence" in d
        assert d["rules_executed"] == 8
        assert d["rules_skipped"] == 3

    def test_zero_total_rules_handled(self) -> None:
        report = BudgetReport(total_rules=0)
        assert report.actual_savings_fraction == 0.0
        assert report.budget_adherence == 1.0
