"""Comprehensive tests for QuantitativeCrossSignalRule.

Tests cover:
- Reinforcing scenarios (positive metric relationships)
- Conflicting scenarios (negative/inconsistent metric relationships)
- Edge cases (missing metrics, boundary values, zero values)
- Partial information (only some metrics available)
- Backward compatibility (rule produces valid Observation objects)
"""

from __future__ import annotations

import pytest

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation
from predictron_engine.reasoning.rules.quantitative_cross_signal import (
    QuantitativeCrossSignalRule,
)


@pytest.fixture
def rule() -> QuantitativeCrossSignalRule:
    return QuantitativeCrossSignalRule()


# ======================================================================
# Helpers
# ======================================================================


def _obs_categories(rule_result: list[Observation]) -> set[str]:
    return {o.category for o in rule_result}


def _obs_statements(rule_result: list[Observation]) -> list[str]:
    return [o.statement for o in rule_result]


# ======================================================================
# Empty / Missing metrics
# ======================================================================


class TestEmptyFeatures:
    def test_empty_features_produces_no_observations(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures()
        result = rule.evaluate(features, [])
        assert result == []

    def test_partial_metrics_no_crash(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=5_000_000,
            growth_rate_pct=50.0,
        )
        result = rule.evaluate(features, [])
        assert isinstance(result, list)

    def test_all_none_metrics_produces_no_observations(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=None,
            customer_count=None,
            team_size_numeric=None,
            burn_rate_usd=None,
            runway_months=None,
            nrr_pct=None,
            churn_rate_pct=None,
            funding_amount_usd=None,
            valuation_usd=None,
            growth_rate_pct=None,
        )
        result = rule.evaluate(features, [])
        assert result == []


# ======================================================================
# Revenue per customer
# ======================================================================


class TestRevPerCustomer:
    def test_enterprise_rpc(self, rule: QuantitativeCrossSignalRule) -> None:
        features = ExtractedFeatures(
            arr_usd=5_000_000,
            customer_count=20,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "rev_per_customer_enterprise" in cats

    def test_mid_market_rpc(self, rule: QuantitativeCrossSignalRule) -> None:
        features = ExtractedFeatures(
            arr_usd=5_000_000,
            customer_count=200,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "rev_per_customer_mid_market" in cats

    def test_smb_rpc(self, rule: QuantitativeCrossSignalRule) -> None:
        features = ExtractedFeatures(
            arr_usd=500_000,
            customer_count=200,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "rev_per_customer_smb" in cats

    def test_zero_customers_skipped(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=5_000_000,
            customer_count=0,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "rev_per_customer_enterprise" not in cats


# ======================================================================
# Revenue per employee
# ======================================================================


class TestRevPerEmployee:
    def test_exceptional_rpe(self, rule: QuantitativeCrossSignalRule) -> None:
        features = ExtractedFeatures(
            arr_usd=20_000_000,
            team_size_numeric=15,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "rev_per_employee_exceptional" in cats

    def test_strong_rpe(self, rule: QuantitativeCrossSignalRule) -> None:
        features = ExtractedFeatures(
            arr_usd=10_000_000,
            team_size_numeric=40,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "rev_per_employee_strong" in cats

    def test_low_rpe(self, rule: QuantitativeCrossSignalRule) -> None:
        features = ExtractedFeatures(
            arr_usd=1_000_000,
            team_size_numeric=100,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "rev_per_employee_low" in cats

    def test_zero_team_skipped(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=5_000_000,
            team_size_numeric=0,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "rev_per_employee_low" not in cats


# ======================================================================
# Growth vs Burn
# ======================================================================


class TestGrowthVsBurn:
    def test_excellent_growth_burn(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            growth_rate_pct=100.0,
            burn_rate_usd=500_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "growth_burn_excellent" in cats

    def test_concerning_growth_burn(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            growth_rate_pct=10.0,
            burn_rate_usd=5_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "growth_burn_concerning" in cats

    def test_zero_burn_skipped(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            growth_rate_pct=50.0,
            burn_rate_usd=0,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "growth_burn_excellent" not in cats
        assert "growth_burn_concerning" not in cats


# ======================================================================
# Burn vs Runway
# ======================================================================


class TestBurnVsRunway:
    def test_critical_burn_runway(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            burn_rate_usd=3_000_000,
            runway_months=3,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "burn_runway_critical" in cats

    def test_warning_burn_runway(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            burn_rate_usd=2_000_000,
            runway_months=10,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "burn_runway_warning" in cats

    def test_sustainable_burn_runway(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            burn_rate_usd=1_000_000,
            runway_months=24,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "burn_runway_sustainable" in cats


# ======================================================================
# NRR / Churn consistency
# ======================================================================


class TestNrrChurnConsistency:
    def test_reinforcing_nrr_churn(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            nrr_pct=120.0,
            churn_rate_pct=2.0,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "nrr_churn_reinforcing" in cats

    def test_conflict_nrr_churn(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            nrr_pct=80.0,
            churn_rate_pct=15.0,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "nrr_churn_conflict" in cats

    def test_masking_nrr_churn(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            nrr_pct=105.0,
            churn_rate_pct=12.0,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "nrr_churn_masking" in cats

    def test_downgrade_nrr_churn(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            nrr_pct=85.0,
            churn_rate_pct=3.0,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "nrr_churn_downgrade" in cats

    def test_no_conflict_when_both_absent(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            nrr_pct=100.0,
            churn_rate_pct=5.0,
        )
        result = rule.evaluate(features, [])
        nrr_cats = [
            o for o in result if o.category.startswith("nrr_churn_")
        ]
        assert len(nrr_cats) == 0


# ======================================================================
# Funding / ARR efficiency
# ======================================================================


class TestFundingArrEfficiency:
    def test_strong_funding_efficiency(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=15_000_000,
            funding_amount_usd=10_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "funding_efficiency_strong" in cats

    def test_weak_funding_efficiency(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=500_000,
            funding_amount_usd=50_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "funding_efficiency_weak" in cats

    def test_zero_funding_skipped(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=5_000_000,
            funding_amount_usd=0,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "funding_efficiency_strong" not in cats


# ======================================================================
# Valuation / ARR multiple
# ======================================================================


class TestValuationArrMultiple:
    def test_high_multiple(self, rule: QuantitativeCrossSignalRule) -> None:
        features = ExtractedFeatures(
            valuation_usd=500_000_000,
            arr_usd=10_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "valuation_arr_high" in cats

    def test_low_multiple(self, rule: QuantitativeCrossSignalRule) -> None:
        features = ExtractedFeatures(
            valuation_usd=40_000_000,
            arr_usd=5_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "valuation_arr_low" in cats

    def test_zero_arr_skipped(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            valuation_usd=100_000_000,
            arr_usd=0,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "valuation_arr_high" not in cats


# ======================================================================
# Funding / Valuation consistency
# ======================================================================


class TestFundingValuationConsistency:
    def test_inconsistent_valuation_below_funding(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            funding_amount_usd=50_000_000,
            valuation_usd=30_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "funding_valuation_inconsistent" in cats

    def test_high_creation(self, rule: QuantitativeCrossSignalRule) -> None:
        features = ExtractedFeatures(
            funding_amount_usd=5_000_000,
            valuation_usd=200_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "funding_valuation_high_creation" in cats


# ======================================================================
# Customer count vs funding stage
# ======================================================================


class TestCustomerCountStage:
    def test_seed_exceeds_expectations(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            customer_count=300,
            funding_stage="seed",
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "customer_count_stage_exceeds" in cats

    def test_seed_below_expectations(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            customer_count=2,
            funding_stage="seed",
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "customer_count_stage_below" in cats

    def test_series_b_below_expectations(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            customer_count=50,
            funding_stage="series_b",
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "customer_count_stage_below" in cats


# ======================================================================
# Team / Revenue consistency
# ======================================================================


class TestTeamRevenueConsistency:
    def test_inconsistent_large_team_low_revenue(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            team_size_numeric=200,
            arr_usd=500_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "team_revenue_inconsistent" in cats

    def test_efficient_small_team_high_revenue(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            team_size_numeric=15,
            arr_usd=15_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "team_revenue_efficient" in cats


# ======================================================================
# Valuation / Weak traction anomaly
# ======================================================================


class TestValWeakTractionAnomaly:
    def test_high_valuation_weak_arr(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            valuation_usd=200_000_000,
            arr_usd=500_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "valuation_weak_traction" in cats

    def test_no_anomaly_when_arr_strong(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            valuation_usd=200_000_000,
            arr_usd=20_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "valuation_weak_traction" not in cats


# ======================================================================
# High ARR, few customers
# ======================================================================


class TestHighArrFewCustomers:
    def test_concentrated_customers(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=5_000_000,
            customer_count=3,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "high_arr_concentrated_customers" in cats

    def test_no_issue_with_many_customers(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=5_000_000,
            customer_count=50,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "high_arr_concentrated_customers" not in cats


# ======================================================================
# High burn, modest growth
# ======================================================================


class TestHighBurnModestGrowth:
    def test_high_burn_low_growth(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            burn_rate_usd=3_000_000,
            growth_rate_pct=15.0,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "high_burn_low_growth" in cats

    def test_no_issue_with_high_growth(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            burn_rate_usd=3_000_000,
            growth_rate_pct=100.0,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "high_burn_low_growth" not in cats


# ======================================================================
# Strong funding, weak execution
# ======================================================================


class TestStrongFundingWeakExecution:
    def test_funding_without_execution(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            funding_amount_usd=30_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "funding_without_execution" in cats

    def test_no_issue_with_execution(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            funding_amount_usd=30_000_000,
            arr_usd=5_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "funding_without_execution" not in cats


# ======================================================================
# ARR vs annual burn
# ======================================================================


class TestArrVsAnnualBurn:
    def test_arr_exceeds_burn(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=20_000_000,
            burn_rate_usd=1_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "arr_exceeds_burn" in cats

    def test_arr_burn_gap(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=1_000_000,
            burn_rate_usd=3_000_000,
        )
        result = rule.evaluate(features, [])
        cats = _obs_categories(result)
        assert "arr_burn_gap" in cats


# ======================================================================
# Observations are valid Observation objects
# ======================================================================


class TestObservationValidity:
    def test_all_observations_valid(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=10_000_000,
            customer_count=100,
            team_size_numeric=30,
            growth_rate_pct=80.0,
            burn_rate_usd=1_500_000,
            runway_months=18,
            nrr_pct=115.0,
            churn_rate_pct=3.0,
            funding_amount_usd=15_000_000,
            valuation_usd=80_000_000,
            funding_stage="series_a",
        )
        result = rule.evaluate(features, [])
        for obs in result:
            assert isinstance(obs, Observation)
            assert obs.dimension == "investment_thesis"
            assert obs.source_rule == "QuantitativeCrossSignalRule"
            assert 0.0 <= obs.confidence <= 1.0
            assert 0.0 <= obs.importance <= 1.0
            assert len(obs.statement) > 0
            assert len(obs.evidence) > 0

    def test_source_rule_set_on_all_observations(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=5_000_000,
            customer_count=50,
            team_size_numeric=20,
            nrr_pct=120.0,
            churn_rate_pct=2.0,
        )
        result = rule.evaluate(features, [])
        for obs in result:
            assert obs.source_rule == "QuantitativeCrossSignalRule"

    def test_evidence_refs_include_metric_values(
        self, rule: QuantitativeCrossSignalRule
    ) -> None:
        features = ExtractedFeatures(
            arr_usd=5_000_000,
            customer_count=50,
        )
        result = rule.evaluate(features, [])
        rpc_obs = [
            o for o in result
            if o.category.startswith("rev_per_customer")
        ]
        assert len(rpc_obs) == 1
        assert any("arr_usd" in e for e in rpc_obs[0].evidence)
        assert any("customer_count" in e for e in rpc_obs[0].evidence)


# ======================================================================
# Rule name
# ======================================================================


class TestRuleName:
    def test_name(self, rule: QuantitativeCrossSignalRule) -> None:
        assert rule.name == "quantitative_cross_signal"
