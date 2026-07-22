"""Tests for the Deterministic Inference Engine (Sprint 17).

Tests validate that the DerivedMetricsEngine correctly:
  - Derives ARR from MRR (and vice versa)
  - Computes revenue per employee
  - Computes LTV/CAC ratio
  - Computes funding efficiency
  - Computes burn multiple
  - Computes runway from cash and burn
  - Does NOT overwrite explicitly extracted values
  - Produces correct traceability logs
  - Handles edge cases (zero, negative, missing values)
"""

from __future__ import annotations

from predictron_engine.extraction.derived.engine import DerivedMetricsEngine
from predictron_engine.extraction.derived.models import DerivedMetricLog
from predictron_engine.extraction.derived.rules import (
    derive_arr_from_mrr,
    derive_arr_per_customer,
    derive_burn_multiple,
    derive_funding_efficiency,
    derive_ltv_cac_ratio,
    derive_mrr_from_arr,
    derive_revenue_per_employee,
    derive_runway_from_cash_and_burn,
)
from predictron_engine.models.extracted_features import ExtractedFeatures


class TestDeriveArrFromMrr:
    """Tests for ARR derivation from MRR."""

    def test_derives_arr_from_mrr(self) -> None:
        features = ExtractedFeatures(mrr_usd=50_000.0)
        logs = derive_arr_from_mrr(features)
        assert len(logs) == 1
        assert logs[0].metric_name == "arr_from_mrr"
        assert logs[0].derived_value == 600_000.0
        assert logs[0].formula == "mrr_usd * 12"
        assert logs[0].confidence == 0.9
        assert "mrr_usd" in logs[0].source_fields

    def test_does_not_overwrite_existing_arr(self) -> None:
        features = ExtractedFeatures(mrr_usd=50_000.0, arr_usd=700_000.0)
        logs = derive_arr_from_mrr(features)
        assert len(logs) == 0

    def test_returns_empty_when_no_mrr(self) -> None:
        features = ExtractedFeatures()
        logs = derive_arr_from_mrr(features)
        assert len(logs) == 0

    def testHandles_large_mrr(self) -> None:
        features = ExtractedFeatures(mrr_usd=5_000_000.0)
        logs = derive_arr_from_mrr(features)
        assert len(logs) == 1
        assert logs[0].derived_value == 60_000_000.0


class TestDeriveMrrFromArr:
    """Tests for MRR derivation from ARR."""

    def test_derives_mrr_from_arr(self) -> None:
        features = ExtractedFeatures(arr_usd=1_200_000.0)
        logs = derive_mrr_from_arr(features)
        assert len(logs) == 1
        assert logs[0].metric_name == "mrr_from_arr"
        assert logs[0].derived_value == 100_000.0
        assert logs[0].formula == "arr_usd / 12"

    def test_does_not_overwrite_existing_mrr(self) -> None:
        features = ExtractedFeatures(arr_usd=1_200_000.0, mrr_usd=90_000.0)
        logs = derive_mrr_from_arr(features)
        assert len(logs) == 0

    def test_returns_empty_when_no_arr(self) -> None:
        features = ExtractedFeatures()
        logs = derive_mrr_from_arr(features)
        assert len(logs) == 0


class TestDeriveRevenuePerEmployee:
    """Tests for revenue per employee derivation."""

    def test_derives_rpe(self) -> None:
        features = ExtractedFeatures(
            arr_usd=6_000_000.0, team_size_numeric=20,
        )
        logs = derive_revenue_per_employee(features)
        assert len(logs) == 1
        assert logs[0].metric_name == "revenue_per_employee"
        assert logs[0].derived_value == 300_000.0
        assert logs[0].confidence == 0.85

    def test_returns_empty_without_arr(self) -> None:
        features = ExtractedFeatures(team_size_numeric=20)
        logs = derive_revenue_per_employee(features)
        assert len(logs) == 0

    def test_returns_empty_without_team(self) -> None:
        features = ExtractedFeatures(arr_usd=6_000_000.0)
        logs = derive_revenue_per_employee(features)
        assert len(logs) == 0

    def test_returns_empty_with_zero_team(self) -> None:
        features = ExtractedFeatures(arr_usd=6_000_000.0, team_size_numeric=0)
        logs = derive_revenue_per_employee(features)
        assert len(logs) == 0

    def test_returns_empty_with_negative_team(self) -> None:
        features = ExtractedFeatures(arr_usd=6_000_000.0, team_size_numeric=-5)
        logs = derive_revenue_per_employee(features)
        assert len(logs) == 0


class TestDeriveLtvCacRatio:
    """Tests for LTV/CAC ratio derivation."""

    def test_derives_ratio(self) -> None:
        features = ExtractedFeatures(ltv_usd=18_000.0, cac_usd=2_400.0)
        logs = derive_ltv_cac_ratio(features)
        assert len(logs) == 1
        assert logs[0].metric_name == "ltv_cac_ratio"
        assert abs(logs[0].derived_value - 7.5) < 0.01
        assert logs[0].formula == "ltv_usd / cac_usd"

    def test_returns_empty_without_ltv(self) -> None:
        features = ExtractedFeatures(cac_usd=2_400.0)
        logs = derive_ltv_cac_ratio(features)
        assert len(logs) == 0

    def test_returns_empty_without_cac(self) -> None:
        features = ExtractedFeatures(ltv_usd=18_000.0)
        logs = derive_ltv_cac_ratio(features)
        assert len(logs) == 0

    def test_returns_empty_with_zero_cac(self) -> None:
        features = ExtractedFeatures(ltv_usd=18_000.0, cac_usd=0.0)
        logs = derive_ltv_cac_ratio(features)
        assert len(logs) == 0

    def test_returns_empty_with_zero_ltv(self) -> None:
        features = ExtractedFeatures(ltv_usd=0.0, cac_usd=2_400.0)
        logs = derive_ltv_cac_ratio(features)
        assert len(logs) == 0


class TestDeriveFundingEfficiency:
    """Tests for funding efficiency derivation."""

    def test_derives_efficiency(self) -> None:
        features = ExtractedFeatures(
            arr_usd=3_200_000.0, funding_amount_usd=1_500_000.0,
        )
        logs = derive_funding_efficiency(features)
        assert len(logs) == 1
        assert logs[0].metric_name == "funding_efficiency"
        assert abs(logs[0].derived_value - 2.1333) < 0.01
        assert logs[0].confidence == 0.8

    def test_returns_empty_without_arr(self) -> None:
        features = ExtractedFeatures(funding_amount_usd=1_500_000.0)
        logs = derive_funding_efficiency(features)
        assert len(logs) == 0

    def test_returns_empty_without_funding(self) -> None:
        features = ExtractedFeatures(arr_usd=3_200_000.0)
        logs = derive_funding_efficiency(features)
        assert len(logs) == 0

    def test_returns_empty_with_zero_arr(self) -> None:
        features = ExtractedFeatures(arr_usd=0.0, funding_amount_usd=1_500_000.0)
        logs = derive_funding_efficiency(features)
        assert len(logs) == 0

    def test_returns_empty_with_zero_funding(self) -> None:
        features = ExtractedFeatures(arr_usd=3_200_000.0, funding_amount_usd=0.0)
        logs = derive_funding_efficiency(features)
        assert len(logs) == 0


class TestDeriveBurnMultiple:
    """Tests for burn multiple derivation."""

    def test_derives_burn_multiple(self) -> None:
        features = ExtractedFeatures(
            burn_rate_usd=200_000.0, arr_usd=2_400_000.0,
        )
        logs = derive_burn_multiple(features)
        assert len(logs) == 1
        assert logs[0].metric_name == "burn_multiple"
        assert abs(logs[0].derived_value - 1.0) < 0.01
        assert "burn_rate_usd * 12" in logs[0].formula

    def test_returns_empty_without_burn(self) -> None:
        features = ExtractedFeatures(arr_usd=2_400_000.0)
        logs = derive_burn_multiple(features)
        assert len(logs) == 0

    def test_returns_empty_without_arr(self) -> None:
        features = ExtractedFeatures(burn_rate_usd=200_000.0)
        logs = derive_burn_multiple(features)
        assert len(logs) == 0

    def test_returns_empty_with_zero_burn(self) -> None:
        features = ExtractedFeatures(burn_rate_usd=0.0, arr_usd=2_400_000.0)
        logs = derive_burn_multiple(features)
        assert len(logs) == 0

    def test_returns_empty_with_zero_arr(self) -> None:
        features = ExtractedFeatures(burn_rate_usd=200_000.0, arr_usd=0.0)
        logs = derive_burn_multiple(features)
        assert len(logs) == 0

    def test_returns_empty_for_unreasonable_ratio(self) -> None:
        features = ExtractedFeatures(burn_rate_usd=10_000_000.0, arr_usd=1_000.0)
        logs = derive_burn_multiple(features)
        assert len(logs) == 0


class TestDeriveRunwayFromCashAndBurn:
    """Tests for runway derivation."""

    def test_derives_runway(self) -> None:
        features = ExtractedFeatures(
            funding_amount_usd=6_000_000.0,
            burn_rate_usd=350_000.0,
        )
        logs = derive_runway_from_cash_and_burn(features)
        assert len(logs) == 1
        assert logs[0].metric_name == "runway_from_cash_and_burn"
        assert logs[0].derived_value == 17.0
        assert logs[0].confidence == 0.6

    def test_does_not_overwrite_existing_runway(self) -> None:
        features = ExtractedFeatures(
            funding_amount_usd=6_000_000.0,
            burn_rate_usd=350_000.0,
            runway_months=8,
        )
        logs = derive_runway_from_cash_and_burn(features)
        assert len(logs) == 0

    def test_returns_empty_without_burn(self) -> None:
        features = ExtractedFeatures(funding_amount_usd=6_000_000.0)
        logs = derive_runway_from_cash_and_burn(features)
        assert len(logs) == 0

    def test_returns_empty_without_funding(self) -> None:
        features = ExtractedFeatures(burn_rate_usd=350_000.0)
        logs = derive_runway_from_cash_and_burn(features)
        assert len(logs) == 0

    def test_returns_empty_with_zero_burn(self) -> None:
        features = ExtractedFeatures(
            funding_amount_usd=6_000_000.0, burn_rate_usd=0.0,
        )
        logs = derive_runway_from_cash_and_burn(features)
        assert len(logs) == 0

    def test_returns_empty_for_unreasonable_runway(self) -> None:
        features = ExtractedFeatures(
            funding_amount_usd=6_000_000.0, burn_rate_usd=1_000.0,
        )
        logs = derive_runway_from_cash_and_burn(features)
        assert len(logs) == 0


class TestDeriveArrPerCustomer:
    """Tests for ACV per customer derivation."""

    def test_derives_acv(self) -> None:
        features = ExtractedFeatures(
            arr_usd=3_200_000.0, customer_count=200,
        )
        logs = derive_arr_per_customer(features)
        assert len(logs) == 1
        assert logs[0].metric_name == "acv_per_customer"
        assert logs[0].derived_value == 16_000.0
        assert logs[0].formula == "arr_usd / customer_count"

    def test_returns_empty_without_arr(self) -> None:
        features = ExtractedFeatures(customer_count=200)
        logs = derive_arr_per_customer(features)
        assert len(logs) == 0

    def test_returns_empty_without_customers(self) -> None:
        features = ExtractedFeatures(arr_usd=3_200_000.0)
        logs = derive_arr_per_customer(features)
        assert len(logs) == 0

    def test_returns_empty_with_zero_customers(self) -> None:
        features = ExtractedFeatures(arr_usd=3_200_000.0, customer_count=0)
        logs = derive_arr_per_customer(features)
        assert len(logs) == 0

    def test_returns_empty_with_zero_arr(self) -> None:
        features = ExtractedFeatures(arr_usd=0.0, customer_count=200)
        logs = derive_arr_per_customer(features)
        assert len(logs) == 0


class TestDerivedMetricsEngine:
    """Integration tests for the full DerivedMetricsEngine."""

    def test_engine_produces_multiple_derivations(self) -> None:
        features = ExtractedFeatures(
            mrr_usd=100_000.0,
            arr_usd=1_200_000.0,
            team_size_numeric=12,
            funding_amount_usd=5_000_000.0,
            burn_rate_usd=200_000.0,
            cac_usd=1_500.0,
            ltv_usd=9_000.0,
            customer_count=50,
        )
        engine = DerivedMetricsEngine()
        enriched, logs = engine.derive(features)

        # Should derive: revenue_per_employee, funding_efficiency,
        # burn_multiple, ltv_cac_ratio, acv_per_customer
        # Should NOT derive ARR-from-MRR (ARR already set)
        # Should NOT derive runway (runway not set but let's check)
        assert len(logs) >= 4

        metric_names = {log.metric_name for log in logs}
        assert "revenue_per_employee" in metric_names
        assert "funding_efficiency" in metric_names
        assert "ltv_cac_ratio" in metric_names
        assert "acv_per_customer" in metric_names

        # ARR should NOT be overwritten (already set)
        assert enriched.arr_usd == 1_200_000.0

    def test_engine_derives_arr_from_mrr_when_no_arr(self) -> None:
        features = ExtractedFeatures(mrr_usd=50_000.0)
        engine = DerivedMetricsEngine()
        enriched, logs = engine.derive(features)

        assert enriched.arr_usd == 600_000.0
        arr_logs = [l for l in logs if l.metric_name == "arr_from_mrr"]
        assert len(arr_logs) == 1

    def test_engine_populates_derived_metrics_log(self) -> None:
        features = ExtractedFeatures(mrr_usd=50_000.0)
        engine = DerivedMetricsEngine()
        enriched, logs = engine.derive(features)

        assert len(enriched.derived_metrics_log) == 1
        assert enriched.derived_metrics_log[0]["metric_name"] == "arr_from_mrr"

    def test_engine_handles_empty_features(self) -> None:
        features = ExtractedFeatures()
        engine = DerivedMetricsEngine()
        enriched, logs = engine.derive(features)
        assert len(logs) == 0
        assert enriched.industry is None

    def test_engine_preserves_all_existing_fields(self) -> None:
        features = ExtractedFeatures(
            industry="fintech",
            business_model="saas",
            customer_type="b2b",
            arr_usd=5_000_000.0,
            team_size_numeric=25,
        )
        engine = DerivedMetricsEngine()
        enriched, _ = engine.derive(features)

        assert enriched.industry == "fintech"
        assert enriched.business_model == "saas"
        assert enriched.customer_type == "b2b"
        assert enriched.arr_usd == 5_000_000.0
        assert enriched.team_size_numeric == 25
        assert enriched.revenue_per_employee_usd == 200_000.0

    def test_derivation_log_has_complete_provenance(self) -> None:
        features = ExtractedFeatures(mrr_usd=83_000.0)
        engine = DerivedMetricsEngine()
        enriched, logs = engine.derive(features)

        assert len(logs) == 1
        log = logs[0]
        assert log.metric_name == "arr_from_mrr"
        assert log.derived_value == 996_000.0
        assert log.source_fields == ["mrr_usd"]
        assert log.source_values == [83_000.0]
        assert log.formula == "mrr_usd * 12"
        assert len(log.explanation) > 0
        assert 0.0 <= log.confidence <= 1.0


class TestDerivedMetricLogModel:
    """Tests for the DerivedMetricLog Pydantic model."""

    def test_model_creation(self) -> None:
        log = DerivedMetricLog(
            metric_name="test_metric",
            derived_value=42.0,
            source_fields=["field_a"],
            source_values=[10.0],
            formula="field_a * 4.2",
            explanation="Test explanation",
            confidence=0.85,
        )
        assert log.metric_name == "test_metric"
        assert log.derived_value == 42.0
        assert log.confidence == 0.85

    def test_model_serialization(self) -> None:
        log = DerivedMetricLog(
            metric_name="test",
            derived_value=100.0,
            source_fields=["a", "b"],
            source_values=[50.0, 2.0],
            formula="a * b",
            explanation="Multiplied",
            confidence=0.9,
        )
        d = log.model_dump()
        assert d["metric_name"] == "test"
        assert d["source_fields"] == ["a", "b"]
        assert isinstance(d, dict)
