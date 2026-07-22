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
  - Maintains the architectural invariant that no rule depends on a derived field
"""

from __future__ import annotations

from predictron_engine.extraction.derived.engine import DerivedMetricsEngine, _METRIC_TO_FIELD
from predictron_engine.extraction.derived.models import DerivedMetricLog
from predictron_engine.extraction.derived.rules import (
    ALL_INFERENCE_RULES,
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


class TestNoCircularDerivations:
    """Architectural invariant: no inference rule depends on a derived field.

    The DerivedMetricsEngine applies rules in a single pass.  Each rule
    reads from the *original* ExtractedFeatures and writes to a separate
    enriched copy.  For this guarantee to hold:

    1. Every rule must declare source_fields that are canonical source
       fields — fields populated by domain extractors, not by the
       inference engine itself.
    2. No rule may declare as a source field a field that is *exclusively*
       a derived output (i.e., a field that no extractor ever populates).
    3. Fields that can be *both* extracted and derived (arr_usd from MRR,
       mrr_usd from ARR, runway_months from funding+burn) are safe
       because the engine only writes them when they are None.

    If any rule violates this invariant the DerivedMetricsEngine could
    silently produce wrong values when it depends on output written by a
    prior rule in the same pass — a correctness bug that is difficult to
    detect in production.

    This test class exists to catch such regressions immediately.
    """

    # Canonical source fields: populated by domain extractors.
    # A rule may only declare these as source_fields.
    _CANONICAL_SOURCE_FIELDS: frozenset[str] = frozenset({
        "mrr_usd",
        "arr_usd",
        "team_size_numeric",
        "funding_amount_usd",
        "burn_rate_usd",
        "cac_usd",
        "ltv_usd",
        "customer_count",
    })

    # Fields that are exclusively derived outputs.  No rule may read
    # from these.  If a rule needs revenue_per_employee, it must
    # recompute it from arr_usd / team_size_numeric.
    _EXCLUSIVELY_DERIVED_FIELDS: frozenset[str] = frozenset({
        "revenue_per_employee_usd",
        "funding_efficiency_ratio",
        "burn_multiple",
        "ltv_cac_ratio",
        "acv_per_customer_usd",
    })

    def test_all_rules_only_declare_canonical_source_fields(self) -> None:
        """Every rule's source_fields must be a subset of canonical source fields.

        If this test fails, a rule is reading from a field that is produced
        by the inference engine rather than by domain extraction.  This
        breaks the single-pass immutability guarantee.
        """
        for rule_name, rule_fn, metric_name in ALL_INFERENCE_RULES:
            features = ExtractedFeatures(
                mrr_usd=50_000.0,
                arr_usd=600_000.0,
                team_size_numeric=20,
                funding_amount_usd=5_000_000.0,
                burn_rate_usd=200_000.0,
                cac_usd=2_000.0,
                ltv_usd=10_000.0,
                customer_count=100,
            )
            logs = rule_fn(features)
            for log_entry in logs:
                for field in log_entry.source_fields:
                    assert field in self._CANONICAL_SOURCE_FIELDS, (
                        f"Rule '{rule_name}' declares source field "
                        f"'{field}' which is not a canonical source field. "
                        f"Rules must only read from domain-extracted fields "
                        f"to prevent circular derivations."
                    )

    def test_no_rule_reads_exclusively_derived_fields(self) -> None:
        """No rule may declare an exclusively-derived field as a source.

        Fields like revenue_per_employee_usd, funding_efficiency_ratio,
        burn_multiple, ltv_cac_ratio, and acv_per_customer_usd are only
        ever produced by inference rules.  If a rule reads from these it
        creates a hidden dependency on a prior rule's output.
        """
        for rule_name, rule_fn, metric_name in ALL_INFERENCE_RULES:
            features = ExtractedFeatures(
                mrr_usd=50_000.0,
                arr_usd=600_000.0,
                team_size_numeric=20,
                funding_amount_usd=5_000_000.0,
                burn_rate_usd=200_000.0,
                cac_usd=2_000.0,
                ltv_usd=10_000.0,
                customer_count=100,
            )
            logs = rule_fn(features)
            for log_entry in logs:
                overlap = set(log_entry.source_fields) & self._EXCLUSIVELY_DERIVED_FIELDS
                assert not overlap, (
                    f"Rule '{rule_name}' reads from exclusively-derived "
                    f"field(s) {overlap}. Rules must recompute ratios from "
                    f"canonical source fields to avoid circular derivations."
                )

    def test_engine_mapping_outputs_are_derived_fields(self) -> None:
        """Every target field in _METRIC_TO_FIELD must be a known derived field.

        This verifies that the engine's write targets are consistent with
        the set of fields the engine is responsible for populating.
        """
        all_known_derived = self._EXCLUSIVELY_DERIVED_FIELDS | {
            "arr_usd",
            "mrr_usd",
            "runway_months",
        }
        for metric_name, target_field in _METRIC_TO_FIELD.items():
            assert target_field in all_known_derived, (
                f"_METRIC_TO_FIELD maps '{metric_name}' to "
                f"'{target_field}' which is not a recognized derived field. "
                f"Update _EXCLUSIVELY_DERIVED_FIELDS or the mapping."
            )

    def test_rules_read_from_original_not_enriched_features(self) -> None:
        """Verify that a rule cannot observe values written by a prior rule.

        This is the runtime manifestation of the architectural invariant.
        We populate only source fields needed for one rule, run the full
        engine, and confirm that a dependent rule does not see values it
        could only obtain from a prior rule's output.

        Concretely: if we provide MRR but not ARR, the engine should
        derive ARR from MRR.  But derive_revenue_per_employee should NOT
        fire because it reads the original features where ARR is None —
        even though the engine just computed ARR.
        """
        features = ExtractedFeatures(
            mrr_usd=100_000.0,
            team_size_numeric=10,
        )
        engine = DerivedMetricsEngine()
        enriched, logs = engine.derive(features)

        metric_names = {log.metric_name for log in logs}

        # ARR should be derived from MRR
        assert "arr_from_mrr" in metric_names
        assert enriched.arr_usd == 1_200_000.0

        # But revenue_per_employee should NOT be derived because the engine
        # reads the *original* features where arr_usd was None.
        assert "revenue_per_employee" not in metric_names
        assert enriched.revenue_per_employee_usd is None

    def test_no_circular_chain_possible(self) -> None:
        """Confirm no rule reads from an exclusively-derived field's output.

        Fields like arr_usd and mrr_usd are dual-purpose — they can be
        populated by domain extractors OR derived by inference rules.
        This is safe because the engine only derives them when the field
        is None.  However, exclusively-derived fields (revenue_per_employee_usd,
        funding_efficiency_ratio, burn_multiple, ltv_cac_ratio,
        acv_per_customer_usd) are only ever produced by inference rules.

        If any rule reads from one of these exclusively-derived fields,
        it creates a hidden dependency on a prior rule's output — a
        correctness bug that breaks single-pass immutability.
        """
        rule_fn_map = {t[0]: t[1] for t in ALL_INFERENCE_RULES}

        for output_rule_name, _, output_metric in ALL_INFERENCE_RULES:
            output_field = _METRIC_TO_FIELD.get(output_metric)
            if output_field is None:
                continue
            if output_field not in self._EXCLUSIVELY_DERIVED_FIELDS:
                continue

            for input_rule_name, _, _ in ALL_INFERENCE_RULES:
                if input_rule_name == output_rule_name:
                    continue
                features = ExtractedFeatures(
                    mrr_usd=50_000.0,
                    arr_usd=600_000.0,
                    team_size_numeric=20,
                    funding_amount_usd=5_000_000.0,
                    burn_rate_usd=200_000.0,
                    cac_usd=2_000.0,
                    ltv_usd=10_000.0,
                    customer_count=100,
                )
                logs = rule_fn_map[input_rule_name](features)
                for log_entry in logs:
                    assert output_field not in log_entry.source_fields, (
                        f"Rule '{input_rule_name}' reads from "
                        f"'{output_field}' which is produced by "
                        f"'{output_rule_name}'. This creates a circular "
                        f"derivation chain."
                    )
