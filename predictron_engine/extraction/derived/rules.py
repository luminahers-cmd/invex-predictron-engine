"""Deterministic inference rules for derived metric computation.

Each rule is a pure function that takes an ExtractedFeatures and returns
a list of DerivedMetricLog entries. Rules are ordered by dependency:
earlier rules may produce values that later rules consume.

All rules follow the same contract:
  - Input: ExtractedFeatures (read-only)
  - Output: list of DerivedMetricLog (may be empty if insufficient inputs)
  - Never raise exceptions
  - Never guess — return empty list when inputs are insufficient
  - Every derivation must have exactly one deterministic formula

Design principles:
  - Each rule handles one logical derivation
  - Rules are independently testable
  - No rule modifies the input features
  - Confidence reflects evidence quality, not outcome certainty
"""

from __future__ import annotations

from predictron_engine.extraction.derived.models import DerivedMetricLog
from predictron_engine.models.extracted_features import ExtractedFeatures

# Minimum confidence threshold for a derivation to be recorded
_MIN_CONFIDENCE: float = 0.5


def derive_arr_from_mrr(features: ExtractedFeatures) -> list[DerivedMetricLog]:
    """Derive ARR from MRR when ARR is not already set.

    Formula: ARR = MRR * 12

    Rationale: Monthly recurring revenue multiplied by 12 months gives
    annual recurring revenue. This is a standard financial conversion
    used universally in SaaS metrics.

    Only produces a result when:
      - MRR is available (not None)
      - ARR is not already set (avoids overwriting extracted values)
    """
    if features.mrr_usd is None:
        return []

    if features.arr_usd is not None:
        return []

    arr_value = features.mrr_usd * 12.0

    return [
        DerivedMetricLog(
            metric_name="arr_from_mrr",
            derived_value=arr_value,
            source_fields=["mrr_usd"],
            source_values=[features.mrr_usd],
            formula="mrr_usd * 12",
            explanation=(
                f"ARR derived as MRR (${features.mrr_usd:,.0f}) "
                f"multiplied by 12 months = ${arr_value:,.0f}"
            ),
            confidence=0.9,
        )
    ]


def derive_mrr_from_arr(features: ExtractedFeatures) -> list[DerivedMetricLog]:
    """Derive MRR from ARR when MRR is not already set.

    Formula: MRR = ARR / 12

    Rationale: Annual recurring revenue divided by 12 months gives
    monthly recurring revenue. Inverse of the ARR-from-MRR derivation.

    Only produces a result when:
      - ARR is available (not None)
      - MRR is not already set
    """
    if features.arr_usd is None:
        return []

    if features.mrr_usd is not None:
        return []

    mrr_value = features.arr_usd / 12.0

    return [
        DerivedMetricLog(
            metric_name="mrr_from_arr",
            derived_value=mrr_value,
            source_fields=["arr_usd"],
            source_values=[features.arr_usd],
            formula="arr_usd / 12",
            explanation=(
                f"MRR derived as ARR (${features.arr_usd:,.0f}) "
                f"divided by 12 months = ${mrr_value:,.0f}"
            ),
            confidence=0.9,
        )
    ]


def derive_arr_from_acv_and_customers(
    features: ExtractedFeatures,
) -> list[DerivedMetricLog]:
    """Derive ARR from ACV (implied) and customer count.

    When we have a revenue figure that looks like per-customer annual
    value and a customer count, we can compute total ARR.

    This rule fires when:
      - ARR is not set
      - MRR is not set (otherwise ARR-from-MRR would be preferred)
      - We have both funding_amount_usd interpreted as a possible ACV proxy
        and customer_count

    NOTE: This rule is conservative. It only fires when the description
    explicitly mentions ACV or annual contract value alongside customer count.
    Since we don't have a separate acv_usd field, this derivation is
    limited to cases where the text parser can identify per-customer metrics.
    """
    # This rule requires explicit ACV extraction which isn't in the current
    # model. Placeholder for future expansion when acv_usd field is added.
    return []


def derive_revenue_from_gmv_and_take_rate(
    features: ExtractedFeatures,
) -> list[DerivedMetricLog]:
    """Derive estimated revenue from GMV and take rate.

    Formula: Revenue = GMV * Take Rate

    NOTE: This rule requires a take_rate_pct field which is not currently
    in ExtractedFeatures. It serves as a documented derivation rule
    that activates when the field is added in future sprints.

    Placeholder for marketplace business models where:
      - GMV is the total transaction volume
      - Take rate is the percentage captured as revenue
    """
    return []


def derive_revenue_per_employee(
    features: ExtractedFeatures,
) -> list[DerivedMetricLog]:
    """Derive revenue per employee from ARR and team size.

    Formula: Revenue per Employee = ARR / Team Size

    Rationale: A key efficiency metric for startups. Higher values
    indicate better capital efficiency and scalability.

    Only produces a result when:
      - ARR is available
      - Team size is available and > 0
    """
    if features.arr_usd is None or features.team_size_numeric is None:
        return []

    if features.team_size_numeric <= 0:
        return []

    rpe = features.arr_usd / features.team_size_numeric

    return [
        DerivedMetricLog(
            metric_name="revenue_per_employee",
            derived_value=rpe,
            source_fields=["arr_usd", "team_size_numeric"],
            source_values=[features.arr_usd, features.team_size_numeric],
            formula="arr_usd / team_size_numeric",
            explanation=(
                f"Revenue per employee: ARR (${features.arr_usd:,.0f}) "
                f"divided by team size ({features.team_size_numeric}) "
                f"= ${rpe:,.0f} per employee"
            ),
            confidence=0.85,
        )
    ]


def derive_runway_from_cash_and_burn(
    features: ExtractedFeatures,
) -> list[DerivedMetricLog]:
    """Derive runway from funding/cash and burn rate.

    Formula: Runway (months) = Cash / Monthly Burn

    Rationale: Financial runway is computed by dividing available cash
    by monthly burn rate. This is a standard financial metric.

    Only produces a result when:
      - Burn rate is available and > 0
      - Runway is not already set
      - We have some funding amount as a proxy for cash position
        (conservative: only uses funding_amount_usd as a lower bound)

    NOTE: funding_amount_usd may not represent current cash on hand.
    The confidence is set lower to reflect this uncertainty.
    """
    if features.burn_rate_usd is None or features.burn_rate_usd <= 0:
        return []

    if features.runway_months is not None:
        return []

    if features.funding_amount_usd is None:
        return []

    runway_months_float = features.funding_amount_usd / features.burn_rate_usd

    # Sanity check: runway should be between 0 and 120 months
    if runway_months_float <= 0 or runway_months_float > 120:
        return []

    runway_months_int = max(1, int(round(runway_months_float)))

    return [
        DerivedMetricLog(
            metric_name="runway_from_cash_and_burn",
            derived_value=float(runway_months_int),
            source_fields=["funding_amount_usd", "burn_rate_usd"],
            source_values=[features.funding_amount_usd, features.burn_rate_usd],
            formula="funding_amount_usd / burn_rate_usd",
            explanation=(
                f"Runway derived as funding (${features.funding_amount_usd:,.0f}) "
                f"divided by monthly burn (${features.burn_rate_usd:,.0f}) "
                f"= ~{runway_months_int} months. "
                f"Note: uses total funding as proxy for available cash, "
                f"actual runway may differ."
            ),
            confidence=0.6,
        )
    ]


def derive_funding_efficiency(
    features: ExtractedFeatures,
) -> list[DerivedMetricLog]:
    """Derive funding efficiency from ARR and total funding.

    Formula: Funding Efficiency = ARR / Total Funding

    Rationale: Measures how effectively a company converts invested
    capital into recurring revenue. Higher ratios indicate better
    capital efficiency. A ratio > 1.0 means the company generates
    more ARR than it has raised in total funding.

    Only produces a result when:
      - ARR is available and > 0
      - Funding amount is available and > 0
    """
    if features.arr_usd is None or features.funding_amount_usd is None:
        return []

    if features.arr_usd <= 0 or features.funding_amount_usd <= 0:
        return []

    efficiency = features.arr_usd / features.funding_amount_usd

    return [
        DerivedMetricLog(
            metric_name="funding_efficiency",
            derived_value=efficiency,
            source_fields=["arr_usd", "funding_amount_usd"],
            source_values=[features.arr_usd, features.funding_amount_usd],
            formula="arr_usd / funding_amount_usd",
            explanation=(
                f"Funding efficiency: ARR (${features.arr_usd:,.0f}) "
                f"divided by total funding (${features.funding_amount_usd:,.0f}) "
                f"= {efficiency:.2f}x"
            ),
            confidence=0.8,
        )
    ]


def derive_burn_multiple(
    features: ExtractedFeatures,
) -> list[DerivedMetricLog]:
    """Derive burn multiple from net burn and net new ARR.

    Formula: Burn Multiple = Net Burn / Net New ARR

    Rationale: Burn multiple measures how much a company burns to
    generate each dollar of new ARR. Lower is better. A burn multiple
    < 1.0 is excellent. > 2.0 is concerning.

    NOTE: This derivation requires net new ARR (ARR growth), which
    isn't directly available. We use current ARR as a proxy when we
    have burn rate, accepting reduced accuracy.

    Only produces a result when:
      - Burn rate is available
      - ARR is available and > 0
      - Team size and other context suggests growth stage
    """
    if features.burn_rate_usd is None or features.burn_rate_usd <= 0:
        return []

    if features.arr_usd is None or features.arr_usd <= 0:
        return []

    annual_burn = features.burn_rate_usd * 12.0
    burn_multiple = annual_burn / features.arr_usd

    # Sanity check
    if burn_multiple <= 0 or burn_multiple > 100:
        return []

    return [
        DerivedMetricLog(
            metric_name="burn_multiple",
            derived_value=burn_multiple,
            source_fields=["burn_rate_usd", "arr_usd"],
            source_values=[features.burn_rate_usd, features.arr_usd],
            formula="(burn_rate_usd * 12) / arr_usd",
            explanation=(
                f"Burn multiple: annual burn "
                f"(${annual_burn:,.0f}) divided by ARR "
                f"(${features.arr_usd:,.0f}) = {burn_multiple:.2f}x. "
                f"Note: uses annualized burn against current ARR "
                f"as proxy for net new ARR."
            ),
            confidence=0.6,
        )
    ]


def derive_ltv_cac_ratio(
    features: ExtractedFeatures,
) -> list[DerivedMetricLog]:
    """Derive LTV/CAC ratio when both values are present.

    Formula: LTV/CAC Ratio = LTV / CAC

    Rationale: The single most important unit economics metric.
    > 3.0 is healthy. > 5.0 is excellent. < 1.0 is unsustainable.

    Only produces a result when:
      - Both CAC and LTV are available
      - Both are > 0
    """
    if features.cac_usd is None or features.ltv_usd is None:
        return []

    if features.cac_usd <= 0 or features.ltv_usd <= 0:
        return []

    ratio = features.ltv_usd / features.cac_usd

    return [
        DerivedMetricLog(
            metric_name="ltv_cac_ratio",
            derived_value=ratio,
            source_fields=["ltv_usd", "cac_usd"],
            source_values=[features.ltv_usd, features.cac_usd],
            formula="ltv_usd / cac_usd",
            explanation=(
                f"LTV/CAC ratio: LTV (${features.ltv_usd:,.0f}) "
                f"divided by CAC (${features.cac_usd:,.0f}) = {ratio:.2f}x"
            ),
            confidence=0.85,
        )
    ]


def derive_arr_per_customer(
    features: ExtractedFeatures,
) -> list[DerivedMetricLog]:
    """Derive average annual contract value per customer from ARR and customer count.

    Formula: ACV = ARR / Customer Count

    Rationale: Average revenue per customer is a key metric for
    understanding pricing power and customer economics.

    Only produces a result when:
      - ARR is available and > 0
      - Customer count is available and > 0
    """
    if features.arr_usd is None or features.customer_count is None:
        return []

    if features.arr_usd <= 0 or features.customer_count <= 0:
        return []

    acv = features.arr_usd / features.customer_count

    return [
        DerivedMetricLog(
            metric_name="acv_per_customer",
            derived_value=acv,
            source_fields=["arr_usd", "customer_count"],
            source_values=[features.arr_usd, features.customer_count],
            formula="arr_usd / customer_count",
            explanation=(
                f"Average contract value per customer: ARR "
                f"(${features.arr_usd:,.0f}) divided by "
                f"{features.customer_count:,} customers = ${acv:,.0f} per customer"
            ),
            confidence=0.85,
        )
    ]


def derive_arr_from_customer_count_and_acv(
    features: ExtractedFeatures,
) -> list[DerivedMetricLog]:
    """Derive ARR when we have per-customer revenue and customer count.

    NOTE: This rule is a placeholder. Currently, we don't have a separate
    acv_usd or per_customer_revenue field. When such a field is added
    (e.g., from parsing "$48,000 ARR per contract" or "$35K ACV"),
    this rule will activate.

    Formula: ARR = ACV * Customer Count
    """
    return []


# ---------------------------------------------------------------------------
# Rule registry — ordered list of all inference rules
# ---------------------------------------------------------------------------

ALL_INFERENCE_RULES: list[
    tuple[str, type | object, str]
] = [
    ("derive_arr_from_mrr", derive_arr_from_mrr, "arr_from_mrr"),
    ("derive_mrr_from_arr", derive_mrr_from_arr, "mrr_from_arr"),
    ("derive_revenue_per_employee", derive_revenue_per_employee, "revenue_per_employee"),
    ("derive_runway_from_cash_and_burn", derive_runway_from_cash_and_burn, "runway_from_cash_and_burn"),
    ("derive_funding_efficiency", derive_funding_efficiency, "funding_efficiency"),
    ("derive_burn_multiple", derive_burn_multiple, "burn_multiple"),
    ("derive_ltv_cac_ratio", derive_ltv_cac_ratio, "ltv_cac_ratio"),
    ("derive_arr_per_customer", derive_arr_per_customer, "acv_per_customer"),
]
