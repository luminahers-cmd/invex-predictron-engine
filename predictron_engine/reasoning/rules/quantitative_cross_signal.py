"""Quantitative cross-signal reasoning rule — analyzes metric relationships.

This rule evaluates structured quantitative metrics from ExtractedFeatures
to detect reinforcing patterns, inconsistencies, and notable combinations
across multiple signals. Unlike single-metric rules, this rule reasons
about *relationships* between metrics — the kind of analysis that
improves quality of reasoning beyond individual threshold checks.

Every observation is fully deterministic and traceable to specific
metric values and named threshold constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.reasoning.rules.base import feature_ref

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation

DIMENSION = "investment_thesis"

# ---------------------------------------------------------------------------
# Revenue-per-customer thresholds
#
# $100K+ per customer indicates enterprise model.
# $10K-$100K indicates mid-market.
# < $10K indicates SMB/high-volume.
# ---------------------------------------------------------------------------

_REVENUE_PER_CUSTOMER_ENTERPRISE: float = 100_000.0
_REVENUE_PER_CUSTOMER_MID_MARKET: float = 10_000.0

# ---------------------------------------------------------------------------
# Revenue-per-employee thresholds
#
# $500K+ is exceptional. $200K+ is strong. $100K+ is acceptable.
# < $50K suggests oversized team or pre-revenue.
# ---------------------------------------------------------------------------

_REVENUE_PER_EMPLOYEE_EXCEPTIONAL: float = 500_000.0
_REVENUE_PER_EMPLOYEE_STRONG: float = 200_000.0
_REVENUE_PER_EMPLOYEE_ACCEPTABLE: float = 100_000.0
_REVENUE_PER_EMPLOYEE_LOW: float = 50_000.0

# ---------------------------------------------------------------------------
# Growth-to-burn efficiency thresholds
#
# Growth rate / burn (in $M) — higher is more efficient.
# > 50% per $1M burn is excellent. > 20% is good. < 10% is concerning.
# ---------------------------------------------------------------------------

_GROWTH_BURN_EXCELLENT: float = 50.0
_GROWTH_BURN_GOOD: float = 20.0
_GROWTH_BURN_CONCERNING: float = 10.0

# ---------------------------------------------------------------------------
# Burn-to-runway sustainability thresholds
#
# burn * runway = remaining capital.
# < 6 months effective runway at current burn = critical.
# ---------------------------------------------------------------------------

_BURN_RUNWAY_CRITICAL_MONTHS: int = 6
_BURN_RUNWAY_WARNING_MONTHS: int = 12

# ---------------------------------------------------------------------------
# Funding-to-ARR efficiency thresholds
#
# ARR / total funding — measures capital efficiency.
# > 1.0 means generating more ARR than total raised. > 0.5 is good.
# ---------------------------------------------------------------------------

_FUNDING_EFFICIENCY_STRONG: float = 1.0
_FUNDING_EFFICIENCY_GOOD: float = 0.5
_FUNDING_EFFICIENCY_WEAK: float = 0.1

# ---------------------------------------------------------------------------
# Valuation-to-ARR thresholds
#
# Valuation / ARR — revenue multiple.
# > 20x is high (growth priced in). < 5x may suggest undervaluation.
# ---------------------------------------------------------------------------

_VALUATION_ARR_HIGH_MULTIPLE: float = 20.0
_VALUATION_ARR_MODERATE_MULTIPLE: float = 10.0
_VALUATION_ARR_LOW_MULTIPLE: float = 5.0

# ---------------------------------------------------------------------------
# Stage-appropriate customer count thresholds
#
# Seed: 5-50 customers expected. Series A: 50-500. Series B+: 200+.
# ---------------------------------------------------------------------------

_STAGE_CUSTOMERS_SEED_LOW: int = 5
_STAGE_CUSTOMERS_SEED_HIGH: int = 200
_STAGE_CUSTOMERS_SERIES_A_LOW: int = 50
_STAGE_CUSTOMERS_SERIES_A_HIGH: int = 1000
_STAGE_CUSTOMERS_SERIES_B_LOW: int = 200

# ---------------------------------------------------------------------------
# Valuation-with-weak-traction anomaly thresholds
#
# Valuation > $100M with ARR < $1M is an anomaly.
# Valuation > $500M with ARR < $5M is an anomaly.
# ---------------------------------------------------------------------------

_VALUATION_ANOMALY_THRESHOLD_1: float = 100_000_000.0
_VALUATION_ANOMALY_ARR_1: float = 1_000_000.0
_VALUATION_ANOMALY_THRESHOLD_2: float = 500_000_000.0
_VALUATION_ANOMALY_ARR_2: float = 5_000_000.0

# ---------------------------------------------------------------------------
# Burn-efficiency thresholds for growth-to-burn ratio
#
# burn_rate * 12 = annual burn. If ARR < annual burn, company is
# consuming more than it generates.
# ---------------------------------------------------------------------------

_ARR_VS_ANNUAL_BURN_SUSTAINABLE: float = 1.0

# ---------------------------------------------------------------------------
# Team-size-to-revenue thresholds
#
# Large teams (>100) with low ARR (<$1M) are unusual.
# Small teams (<20) with high ARR (>$10M) are efficient.
# ---------------------------------------------------------------------------

_TEAM_LARGE_THRESHOLD: int = 100
_TEAM_SMALL_THRESHOLD: int = 20
_REVENUE_LOW_THRESHOLD: float = 1_000_000.0
_REVENUE_HIGH_THRESHOLD: float = 10_000_000.0


class QuantitativeCrossSignalRule:
    """Analyzes relationships between structured quantitative metrics.

    Produces cross-signal observations that detect:
    - Reinforcing metric patterns (e.g. high ARR + low churn = strong unit economics)
    - Inconsistent metric patterns (e.g. high valuation + weak traction)
    - Notable metric combinations (e.g. high burn + short runway = urgency)

    Each observation is deterministic and fully traceable to specific
    metric values and named threshold constants.
    """

    @property
    def name(self) -> str:
        return "quantitative_cross_signal"

    def evaluate(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        observations: list[Obs] = []
        observations.extend(self._rev_per_customer(features))
        observations.extend(self._rev_per_employee(features))
        observations.extend(self._growth_vs_burn(features))
        observations.extend(self._burn_vs_runway(features))
        observations.extend(self._nrr_churn_consistency(features))
        observations.extend(self._funding_arr_efficiency(features))
        observations.extend(self._valuation_arr_multiple(features))
        observations.extend(self._funding_valuation_consistency(features))
        observations.extend(self._customer_count_stage(features))
        observations.extend(self._team_revenue_consistency(features))
        observations.extend(self._val_weak_traction_anomaly(features))
        observations.extend(self._high_arr_few_customers(features))
        observations.extend(self._high_burn_modest_growth(features))
        observations.extend(self._strong_funding_weak_execution(features))
        observations.extend(self._arr_vs_annual_burn(features))
        return observations

    # ------------------------------------------------------------------
    # Reinforcing patterns
    # ------------------------------------------------------------------

    def _rev_per_customer(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """ARR / customer count — revenue concentration metric."""
        from predictron_engine.models.report import Observation as Obs

        if features.arr_usd is None or features.customer_count is None:
            return []
        if features.customer_count == 0:
            return []

        refs = [
            feature_ref("arr_usd", features.arr_usd),
            feature_ref("customer_count", features.customer_count),
        ]
        rpc = features.arr_usd / features.customer_count

        if rpc >= _REVENUE_PER_CUSTOMER_ENTERPRISE:
            statement = (
                f"Revenue per customer of ${rpc:,.0f} "
                f"(ARR ${features.arr_usd / 1_000_000:.1f}M / "
                f"{features.customer_count:,} customers) indicates "
                f"enterprise-grade customer value."
            )
            confidence = 0.8
            importance = 0.8
            category = "rev_per_customer_enterprise"
        elif rpc >= _REVENUE_PER_CUSTOMER_MID_MARKET:
            statement = (
                f"Revenue per customer of ${rpc:,.0f} "
                f"(ARR ${features.arr_usd / 1_000_000:.1f}M / "
                f"{features.customer_count:,} customers) indicates "
                f"mid-market customer value."
            )
            confidence = 0.75
            importance = 0.7
            category = "rev_per_customer_mid_market"
        else:
            statement = (
                f"Revenue per customer of ${rpc:,.0f} "
                f"(ARR ${features.arr_usd / 1_000_000:.1f}M / "
                f"{features.customer_count:,} customers) indicates "
                f"SMB or high-volume model."
            )
            confidence = 0.7
            importance = 0.6
            category = "rev_per_customer_smb"

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _rev_per_employee(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """ARR / team size — capital efficiency metric."""
        from predictron_engine.models.report import Observation as Obs

        if features.arr_usd is None or features.team_size_numeric is None:
            return []
        if features.team_size_numeric == 0:
            return []

        refs = [
            feature_ref("arr_usd", features.arr_usd),
            feature_ref("team_size_numeric", features.team_size_numeric),
        ]
        rpe = features.arr_usd / features.team_size_numeric

        if rpe >= _REVENUE_PER_EMPLOYEE_EXCEPTIONAL:
            statement = (
                f"Revenue per employee of ${rpe / 1_000:.0f}K "
                f"(${features.arr_usd / 1_000_000:.1f}M ARR / "
                f"{features.team_size_numeric} employees) indicates "
                f"exceptional capital efficiency."
            )
            confidence = 0.8
            importance = 0.85
            category = "rev_per_employee_exceptional"
        elif rpe >= _REVENUE_PER_EMPLOYEE_STRONG:
            statement = (
                f"Revenue per employee of ${rpe / 1_000:.0f}K "
                f"(${features.arr_usd / 1_000_000:.1f}M ARR / "
                f"{features.team_size_numeric} employees) indicates "
                f"strong team productivity."
            )
            confidence = 0.75
            importance = 0.75
            category = "rev_per_employee_strong"
        elif rpe >= _REVENUE_PER_EMPLOYEE_ACCEPTABLE:
            statement = (
                f"Revenue per employee of ${rpe / 1_000:.0f}K "
                f"(${features.arr_usd / 1_000_000:.1f}M ARR / "
                f"{features.team_size_numeric} employees) indicates "
                f"acceptable team efficiency."
            )
            confidence = 0.7
            importance = 0.65
            category = "rev_per_employee_acceptable"
        else:
            statement = (
                f"Revenue per employee of ${rpe / 1_000:.0f}K "
                f"(${features.arr_usd / 1_000_000:.1f}M ARR / "
                f"{features.team_size_numeric} employees) is low, "
                f"suggesting team may be oversized for current revenue."
            )
            confidence = 0.7
            importance = 0.7
            category = "rev_per_employee_low"

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _growth_vs_burn(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """Growth rate relative to burn — efficiency of growth spend."""
        from predictron_engine.models.report import Observation as Obs

        if features.growth_rate_pct is None or features.burn_rate_usd is None:
            return []

        refs = [
            feature_ref("growth_rate_pct", features.growth_rate_pct),
            feature_ref("burn_rate_usd", features.burn_rate_usd),
        ]

        burn_m = features.burn_rate_usd / 1_000_000.0
        if burn_m <= 0:
            return []

        ratio = features.growth_rate_pct / burn_m

        if ratio >= _GROWTH_BURN_EXCELLENT:
            statement = (
                f"Growth-to-burn efficiency is excellent: {features.growth_rate_pct:.0f}% "
                f"growth with ${burn_m:.1f}M/mo burn "
                f"({ratio:.1f}% growth per $1M burn)."
            )
            confidence = 0.8
            importance = 0.85
            category = "growth_burn_excellent"
        elif ratio >= _GROWTH_BURN_GOOD:
            statement = (
                f"Growth-to-burn efficiency is good: {features.growth_rate_pct:.0f}% "
                f"growth with ${burn_m:.1f}M/mo burn "
                f"({ratio:.1f}% growth per $1M burn)."
            )
            confidence = 0.75
            importance = 0.75
            category = "growth_burn_good"
        elif ratio >= _GROWTH_BURN_CONCERNING:
            statement = (
                f"Growth-to-burn efficiency is moderate: {features.growth_rate_pct:.0f}% "
                f"growth with ${burn_m:.1f}M/mo burn "
                f"({ratio:.1f}% growth per $1M burn)."
            )
            confidence = 0.7
            importance = 0.7
            category = "growth_burn_moderate"
        else:
            statement = (
                f"Growth-to-burn efficiency is concerning: {features.growth_rate_pct:.0f}% "
                f"growth with ${burn_m:.1f}M/mo burn "
                f"({ratio:.1f}% growth per $1M burn) suggests inefficient "
                f"capital deployment."
            )
            confidence = 0.75
            importance = 0.8
            category = "growth_burn_concerning"

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _burn_vs_runway(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """Burn rate relative to runway — sustainability check."""
        from predictron_engine.models.report import Observation as Obs

        if features.burn_rate_usd is None or features.runway_months is None:
            return []

        refs = [
            feature_ref("burn_rate_usd", features.burn_rate_usd),
            feature_ref("runway_months", features.runway_months),
        ]

        remaining = features.burn_rate_usd * features.runway_months
        burn_m = features.burn_rate_usd / 1_000_000.0

        if features.runway_months < _BURN_RUNWAY_CRITICAL_MONTHS:
            statement = (
                f"Burn of ${burn_m:.1f}M/mo with only "
                f"{features.runway_months} months runway leaves "
                f"${remaining / 1_000_000:.1f}M remaining capital — "
                f"critical sustainability risk."
            )
            confidence = 0.85
            importance = 0.9
            category = "burn_runway_critical"
        elif features.runway_months < _BURN_RUNWAY_WARNING_MONTHS:
            statement = (
                f"Burn of ${burn_m:.1f}M/mo with "
                f"{features.runway_months} months runway "
                f"(${remaining / 1_000_000:.1f}M remaining) — "
                f"fundraising should be planned soon."
            )
            confidence = 0.8
            importance = 0.8
            category = "burn_runway_warning"
        else:
            statement = (
                f"Burn of ${burn_m:.1f}M/mo with "
                f"{features.runway_months} months runway "
                f"(${remaining / 1_000_000:.1f}M remaining) — "
                f"sustainable for current growth phase."
            )
            confidence = 0.75
            importance = 0.7
            category = "burn_runway_sustainable"

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _nrr_churn_consistency(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """NRR and churn rate consistency check."""
        from predictron_engine.models.report import Observation as Obs

        if features.nrr_pct is None or features.churn_rate_pct is None:
            return []

        refs = [
            feature_ref("nrr_pct", features.nrr_pct),
            feature_ref("churn_rate_pct", features.churn_rate_pct),
        ]

        # NRR > 100% with low churn is reinforcing
        if features.nrr_pct >= 110 and features.churn_rate_pct <= 5:
            statement = (
                f"NRR of {features.nrr_pct:.0f}% with churn of "
                f"{features.churn_rate_pct:.1f}% is strongly reinforcing: "
                f"excellent retention with net expansion from existing customers."
            )
            confidence = 0.85
            importance = 0.85
            category = "nrr_churn_reinforcing"
        # NRR < 100% with high churn — consistent negative signal
        elif features.nrr_pct < 90 and features.churn_rate_pct > 10:
            statement = (
                f"NRR of {features.nrr_pct:.0f}% with churn of "
                f"{features.churn_rate_pct:.1f}% confirms severe retention "
                f"issues: net contraction with high customer loss."
            )
            confidence = 0.85
            importance = 0.85
            category = "nrr_churn_conflict"
        # NRR > 100% but high churn — inconsistent (expansion masks churn)
        elif features.nrr_pct >= 100 and features.churn_rate_pct > 10:
            statement = (
                f"NRR of {features.nrr_pct:.0f}% contradicts churn of "
                f"{features.churn_rate_pct:.1f}%: expansion from remaining "
                f"customers masks significant customer loss."
            )
            confidence = 0.75
            importance = 0.8
            category = "nrr_churn_masking"
        # Low NRR with low churn — unusual (downgrades, not churn)
        elif features.nrr_pct < 90 and features.churn_rate_pct <= 5:
            statement = (
                f"NRR of {features.nrr_pct:.0f}% with churn of only "
                f"{features.churn_rate_pct:.1f}%: customers stay but "
                f"spend less, suggesting pricing or value delivery issues."
            )
            confidence = 0.7
            importance = 0.7
            category = "nrr_churn_downgrade"
        else:
            return []

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _funding_arr_efficiency(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """ARR relative to total funding — capital efficiency."""
        from predictron_engine.models.report import Observation as Obs

        if features.arr_usd is None or features.funding_amount_usd is None:
            return []

        if features.funding_amount_usd == 0:
            return []

        refs = [
            feature_ref("arr_usd", features.arr_usd),
            feature_ref("funding_amount_usd", features.funding_amount_usd),
        ]

        ratio = features.arr_usd / features.funding_amount_usd

        if ratio >= _FUNDING_EFFICIENCY_STRONG:
            statement = (
                f"Funding efficiency is strong: ARR of "
                f"${features.arr_usd / 1_000_000:.1f}M against "
                f"${features.funding_amount_usd / 1_000_000:.1f}M total "
                f"funding ({ratio:.2f}x) — generating more revenue than raised."
            )
            confidence = 0.8
            importance = 0.85
            category = "funding_efficiency_strong"
        elif ratio >= _FUNDING_EFFICIENCY_GOOD:
            statement = (
                f"Funding efficiency is good: ARR of "
                f"${features.arr_usd / 1_000_000:.1f}M against "
                f"${features.funding_amount_usd / 1_000_000:.1f}M total "
                f"funding ({ratio:.2f}x)."
            )
            confidence = 0.75
            importance = 0.75
            category = "funding_efficiency_good"
        elif ratio >= _FUNDING_EFFICIENCY_WEAK:
            statement = (
                f"Funding efficiency is moderate: ARR of "
                f"${features.arr_usd / 1_000_000:.1f}M against "
                f"${features.funding_amount_usd / 1_000_000:.1f}M total "
                f"funding ({ratio:.2f}x) — capital efficiency could improve."
            )
            confidence = 0.7
            importance = 0.65
            category = "funding_efficiency_moderate"
        else:
            statement = (
                f"Funding efficiency is weak: ARR of "
                f"${features.arr_usd / 1_000_000:.1f}M against "
                f"${features.funding_amount_usd / 1_000_000:.1f}M total "
                f"funding ({ratio:.2f}x) — significant capital deployed "
                f"with limited revenue return."
            )
            confidence = 0.75
            importance = 0.8
            category = "funding_efficiency_weak"

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _valuation_arr_multiple(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """Valuation relative to ARR — revenue multiple analysis."""
        from predictron_engine.models.report import Observation as Obs

        if features.valuation_usd is None or features.arr_usd is None:
            return []
        if features.arr_usd == 0:
            return []

        refs = [
            feature_ref("valuation_usd", features.valuation_usd),
            feature_ref("arr_usd", features.arr_usd),
        ]

        multiple = features.valuation_usd / features.arr_usd

        if multiple >= _VALUATION_ARR_HIGH_MULTIPLE:
            statement = (
                f"Revenue multiple of {multiple:.1f}x "
                f"(${features.valuation_usd / 1_000_000:.0f}M valuation / "
                f"${features.arr_usd / 1_000_000:.1f}M ARR) indicates high "
                f"growth expectations priced into valuation."
            )
            confidence = 0.75
            importance = 0.75
            category = "valuation_arr_high"
        elif multiple >= _VALUATION_ARR_MODERATE_MULTIPLE:
            statement = (
                f"Revenue multiple of {multiple:.1f}x "
                f"(${features.valuation_usd / 1_000_000:.0f}M valuation / "
                f"${features.arr_usd / 1_000_000:.1f}M ARR) indicates "
                f"moderate growth expectations."
            )
            confidence = 0.7
            importance = 0.7
            category = "valuation_arr_moderate"
        elif multiple >= _VALUATION_ARR_LOW_MULTIPLE:
            statement = (
                f"Revenue multiple of {multiple:.1f}x "
                f"(${features.valuation_usd / 1_000_000:.0f}M valuation / "
                f"${features.arr_usd / 1_000_000:.1f}M ARR) indicates "
                f"reasonable valuation relative to revenue."
            )
            confidence = 0.7
            importance = 0.65
            category = "valuation_arr_low"
        else:
            statement = (
                f"Revenue multiple of {multiple:.1f}x "
                f"(${features.valuation_usd / 1_000_000:.0f}M valuation / "
                f"${features.arr_usd / 1_000_000:.1f}M ARR) is low, "
                f"suggesting potential undervaluation or execution concerns."
            )
            confidence = 0.7
            importance = 0.7
            category = "valuation_arr_very_low"

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _funding_valuation_consistency(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """Check if funding amount and valuation are internally consistent."""
        from predictron_engine.models.report import Observation as Obs

        if features.funding_amount_usd is None or features.valuation_usd is None:
            return []
        if features.funding_amount_usd == 0:
            return []

        refs = [
            feature_ref("funding_amount_usd", features.funding_amount_usd),
            feature_ref("valuation_usd", features.valuation_usd),
        ]

        ratio = features.valuation_usd / features.funding_amount_usd

        # Valuation below funding is unusual
        if ratio < 1.0:
            statement = (
                f"Valuation of ${features.valuation_usd / 1_000_000:.0f}M "
                f"is below total funding of "
                f"${features.funding_amount_usd / 1_000_000:.1f}M "
                f"({ratio:.2f}x) — company may be underwater or "
                f"significantly diluted."
            )
            confidence = 0.8
            importance = 0.85
            category = "funding_valuation_inconsistent"
        # Very high valuation relative to funding
        elif ratio >= 20.0:
            statement = (
                f"Valuation of ${features.valuation_usd / 1_000_000:.0f}M "
                f"is {ratio:.1f}x total funding of "
                f"${features.funding_amount_usd / 1_000_000:.1f}M — "
                f"strong value creation, but verify traction supports valuation."
            )
            confidence = 0.7
            importance = 0.7
            category = "funding_valuation_high_creation"
        else:
            return []

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _customer_count_stage(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """Customer count relative to funding stage — stage appropriateness."""
        from predictron_engine.models.report import Observation as Obs

        if features.customer_count is None or features.funding_stage is None:
            return []

        refs = [
            feature_ref("customer_count", features.customer_count),
            feature_ref("funding_stage", features.funding_stage),
        ]

        stage = features.funding_stage.lower()
        count = features.customer_count

        # Stage-appropriate expectations
        if stage in ("pre_seed", "angel"):
            if count >= _STAGE_CUSTOMERS_SEED_HIGH:
                statement = (
                    f"Customer count of {count:,} exceeds typical "
                    f"pre-seed/angel expectations — strong early traction "
                    f"for the stage."
                )
                confidence = 0.7
                importance = 0.7
                category = "customer_count_stage_exceeds"
            else:
                return []
        elif stage in ("seed",):
            if count >= _STAGE_CUSTOMERS_SEED_HIGH:
                statement = (
                    f"Customer count of {count:,} exceeds typical seed "
                    f"expectations ({_STAGE_CUSTOMERS_SEED_LOW}-"
                    f"{_STAGE_CUSTOMERS_SEED_HIGH} range) — "
                    f"strong traction for the stage."
                )
                confidence = 0.7
                importance = 0.7
                category = "customer_count_stage_exceeds"
            elif count < _STAGE_CUSTOMERS_SEED_LOW:
                statement = (
                    f"Customer count of {count:,} is below typical seed "
                    f"expectations (minimum ~{_STAGE_CUSTOMERS_SEED_LOW}) — "
                    f"customer acquisition may need attention."
                )
                confidence = 0.65
                importance = 0.65
                category = "customer_count_stage_below"
            else:
                return []
        elif stage in ("series_a",):
            if count >= _STAGE_CUSTOMERS_SERIES_A_HIGH:
                statement = (
                    f"Customer count of {count:,} exceeds typical Series A "
                    f"expectations — strong market penetration for the stage."
                )
                confidence = 0.7
                importance = 0.7
                category = "customer_count_stage_exceeds"
            elif count < _STAGE_CUSTOMERS_SERIES_A_LOW:
                statement = (
                    f"Customer count of {count:,} is below typical Series A "
                    f"expectations (minimum ~{_STAGE_CUSTOMERS_SERIES_A_LOW}) — "
                    f"may indicate early-stage traction despite advanced funding."
                )
                confidence = 0.65
                importance = 0.7
                category = "customer_count_stage_below"
            else:
                return []
        elif stage in ("series_b", "series_c", "growth"):
            if count < _STAGE_CUSTOMERS_SERIES_B_LOW:
                statement = (
                    f"Customer count of {count:,} is below typical "
                    f"{stage.replace('_', ' ')} expectations "
                    f"(minimum ~{_STAGE_CUSTOMERS_SERIES_B_LOW}) — "
                    f"traction may not justify the funding stage."
                )
                confidence = 0.7
                importance = 0.75
                category = "customer_count_stage_below"
            else:
                return []
        else:
            return []

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _team_revenue_consistency(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """Large team with low revenue or small team with high revenue."""
        from predictron_engine.models.report import Observation as Obs

        if features.team_size_numeric is None or features.arr_usd is None:
            return []

        refs = [
            feature_ref("team_size_numeric", features.team_size_numeric),
            feature_ref("arr_usd", features.arr_usd),
        ]

        team = features.team_size_numeric
        arr = features.arr_usd

        if team >= _TEAM_LARGE_THRESHOLD and arr < _REVENUE_LOW_THRESHOLD:
            rpe = arr / max(team, 1)
            statement = (
                f"Team of {team} with ARR of only "
                f"${arr / 1_000_000:.1f}M "
                f"(${rpe / 1_000:.0f}K per employee) — "
                f"large team with limited revenue suggests either heavy "
                f"R&D investment or operational inefficiency."
            )
            confidence = 0.75
            importance = 0.8
            category = "team_revenue_inconsistent"
        elif team <= _TEAM_SMALL_THRESHOLD and arr >= _REVENUE_HIGH_THRESHOLD:
            rpe = arr / max(team, 1)
            statement = (
                f"Lean team of {team} generating "
                f"${arr / 1_000_000:.1f}M ARR "
                f"(${rpe / 1_000:.0f}K per employee) — "
                f"exceptional team productivity and capital efficiency."
            )
            confidence = 0.8
            importance = 0.8
            category = "team_revenue_efficient"
        else:
            return []

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _val_weak_traction_anomaly(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """High valuation with weak traction — potential anomaly."""
        from predictron_engine.models.report import Observation as Obs

        if features.valuation_usd is None:
            return []

        has_weak_traction = (
            (features.arr_usd is not None and features.arr_usd < _VALUATION_ANOMALY_ARR_1)
            or (
                features.arr_usd is None
                and (features.revenue_amount_signals is None
                     or not features.revenue_amount_signals)
            )
        )

        if not has_weak_traction:
            return []

        refs = [feature_ref("valuation_usd", features.valuation_usd)]

        if features.arr_usd is not None:
            refs.append(feature_ref("arr_usd", features.arr_usd))

        if features.valuation_usd >= _VALUATION_ANOMALY_THRESHOLD_2:
            statement = (
                f"Valuation of ${features.valuation_usd / 1_000_000:.0f}M "
                f"with ARR below "
                f"${_VALUATION_ANOMALY_ARR_2 / 1_000_000:.0f}M — "
                f"high valuation appears disconnected from revenue traction. "
                f"Valuation may be driven by team, technology, or market "
                f"potential rather than proven metrics."
            )
            confidence = 0.75
            importance = 0.85
            category = "valuation_weak_traction"
        elif features.valuation_usd >= _VALUATION_ANOMALY_THRESHOLD_1:
            statement = (
                f"Valuation of ${features.valuation_usd / 1_000_000:.0f}M "
                f"with ARR below "
                f"${_VALUATION_ANOMALY_ARR_1 / 1_000_000:.0f}M — "
                f"valuation significantly exceeds revenue evidence. "
                f"Verify that traction supports the valuation."
            )
            confidence = 0.7
            importance = 0.8
            category = "valuation_weak_traction"
        else:
            return []

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _high_arr_few_customers(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """High ARR with very few customers — extreme concentration."""
        from predictron_engine.models.report import Observation as Obs

        if features.arr_usd is None or features.customer_count is None:
            return []

        if features.customer_count == 0:
            return []

        if features.customer_count > 10:
            return []

        rpc = features.arr_usd / features.customer_count

        refs = [
            feature_ref("arr_usd", features.arr_usd),
            feature_ref("customer_count", features.customer_count),
        ]

        if features.arr_usd >= 1_000_000 and features.customer_count <= 5:
            statement = (
                f"ARR of ${features.arr_usd / 1_000_000:.1f}M from only "
                f"{features.customer_count} customer(s) "
                f"(${rpc / 1_000:.0f}K per customer) — "
                f"extreme customer concentration risk. Loss of a single "
                f"customer would significantly impact revenue."
            )
            confidence = 0.8
            importance = 0.85
            category = "high_arr_concentrated_customers"
        else:
            return []

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _high_burn_modest_growth(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """High burn rate with modest growth — capital inefficiency."""
        from predictron_engine.models.report import Observation as Obs

        if features.burn_rate_usd is None or features.growth_rate_pct is None:
            return []

        refs = [
            feature_ref("burn_rate_usd", features.burn_rate_usd),
            feature_ref("growth_rate_pct", features.growth_rate_pct),
        ]

        burn_m = features.burn_rate_usd / 1_000_000.0

        if burn_m >= 2.0 and features.growth_rate_pct < 30:
            statement = (
                f"Monthly burn of ${burn_m:.1f}M with only "
                f"{features.growth_rate_pct:.0f}% growth — "
                f"high capital consumption with modest growth trajectory. "
                f"Growth should accelerate to justify the burn rate."
            )
            confidence = 0.8
            importance = 0.85
            category = "high_burn_low_growth"
        elif burn_m >= 5.0 and features.growth_rate_pct < 50:
            statement = (
                f"Monthly burn of ${burn_m:.1f}M with "
                f"{features.growth_rate_pct:.0f}% growth — "
                f"very high burn rate not yet matched by proportional growth."
            )
            confidence = 0.8
            importance = 0.85
            category = "high_burn_low_growth"
        else:
            return []

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _strong_funding_weak_execution(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """Strong funding without execution signals — capital without traction."""
        from predictron_engine.models.report import Observation as Obs

        if features.funding_amount_usd is None:
            return []

        if features.funding_amount_usd < 10_000_000:
            return []

        refs = [feature_ref("funding_amount_usd", features.funding_amount_usd)]

        has_execution = bool(
            features.arr_usd is not None and features.arr_usd > 0
        )
        has_growth = features.growth_rate_pct is not None and features.growth_rate_pct > 20
        has_customers = (
            features.customer_count is not None and features.customer_count > 10
        )

        if not has_execution and not has_growth and not has_customers:
            statement = (
                f"Total funding of "
                f"${features.funding_amount_usd / 1_000_000:.1f}M without "
                f"evidence of revenue, growth, or meaningful customer base — "
                f"capital has been deployed but execution signals are absent."
            )
            confidence = 0.7
            importance = 0.8
            category = "funding_without_execution"
        else:
            return []

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]

    def _arr_vs_annual_burn(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        """ARR relative to annual burn — revenue vs. consumption rate."""
        from predictron_engine.models.report import Observation as Obs

        if features.arr_usd is None or features.burn_rate_usd is None:
            return []

        refs = [
            feature_ref("arr_usd", features.arr_usd),
            feature_ref("burn_rate_usd", features.burn_rate_usd),
        ]

        annual_burn = features.burn_rate_usd * 12
        if annual_burn == 0:
            return []

        ratio = features.arr_usd / annual_burn

        if ratio >= _ARR_VS_ANNUAL_BURN_SUSTAINABLE:
            statement = (
                f"ARR of ${features.arr_usd / 1_000_000:.1f}M exceeds "
                f"annual burn of ${annual_burn / 1_000_000:.1f}M "
                f"({ratio:.2f}x) — company generates more revenue than "
                f"it consumes."
            )
            confidence = 0.8
            importance = 0.85
            category = "arr_exceeds_burn"
        elif ratio >= 0.5:
            statement = (
                f"ARR of ${features.arr_usd / 1_000_000:.1f}M covers "
                f"{ratio:.0%} of annual burn of "
                f"${annual_burn / 1_000_000:.1f}M — revenue partially "
                f"funds operations but gap remains."
            )
            confidence = 0.75
            importance = 0.75
            category = "arr_partial_burn_coverage"
        else:
            statement = (
                f"ARR of ${features.arr_usd / 1_000_000:.1f}M covers only "
                f"{ratio:.0%} of annual burn of "
                f"${annual_burn / 1_000_000:.1f}M — significant gap between "
                f"revenue and consumption requires external funding."
            )
            confidence = 0.75
            importance = 0.8
            category = "arr_burn_gap"

        return [
            Obs(
                dimension=DIMENSION,
                category=category,
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeCrossSignalRule",
            )
        ]



