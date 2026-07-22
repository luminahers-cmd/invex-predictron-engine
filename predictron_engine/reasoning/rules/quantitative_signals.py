"""Quantitative signals reasoning rule — consumes structured metrics.

This rule evaluates structured quantitative fields (ARR, runway, NRR,
churn, growth, burn rate, CAC/LTV, funding, valuation, customer count,
team size, market size) to produce quantitative observations.

All thresholds are named constants with documented rationale.
All logic is deterministic and fully explainable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.reasoning.rules.base import feature_ref

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation

DIMENSION = "traction_signals"

# ---------------------------------------------------------------------------
# ARR / Revenue thresholds
#
# ARR below $1M is typically pre-scale. $1M-$10M indicates meaningful
# traction. $10M+ signals strong revenue maturity for startups.
# ---------------------------------------------------------------------------

_ARR_PRE_SCALE_THRESHOLD: float = 1_000_000.0
_ARR_MEANINGFUL_TRACTION_THRESHOLD: float = 10_000_000.0
_ARR_STRONG_TRACTION_THRESHOLD: float = 50_000_000.0

# ---------------------------------------------------------------------------
# MRR thresholds
#
# MRR < $83K corresponds to ARR < $1M (pre-scale).
# MRR $83K-$833K corresponds to ARR $1M-$10M.
# ---------------------------------------------------------------------------

_MRR_PRE_SCALE_THRESHOLD: float = 83_000.0
_MRR_MEANINGFUL_THRESHOLD: float = 833_000.0

# ---------------------------------------------------------------------------
# Growth rate thresholds
#
# Growth rate interpretation:
#   < 20% — slow growth, below venture expectations
#   20-50% — moderate growth
#   50-100% — strong growth
#   100%+ — exceptional growth
# ---------------------------------------------------------------------------

_GROWTH_SLOW_THRESHOLD: float = 20.0
_GROWTH_MODERATE_THRESHOLD: float = 50.0
_GROWTH_STRONG_THRESHOLD: float = 100.0

# ---------------------------------------------------------------------------
# NRR (Net Revenue Retention) thresholds
#
# NRR > 100% means net expansion (existing customers grow).
# NRR > 110% is strong. NRR > 130% is exceptional.
# NRR < 90% indicates net contraction.
# ---------------------------------------------------------------------------

_NRR_CONTRACTION_THRESHOLD: float = 90.0
_NRR_NEUTRAL_THRESHOLD: float = 100.0
_NRR_STRONG_THRESHOLD: float = 110.0
_NRR_EXCEPTIONAL_THRESHOLD: float = 130.0

# ---------------------------------------------------------------------------
# Churn rate thresholds
#
# Monthly churn: < 2% excellent, 2-5% acceptable, 5-10% concerning,
# > 10% severe.
# ---------------------------------------------------------------------------

_CHURN_EXCELLENT_THRESHOLD: float = 2.0
_CHURN_ACCEPTABLE_THRESHOLD: float = 5.0
_CHURN_CONCERNING_THRESHOLD: float = 10.0

# ---------------------------------------------------------------------------
# Runway thresholds
#
# < 6 months: critical risk. 6-12 months: elevated risk.
# 12-18 months: adequate. 18-24 months: comfortable. > 24 months: strong.
# ---------------------------------------------------------------------------

_RUNWAY_CRITICAL_MONTHS: int = 6
_RUNWAY_ELEVATED_MONTHS: int = 12
_RUNWAY_ADEQUATE_MONTHS: int = 18
_RUNWAY_COMFORTABLE_MONTHS: int = 24

# ---------------------------------------------------------------------------
# Burn rate thresholds
#
# Burn > $5M/month is very high. $1M-$5M/month is high.
# ---------------------------------------------------------------------------

_BURN_HIGH_MONTHLY: float = 1_000_000.0
_BURN_VERY_HIGH_MONTHLY: float = 5_000_000.0

# ---------------------------------------------------------------------------
# CAC/LTV ratio thresholds
#
# LTV/CAC < 1.0 means losing money per customer.
# 1.0-3.0 is below best practice. 3.0-5.0 is healthy. > 5.0 is excellent.
# ---------------------------------------------------------------------------

_LTV_CAC_UNSUSTAINABLE: float = 1.0
_LTV_CAC_BELOW_BEST_PRACTICE: float = 3.0
_LTV_CAC_HEALTHY: float = 5.0

# ---------------------------------------------------------------------------
# Customer count thresholds
#
# < 10 customers: very early. 10-50: early traction.
# 50-200: growing. 200-1000: scaling. > 1000: established.
# ---------------------------------------------------------------------------

_CUSTOMERS_VERY_EARLY: int = 10
_CUSTOMERS_EARLY_TRACTION: int = 50
_CUSTOMERS_GROWING: int = 200
_CUSTOMERS_SCALING: int = 1000

# ---------------------------------------------------------------------------
# Team size thresholds
#
# Used for consistency checks against revenue.
# ---------------------------------------------------------------------------

_TEAM_SOLO: int = 10
_TEAM_SMALL: int = 50
_TEAM_MEDIUM: int = 200

# ---------------------------------------------------------------------------
# Funding vs valuation thresholds
#
# Valuation/funding ratio < 1.0 means underwater (unusual).
# Ratio 1.0-3.0 is modest. > 3.0 indicates strong value creation.
# ---------------------------------------------------------------------------

_VALUATION_FUNDING_UNDERWATER: float = 1.0
_VALUATION_FUNDING_MODEST: float = 3.0
_VALUATION_FUNDING_STRONG: float = 5.0


class QuantitativeSignalsRule:
    """Produces observations from structured quantitative metrics.

    Evaluates ARR, growth, NRR, churn, runway, burn, unit economics,
    funding, valuation, customer count, and team size to produce
    quantitative observations about the startup's financial health
    and traction quality.

    Each observation is traceable to extracted structured metrics
    and deterministic thresholds.
    """

    @property
    def name(self) -> str:
        return "quantitative_signals"

    def evaluate(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        observations: list[Obs] = []
        observations.extend(self._evaluate_arr(features))
        observations.extend(self._evaluate_mrr(features))
        observations.extend(self._evaluate_growth(features))
        observations.extend(self._evaluate_nrr(features))
        observations.extend(self._evaluate_churn(features))
        observations.extend(self._evaluate_runway(features))
        observations.extend(self._evaluate_burn_rate(features))
        observations.extend(self._evaluate_unit_economics(features))
        observations.extend(self._evaluate_funding_valuation(features))
        observations.extend(self._evaluate_customer_traction(features))
        observations.extend(self._evaluate_team_revenue_consistency(features))
        observations.extend(self._evaluate_market_size(features))
        return observations

    def _evaluate_arr(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if features.arr_usd is None:
            return []

        refs = [feature_ref("arr_usd", features.arr_usd)]
        arr = features.arr_usd

        if arr >= _ARR_STRONG_TRACTION_THRESHOLD:
            statement = (
                f"ARR of ${arr / 1_000_000:.1f}M indicates strong revenue "
                f"maturity and significant scale."
            )
            confidence = 0.85
            importance = 0.9
        elif arr >= _ARR_MEANINGFUL_TRACTION_THRESHOLD:
            statement = (
                f"ARR of ${arr / 1_000_000:.1f}M indicates meaningful "
                f"revenue traction with room for continued growth."
            )
            confidence = 0.8
            importance = 0.85
        elif arr >= _ARR_PRE_SCALE_THRESHOLD:
            statement = (
                f"ARR of ${arr / 1_000_000:.1f}M indicates pre-scale "
                f"revenue with early traction."
            )
            confidence = 0.75
            importance = 0.7
        else:
            statement = (
                f"ARR of ${arr:,.0f} is below $1M, indicating very early "
                f"revenue stage."
            )
            confidence = 0.7
            importance = 0.6

        return [
            Obs(
                dimension=DIMENSION,
                category="quantitative_arr",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeSignalsRule",
            )
        ]

    def _evaluate_mrr(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if features.mrr_usd is None:
            return []

        refs = [feature_ref("mrr_usd", features.mrr_usd)]
        mrr = features.mrr_usd

        if mrr >= _MRR_MEANINGFUL_THRESHOLD:
            statement = (
                f"MRR of ${mrr / 1_000:.0f}K indicates strong monthly "
                f"recurring revenue base."
            )
            confidence = 0.8
            importance = 0.8
        elif mrr >= _MRR_PRE_SCALE_THRESHOLD:
            statement = (
                f"MRR of ${mrr / 1_000:.0f}K indicates early recurring "
                f"revenue traction."
            )
            confidence = 0.75
            importance = 0.7
        else:
            statement = (
                f"MRR of ${mrr:,.0f} is below $83K, indicating very early "
                f"monthly recurring revenue."
            )
            confidence = 0.7
            importance = 0.6

        return [
            Obs(
                dimension=DIMENSION,
                category="quantitative_mrr",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeSignalsRule",
            )
        ]

    def _evaluate_growth(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if features.growth_rate_pct is None:
            return []

        refs = [feature_ref("growth_rate_pct", features.growth_rate_pct)]
        growth = features.growth_rate_pct

        if growth >= _GROWTH_STRONG_THRESHOLD:
            statement = (
                f"Growth rate of {growth:.0f}% indicates exceptional "
                f"momentum and strong market demand."
            )
            confidence = 0.8
            importance = 0.85
        elif growth >= _GROWTH_MODERATE_THRESHOLD:
            statement = (
                f"Growth rate of {growth:.0f}% indicates strong growth "
                f"momentum."
            )
            confidence = 0.75
            importance = 0.8
        elif growth >= _GROWTH_SLOW_THRESHOLD:
            statement = (
                f"Growth rate of {growth:.0f}% indicates moderate growth, "
                f"potentially below venture expectations."
            )
            confidence = 0.7
            importance = 0.65
        else:
            statement = (
                f"Growth rate of {growth:.0f}% indicates slow growth, "
                f"which may concern investors."
            )
            confidence = 0.7
            importance = 0.6

        return [
            Obs(
                dimension=DIMENSION,
                category="quantitative_growth",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeSignalsRule",
            )
        ]

    def _evaluate_nrr(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if features.nrr_pct is None:
            return []

        refs = [feature_ref("nrr_pct", features.nrr_pct)]
        nrr = features.nrr_pct

        if nrr >= _NRR_EXCEPTIONAL_THRESHOLD:
            statement = (
                f"NRR of {nrr:.0f}% indicates exceptional net revenue "
                f"retention with strong expansion from existing customers."
            )
            confidence = 0.85
            importance = 0.85
        elif nrr >= _NRR_STRONG_THRESHOLD:
            statement = (
                f"NRR of {nrr:.0f}% indicates strong net revenue retention "
                f"with meaningful expansion."
            )
            confidence = 0.8
            importance = 0.8
        elif nrr >= _NRR_NEUTRAL_THRESHOLD:
            statement = (
                f"NRR of {nrr:.0f}% indicates neutral to slightly positive "
                f"net revenue retention."
            )
            confidence = 0.75
            importance = 0.65
        elif nrr >= _NRR_CONTRACTION_THRESHOLD:
            statement = (
                f"NRR of {nrr:.0f}% indicates mild net revenue contraction, "
                f"suggesting some customer churn or downgrades."
            )
            confidence = 0.7
            importance = 0.7
        else:
            statement = (
                f"NRR of {nrr:.0f}% indicates significant net revenue "
                f"contraction, a serious concern for recurring revenue models."
            )
            confidence = 0.75
            importance = 0.8

        return [
            Obs(
                dimension=DIMENSION,
                category="quantitative_nrr",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeSignalsRule",
            )
        ]

    def _evaluate_churn(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if features.churn_rate_pct is None:
            return []

        refs = [feature_ref("churn_rate_pct", features.churn_rate_pct)]
        churn = features.churn_rate_pct

        if churn <= _CHURN_EXCELLENT_THRESHOLD:
            statement = (
                f"Churn rate of {churn:.1f}% indicates excellent customer "
                f"retention."
            )
            confidence = 0.8
            importance = 0.8
        elif churn <= _CHURN_ACCEPTABLE_THRESHOLD:
            statement = (
                f"Churn rate of {churn:.1f}% indicates acceptable customer "
                f"retention with room for improvement."
            )
            confidence = 0.75
            importance = 0.7
        elif churn <= _CHURN_CONCERNING_THRESHOLD:
            statement = (
                f"Churn rate of {churn:.1f}% indicates concerning customer "
                f"retention that may impact growth."
            )
            confidence = 0.7
            importance = 0.75
        else:
            statement = (
                f"Churn rate of {churn:.1f}% indicates severe customer "
                f"retention issues that threaten business viability."
            )
            confidence = 0.75
            importance = 0.85

        return [
            Obs(
                dimension=DIMENSION,
                category="quantitative_churn",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeSignalsRule",
            )
        ]

    def _evaluate_runway(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if features.runway_months is None:
            return []

        refs = [feature_ref("runway_months", features.runway_months)]
        runway = features.runway_months

        if runway >= _RUNWAY_COMFORTABLE_MONTHS:
            statement = (
                f"Runway of {runway} months provides strong financial "
                f"runway for execution and growth."
            )
            confidence = 0.8
            importance = 0.75
        elif runway >= _RUNWAY_ADEQUATE_MONTHS:
            statement = (
                f"Runway of {runway} months provides adequate financial "
                f"runway for near-term plans."
            )
            confidence = 0.75
            importance = 0.7
        elif runway >= _RUNWAY_ELEVATED_MONTHS:
            statement = (
                f"Runway of {runway} months indicates elevated funding "
                f"risk. Fundraising should be planned."
            )
            confidence = 0.75
            importance = 0.8
        else:
            statement = (
                f"Runway of {runway} months indicates critical funding "
                f"risk. Immediate fundraising attention required."
            )
            confidence = 0.8
            importance = 0.9

        return [
            Obs(
                dimension=DIMENSION,
                category="quantitative_runway",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeSignalsRule",
            )
        ]

    def _evaluate_burn_rate(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if features.burn_rate_usd is None:
            return []

        refs = [feature_ref("burn_rate_usd", features.burn_rate_usd)]
        burn = features.burn_rate_usd

        if burn >= _BURN_VERY_HIGH_MONTHLY:
            statement = (
                f"Monthly burn rate of ${burn / 1_000_000:.1f}M is very "
                f"high, requiring strong revenue or fundraising support."
            )
            confidence = 0.8
            importance = 0.85
        elif burn >= _BURN_HIGH_MONTHLY:
            statement = (
                f"Monthly burn rate of ${burn / 1_000:.0f}K is high but "
                f"common for scaling-stage companies."
            )
            confidence = 0.75
            importance = 0.7
        else:
            statement = (
                f"Monthly burn rate of ${burn:,.0f} is relatively low, "
                f"indicating capital efficiency."
            )
            confidence = 0.7
            importance = 0.65

        return [
            Obs(
                dimension=DIMENSION,
                category="quantitative_burn_rate",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeSignalsRule",
            )
        ]

    def _evaluate_unit_economics(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        observations: list[Obs] = []

        if features.cac_usd is not None and features.ltv_usd is not None:
            refs = [
                feature_ref("cac_usd", features.cac_usd),
                feature_ref("ltv_usd", features.ltv_usd),
            ]
            ratio = features.ltv_usd / features.cac_usd

            if ratio >= _LTV_CAC_HEALTHY:
                statement = (
                    f"LTV/CAC ratio of {ratio:.1f}x indicates excellent "
                    f"unit economics and strong capital efficiency."
                )
                confidence = 0.8
                importance = 0.85
            elif ratio >= _LTV_CAC_BELOW_BEST_PRACTICE:
                statement = (
                    f"LTV/CAC ratio of {ratio:.1f}x indicates healthy "
                    f"unit economics above best practice minimum."
                )
                confidence = 0.75
                importance = 0.75
            elif ratio >= _LTV_CAC_UNSUSTAINABLE:
                statement = (
                    f"LTV/CAC ratio of {ratio:.1f}x is below best "
                    f"practice, suggesting unit economics need improvement."
                )
                confidence = 0.7
                importance = 0.7
            else:
                statement = (
                    f"LTV/CAC ratio of {ratio:.1f}x indicates the company "
                    f"loses money per customer acquired — unsustainable."
                )
                confidence = 0.75
                importance = 0.85

            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="quantitative_unit_economics",
                    statement=statement,
                    evidence=refs,
                    confidence=confidence,
                    importance=importance,
                    source_rule="QuantitativeSignalsRule",
                )
            )

        elif features.cac_usd is not None:
            refs = [feature_ref("cac_usd", features.cac_usd)]
            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="quantitative_unit_economics",
                    statement=(
                        f"CAC of ${features.cac_usd:,.0f} identified "
                        f"without corresponding LTV for ratio assessment."
                    ),
                    evidence=refs,
                    confidence=0.6,
                    importance=0.5,
                    source_rule="QuantitativeSignalsRule",
                )
            )

        return observations

    def _evaluate_funding_valuation(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        observations: list[Obs] = []

        if (
            features.funding_amount_usd is not None
            and features.valuation_usd is not None
        ):
            refs = [
                feature_ref("funding_amount_usd", features.funding_amount_usd),
                feature_ref("valuation_usd", features.valuation_usd),
            ]
            ratio = features.valuation_usd / features.funding_amount_usd

            if ratio >= _VALUATION_FUNDING_STRONG:
                statement = (
                    f"Valuation of ${features.valuation_usd / 1_000_000:.0f}M "
                    f"against ${features.funding_amount_usd / 1_000_000:.0f}M "
                    f"funding ({ratio:.1f}x) indicates strong value creation."
                )
                confidence = 0.8
                importance = 0.8
            elif ratio >= _VALUATION_FUNDING_MODEST:
                statement = (
                    f"Valuation of ${features.valuation_usd / 1_000_000:.0f}M "
                    f"against ${features.funding_amount_usd / 1_000_000:.0f}M "
                    f"funding ({ratio:.1f}x) indicates modest value creation."
                )
                confidence = 0.75
                importance = 0.7
            elif ratio >= _VALUATION_FUNDING_UNDERWATER:
                statement = (
                    f"Valuation of ${features.valuation_usd / 1_000_000:.0f}M "
                    f"barely exceeds ${features.funding_amount_usd / 1_000_000:.0f}M "
                    f"funding ({ratio:.1f}x), indicating limited value creation "
                    f"relative to capital deployed."
                )
                confidence = 0.7
                importance = 0.7
            else:
                statement = (
                    f"Valuation of ${features.valuation_usd / 1_000_000:.0f}M "
                    f"is below ${features.funding_amount_usd / 1_000_000:.0f}M "
                    f"funding ({ratio:.1f}x), suggesting significant dilution "
                    f"or value impairment."
                )
                confidence = 0.75
                importance = 0.8

            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="quantitative_funding_valuation",
                    statement=statement,
                    evidence=refs,
                    confidence=confidence,
                    importance=importance,
                    source_rule="QuantitativeSignalsRule",
                )
            )

        elif features.funding_amount_usd is not None:
            refs = [feature_ref("funding_amount_usd", features.funding_amount_usd)]
            funding = features.funding_amount_usd
            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="quantitative_funding",
                    statement=(
                        f"Total funding of ${funding / 1_000_000:.1f}M "
                        f"identified."
                    ),
                    evidence=refs,
                    confidence=0.7,
                    importance=0.55,
                    source_rule="QuantitativeSignalsRule",
                )
            )

        return observations

    def _evaluate_customer_traction(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if features.customer_count is None:
            return []

        refs = [feature_ref("customer_count", features.customer_count)]
        count = features.customer_count

        if count >= _CUSTOMERS_SCALING:
            statement = (
                f"Customer count of {count:,} indicates established "
                f"market presence and strong traction."
            )
            confidence = 0.8
            importance = 0.8
        elif count >= _CUSTOMERS_GROWING:
            statement = (
                f"Customer count of {count:,} indicates growing market "
                f"penetration."
            )
            confidence = 0.75
            importance = 0.75
        elif count >= _CUSTOMERS_EARLY_TRACTION:
            statement = (
                f"Customer count of {count:,} indicates early traction "
                f"with meaningful market validation."
            )
            confidence = 0.7
            importance = 0.7
        elif count >= _CUSTOMERS_VERY_EARLY:
            statement = (
                f"Customer count of {count:,} indicates very early "
                f"customer acquisition."
            )
            confidence = 0.65
            importance = 0.6
        else:
            statement = (
                f"Customer count of {count} indicates pre-launch or "
                f"very early stage customer development."
            )
            confidence = 0.6
            importance = 0.5

        return [
            Obs(
                dimension=DIMENSION,
                category="quantitative_customer_count",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeSignalsRule",
            )
        ]

    def _evaluate_team_revenue_consistency(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if features.team_size_numeric is None or features.arr_usd is None:
            return []

        refs = [
            feature_ref("team_size_numeric", features.team_size_numeric),
            feature_ref("arr_usd", features.arr_usd),
        ]

        team = features.team_size_numeric
        arr = features.arr_usd
        revenue_per_employee = arr / max(team, 1)

        if team <= _TEAM_SOLO and arr >= _ARR_MEANINGFUL_TRACTION_THRESHOLD:
            statement = (
                f"Team of {team} generating ${arr / 1_000_000:.1f}M ARR "
                f"(${revenue_per_employee / 1_000:.0f}K per employee) "
                f"indicates exceptional capital efficiency."
            )
            confidence = 0.8
            importance = 0.8
        elif revenue_per_employee >= 500_000:
            statement = (
                f"Revenue per employee of ${revenue_per_employee / 1_000:.0f}K "
                f"(team of {team}, ARR ${arr / 1_000_000:.1f}M) indicates "
                f"strong team productivity."
            )
            confidence = 0.75
            importance = 0.7
        elif revenue_per_employee >= 100_000:
            statement = (
                f"Revenue per employee of ${revenue_per_employee / 1_000:.0f}K "
                f"(team of {team}, ARR ${arr / 1_000_000:.1f}M) indicates "
                f"acceptable team efficiency."
            )
            confidence = 0.7
            importance = 0.6
        else:
            statement = (
                f"Revenue per employee of ${revenue_per_employee / 1_000:.0f}K "
                f"(team of {team}, ARR ${arr / 1_000_000:.1f}M) is low, "
                f"suggesting team may be oversized for current revenue."
            )
            confidence = 0.65
            importance = 0.6

        return [
            Obs(
                dimension=DIMENSION,
                category="quantitative_team_efficiency",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeSignalsRule",
            )
        ]

    def _evaluate_market_size(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if features.market_size_usd is None:
            return []

        refs = [feature_ref("market_size_usd", features.market_size_usd)]
        market = features.market_size_usd

        if market >= 1_000_000_000_000:
            statement = (
                f"Addressable market of ${market / 1_000_000_000_000:.0f}T "
                f"represents an exceptionally large market opportunity."
            )
            confidence = 0.8
            importance = 0.8
        elif market >= 100_000_000_000:
            statement = (
                f"Addressable market of ${market / 1_000_000_000:.0f}B "
                f"represents a large market opportunity."
            )
            confidence = 0.75
            importance = 0.75
        elif market >= 10_000_000_000:
            statement = (
                f"Addressable market of ${market / 1_000_000_000:.1f}B "
                f"represents a substantial market opportunity."
            )
            confidence = 0.7
            importance = 0.7
        else:
            statement = (
                f"Addressable market of ${market / 1_000_000:.0f}M "
                f"represents a focused market opportunity."
            )
            confidence = 0.65
            importance = 0.6

        return [
            Obs(
                dimension=DIMENSION,
                category="quantitative_market_size",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="QuantitativeSignalsRule",
            )
        ]
