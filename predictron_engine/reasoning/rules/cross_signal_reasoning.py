"""Cross-signal reasoning rule — synthesizes observations across dimensions.

This rule analyzes observations and features from multiple dimensions
to detect reinforcing patterns, conflicting signals, and investment
thesis indicators that no single-dimension rule can identify.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.reasoning.rules.base import feature_ref

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation

DIMENSION = "investment_thesis"

_REINFORCING_WEIGHTS: dict[str, float] = {
    "market_product_fit": 0.85,
    "team_market_alignment": 0.80,
    "business_model_market_fit": 0.80,
    "traction_business_model_consistency": 0.75,
    "competition_differentiation": 0.70,
    "technology_market_alignment": 0.65,
    "founder_domain_alignment": 0.75,
    "scale_evidence": 0.70,
}

_CONFLICT_WEIGHTS: dict[str, float] = {
    "enterprise_product_no_enterprise_customers": 0.65,
    "high_competition_no_differentiation": 0.70,
    "complex_technology_weak_team": 0.75,
    "no_revenue_high_funding": 0.50,
    "consumer_product_b2b_signals": 0.60,
    "pre_revenue_mature_market": 0.55,
}


class CrossSignalReasoningRule:
    """Produces cross-dimensional observations by analyzing signal patterns.

    Examines combinations of observations and features across all dimensions
    to identify reinforcing strengths, conflicting signals, and investment
    thesis indicators.
    """

    @property
    def name(self) -> str:
        return "cross_signal_reasoning"

    def evaluate(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        observations: list[Obs] = []
        observations.extend(self._detect_market_product_fit(features))
        observations.extend(self._detect_team_market_alignment(features))
        observations.extend(self._detect_business_model_market_fit(features))
        observations.extend(self._detect_traction_consistency(features))
        observations.extend(self._detect_competition_differentiation(features))
        observations.extend(self._detect_technology_market_alignment(features))
        observations.extend(self._detect_founder_domain_alignment(features))
        observations.extend(self._detect_scale_evidence(features))
        observations.extend(self._detect_conflicts(features))
        return observations

    def _detect_market_product_fit(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if not features.industry or not features.product_category:
            return []

        refs: list[str] = [
            feature_ref("industry", features.industry),
            feature_ref("product_category", features.product_category),
        ]

        fit_score = 0.0
        signals: list[str] = []

        if features.industry and features.product_category:
            fit_score += 0.3
            signals.append("industry_product_category_match")

        if features.enterprise_orientation and features.customer_type:
            if features.enterprise_orientation == "enterprise" and features.customer_type == "b2b":
                fit_score += 0.3
                signals.append("enterprise_b2b_alignment")
            elif features.enterprise_orientation == "consumer" and features.customer_type == "b2c":
                fit_score += 0.3
                signals.append("consumer_b2c_alignment")

        if features.market_maturity and features.product_maturity:
            maturity_pairs = {
                ("emerging", "beta"): 0.2,
                ("emerging", "concept"): 0.15,
                ("growth", "growth"): 0.3,
                ("growth", "beta"): 0.25,
                ("mature", "mature"): 0.2,
                ("mature", "growth"): 0.15,
            }
            pair_key = (features.market_maturity, features.product_maturity)
            if pair_key in maturity_pairs:
                fit_score += maturity_pairs[pair_key]
                signals.append(f"maturity_alignment_{pair_key[0]}_{pair_key[1]}")

        if features.target_workflow and features.primary_capabilities:
            fit_score += 0.15
            signals.append("workflow_capability_alignment")

        if fit_score >= 0.5:
            statement = (
                f"Strong market-product fit detected: {features.industry} market "
                f"with {features.product_category} product. Alignment signals: "
                f"{', '.join(signals[:3])}."
            )
            confidence = min(0.5 + fit_score * 0.5, 0.9)
            importance = 0.8
        elif fit_score >= 0.3:
            statement = (
                f"Moderate market-product fit between {features.industry} and "
                f"{features.product_category}. Some alignment signals present."
            )
            confidence = 0.55
            importance = 0.6
        else:
            return []

        return [
            Obs(
                dimension=DIMENSION,
                category="market_product_fit",
                statement=statement,
                evidence=refs + [f"signal:{s}" for s in signals],
                confidence=confidence,
                importance=importance,
                source_rule="CrossSignalReasoningRule",
            )
        ]

    def _detect_team_market_alignment(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if not features.industry or features.founder_profile_count == 0:
            return []

        refs: list[str] = [
            feature_ref("industry", features.industry),
            feature_ref("founder_profile_count", features.founder_profile_count),
        ]

        alignment_score = 0.0
        signals: list[str] = []

        if features.domain_expertise_signals:
            alignment_score += min(len(features.domain_expertise_signals) * 0.15, 0.4)
            signals.append("domain_expertise")
            refs.append(feature_ref("domain_expertise_signals", features.domain_expertise_signals))

        if features.founder_market_fit_signals:
            alignment_score += min(
                len(features.founder_market_fit_signals) * 0.15, 0.35,
            )
            signals.append("founder_market_fit")
            fmfs_ref = feature_ref(
                "founder_market_fit_signals",
                features.founder_market_fit_signals,
            )
            refs.append(fmfs_ref)

        if features.serial_founder_indicators:
            alignment_score += 0.15
            signals.append("serial_founder")

        if features.founder_team_type == "technical":
            tech_heavy = {
                "ai_ml", "cybersecurity",
                "cloud_infrastructure", "data_engineering",
            }
            tech_match = (
                features.primary_technology_domain in tech_heavy
                or features.technical_complexity in ("high", "moderate")
            )
            if tech_match:
                alignment_score += 0.15
                signals.append("technical_founder_technical_market")

        if alignment_score < 0.2:
            return []

        statement = (
            f"Team-market alignment: {features.founder_profile_count} founder(s) "
            f"in {features.industry} with alignment signals: "
            f"{', '.join(signals)}."
        )

        return [
            Obs(
                dimension=DIMENSION,
                category="team_market_alignment",
                statement=statement,
                evidence=refs,
                confidence=min(0.45 + alignment_score * 0.4, 0.85),
                importance=0.75,
                source_rule="CrossSignalReasoningRule",
            )
        ]

    def _detect_business_model_market_fit(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if not features.business_model or not features.industry:
            return []

        refs: list[str] = [
            feature_ref("business_model", features.business_model),
            feature_ref("industry", features.industry),
        ]

        fit_score = 0.0
        signals: list[str] = []

        if features.revenue_model:
            refs.append(feature_ref("revenue_model", features.revenue_model))
            saas_sub = (
                features.business_model == "saas"
                and features.revenue_model in ("subscription", "usage_based")
            )
            mp_comm = (
                features.business_model == "marketplace"
                and features.revenue_model in ("commission", "transaction_fee")
            )
            if saas_sub:
                fit_score += 0.25
                signals.append("saas_subscription_alignment")
            elif mp_comm:
                fit_score += 0.25
                signals.append("marketplace_commission_alignment")

        if features.recurring_revenue_signal:
            refs.append(feature_ref("recurring_revenue_signal", features.recurring_revenue_signal))
            if features.recurring_revenue_signal == "recurring":
                fit_score += 0.2
                signals.append("recurring_revenue")

        if features.customer_acquisition_model:
            cam_ref = feature_ref(
                "customer_acquisition_model",
                features.customer_acquisition_model,
            )
            refs.append(cam_ref)
            b2b_sales = (
                features.customer_type == "b2b"
                and features.customer_acquisition_model in ("sales_led", "hybrid")
            )
            b2c_plg = (
                features.customer_type == "b2c"
                and features.customer_acquisition_model == "product_led"
            )
            if b2b_sales:
                fit_score += 0.15
                signals.append("b2b_sales_alignment")
            elif b2c_plg:
                fit_score += 0.15
                signals.append("b2c_plg_alignment")

        if features.unit_economics_indicators:
            fit_score += min(len(features.unit_economics_indicators) * 0.1, 0.25)
            signals.append("unit_economics_signals")

        if features.platform_characteristics:
            fit_score += 0.1
            signals.append("platform_characteristics")

        if fit_score < 0.2:
            return []

        statement = (
            f"Business model-market fit: {features.business_model} model in "
            f"{features.industry} with fit signals: {', '.join(signals)}."
        )

        return [
            Obs(
                dimension=DIMENSION,
                category="business_model_market_fit",
                statement=statement,
                evidence=refs + [f"signal:{s}" for s in signals],
                confidence=min(0.5 + fit_score * 0.4, 0.85),
                importance=0.75,
                source_rule="CrossSignalReasoningRule",
            )
        ]

    def _detect_traction_consistency(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        refs: list[str] = []
        consistency_score = 0.0
        signals: list[str] = []

        has_revenue_evidence = bool(features.revenue_amount_signals or features.arr_mrr_signals)
        has_customer_evidence = bool(
            features.customer_count_signals
            or features.paying_customer_signals
            or features.enterprise_customer_signals
        )
        has_growth_evidence = bool(features.growth_signals or features.expansion_signals)
        has_funding_evidence = bool(features.funding_amount_signals or features.investor_signals)

        if features.has_revenue is not None:
            refs.append(feature_ref("has_revenue", features.has_revenue))

        if has_revenue_evidence:
            consistency_score += 0.2
            signals.append("revenue_evidence")
            refs.append(feature_ref("revenue_amount_signals", features.revenue_amount_signals))

        if has_customer_evidence:
            consistency_score += 0.2
            signals.append("customer_evidence")

        if has_growth_evidence:
            consistency_score += 0.2
            signals.append("growth_evidence")

        if has_funding_evidence:
            consistency_score += 0.15
            signals.append("funding_evidence")

        if features.retention_signals:
            consistency_score += 0.15
            signals.append("retention_evidence")

        if features.engagement_signals:
            consistency_score += 0.1
            signals.append("engagement_evidence")

        if consistency_score < 0.3:
            return []

        evidence_count = len(signals)
        if evidence_count >= 4:
            statement = (
                f"Strong traction consistency: {evidence_count} independent "
                f"evidence categories support the traction narrative. "
                f"Signals include: {', '.join(signals[:4])}."
            )
            confidence = 0.75
            importance = 0.8
        else:
            statement = (
                f"Moderate traction consistency: {evidence_count} evidence "
                f"categories detected. Signals: {', '.join(signals)}."
            )
            confidence = 0.6
            importance = 0.65

        return [
            Obs(
                dimension=DIMENSION,
                category="traction_consistency",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="CrossSignalReasoningRule",
            )
        ]

    def _detect_competition_differentiation(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        has_competition = bool(
            features.direct_competitor_signals or features.indirect_competitor_signals
        )
        has_differentiation = bool(
            features.differentiation_signals
            or features.competitive_moat_indicators
        )

        if not has_competition:
            return []

        refs: list[str] = [
            feature_ref("direct_competitors", len(features.direct_competitor_signals)),
            feature_ref("indirect_competitors", len(features.indirect_competitor_signals)),
        ]

        competitor_count = (
            len(features.direct_competitor_signals)
            + len(features.indirect_competitor_signals)
        )

        if has_differentiation:
            diff_count = (
                len(features.differentiation_signals)
                + len(features.competitive_moat_indicators)
            )
            refs.append(feature_ref("differentiation_count", diff_count))
            moat_ref = feature_ref(
                "moat_count",
                len(features.competitive_moat_indicators),
            )
            refs.append(moat_ref)

            ratio = diff_count / max(competitor_count, 1)
            if ratio >= 1.0 or len(features.competitive_moat_indicators) >= 2:
                statement = (
                    f"Strong differentiation vs {competitor_count} competitors: "
                    f"{diff_count} differentiation indicators including "
                    f"{len(features.competitive_moat_indicators)} moat signals."
                )
                confidence = 0.7
                importance = 0.8
            else:
                statement = (
                    f"Moderate differentiation vs {competitor_count} competitors: "
                    f"{diff_count} differentiation signals present."
                )
                confidence = 0.55
                importance = 0.6
        else:
            statement = (
                f"Limited differentiation detected against {competitor_count} "
                f"competitors. No strong moat or differentiation indicators found."
            )
            confidence = 0.6
            importance = 0.75
            refs.append(feature_ref("differentiation_count", 0))

        return [
            Obs(
                dimension=DIMENSION,
                category="competition_differentiation",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="CrossSignalReasoningRule",
            )
        ]

    def _detect_technology_market_alignment(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if not features.primary_technology_domain or not features.industry:
            return []

        refs: list[str] = [
            feature_ref("primary_technology_domain", features.primary_technology_domain),
            feature_ref("industry", features.industry),
        ]

        alignment_score = 0.0
        signals: list[str] = []

        if features.infrastructure_maturity:
            refs.append(feature_ref("infrastructure_maturity", features.infrastructure_maturity))
            if features.infrastructure_maturity in ("mature", "enterprise_grade"):
                alignment_score += 0.2
                signals.append("mature_infrastructure")

        if features.engineering_maturity:
            refs.append(feature_ref("engineering_maturity", features.engineering_maturity))
            if features.engineering_maturity in ("established", "sophisticated"):
                alignment_score += 0.2
                signals.append("established_engineering")

        if features.api_strategy and features.api_strategy != "unknown":
            alignment_score += 0.15
            signals.append(f"api_strategy_{features.api_strategy}")

        if features.security_signals:
            alignment_score += min(len(features.security_signals) * 0.05, 0.15)
            signals.append("security_signals")

        if features.scalability_indicators:
            alignment_score += min(len(features.scalability_indicators) * 0.05, 0.15)
            signals.append("scalability_signals")

        if features.cloud_infrastructure_signals:
            alignment_score += 0.1
            signals.append("cloud_infrastructure")

        if alignment_score < 0.25:
            return []

        statement = (
            f"Technology-market alignment: {features.primary_technology_domain} "
            f"technology supporting {features.industry} market. "
            f"Signals: {', '.join(signals[:3])}."
        )

        return [
            Obs(
                dimension=DIMENSION,
                category="technology_market_alignment",
                statement=statement,
                evidence=refs + [f"signal:{s}" for s in signals],
                confidence=min(0.5 + alignment_score * 0.4, 0.85),
                importance=0.7,
                source_rule="CrossSignalReasoningRule",
            )
        ]

    def _detect_founder_domain_alignment(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if features.founder_profile_count == 0:
            return []

        refs: list[str] = [
            feature_ref("founder_profile_count", features.founder_profile_count),
        ]

        alignment_signals: list[str] = []

        if features.founder_market_fit_signals:
            fmfs_ref = feature_ref(
                "founder_market_fit_signals",
                features.founder_market_fit_signals,
            )
            refs.append(fmfs_ref)
            alignment_signals.append("founder_market_fit")

        if features.domain_expertise_signals:
            refs.append(feature_ref("domain_expertise_signals", features.domain_expertise_signals))
            alignment_signals.append("domain_expertise")

        if features.founder_team_type and features.primary_technology_domain:
            technical_domains = {
                "ai_ml", "cybersecurity",
                "cloud_infrastructure", "data_engineering", "iot",
            }
            is_technical = (
                features.founder_team_type == "technical"
                and features.primary_technology_domain in technical_domains
            )
            is_biz_b2b = (
                features.founder_team_type == "business"
                and features.customer_type == "b2b"
            )
            if is_technical:
                alignment_signals.append("technical_founder_tech_market")
            elif is_biz_b2b:
                alignment_signals.append("business_founder_b2b")

        if features.leadership_roles:
            refs.append(feature_ref("leadership_roles", features.leadership_roles))
            alignment_signals.append("leadership_roles_present")

        if len(alignment_signals) < 2:
            return []

        statement = (
            f"Strong founder domain alignment: {len(alignment_signals)} "
            f"signals ({', '.join(alignment_signals)}) indicate deep "
            f"relevance to the target market."
        )

        return [
            Obs(
                dimension=DIMENSION,
                category="founder_domain_alignment",
                statement=statement,
                evidence=refs,
                confidence=min(0.5 + len(alignment_signals) * 0.1, 0.85),
                importance=0.75,
                source_rule="CrossSignalReasoningRule",
            )
        ]

    def _detect_scale_evidence(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        refs: list[str] = []
        scale_signals: list[str] = []

        if features.scalability_indicators:
            refs.append(feature_ref("scalability_indicators", features.scalability_indicators))
            scale_signals.extend(features.scalability_indicators[:3])

        if features.network_effects_signals:
            refs.append(feature_ref("network_effects_signals", features.network_effects_signals))
            scale_signals.append("network_effects")

        if features.platform_characteristics:
            refs.append(feature_ref("platform_characteristics", features.platform_characteristics))
            scale_signals.append("platform_characteristics")

        if features.switching_cost_indicators:
            sci_ref = feature_ref(
                "switching_cost_indicators",
                features.switching_cost_indicators,
            )
            refs.append(sci_ref)
            scale_signals.append("switching_costs")

        if features.growth_signals:
            refs.append(feature_ref("growth_signals", features.growth_signals))
            scale_signals.append("growth_evidence")

        if features.expansion_signals:
            refs.append(feature_ref("expansion_signals", features.expansion_signals))
            scale_signals.append("expansion_evidence")

        if len(scale_signals) < 2:
            return []

        statement = (
            f"Scale evidence detected: {len(scale_signals)} signals "
            f"({', '.join(scale_signals[:4])}) indicate potential for "
            f"significant growth."
        )

        return [
            Obs(
                dimension=DIMENSION,
                category="scale_evidence",
                statement=statement,
                evidence=refs,
                confidence=min(0.5 + len(scale_signals) * 0.07, 0.8),
                importance=0.7,
                source_rule="CrossSignalReasoningRule",
            )
        ]

    def _detect_conflicts(
        self, features: ExtractedFeatures
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        observations: list[Obs] = []

        if (features.customer_type == "b2b" and features.enterprise_orientation == "enterprise"
                and not features.enterprise_customer_signals
                and not features.paying_customer_signals):
            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="signal_conflict",
                    statement=(
                        "Conflict: Enterprise B2B positioning detected but no "
                        "enterprise customer evidence found. May indicate pre-revenue "
                        "or early-stage with unvalidated enterprise assumption."
                    ),
                    evidence=[
                        feature_ref("customer_type", features.customer_type),
                        feature_ref(
                            "enterprise_orientation",
                            features.enterprise_orientation,
                        ),
                        feature_ref(
                            "enterprise_customer_signals",
                            features.enterprise_customer_signals,
                        ),
                    ],
                    confidence=0.6,
                    importance=0.7,
                    source_rule="CrossSignalReasoningRule",
                )
            )

        if (features.direct_competitor_signals
                and not features.differentiation_signals
                and not features.competitive_moat_indicators):
            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="signal_conflict",
                    statement=(
                        f"Conflict: {len(features.direct_competitor_signals)} "
                        f"direct competitors detected but no differentiation or "
                        f"moat signals found. Competitive position is weak."
                    ),
                    evidence=[
                        feature_ref(
                            "direct_competitor_signals",
                            features.direct_competitor_signals,
                        ),
                    ],
                    confidence=0.65,
                    importance=0.8,
                    source_rule="CrossSignalReasoningRule",
                )
            )

        if (features.primary_technology_domain
                and features.technical_complexity in ("high",)
                and features.engineering_strength == "weak"):
            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="signal_conflict",
                    statement=(
                        "Conflict: High technical complexity with weak engineering "
                        "strength suggests execution risk."
                    ),
                    evidence=[
                        feature_ref("technical_complexity", features.technical_complexity),
                        feature_ref("engineering_strength", features.engineering_strength),
                    ],
                    confidence=0.7,
                    importance=0.8,
                    source_rule="CrossSignalReasoningRule",
                )
            )

        if (features.funding_amount_signals
                and not features.revenue_amount_signals
                and not features.arr_mrr_signals
                and features.has_revenue is False):
            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="signal_conflict",
                    statement=(
                        "Conflict: Significant funding signals detected but no "
                        "revenue evidence. Capital deployed without revenue "
                        "validation may indicate high burn risk."
                    ),
                    evidence=[
                        feature_ref("funding_amount_signals", features.funding_amount_signals),
                        feature_ref("has_revenue", features.has_revenue),
                    ],
                    confidence=0.55,
                    importance=0.65,
                    source_rule="CrossSignalReasoningRule",
                )
            )

        if (features.customer_type == "b2b"
                and features.enterprise_orientation == "consumer"):
            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="signal_conflict",
                    statement=(
                        "Conflict: B2B customer type conflicts with consumer "
                        "orientation signal. Customer targeting may be unclear."
                    ),
                    evidence=[
                        feature_ref("customer_type", features.customer_type),
                        feature_ref("enterprise_orientation", features.enterprise_orientation),
                    ],
                    confidence=0.5,
                    importance=0.6,
                    source_rule="CrossSignalReasoningRule",
                )
            )

        if (features.business_model == "marketplace"
                and features.customer_type == "b2b"
                and not features.platform_characteristics
                and not features.network_effects_signals):
            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="signal_conflict",
                    statement=(
                        "Conflict: Marketplace model detected but no platform or "
                        "network effects evidence. Marketplace viability is unclear."
                    ),
                    evidence=[
                        feature_ref("business_model", features.business_model),
                        feature_ref("customer_type", features.customer_type),
                    ],
                    confidence=0.55,
                    importance=0.65,
                    source_rule="CrossSignalReasoningRule",
                )
            )

        if (features.revenue_model == "subscription"
                and features.recurring_revenue_signal == "one_time"
                and features.customer_type == "b2b"):
            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="signal_conflict",
                    statement=(
                        "Conflict: Subscription revenue model conflicts with "
                        "one-time revenue signals. Revenue model consistency "
                        "needs verification."
                    ),
                    evidence=[
                        feature_ref("revenue_model", features.revenue_model),
                        feature_ref("recurring_revenue_signal", features.recurring_revenue_signal),
                    ],
                    confidence=0.6,
                    importance=0.7,
                    source_rule="CrossSignalReasoningRule",
                )
            )

        if (features.founder_profile_count >= 2
                and not features.domain_expertise_signals
                and not features.founder_market_fit_signals
                and features.funding_stage in ("series_a", "series_b", "growth")):
            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="signal_conflict",
                    statement=(
                        "Conflict: Multiple founders detected but no domain "
                        "expertise or market fit signals at advanced funding "
                        "stage. Team-market alignment is uncertain."
                    ),
                    evidence=[
                        feature_ref("founder_profile_count", features.founder_profile_count),
                        feature_ref("funding_stage", features.funding_stage),
                    ],
                    confidence=0.5,
                    importance=0.6,
                    source_rule="CrossSignalReasoningRule",
                )
            )

        if (features.scalability_indicators
                and features.growth_signals
                and not features.customer_count_signals
                and not features.revenue_amount_signals
                and not features.arr_mrr_signals):
            observations.append(
                Obs(
                    dimension=DIMENSION,
                    category="signal_conflict",
                    statement=(
                        "Conflict: Scalability and growth signals detected but "
                        "no customer or revenue evidence. Growth claims may be "
                        "premature or unsubstantiated."
                    ),
                    evidence=[
                        feature_ref("scalability_indicators", features.scalability_indicators),
                        feature_ref("growth_signals", features.growth_signals),
                    ],
                    confidence=0.55,
                    importance=0.7,
                    source_rule="CrossSignalReasoningRule",
                )
            )

        return observations
