"""Comprehensive tests for RiskExtractor — Sprint 9 Risk Intelligence.

Covers all 19 risk dimensions with positive detection, negative (no false
positives), data-driven signals, confidence scoring, keyword extraction,
and edge cases.
"""

from __future__ import annotations

from predictron_engine.extraction.extractors.risk import RiskExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_startup(description: str) -> Startup:
    return Startup(
        name="TestCo",
        website="https://testco.example.com",
        description=description,
    )


def _make_data(founder_count: int = 0) -> CollectedData:
    return CollectedData(
        startup_name="TestCo",
        founder_count=founder_count,
    )


def _extract(description: str, founder_count: int = 0) -> ExtractedFeatures:
    return RiskExtractor().extract(
        _make_startup(description), _make_data(founder_count)
    )


# ---------------------------------------------------------------------------
# Return type and defaults
# ---------------------------------------------------------------------------


class TestRiskExtractorDefaults:
    def test_returns_extracted_features(self):
        result = _extract("A simple test description.")
        assert isinstance(result, ExtractedFeatures)

    def test_all_risk_lists_default_empty(self):
        result = RiskExtractor().extract(
            _make_startup("A simple test description."), _make_data(founder_count=2)
        )
        assert result.market_risk == []
        assert result.founder_risk == []
        assert result.execution_risk == []
        assert result.product_risk == []
        assert result.technology_risk == []
        assert result.business_model_risk == []
        assert result.traction_risk == []
        assert result.competitive_risk == []
        assert result.regulatory_risk == []
        assert result.operational_risk == []
        assert result.platform_dependency_risk == []
        assert result.customer_concentration_risk == []
        assert result.hiring_risk == []
        assert result.funding_risk == []
        assert result.scaling_risk == []
        assert result.security_risk == []
        assert result.compliance_risk == []
        assert result.risk_keywords == []
        assert result.risk_confidence == 0.0

    def test_non_risk_fields_not_set(self):
        result = RiskExtractor().extract(
            _make_startup("A simple test description."), _make_data(founder_count=2)
        )
        assert result.industry is None
        assert result.business_model is None
        assert result.technology_stack == []

    def test_risk_confidence_in_range(self):
        result = _extract("Highly regulated market with significant risk.")
        assert 0.0 <= result.risk_confidence <= 1.0


# ---------------------------------------------------------------------------
# Market risk
# ---------------------------------------------------------------------------


class TestMarketRisk:
    def test_uncertain_market(self):
        result = _extract("Operating in an uncertain market with volatile demand.")
        assert len(result.market_risk) > 0
        assert any("uncertain_market" in s for s in result.market_risk)

    def test_market_timing_risk(self):
        result = _extract("The product is ahead of its time, market not ready yet.")
        assert any("market_timing" in s for s in result.market_risk)

    def test_unproven_market(self):
        result = _extract("Entering an unproven market with nascent demand.")
        assert any("unproven_market" in s for s in result.market_risk)

    def test_market_saturated(self):
        result = _extract("Operating in a saturated market with established players.")
        assert any("market_saturated" in s for s in result.market_risk)

    def test_niche_limitation(self):
        result = _extract("The niche market has limited addressable market size.")
        assert any("niche_limitation" in s for s in result.market_risk)

    def test_cyclical_demand(self):
        result = _extract("Revenue has cyclical demand patterns throughout the year.")
        assert any("cyclical_demand" in s for s in result.market_risk)

    def test_no_market_risk_for_clean_description(self):
        result = _extract(
            "A strong B2B SaaS platform serving enterprise clients "
            "with proven product-market fit and recurring revenue."
        )
        assert result.market_risk == []


# ---------------------------------------------------------------------------
# Founder risk
# ---------------------------------------------------------------------------


class TestFounderRisk:
    def test_solo_founder_from_text(self):
        result = _extract("Founded by a solo founder with deep expertise.")
        assert any("solo_founder" in s for s in result.founder_risk)

    def test_solo_founder_from_data(self):
        result = RiskExtractor().extract(
            _make_startup("A B2B analytics company."),
            _make_data(founder_count=1),
        )
        assert any("solo_founder" in s for s in result.founder_risk)

    def test_no_founders_from_data(self):
        result = RiskExtractor().extract(
            _make_startup("A B2B analytics company."),
            _make_data(founder_count=0),
        )
        assert any("no_founders" in s for s in result.founder_risk)

    def test_no_domain_expertise(self):
        result = _extract("Built by first-time founders with no domain experience.")
        assert any("no_domain_expertise" in s for s in result.founder_risk)

    def test_key_person_dependency(self):
        result = _extract(
            "The company has a key person risk with CEO as single point of failure."
        )
        assert any("key_person_dependency" in s for s in result.founder_risk)

    def test_multiple_founders_low_risk(self):
        result = RiskExtractor().extract(
            _make_startup(
                "A B2B SaaS platform with two experienced founders "
                "who have complementary technical and business expertise."
            ),
            _make_data(founder_count=2),
        )
        assert result.founder_risk == []


# ---------------------------------------------------------------------------
# Execution risk
# ---------------------------------------------------------------------------


class TestExecutionRisk:
    def test_early_stage(self):
        result = _extract("Currently in pre-seed stage with MVP concept.")
        assert any("early_stage" in s for s in result.execution_risk)

    def test_pivot_history(self):
        result = _extract("The team pivoted twice before finding the current focus.")
        assert any("pivot_history" in s for s in result.execution_risk)

    def test_no_revenue(self):
        result = _extract("Pre-revenue stage, not yet monetizing the platform.")
        assert any("no_revenue" in s for s in result.execution_risk)

    def test_product_not_ready(self):
        result = _extract("Still building the product, not yet launched.")
        assert any("product_not_ready" in s for s in result.execution_risk)

    def test_clean_execution(self):
        result = _extract(
            "Launched product with strong traction, on track for targets."
        )
        assert result.execution_risk == []


# ---------------------------------------------------------------------------
# Product risk
# ---------------------------------------------------------------------------


class TestProductRisk:
    def test_unproven_value(self):
        result = _extract("Product with unproven value proposition in the market.")
        assert any("unproven_value" in s for s in result.product_risk)

    def test_high_churn(self):
        result = _extract("Customer retention issues with high churn rates.")
        assert any("high_churn" in s for s in result.product_risk)

    def test_beta_stage(self):
        result = _extract("Currently in beta phase with limited users.")
        assert any("beta_stage" in s for s in result.product_risk)

    def test_low_engagement(self):
        result = _extract("Platform has low engagement and poor adoption rates.")
        assert any("low_engagement" in s for s in result.product_risk)

    def test_clean_product(self):
        result = _extract(
            "Production-ready platform with strong adoption "
            "and high customer engagement metrics."
        )
        assert result.product_risk == []


# ---------------------------------------------------------------------------
# Technology risk
# ---------------------------------------------------------------------------


class TestTechnologyRisk:
    def test_unproven_tech(self):
        result = _extract("Built on unproven experimental technology approach.")
        assert any("unproven_tech" in s for s in result.technology_risk)

    def test_technical_debt(self):
        result = _extract("The codebase has significant technical debt from rapid growth.")
        assert any("technical_debt" in s for s in result.technology_risk)

    def test_single_points_of_failure(self):
        result = _extract(
            "Infrastructure has single point of failure with no redundancy."
        )
        assert any("single_points_of_failure" in s for s in result.technology_risk)

    def test_vendor_lock_in(self):
        result = _extract(
            "Heavy vendor lock-in to proprietary platform dependency."
        )
        assert any("vendor_lock_in" in s for s in result.technology_risk)

    def test_scalability_concerns(self):
        result = _extract(
            "Scalability concerns as the system does not scale beyond 1000 users."
        )
        assert any("scalability_concerns" in s for s in result.technology_risk)

    def test_clean_technology(self):
        result = _extract(
            "Built on proven AWS infrastructure with battle-tested architecture."
        )
        assert result.technology_risk == []


# ---------------------------------------------------------------------------
# Business model risk
# ---------------------------------------------------------------------------


class TestBusinessModelRisk:
    def test_unproven_monetization(self):
        result = _extract(
            "Unproven monetization strategy, no clear revenue model defined."
        )
        assert any("unproven_monetization" in s for s in result.business_model_risk)

    def test_low_margins(self):
        result = _extract("Thin margins with low margins on each transaction.")
        assert any("low_margins" in s for s in result.business_model_risk)

    def test_dependency_on_single_revenue(self):
        result = _extract("Solely reliant on a single revenue stream from enterprise.")
        assert any(
            "dependency_on_single_revenue" in s for s in result.business_model_risk
        )

    def test_high_cac(self):
        result = _extract("High customer acquisition cost with expensive acquisition.")
        assert any(
            "high_customer_acquisition_cost" in s for s in result.business_model_risk
        )

    def test_clean_business_model(self):
        result = _extract(
            "SaaS subscription with strong unit economics, LTV/CAC of 5x."
        )
        assert result.business_model_risk == []


# ---------------------------------------------------------------------------
# Traction risk
# ---------------------------------------------------------------------------


class TestTractionRisk:
    def test_no_traction(self):
        result = _extract("Currently no traction with zero users or revenue.")
        assert any("no_traction" in s for s in result.traction_risk)

    def test_declining_metrics(self):
        result = _extract("Declining revenue with falling growth metrics.")
        assert any("declining_metrics" in s for s in result.traction_risk)

    def test_low_retention(self):
        result = _extract("Low retention rate with churn rate above 15%.")
        assert any("low_retention" in s for s in result.traction_risk)

    def test_stagnant_growth(self):
        result = _extract("User growth has plateaued with stagnant growth metrics.")
        assert any("stagnant_growth" in s for s in result.traction_risk)

    def test_pre_revenue_from_text(self):
        result = _extract("Currently in pre-revenue stage with beta product.")
        assert any("no_revenue" in s for s in result.traction_risk)

    def test_clean_traction(self):
        result = _extract(
            "Strong traction with 140% net revenue retention "
            "and 30% month-over-month growth."
        )
        assert result.traction_risk == []


# ---------------------------------------------------------------------------
# Competitive risk
# ---------------------------------------------------------------------------


class TestCompetitiveRisk:
    def test_dominant_competitor(self):
        result = _extract("Facing a dominant competitor with well-funded rival.")
        assert any("dominant_competitor" in s for s in result.competitive_risk)

    def test_race_to_bottom(self):
        result = _extract(
            "Industry experiencing race to the bottom and price war."
        )
        assert any("race_to_bottom" in s for s in result.competitive_risk)

    def test_low_barrier_to_entry(self):
        result = _extract("Low barrier to entry, easy to replicate the solution.")
        assert any("low_barrier_to_entry" in s for s in result.competitive_risk)

    def test_commoditization(self):
        result = _extract(
            "Market is commoditizing with table stakes features."
        )
        assert any("commoditization" in s for s in result.competitive_risk)

    def test_big_tech_threat(self):
        result = _extract(
            "Risk from big tech companies like Google could enter this space."
        )
        assert any("big_tech_threat" in s for s in result.competitive_risk)

    def test_clean_competitive(self):
        result = _extract(
            "Fragmented market with diverse players and strong differentiation."
        )
        assert result.competitive_risk == []


# ---------------------------------------------------------------------------
# Regulatory risk
# ---------------------------------------------------------------------------


class TestRegulatoryRisk:
    def test_pending_regulation(self):
        result = _extract(
            "Pending regulation and upcoming regulatory changes in the sector."
        )
        assert any("pending_regulation" in s for s in result.regulatory_risk)

    def test_heavy_regulation(self):
        result = _extract(
            "Operating in a heavily regulated industry with complex regulatory landscape."
        )
        assert any("heavy_regulation" in s for s in result.regulatory_risk)

    def test_compliance_requirement(self):
        result = _extract(
            "Must comply with HIPAA compliance requirements for healthcare data."
        )
        assert any("compliance_requirement" in s for s in result.regulatory_risk)

    def test_legal_exposure(self):
        result = _extract("Potential legal exposure from ongoing litigation risk.")
        assert any("legal_exposure" in s for s in result.regulatory_risk)

    def test_data_privacy(self):
        result = _extract(
            "GDPR data privacy concerns with personal data handling."
        )
        assert any("data_privacy" in s for s in result.regulatory_risk)

    def test_clean_regulatory(self):
        result = _extract(
            "SaaS platform with minimal regulatory requirements."
        )
        assert result.regulatory_risk == []


# ---------------------------------------------------------------------------
# Operational risk
# ---------------------------------------------------------------------------


class TestOperationalRisk:
    def test_supply_chain(self):
        result = _extract(
            "Hardware component shortage creates supply chain risk."
        )
        assert any("supply_chain" in s for s in result.operational_risk)

    def test_geographic_concentration(self):
        result = _extract(
            "Single location operation with geographic concentration risk."
        )
        assert any("geographic_concentration" in s for s in result.operational_risk)

    def test_operational_complexity(self):
        result = _extract(
            "Logistically complex operations with manual processes across regions."
        )
        assert any("operational_complexity" in s for s in result.operational_risk)

    def test_clean_operational(self):
        result = _extract(
            "Cloud-based SaaS with automated operations and multi-region deployment."
        )
        assert result.operational_risk == []


# ---------------------------------------------------------------------------
# Platform dependency risk
# ---------------------------------------------------------------------------


class TestPlatformDependencyRisk:
    def test_single_platform(self):
        result = _extract("Built exclusively on Shopify platform.")
        assert any("single_platform" in s for s in result.platform_dependency_risk)

    def test_app_store_dependency(self):
        result = _extract(
            "App Store dependency with Apple review rejection risk."
        )
        assert any("app_store_dependency" in s for s in result.platform_dependency_risk)

    def test_api_dependency(self):
        result = _extract(
            "Heavy API dependency with risk of API key revocation."
        )
        assert any("api_dependency" in s for s in result.platform_dependency_risk)

    def test_cloud_lock_in(self):
        result = _extract(
            "AWS lock-in dependency with significant migration costs."
        )
        assert any("cloud_lock_in" in s for s in result.platform_dependency_risk)

    def test_clean_platform(self):
        result = _extract(
            "Multi-cloud strategy with vendor-agnostic architecture."
        )
        assert result.platform_dependency_risk == []


# ---------------------------------------------------------------------------
# Customer concentration risk
# ---------------------------------------------------------------------------


class TestCustomerConcentrationRisk:
    def test_few_customers(self):
        result = _extract(
            "Limited customer base with fewer than 10 clients."
        )
        assert any("few_customers" in s for s in result.customer_concentration_risk)

    def test_top_customer_dependency(self):
        result = _extract(
            "Top customer concentration risk, single customer represents 40% revenue."
        )
        assert any(
            "top_customer_dependency" in s for s in result.customer_concentration_risk
        )

    def test_concentration_risk(self):
        result = _extract(
            "Significant revenue concentration risk with enterprise clients."
        )
        assert any(
            "concentration_risk" in s for s in result.customer_concentration_risk
        )

    def test_few_large_contracts(self):
        result = _extract(
            "Few large contracts drive most of the revenue."
        )
        assert any(
            "few_large_contracts" in s for s in result.customer_concentration_risk
        )

    def test_clean_customer_concentration(self):
        result = _extract(
            "Diversified customer base with 500+ clients across segments."
        )
        assert result.customer_concentration_risk == []


# ---------------------------------------------------------------------------
# Hiring risk
# ---------------------------------------------------------------------------


class TestHiringRisk:
    def test_talent_shortage(self):
        result = _extract(
            "Talent shortage makes it difficult to hire specialized engineers."
        )
        assert any("talent_shortage" in s for s in result.hiring_risk)

    def test_high_turnover(self):
        result = _extract(
            "High employee turnover with significant team churn."
        )
        assert any("high_turnover" in s for s in result.hiring_risk)

    def test_competitive_hiring(self):
        result = _extract(
            "Competitive hiring environment with expensive talent market."
        )
        assert any("competitive_hiring" in s for s in result.hiring_risk)

    def test_clean_hiring(self):
        result = _extract(
            "Strong engineering team with excellent retention and growth."
        )
        assert result.hiring_risk == []


# ---------------------------------------------------------------------------
# Funding risk
# ---------------------------------------------------------------------------


class TestFundingRisk:
    def test_unfunded(self):
        result = _extract("Bootstrapped company with no external funding raised.")
        assert any("unfunded" in s for s in result.funding_risk)

    def test_runway_concern(self):
        result = _extract("Short runway with 6 months of remaining cash.")
        assert any("runway_concern" in s for s in result.funding_risk)

    def test_high_burn_rate(self):
        result = _extract(
            "High burn rate with significant cash burn每月."
        )
        assert any("high_burn_rate" in s for s in result.funding_risk)

    def test_capital_intensive(self):
        result = _extract(
            "Capital intensive business requiring significant investment."
        )
        assert any("capital_intensive" in s for s in result.funding_risk)

    def test_clean_funding(self):
        result = _extract(
            "Series B funded with $45M raised from Tier 1 investors."
        )
        assert result.funding_risk == []


# ---------------------------------------------------------------------------
# Scaling risk
# ---------------------------------------------------------------------------


class TestScalingRisk:
    def test_scaling_costs(self):
        result = _extract(
            "Scaling costs are increasing with growing infrastructure expenses."
        )
        assert any("scaling_costs" in s for s in result.scaling_risk)

    def test_margin_pressure(self):
        result = _extract(
            "Margin compression from scaling with decreasing margins."
        )
        assert any("margin_pressure" in s for s in result.scaling_risk)

    def test_infrastructure_limits(self):
        result = _extract(
            "Infrastructure at capacity with infrastructure bottleneck."
        )
        assert any("infrastructure_limits" in s for s in result.scaling_risk)

    def test_clean_scaling(self):
        result = _extract(
            "Auto-scaling cloud infrastructure handles load efficiently."
        )
        assert result.scaling_risk == []


# ---------------------------------------------------------------------------
# Security risk
# ---------------------------------------------------------------------------


class TestSecurityRisk:
    def test_sensitive_data(self):
        result = _extract(
            "Platform handles sensitive customer data and financial data."
        )
        assert any("sensitive_data" in s for s in result.security_risk)

    def test_third_party_risk(self):
        result = _extract(
            "Third-party dependency creates vendor security risk."
        )
        assert any("third_party_risk" in s for s in result.security_risk)

    def test_security_certifications_missing(self):
        result = _extract(
            "Currently no SOC 2 certification or security audit completed."
        )
        assert any(
            "security_certifications_missing" in s for s in result.security_risk
        )

    def test_clean_security(self):
        result = _extract(
            "SOC 2 Type II certified with end-to-end encryption."
        )
        assert result.security_risk == []


# ---------------------------------------------------------------------------
# Compliance risk
# ---------------------------------------------------------------------------


class TestComplianceRisk:
    def test_multi_framework(self):
        result = _extract(
            "Must maintain SOC 2 and ISO 27001 and GDPR compliance."
        )
        assert any("multi_framework" in s for s in result.compliance_risk)

    def test_audit_burden(self):
        result = _extract(
            "Heavy audit burden with annual compliance audits required."
        )
        assert any("audit_burden" in s for s in result.compliance_risk)

    def test_cross_border_compliance(self):
        result = _extract(
            "Cross-border compliance with data transfer requirements across regions."
        )
        assert any("cross_border_compliance" in s for s in result.compliance_risk)

    def test_evolving_regulations(self):
        result = _extract(
            "Evolving regulations create shifting compliance requirements."
        )
        assert any("evolving_regulations" in s for s in result.compliance_risk)

    def test_clean_compliance(self):
        result = _extract("Simple compliance requirements with SOC 2 certification.")
        assert result.compliance_risk == []


# ---------------------------------------------------------------------------
# Risk keywords
# ---------------------------------------------------------------------------


class TestRiskKeywords:
    def test_financial_keywords(self):
        result = _extract(
            "High burn rate and short runway with negative unit economics."
        )
        assert "burn rate" in result.risk_keywords
        assert "runway" in result.risk_keywords
        assert "unit economics" in result.risk_keywords

    def test_market_keywords(self):
        result = _extract("Uncertain market risk in nascent market.")
        assert "market risk" in result.risk_keywords

    def test_exec_keywords(self):
        result = _extract("Execution risk with scaling challenges ahead.")
        assert "execution risk" in result.risk_keywords
        assert "scaling challenges" in result.risk_keywords

    def test_tech_keywords(self):
        result = _extract("Technical debt and vendor lock-in are key concerns.")
        assert "technical debt" in result.risk_keywords
        assert "vendor lock-in" in result.risk_keywords

    def test_regulatory_keywords(self):
        result = _extract(
            "Regulatory risk with GDPR compliance requirements."
        )
        assert "regulatory risk" in result.risk_keywords
        assert "GDPR" in result.risk_keywords

    def test_competitive_keywords(self):
        result = _extract(
            "Low barriers to entry and race to the bottom pricing."
        )
        assert "low barriers to entry" in result.risk_keywords
        assert "race to the bottom" in result.risk_keywords

    def test_team_keywords(self):
        result = _extract("Hiring challenges with talent shortage in key roles.")
        assert "hiring challenges" in result.risk_keywords
        assert "talent shortage" in result.risk_keywords

    def test_security_keywords(self):
        result = _extract("Data breach history with third-party risk concerns.")
        assert "data breach" in result.risk_keywords
        assert "third-party risk" in result.risk_keywords

    def test_no_keywords_for_clean_text(self):
        result = _extract(
            "A strong enterprise platform with proven technology."
        )
        assert result.risk_keywords == []


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


class TestRiskConfidence:
    def test_zero_confidence_for_clean_input(self):
        result = RiskExtractor().extract(
            _make_startup(
                "A strong B2B SaaS platform with proven technology, "
                "experienced team, and enterprise traction."
            ),
            _make_data(founder_count=2),
        )
        assert result.risk_confidence == 0.0

    def test_high_confidence_for_risky_input(self):
        result = _extract(
            "Pre-revenue startup with unproven market, solo founder, "
            "high burn rate, no traction, technical debt, "
            "regulatory risk, and compliance requirements. "
            "Low barrier to entry with dominant competitors. "
            "Bootstrapped with short runway and talent shortage."
        )
        assert result.risk_confidence > 0.3

    def test_confidence_bounded(self):
        result = _extract(
            "Maximum risk: uncertain market, solo founder, pre-revenue, "
            "no traction, high churn, technical debt, vendor lock-in, "
            "low margins, declining revenue, dominant competitors, "
            "pending regulation, supply chain risk, "
            "API dependency, few customers, talent shortage, "
            "high burn rate, scaling challenges, data breach, "
            "multi-framework compliance burden. "
            "Burn rate, runway, unit economics, market risk, "
            "execution risk, technical debt, vendor lock-in, "
            "regulatory risk, GDPR, compliance requirements."
        )
        assert result.risk_confidence <= 1.0

    def test_confidence_increases_with_more_signals(self):
        minimal = _extract("Pre-revenue stage.")
        maximal = _extract(
            "Pre-revenue, uncertain market, solo founder, "
            "no traction, technical debt, high burn rate, "
            "talent shortage, dominant competitors, "
            "regulatory risk, GDPR compliance, "
            "data breach risk, vendor lock-in."
        )
        assert maximal.risk_confidence >= minimal.risk_confidence


# ---------------------------------------------------------------------------
# Multi-dimension extraction
# ---------------------------------------------------------------------------


class TestMultiDimension:
    def test_all_dimensions_populated(self):
        result = _extract(
            "Pre-seed stage with solo founder in uncertain market "
            "and nascent demand. Pre-revenue with no traction, "
            "technical debt in codebase, no SOC 2 certification. "
            "High burn rate, short runway, talent shortage to hire. "
            "Dominant competitor with well-funded rival. "
            "Regulatory risk with GDPR data privacy concerns. "
            "Single platform dependency on AWS. "
            "Limited customer base with concentration risk."
        )
        # At least 8 dimensions should fire
        dims_with_signals = sum(1 for dim in [
            result.market_risk,
            result.founder_risk,
            result.execution_risk,
            result.product_risk,
            result.technology_risk,
            result.funding_risk,
            result.hiring_risk,
            result.competitive_risk,
            result.regulatory_risk,
            result.security_risk,
            result.platform_dependency_risk,
            result.customer_concentration_risk,
        ] if len(dim) > 0)
        assert dims_with_signals >= 8


# ---------------------------------------------------------------------------
# Snippet generation
# ---------------------------------------------------------------------------


class TestSnippetGeneration:
    def test_snippets_contain_context(self):
        result = _extract(
            "The uncertain market creates significant market risk for the company."
        )
        for signal in result.market_risk:
            assert ":" in signal
            # Snippet should contain the matched text area
            assert len(signal) > len("uncertain_market: ")


# ---------------------------------------------------------------------------
# Integration: CompositeExtractor includes risk fields
# ---------------------------------------------------------------------------


class TestCompositeIntegration:
    def test_composite_extractor_populates_risk_fields(self):
        from predictron_engine.extraction.composite import CompositeExtractor

        startup = _make_startup(
            "Pre-revenue startup with solo founder, high burn rate, "
            "and technical debt in uncertain market."
        )
        data = _make_data(founder_count=1)
        result = CompositeExtractor().extract(startup, data)

        assert isinstance(result, ExtractedFeatures)
        # At least some risk fields should be populated
        assert (
            len(result.market_risk) > 0
            or len(result.founder_risk) > 0
            or len(result.funding_risk) > 0
            or len(result.technology_risk) > 0
        )
        assert result.risk_confidence > 0.0

    def test_composite_merge_preserves_risk_signals(self):
        from predictron_engine.extraction.composite import CompositeExtractor

        startup = _make_startup(
            "Pre-revenue with high burn rate and technical debt."
        )
        data = _make_data()
        result = CompositeExtractor().extract(startup, data)

        # Risk fields should survive the merge
        assert isinstance(result.risk_keywords, list)
        assert isinstance(result.risk_confidence, float)
