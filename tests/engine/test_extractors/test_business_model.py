"""Comprehensive tests for BusinessModelExtractor (Sprint 4).

Covers all 19 extraction dimensions with weighted scoring validation,
ambiguity handling, edge cases, and benchmark case alignment.
"""

from __future__ import annotations

import pytest

from predictron_engine.extraction.extractors.business_model import (
    BusinessModelExtractor,
)
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


def _extract(desc: str, name: str = "TestCo") -> ExtractedFeatures:
    startup = Startup(
        name=name,
        website=f"https://{name.lower()}.example.com",
        description=desc,
    )
    from predictron_engine.models.collected_data import CollectedData

    data = CollectedData(
        startup_name=name,
        website_domain=f"{name.lower()}.example.com",
        description_tokens=[],
        description_word_count=len(desc.split()),
        has_website=True,
        has_pitch_deck=False,
        founder_count=0,
        url_metadata={},
        enrichment_signals={},
    )
    return BusinessModelExtractor().extract(startup, data)


class TestBusinessModelExtractorReturnType:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = BusinessModelExtractor().extract(
            sample_startup, sample_collected_data
        )
        assert isinstance(result, ExtractedFeatures)

    def test_all_new_fields_present(self, sample_collected_data):
        startup = Startup(
            name="FullCo",
            website="https://fullco.example.com",
            description=(
                "An enterprise SaaS platform with monthly subscription "
                "pricing, annual contracts, and recurring revenue."
            ),
        )
        result = BusinessModelExtractor().extract(startup, sample_collected_data)
        assert result.business_model is not None
        assert result.secondary_business_model is None
        assert result.customer_type is not None
        assert result.revenue_model is not None
        assert result.pricing_model is not None
        assert result.monetization_strategy is not None
        assert result.business_model_confidence > 0.0


class TestPrimaryBusinessModel:
    @pytest.mark.parametrize(
        "description,expected",
        [
            (
                "An enterprise saas platform with monthly subscription pricing "
                "and annual contracts for mid-market enterprises.",
                "saas",
            ),
            (
                "A two-sided marketplace with platform fee and take rate model "
                "connecting manufacturers with buyers.",
                "marketplace",
            ),
            (
                "Embedded payment infrastructure handling split payments, "
                "escrow, and transaction processing with per-transaction fees.",
                "transactional",
            ),
            (
                "FDA-cleared AI diagnostic tools with per-study licensing "
                "and enterprise hospital contracts.",
                "licensing",
            ),
            (
                "Robotics-as-a-service model with per-robot monthly fees "
                "for warehouse automation.",
                "paas",
            ),
            (
                "Consumer mobile application with freemium model and "
                "a $9.99/month premium subscription.",
                "saas",
            ),
            (
                "Open-source-first CI/CD platform converting open-source "
                "users to paid plans at $299/month.",
                "saas",
            ),
            (
                "Enterprise compliance management platform with ACV of "
                "$120,000 and 98% gross retention.",
                "saas",
            ),
            (
                "GPU-optimized model serving infrastructure with usage-based "
                "pricing averaging $18,000/month.",
                "paas",
            ),
            (
                "Enterprise carbon accounting software with ARR of $3.5M "
                "and 95% gross retention.",
                "saas",
            ),
        ],
    )
    def test_primary_business_model(self, description, expected):
        result = _extract(description)
        assert result.business_model == expected


class TestSecondaryBusinessModel:
    def test_hybrid_saas_marketplace(self):
        desc = (
            "A SaaS analytics platform with subscription pricing that also "
            "operates a marketplace connecting data providers with enterprises. "
            "The marketplace takes a 15% platform fee on transactions."
        )
        result = _extract(desc)
        assert result.business_model == "saas"
        assert result.secondary_business_model == "marketplace"

    def test_no_secondary_when_single_model(self):
        desc = (
            "An enterprise saas platform with monthly subscription pricing "
            "and annual contracts for mid-market enterprises."
        )
        result = _extract(desc)
        assert result.secondary_business_model is None

    def test_no_secondary_when_primary_is_none(self):
        result = _extract("A generic technology company.")
        assert result.business_model is None
        assert result.secondary_business_model is None


class TestCustomerTypeClassification:
    @pytest.mark.parametrize(
        "description,expected",
        [
            (
                "An enterprise B2B platform for businesses and organizations.",
                "b2b",
            ),
            (
                "A consumer app for personal use by individual end users.",
                "b2c",
            ),
            (
                "A two-sided B2B marketplace connecting manufacturers "
                "with small and mid-size manufacturing facilities.",
                "b2b",
            ),
            (
                "A consumer mobile application that combines fitness "
                "tracking with social networking for individual users.",
                "b2c",
            ),
        ],
    )
    def test_customer_type(self, description, expected):
        result = _extract(description)
        assert result.customer_type == expected

    def test_b2b2c_detection(self):
        desc = (
            "A B2B2C two-sided platform connecting businesses with "
            "consumer end users through a marketplace."
        )
        result = _extract(desc)
        assert result.customer_type == "b2b2c"

    def test_b2g_detection(self):
        desc = (
            "A government B2G platform for federal public sector "
            "procurement and defense compliance."
        )
        result = _extract(desc)
        assert result.customer_type == "b2g"


class TestRevenueModel:
    @pytest.mark.parametrize(
        "description,expected",
        [
            (
                "Subscription-based SaaS with monthly recurring revenue "
                "and annual contract values of $48,000 ARR.",
                "subscription",
            ),
            (
                "Per-study licensing with enterprise hospital contracts "
                "and patent licensing for medical imaging.",
                "licensing",
            ),
            (
                "Embedded payment infrastructure handling $2.1B in annual "
                "payment volume with a take rate of 0.8%.",
                "transaction_fee",
            ),
            (
                "Two-sided B2B marketplace with 12% marketplace take rate "
                "on $180M in GMV annually.",
                "commission",
            ),
            (
                "Open-source-first platform converting free users to paid "
                "plans. Freemium model with $299/month pricing.",
                "freemium",
            ),
            (
                "Usage-based pricing with average customer spend of "
                "$18,000/month across inference requests.",
                "usage_based",
            ),
        ],
    )
    def test_revenue_model(self, description, expected):
        result = _extract(description)
        assert result.revenue_model == expected


class TestPricingModel:
    @pytest.mark.parametrize(
        "description,expected",
        [
            (
                "Annual contract values averaging $48,000 ARR with "
                "95% gross retention for enterprise clients.",
                "annual_contract",
            ),
            (
                "Per-study licensing with enterprise hospital contracts "
                "and 97.3% detection sensitivity.",
                "per_study",
            ),
            (
                "Usage-based pricing with average customer spend of "
                "$18,000/month for GPU inference processing.",
                "usage_based",
            ),
            (
                "ACV of $120,000 with 98% gross retention "
                "and annual contract structure.",
                "annual_contract",
            ),
            (
                "Freemium model converting open-source users to paid "
                "plans at $299/month with tiered pricing.",
                "tiered",
            ),
        ],
    )
    def test_pricing_model(self, description, expected):
        result = _extract(description)
        assert result.pricing_model == expected


class TestMonetizationStrategy:
    def test_subscription_monetization(self):
        desc = (
            "Enterprise saas platform with subscription-based monthly "
            "pricing and recurring revenue from enterprise clients."
        )
        result = _extract(desc)
        assert result.monetization_strategy == "subscription"

    def test_commission_monetization(self):
        desc = (
            "Two-sided B2B marketplace with marketplace take rate of 12% "
            "and GMV of $180M annually with platform fees."
        )
        result = _extract(desc)
        assert result.monetization_strategy == "commission"

    def test_freemium_monetization(self):
        desc = (
            "Consumer mobile app with freemium model. Free tier and "
            "open-source-first approach with community growth."
        )
        result = _extract(desc)
        assert result.monetization_strategy == "freemium"

    def test_licensing_monetization(self):
        desc = (
            "Medical imaging AI with per-study licensing, patent licensing "
            "for neural network architecture and enterprise licensing."
        )
        result = _extract(desc)
        assert result.monetization_strategy == "licensing"


class TestCustomerAcquisitionModel:
    def test_product_led_acquisition(self):
        desc = (
            "Open-source-first CI/CD platform with 14,000 GitHub stars "
            "and 850 contributing developers. Converting free users to "
            "paid plans through freemium and community engagement."
        )
        result = _extract(desc)
        assert result.customer_acquisition_model == "product_led"

    def test_sales_led_acquisition(self):
        desc = (
            "Enterprise compliance platform with direct sales team. "
            "Sales professionals and account executives selling to "
            "95 financial institutions including Fortune 500 clients."
        )
        result = _extract(desc)
        assert result.customer_acquisition_model == "sales_led"


class TestSalesMotion:
    def test_enterprise_sales_motion(self):
        desc = (
            "Enterprise clients including Fortune 500 companies. "
            "Direct sales team with account executives targeting "
            "85 enterprise customers with $48,000 ARR contracts."
        )
        result = _extract(desc)
        assert result.sales_motion == "enterprise_sales"

    def test_product_led_sales_motion(self):
        desc = (
            "Open-source-first platform with 14,000 GitHub stars. "
            "Freemium model converting to paid plans. "
            "420,000 monthly active users with self-serve onboarding."
        )
        result = _extract(desc)
        assert result.sales_motion == "product_led"


class TestDistributionModel:
    def test_direct_distribution(self):
        desc = (
            "Direct sales team selling enterprise saas to "
            "85 enterprise clients with annual contracts."
        )
        result = _extract(desc)
        assert result.distribution_model == "direct"

    def test_api_distribution(self):
        desc = (
            "API-first payment infrastructure with developer platform "
            "and REST API for marketplace businesses."
        )
        result = _extract(desc)
        assert result.distribution_model == "api"

    def test_app_store_distribution(self):
        desc = (
            "Consumer mobile application available on iOS and Android. "
            "Fitness tracking with social networking features."
        )
        result = _extract(desc)
        assert result.distribution_model == "app_store"

    def test_open_source_distribution(self):
        desc = (
            "Open-source-first CI/CD platform with GitHub stars and "
            "contributing developers community."
        )
        result = _extract(desc)
        assert result.distribution_model == "open_source"


class TestValuePropositionSignals:
    def test_efficiency_and_insights(self):
        desc = (
            "Automated analytics platform providing real-time insight "
            "generation and dashboard reporting for business intelligence."
        )
        result = _extract(desc)
        assert "efficiency_gain" in result.value_proposition_signals
        assert "data_insights" in result.value_proposition_signals

    def test_compliance_and_risk(self):
        desc = (
            "Enterprise compliance platform automating regulatory tracking "
            "and audit preparation for security and risk mitigation."
        )
        result = _extract(desc)
        assert "risk_mitigation" in result.value_proposition_signals

    def test_scalability_signal(self):
        desc = (
            "Cloud-native SaaS platform with auto-scaling infrastructure "
            "for growing enterprise clients."
        )
        result = _extract(desc)
        assert "scalability" in result.value_proposition_signals


class TestRecurringRevenueSignal:
    def test_recurring_revenue(self):
        desc = (
            "Subscription-based SaaS with monthly recurring revenue "
            "and annual contracts. 140% net revenue retention."
        )
        result = _extract(desc)
        assert result.recurring_revenue_signal == "recurring"

    def test_transactional_revenue(self):
        desc = (
            "Per-transaction payment processing with take rate model "
            "and transaction fees on payment volume."
        )
        result = _extract(desc)
        assert result.recurring_revenue_signal == "transactional"

    def test_mixed_revenue(self):
        desc = (
            "Subscription revenue with recurring MRR and annual "
            "contracts, plus per-transaction fees on payment volume."
        )
        result = _extract(desc)
        assert result.recurring_revenue_signal == "mixed"

    def test_unknown_revenue(self):
        result = _extract("A technology company building products.")
        assert result.recurring_revenue_signal == "unknown"


class TestMarketplaceDynamics:
    def test_marketplace_dynamics_detected(self):
        desc = (
            "Two-sided marketplace with take rate, platform fee, "
            "and GMV metrics. Supply and demand dynamics with "
            "supplier and buyer listings."
        )
        result = _extract(desc)
        assert len(result.marketplace_dynamics) > 0
        assert "two_sided_marketplace" in result.marketplace_dynamics
        assert "take_rate_model" in result.marketplace_dynamics

    def test_no_marketplace_dynamics(self):
        result = _extract(
            "Enterprise saas platform with subscription pricing."
        )
        assert result.marketplace_dynamics == []


class TestNetworkEffects:
    def test_network_effects_detected(self):
        desc = (
            "Platform with strong network effects and viral growth. "
            "Ecosystem lock-in through developer platform and "
            "flywheel dynamics."
        )
        result = _extract(desc)
        assert len(result.network_effects_signals) > 0

    def test_no_network_effects(self):
        result = _extract("Simple mobile application for personal use.")
        assert result.network_effects_signals == []


class TestPlatformCharacteristics:
    def test_platform_characteristics_detected(self):
        desc = (
            "API-first, cloud-native, multi-tenant developer platform "
            "with self-serve and usage-based scaling."
        )
        result = _extract(desc)
        assert len(result.platform_characteristics) > 0
        assert "api_first" in result.platform_characteristics
        assert "cloud_native" in result.platform_characteristics

    def test_no_platform_characteristics(self):
        result = _extract("A simple consulting services company.")
        assert result.platform_characteristics == []


class TestSwitchingCostIndicators:
    def test_switching_costs_detected(self):
        desc = (
            "Enterprise compliance platform with deep ERP integration "
            "creating workflow integration lock-in and vendor lock."
        )
        result = _extract(desc)
        assert len(result.switching_cost_indicators) > 0

    def test_no_switching_costs(self):
        result = _extract("Consumer mobile app for fitness tracking.")
        assert result.switching_cost_indicators == []


class TestUnitEconomicsIndicators:
    def test_unit_economics_detected(self):
        desc = (
            "SaaS platform with LTV/CAC ratio of 4.2x, "
            "95% gross retention and $3.5M ARR. "
            "Net revenue retention of 140%."
        )
        result = _extract(desc)
        assert len(result.unit_economics_indicators) > 0
        assert "ltv_cac_ratio" in result.unit_economics_indicators
        assert "gross_retention" in result.unit_economics_indicators

    def test_no_unit_economics(self):
        result = _extract("A technology startup building products.")
        assert result.unit_economics_indicators == []


class TestBusinessModelMaturity:
    @pytest.mark.parametrize(
        "description,expected",
        [
            ("Pre-revenue on commercial tier, in beta launch phase.", "nascent"),
            (
                "Pre-seed stage with $1.5M raised from angels. "
                "420,000 monthly active users with 38,000 premium subscribers.",
                "early",
            ),
            (
                "$4.2M ARR with 140% net revenue retention. "
                "85 enterprise clients. Series A funded.",
                "growth",
            ),
            (
                "Series C stage with $85M raised. "
                "95 financial institutions including top-20 US banks.",
                "growth",
            ),
        ],
    )
    def test_maturity_classification(self, description, expected):
        result = _extract(description)
        assert result.business_model_maturity == expected


class TestBusinessModelKeywords:
    def test_saas_keywords(self):
        desc = (
            "Enterprise B2B SaaS platform with subscription pricing, "
            "annual contracts, and recurring revenue."
        )
        result = _extract(desc)
        assert "saas" in result.business_model_keywords
        assert "subscription" in result.business_model_keywords
        assert "b2b" in result.business_model_keywords
        assert "recurring_revenue" in result.business_model_keywords

    def test_marketplace_keywords(self):
        desc = (
            "Two-sided marketplace with take rate model, GMV metrics, "
            "and network effects."
        )
        result = _extract(desc)
        assert "marketplace" in result.business_model_keywords
        assert "two_sided" in result.business_model_keywords
        assert "take_rate" in result.business_model_keywords


class TestBusinessModelConfidence:
    def test_high_confidence_rich_description(self):
        desc = (
            "Enterprise B2B SaaS platform providing real-time analytics "
            "and business intelligence for mid-market enterprises. "
            "Revenue is subscription-based with annual contracts averaging "
            "$48,000 ARR. 85 enterprise clients with 140% net revenue "
            "retention. $4.2M ARR."
        )
        result = _extract(desc)
        assert result.business_model_confidence >= 0.5

    def test_low_confidence_sparse_description(self):
        result = _extract("A technology company.")
        assert result.business_model_confidence < 0.3

    def test_confidence_range(self):
        desc = (
            "Enterprise saas with subscription pricing, annual contracts, "
            "recurring revenue, and usage-based scaling."
        )
        result = _extract(desc)
        assert 0.0 <= result.business_model_confidence <= 1.0


class TestNoMatchReturnsNone:
    def test_returns_none_for_generic_description(self, sample_collected_data):
        startup = Startup(
            name="GenericCo",
            website="https://genericco.example.com",
            description="A sample startup for testing purposes.",
        )
        result = BusinessModelExtractor().extract(startup, sample_collected_data)
        assert result.business_model is None
        assert result.customer_type is None
        assert result.revenue_model is None
        assert result.pricing_model is None
        assert result.monetization_strategy is None
        assert result.business_model_confidence < 0.2


class TestBenchmarkCaseAlignment:
    """Validate extraction against all 10 benchmark cases."""

    def test_b2b_saas_analytix_cloud(self):
        desc = (
            "Analytix Cloud is a B2B SaaS platform providing real-time "
            "analytics and business intelligence for mid-market enterprises. "
            "The platform integrates with existing data warehouses and "
            "delivers automated insight generation through proprietary "
            "query optimization. Revenue is subscription-based with annual "
            "contracts averaging $48,000 ARR. The company has 85 enterprise "
            "clients and has achieved $4.2M ARR with 140% net revenue "
            "retention. Founded in 2021, the team of 35 engineers and "
            "sales professionals operates from San Francisco with a remote "
            "engineering hub in Austin."
        )
        result = _extract(desc, "Analytix Cloud")
        assert result.business_model == "saas"
        assert result.customer_type == "b2b"
        assert result.revenue_model == "subscription"
        assert result.recurring_revenue_signal == "recurring"
        assert result.sales_motion == "enterprise_sales"
        assert result.business_model_maturity == "growth"

    def test_healthcare_ai_medvision(self):
        desc = (
            "MedVision AI develops FDA-cleared AI diagnostic tools for "
            "medical imaging. The platform analyzes X-rays, MRIs, and CT "
            "scans to assist radiologists in detecting anomalies with "
            "97.3% sensitivity. The company holds 3 patents on its "
            "convolutional neural network architecture and has completed "
            "clinical trials across 12 hospital systems. Revenue model is "
            "per-study licensing with enterprise hospital contracts. "
            "Currently generating $2.8M ARR from 40 hospital clients. "
            "Series A funded with $12M raised from healthcare-focused VCs. "
            "Based in Boston with regulatory and clinical teams."
        )
        result = _extract(desc, "MedVision AI")
        assert result.business_model == "licensing"
        assert result.customer_type == "b2b"
        assert result.revenue_model == "licensing"
        assert result.pricing_model == "per_study"
        assert result.recurring_revenue_signal == "recurring"
        assert result.sales_motion == "enterprise_sales"

    def test_fintech_paybridge(self):
        desc = (
            "PayBridge provides embedded payment infrastructure for "
            "marketplace and platform businesses. The API-first solution "
            "handles split payments, escrow, KYC compliance, and multi-"
            "currency settlement across 35 countries. Processing $2.1B in "
            "annual payment volume with a take rate of 0.8%. The company "
            "serves 280 marketplace clients including 12 in the Fortune "
            "500. Founded in 2019, team of 120 across London, Singapore, "
            "and New York. Raised $45M Series B from Tier 1 fintech investors."
        )
        result = _extract(desc, "PayBridge")
        assert result.business_model == "transactional"
        assert result.customer_type == "b2b"
        assert result.revenue_model == "transaction_fee"
        assert result.recurring_revenue_signal == "transactional"
        assert result.sales_motion == "enterprise_sales"

    def test_devtools_shipkit(self):
        desc = (
            "ShipKit is an open-source-first CI/CD platform designed for "
            "monorepo architectures. The platform provides incremental "
            "builds, intelligent test parallelization, and deployment "
            "orchestration for teams running microservices. The open-source "
            "core has 14,000 GitHub stars and 850 contributing developers. "
            "Commercial features include audit logging, SSO, and priority "
            "support. Currently converting 340 open-source users to paid "
            "plans at $299/month. Pre-revenue on commercial tier, "
            "currently in beta launch phase. Founded in 2023 by two "
            "ex-GitHub engineers based in Seattle."
        )
        result = _extract(desc, "ShipKit")
        assert result.business_model == "saas"
        assert result.customer_type == "b2b"
        assert result.revenue_model == "freemium"
        assert result.recurring_revenue_signal == "recurring"
        assert result.sales_motion == "product_led"
        assert result.distribution_model == "open_source"

    def test_marketplace_supplyhub(self):
        desc = (
            "SupplyHub operates a two-sided B2B marketplace connecting "
            "industrial equipment manufacturers with small and mid-size "
            "manufacturing facilities. The platform handles quoting, "
            "procurement, and logistics for MRO supplies across North "
            "America. Marketplace take rate is 12% on transactions. "
            "Currently facilitating $180M in GMV annually with 2,400 "
            "supplier listings and 8,500 active buyer accounts. Revenue "
            "of $21.6M with unit economics showing LTV/CAC of 4.2x. "
            "Raised $30M Series A. Founded in 2020, team of 65 based "
            "in Chicago."
        )
        result = _extract(desc, "SupplyHub")
        assert result.business_model == "marketplace"
        assert result.customer_type == "b2b"
        assert result.revenue_model == "commission"
        assert result.recurring_revenue_signal == "transactional"
        assert len(result.marketplace_dynamics) > 0

    def test_consumer_app_fitsocial(self):
        desc = (
            "FitSocial is a consumer mobile application that combines "
            "fitness tracking with social networking. Users log workouts, "
            "share progress, and participate in community challenges. "
            "Monetization is through a freemium model with a $9.99/month "
            "premium subscription and sponsored brand partnerships. "
            "Currently has 420,000 monthly active users with 38,000 "
            "premium subscribers. Retention at D30 is 42%. Available on "
            "iOS and Android. Founded in 2022, team of 18, based in "
            "Los Angeles. Pre-seed stage with $1.5M raised from angels."
        )
        result = _extract(desc, "FitSocial")
        assert result.business_model == "saas"
        assert result.customer_type == "b2c"
        assert result.revenue_model in ("subscription", "freemium")
        assert result.distribution_model == "app_store"

    def test_climate_tech_carbonlens(self):
        desc = (
            "CarbonLens provides enterprise-grade carbon accounting and "
            "emissions tracking software for Scope 1, 2, and 3 emissions. "
            "The platform integrates with ERP systems, supply chain tools, "
            "and IoT sensors to automate emissions data collection and "
            "reporting. Aligned with GHG Protocol and SEC climate disclosure "
            "requirements. Serving 60 enterprise clients across "
            "manufacturing, logistics, and energy sectors. ARR of $3.5M "
            "with 95% gross retention. Raised $8M seed from climate-focused "
            "VCs. Founded in 2022, team of 28, headquartered in Berlin "
            "with operations in the US."
        )
        result = _extract(desc, "CarbonLens")
        assert result.business_model == "saas"
        assert result.customer_type == "b2b"
        assert result.revenue_model == "subscription"
        assert result.recurring_revenue_signal == "recurring"
        assert result.sales_motion == "enterprise_sales"

    def test_robotics_autoware(self):
        desc = (
            "AutoWare Robotics builds autonomous mobile robots for "
            "warehouse and fulfillment center operations. The robots use "
            "LiDAR-based SLAM navigation and handle goods-to-person picking, "
            "inventory scanning, and pallet transport. The company offers "
            "a Robotics-as-a-Service model with per-robot monthly fees "
            "covering hardware, maintenance, and software updates. Current "
            "fleet of 800 deployed robots across 15 customer facilities. "
            "Revenue model generates $6.4M ARR from RaaS contracts. "
            "Hardware costs funded through $52M in total funding including "
            "a Series B. Founded in 2019, team of 110 in Munich and Detroit."
        )
        result = _extract(desc, "AutoWare Robotics")
        assert result.business_model == "paas"
        assert result.customer_type == "b2b"
        assert result.revenue_model == "subscription"
        assert result.recurring_revenue_signal == "recurring"
        assert result.business_model_maturity == "growth"

    def test_enterprise_software_complianceos(self):
        desc = (
            "ComplianceOS is an enterprise compliance management platform "
            "that automates regulatory tracking, policy management, and "
            "audit preparation for financial services companies. The "
            "platform covers SOC 2, PCI DSS, GDPR, and CCPA compliance "
            "workflows with automated evidence collection and control "
            "monitoring. Serving 95 financial institutions including 8 "
            "top-20 US banks. ACV of $120,000 with 98% gross retention. "
            "Series C stage with $85M raised. Team of 200 across New York, "
            "Charlotte, and remote. Founded in 2018."
        )
        result = _extract(desc, "ComplianceOS")
        assert result.business_model == "saas"
        assert result.customer_type == "b2b"
        assert result.revenue_model == "subscription"
        assert result.recurring_revenue_signal == "recurring"
        assert result.sales_motion == "enterprise_sales"

    def test_ai_infrastructure_inferencelabs(self):
        desc = (
            "Inference Labs provides GPU-optimized model serving "
            "infrastructure for teams deploying large language models in "
            "production. The platform handles auto-scaling, model versioning, "
            "A/B testing, and cost optimization across multi-cloud GPU "
            "clusters. Supports PyTorch, TensorFlow, and ONNX models. "
            "Processing over 2 billion inference requests daily for 150 "
            "enterprise customers. Usage-based pricing with average customer "
            "spend of $18,000/month. Raised $65M Series A. Founded in 2023 "
            "by former infrastructure leads from major AI labs. Team of 75 "
            "in San Francisco and Toronto."
        )
        result = _extract(desc, "Inference Labs")
        assert result.business_model == "paas"
        assert result.customer_type == "b2b"
        assert result.revenue_model == "usage_based"
        assert result.recurring_revenue_signal == "recurring"
        assert result.sales_motion == "enterprise_sales"
