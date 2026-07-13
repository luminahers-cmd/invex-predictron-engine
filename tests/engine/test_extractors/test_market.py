from __future__ import annotations

from predictron_engine.extraction.extractors.market import MarketExtractor
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


def _make_startup(description: str, name: str = "TestCo") -> Startup:
    return Startup(
        name=name,
        website="https://test.example.com",
        description=description,
    )


class TestMarketExtractorReturns:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_only_populates_market_fields(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.business_model is None
        assert result.funding_stage is None
        assert result.technology_stack == []
        assert result.founded_year is None
        assert result.has_revenue is None


class TestIndustryClassification:
    def test_fintech(self, sample_collected_data):
        startup = _make_startup(
            "A fintech company providing banking and payments solutions."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "fintech"
        assert result.industry_confidence > 0.3

    def test_healthtech(self, sample_collected_data):
        startup = _make_startup(
            "MedVision AI develops FDA-cleared AI diagnostic tools for "
            "medical imaging. The platform analyzes X-rays, MRIs, and CT "
            "scans to assist radiologists."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "healthtech"
        assert result.industry_confidence > 0.3

    def test_enterprise_saas(self, sample_collected_data):
        startup = _make_startup(
            "An enterprise saas crm platform for b2b companies "
            "providing subscription-based customer management."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "enterprise_saas"

    def test_marketplace(self, sample_collected_data):
        startup = _make_startup(
            "A two-sided marketplace connecting suppliers with buyers "
            "through network effects and platform fees."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "marketplace"

    def test_consumer_tech(self, sample_collected_data):
        startup = _make_startup(
            "A consumer mobile app that combines fitness tracking with "
            "social networking. Users log workouts and share progress."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "consumer_tech"

    def test_climate_tech(self, sample_collected_data):
        startup = _make_startup(
            "CarbonLens provides enterprise-grade carbon accounting and "
            "emissions tracking software for Scope 1, 2, and 3 emissions "
            "aligned with GHG Protocol."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "climate_tech"

    def test_hardware_robotics(self, sample_collected_data):
        startup = _make_startup(
            "AutoWare Robotics builds autonomous mobile robots for "
            "warehouse operations using LiDAR-based SLAM navigation."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "hardware"

    def test_ai_ml(self, sample_collected_data):
        startup = _make_startup(
            "Inference Labs provides GPU-optimized model serving "
            "infrastructure for teams deploying large language models "
            "in production with PyTorch and TensorFlow."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "ai_ml"

    def test_cybersecurity(self, sample_collected_data):
        startup = _make_startup(
            "A cybersecurity platform providing threat detection and "
            "infosec compliance for enterprise organizations."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "cybersecurity"

    def test_no_match_returns_none(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.industry is None
        assert result.industry_confidence == 0.0

    def test_confidence_scales_with_evidence(self, sample_collected_data):
        weak = _make_startup("A software company.")
        strong = _make_startup(
            "A fintech platform providing embedded payment processing "
            "with split payments, escrow, KYC compliance, and "
            "multi-currency settlement across 35 countries."
        )
        weak_result = MarketExtractor().extract(weak, sample_collected_data)
        strong_result = MarketExtractor().extract(strong, sample_collected_data)
        assert strong_result.industry_confidence > weak_result.industry_confidence


class TestSubIndustry:
    def test_fintech_payments(self, sample_collected_data):
        startup = _make_startup(
            "A fintech company providing embedded payment processing "
            "and payment infrastructure for marketplace platforms."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "fintech"
        assert result.sub_industry == "embedded_payments"

    def test_healthtech_diagnostic_imaging(self, sample_collected_data):
        startup = _make_startup(
            "A healthtech company building diagnostic imaging tools "
            "for medical imaging and radiology workflows."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.sub_industry == "diagnostic_imaging"

    def test_enterprise_saas_compliance(self, sample_collected_data):
        startup = _make_startup(
            "An enterprise saas compliance management platform for "
            "regulatory tracking, audit preparation, and policy management."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.sub_industry == "compliance"

    def test_ai_ml_infrastructure(self, sample_collected_data):
        startup = _make_startup(
            "An AI platform providing model serving and inference "
            "infrastructure for large language model deployment."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.sub_industry == "ai_infrastructure"

    def test_hardware_robotics(self, sample_collected_data):
        startup = _make_startup(
            "Building autonomous mobile robots for warehouse automation "
            "using LiDAR SLAM navigation and fleet management."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.sub_industry == "robotics"

    def test_no_sub_industry_for_weak_signal(self, sample_collected_data):
        startup = _make_startup("A fintech startup.")
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.sub_industry is None

    def test_no_sub_industry_without_primary(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.sub_industry is None


class TestGeography:
    def test_north_america(self, sample_collected_data):
        startup = _make_startup(
            "Based in San Francisco with operations across North America "
            "serving enterprise clients in the United States."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.geography == "north_america"

    def test_europe(self, sample_collected_data):
        startup = _make_startup(
            "Headquartered in Berlin with teams in London and Paris "
            "serving European markets."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.geography == "europe"

    def test_asia_pacific(self, sample_collected_data):
        startup = _make_startup(
            "Operating across Asia Pacific with offices in Singapore "
            "and Tokyo, serving the APAC region."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.geography == "asia_pacific"

    def test_global(self, sample_collected_data):
        startup = _make_startup(
            "Operating globally across 35 countries with international "
            "customers worldwide."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.geography == "global"

    def test_no_geography(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.geography is None


class TestCustomerType:
    def test_b2b(self, sample_collected_data):
        startup = _make_startup(
            "An enterprise SaaS platform serving B2B companies and "
            "organizations with subscription-based tools."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.customer_type == "b2b"

    def test_b2c(self, sample_collected_data):
        startup = _make_startup(
            "A consumer mobile app with 420,000 monthly active users "
            "and 38,000 premium subscribers."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.customer_type == "b2c"

    def test_b2b2c(self, sample_collected_data):
        startup = _make_startup(
            "A two-sided platform connecting businesses with consumers "
            "through a marketplace model."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.customer_type == "b2b2c"

    def test_b2g(self, sample_collected_data):
        startup = _make_startup(
            "A government technology platform providing public sector "
            "solutions for federal agencies."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.customer_type == "b2g"

    def test_no_customer_type(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.customer_type is None


class TestCustomerSegment:
    def test_enterprise_segment(self, sample_collected_data):
        startup = _make_startup(
            "Serving enterprise clients including Fortune 500 companies "
            "and large organizations across industries."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.customer_segment == "enterprise"

    def test_mid_market_segment(self, sample_collected_data):
        startup = _make_startup(
            "Providing tools for mid-market businesses with 100-1000 "
            "employees seeking affordable solutions."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.customer_segment == "mid_market"

    def test_developer_segment(self, sample_collected_data):
        startup = _make_startup(
            "Developer tools for engineering teams building with "
            "open-source software and GitHub workflows."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.customer_segment == "developer"

    def test_no_segment(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.customer_segment is None


class TestEnterpriseOrientation:
    def test_enterprise(self, sample_collected_data):
        startup = _make_startup(
            "Enterprise SaaS platform with annual contracts, gross "
            "retention metrics, and Fortune 500 clients."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.enterprise_orientation == "enterprise"

    def test_consumer(self, sample_collected_data):
        startup = _make_startup(
            "A consumer mobile app with monthly active users, "
            "freemium model, and social features."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.enterprise_orientation == "consumer"

    def test_hybrid(self, sample_collected_data):
        startup = _make_startup(
            "An enterprise platform serving both business clients "
            "and consumer end users through a freemium B2B2C model "
            "with subscription revenue and monthly active users."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.enterprise_orientation == "hybrid"

    def test_no_orientation(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.enterprise_orientation is None


class TestTargetMarket:
    def test_enterprise_b2b_fintech(self, sample_collected_data):
        startup = _make_startup(
            "Enterprise fintech platform serving B2B companies in "
            "financial services with annual contracts."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.target_market is not None
        assert "enterprise" in result.target_market
        assert "businesses" in result.target_market

    def test_consumer_b2c(self, sample_collected_data):
        startup = _make_startup(
            "A consumer mobile app for fitness tracking with "
            "monthly active users and freemium model."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.target_market is not None
        assert "consumers" in result.target_market

    def test_no_target_market(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.target_market is None


class TestMarketMaturity:
    def test_emerging_market(self, sample_collected_data):
        startup = _make_startup(
            "Operating in an emerging market with first mover advantage "
            "and novel technology approach."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.market_maturity == "emerging"

    def test_growth_market(self, sample_collected_data):
        startup = _make_startup(
            "Rapidly growing market with high growth rates and "
            "increasing demand for automation solutions."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.market_maturity == "growth"

    def test_mature_market(self, sample_collected_data):
        startup = _make_startup(
            "A well-established market with mature incumbents and "
            "consolidated competitive dynamics."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.market_maturity == "mature"

    def test_no_maturity(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.market_maturity is None


class TestMarketKeywords:
    def test_saas_keywords(self, sample_collected_data):
        startup = _make_startup(
            "A cloud-native SaaS platform with API-first architecture "
            "providing real-time analytics."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert "saas" in result.market_keywords
        assert "cloud_native" in result.market_keywords
        assert "api_first" in result.market_keywords
        assert "real_time" in result.market_keywords

    def test_ai_keywords(self, sample_collected_data):
        startup = _make_startup(
            "An AI-powered platform using machine learning for "
            "automated analytics and enterprise integration."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert "ai_powered" in result.market_keywords

    def test_no_keywords(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.market_keywords == []


class TestMarketSignals:
    def test_regulatory_signal(self, sample_collected_data):
        startup = _make_startup(
            "Compliance platform for SOC 2 and GDPR regulatory requirements."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert "regulatory_complexity" in result.market_signals

    def test_network_effects_signal(self, sample_collected_data):
        startup = _make_startup(
            "A marketplace with strong network effects and viral growth."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert "network_effects_present" in result.market_signals

    def test_recurring_revenue_signal(self, sample_collected_data):
        startup = _make_startup(
            "Subscription-based model with recurring revenue and "
            "strong net revenue retention."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert "recurring_revenue_signal" in result.market_signals

    def test_no_signals(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.market_signals == []


class TestMarketCharacteristics:
    def test_network_effects(self, sample_collected_data):
        startup = _make_startup(
            "Two-sided marketplace with network effects and flywheel dynamics."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert "network_effects" in result.market_characteristics

    def test_recurring_revenue(self, sample_collected_data):
        startup = _make_startup(
            "SaaS subscription model with recurring revenue and ARR."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert "recurring_revenue" in result.market_characteristics

    def test_regulatory_moat(self, sample_collected_data):
        startup = _make_startup(
            "Compliance platform with SOC 2 certification and "
            "ISO 27001 regulatory requirements."
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert "regulatory_moat" in result.market_characteristics

    def test_no_characteristics(self, sample_startup, sample_collected_data):
        result = MarketExtractor().extract(sample_startup, sample_collected_data)
        assert result.market_characteristics == []


class TestBenchmarkCases:
    """Validate MarketExtractor against known benchmark startup profiles."""

    def test_b2b_saas_analytics(self, sample_collected_data):
        startup = _make_startup(
            "Analytix Cloud is a B2B SaaS platform providing real-time "
            "analytics and business intelligence for mid-market enterprises. "
            "Revenue is subscription-based with annual contracts. The company "
            "has 85 enterprise clients and has achieved $4.2M ARR with 140% "
            "net revenue retention. Based in San Francisco.",
            name="Analytix Cloud",
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "enterprise_saas"
        assert result.customer_type == "b2b"
        assert result.geography == "north_america"
        assert result.enterprise_orientation == "enterprise"
        assert result.industry_confidence > 0.3

    def test_healthcare_ai(self, sample_collected_data):
        startup = _make_startup(
            "MedVision AI develops FDA-cleared AI diagnostic tools for "
            "medical imaging. The platform analyzes X-rays, MRIs, and CT "
            "scans to assist radiologists. Revenue model is per-study "
            "licensing with enterprise hospital contracts. Based in Boston.",
            name="MedVision AI",
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "healthtech"
        assert result.sub_industry == "diagnostic_imaging"
        assert result.customer_type == "b2b"
        assert result.geography == "north_america"

    def test_fintech_payments(self, sample_collected_data):
        startup = _make_startup(
            "PayBridge provides embedded payment infrastructure for "
            "marketplace and platform businesses. The API-first solution "
            "handles split payments, escrow, KYC compliance, and multi-"
            "currency settlement across 35 countries. Processing $2.1B in "
            "annual payment volume. Raised $45M Series B. Based in London, "
            "Singapore, and New York.",
            name="PayBridge",
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "fintech"
        assert result.sub_industry == "embedded_payments"
        assert result.customer_type == "b2b"

    def test_climate_tech(self, sample_collected_data):
        startup = _make_startup(
            "CarbonLens provides enterprise-grade carbon accounting and "
            "emissions tracking software for Scope 1, 2, and 3 emissions. "
            "The platform integrates with ERP systems and is aligned with "
            "GHG Protocol and SEC climate disclosure requirements. Based "
            "in Berlin with operations in the US.",
            name="CarbonLens",
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "climate_tech"
        assert result.customer_type == "b2b"

    def test_robotics(self, sample_collected_data):
        startup = _make_startup(
            "AutoWare Robotics builds autonomous mobile robots for "
            "warehouse and fulfillment center operations. The robots use "
            "LiDAR-based SLAM navigation. Robotics-as-a-Service model "
            "serving enterprise customer facilities. Fleet of 800 deployed "
            "robots. Based in Munich and Detroit.",
            name="AutoWare Robotics",
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "hardware"
        assert result.sub_industry == "robotics"
        assert result.customer_type == "b2b"

    def test_ai_infrastructure(self, sample_collected_data):
        startup = _make_startup(
            "Inference Labs provides GPU-optimized model serving "
            "infrastructure for teams deploying large language models in "
            "production. Supports PyTorch, TensorFlow, and ONNX models. "
            "Processing over 2 billion inference requests daily. Based in "
            "San Francisco and Toronto.",
            name="Inference Labs",
        )
        result = MarketExtractor().extract(startup, sample_collected_data)
        assert result.industry == "ai_ml"
        assert result.sub_industry == "ai_infrastructure"
        assert result.customer_type == "b2b"
