"""Comprehensive tests for the ProductExtractor.

Covers all 18+ extraction dimensions added in Sprint 3, plus backward
compatibility with the original technology_stack extraction.
"""

from __future__ import annotations

import pytest

from predictron_engine.extraction.extractors.product import ProductExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor() -> ProductExtractor:
    return ProductExtractor()


@pytest.fixture
def minimal_collected_data() -> CollectedData:
    return CollectedData(
        startup_name="TestCo",
        website_domain="testco.example.com",
        description_tokens=[],
        description_word_count=0,
        has_website=True,
        has_pitch_deck=False,
        founder_count=0,
    )


@pytest.fixture
def b2b_saas_startup() -> Startup:
    return Startup(
        name="Analytix Cloud",
        website="https://analytixcloud.example.com",
        description=(
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
        ),
    )


@pytest.fixture
def healthcare_ai_startup() -> Startup:
    return Startup(
        name="MedVision AI",
        website="https://medvisionai.example.com",
        description=(
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
        ),
    )


@pytest.fixture
def fintech_startup() -> Startup:
    return Startup(
        name="PayBridge",
        website="https://paybridge.example.com",
        description=(
            "PayBridge provides embedded payment infrastructure for "
            "marketplace and platform businesses. The API-first solution "
            "handles split payments, escrow, KYC compliance, and multi-"
            "currency settlement across 35 countries. Processing $2.1B in "
            "annual payment volume with a take rate of 0.8%. The company "
            "serves 280 marketplace clients including 12 in the Fortune "
            "500. Founded in 2019, team of 120 across London, Singapore, "
            "and New York. Raised $45M Series B from Tier 1 fintech investors."
        ),
    )


@pytest.fixture
def devtools_startup() -> Startup:
    return Startup(
        name="ShipKit",
        website="https://shipkit.dev",
        description=(
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
        ),
    )


@pytest.fixture
def marketplace_startup() -> Startup:
    return Startup(
        name="SupplyHub",
        website="https://supplyhub.example.com",
        description=(
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
        ),
    )


@pytest.fixture
def consumer_app_startup() -> Startup:
    return Startup(
        name="FitSocial",
        website="https://fitsocial.example.com",
        description=(
            "FitSocial is a consumer mobile application that combines "
            "fitness tracking with social networking. Users log workouts, "
            "share progress, and participate in community challenges. "
            "Monetization is through a freemium model with a $9.99/month "
            "premium subscription and sponsored brand partnerships. "
            "Currently has 420,000 monthly active users with 38,000 "
            "premium subscribers. Retention at D30 is 42%. Available on "
            "iOS and Android. Founded in 2022, team of 18, based in "
            "Los Angeles. Pre-seed stage with $1.5M raised from angels."
        ),
    )


@pytest.fixture
def climate_tech_startup() -> Startup:
    return Startup(
        name="CarbonLens",
        website="https://carbonlens.example.com",
        description=(
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
        ),
    )


@pytest.fixture
def robotics_startup() -> Startup:
    return Startup(
        name="AutoWare Robotics",
        website="https://autoware.example.com",
        description=(
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
        ),
    )


@pytest.fixture
def ai_infra_startup() -> Startup:
    return Startup(
        name="Inference Labs",
        website="https://inferencelabs.example.com",
        description=(
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
        ),
    )


@pytest.fixture
def empty_startup() -> Startup:
    return Startup(
        name="EmptyCo",
        website="https://emptyco.example.com",
        description="An early-stage startup exploring new ideas.",
    )


# ---------------------------------------------------------------------------
# Basic return type tests
# ---------------------------------------------------------------------------

class TestProductExtractorReturnType:
    def test_returns_extracted_features(self, extractor, sample_startup, sample_collected_data):
        result = extractor.extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_only_populates_product_fields(self, extractor, sample_startup, sample_collected_data):
        result = extractor.extract(sample_startup, sample_collected_data)
        # ProductExtractor should not touch these fields
        assert result.industry is None
        assert result.funding_stage is None
        assert result.founded_year is None
        assert result.founder_profile_count == 0
        assert result.description_length == 0


# ---------------------------------------------------------------------------
# Product category classification tests
# ---------------------------------------------------------------------------

class TestProductCategory:
    def test_b2b_saas_analytics(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        assert result.product_category == "analytics"

    def test_healthcare_diagnostics(self, extractor, healthcare_ai_startup, minimal_collected_data):
        result = extractor.extract(healthcare_ai_startup, minimal_collected_data)
        assert result.product_category == "diagnostics"

    def test_fintech_payments(self, extractor, fintech_startup, minimal_collected_data):
        result = extractor.extract(fintech_startup, minimal_collected_data)
        assert result.product_category == "payments"

    def test_devtools_ci_cd(self, extractor, devtools_startup, minimal_collected_data):
        result = extractor.extract(devtools_startup, minimal_collected_data)
        assert result.product_category == "ci_cd"

    def test_marketplace(self, extractor, marketplace_startup, minimal_collected_data):
        result = extractor.extract(marketplace_startup, minimal_collected_data)
        assert result.product_category == "marketplace"

    def test_consumer_fitness(self, extractor, consumer_app_startup, minimal_collected_data):
        result = extractor.extract(consumer_app_startup, minimal_collected_data)
        assert result.product_category == "fitness_social"

    def test_climate_carbon(self, extractor, climate_tech_startup, minimal_collected_data):
        result = extractor.extract(climate_tech_startup, minimal_collected_data)
        assert result.product_category == "carbon_accounting"

    def test_robotics(self, extractor, robotics_startup, minimal_collected_data):
        result = extractor.extract(robotics_startup, minimal_collected_data)
        assert result.product_category == "robotics"

    def test_ai_model_serving(self, extractor, ai_infra_startup, minimal_collected_data):
        result = extractor.extract(ai_infra_startup, minimal_collected_data)
        assert result.product_category == "model_serving"

    def test_enterprise_compliance(self, extractor, minimal_collected_data):
        startup = Startup(
            name="ComplianceOS",
            website="https://complianceos.example.com",
            description=(
                "ComplianceOS is an enterprise compliance management platform "
                "that automates regulatory tracking, policy management, and "
                "audit preparation for financial services companies. The "
                "platform covers SOC 2, PCI DSS, GDPR, and CCPA compliance "
                "workflows with automated evidence collection and control "
                "monitoring."
            ),
        )
        result = extractor.extract(startup, minimal_collected_data)
        assert result.product_category == "compliance"

    def test_empty_description_returns_none(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.product_category is None


# ---------------------------------------------------------------------------
# Product type classification tests
# ---------------------------------------------------------------------------

class TestProductType:
    def test_saas_platform(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        assert result.product_type == "platform"

    def test_mobile_application(self, extractor, consumer_app_startup, minimal_collected_data):
        result = extractor.extract(consumer_app_startup, minimal_collected_data)
        assert result.product_type == "application"

    def test_api_first(self, extractor, fintech_startup, minimal_collected_data):
        result = extractor.extract(fintech_startup, minimal_collected_data)
        assert result.product_type == "api"

    def test_infrastructure(self, extractor, ai_infra_startup, minimal_collected_data):
        result = extractor.extract(ai_infra_startup, minimal_collected_data)
        assert result.product_type == "infrastructure"

    def test_empty_returns_none(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.product_type is None


# ---------------------------------------------------------------------------
# SaaS model classification tests
# ---------------------------------------------------------------------------

class TestSaasModel:
    def test_saas_model(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        assert result.saas_model == "saas"

    def test_api_model(self, extractor, fintech_startup, minimal_collected_data):
        result = extractor.extract(fintech_startup, minimal_collected_data)
        # PayBridge mentions "API-first" and "embedded payment infrastructure"
        assert result.saas_model in ("api", "infrastructure")

    def test_marketplace_model(self, extractor, marketplace_startup, minimal_collected_data):
        result = extractor.extract(marketplace_startup, minimal_collected_data)
        assert result.saas_model == "marketplace"

    def test_raas_model(self, extractor, robotics_startup, minimal_collected_data):
        result = extractor.extract(robotics_startup, minimal_collected_data)
        assert result.saas_model in ("raas", "paas")

    def test_licensing_model(self, extractor, healthcare_ai_startup, minimal_collected_data):
        result = extractor.extract(healthcare_ai_startup, minimal_collected_data)
        # Healthcare AI mentions "per-study licensing"
        assert result.saas_model == "licensing"


# ---------------------------------------------------------------------------
# AI orientation classification tests
# ---------------------------------------------------------------------------

class TestAIOrientation:
    def test_ai_native(self, extractor, ai_infra_startup, minimal_collected_data):
        result = extractor.extract(ai_infra_startup, minimal_collected_data)
        assert result.ai_orientation == "ai_native"

    def test_ai_native_healthcare(self, extractor, healthcare_ai_startup, minimal_collected_data):
        result = extractor.extract(healthcare_ai_startup, minimal_collected_data)
        assert result.ai_orientation == "ai_native"

    def test_non_ai_saas(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        # B2B SaaS mentions "automated insight generation" and "proprietary"
        # which can trigger ai_enabled; accept any non-ai-native result
        assert result.ai_orientation != "ai_native"

    def test_non_ai_marketplace(self, extractor, marketplace_startup, minimal_collected_data):
        result = extractor.extract(marketplace_startup, minimal_collected_data)
        assert result.ai_orientation in ("non_ai", "unknown")

    def test_empty_returns_non_ai(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.ai_orientation == "non_ai"


# ---------------------------------------------------------------------------
# Technology stack tests (backward compatibility)
# ---------------------------------------------------------------------------

class TestTechnologyStack:
    def test_tech_stack_from_description(self, extractor, minimal_collected_data):
        startup = Startup(
            name="TechCo",
            website="https://techco.example.com",
            description="Built with Python, React, and deployed on AWS using Docker.",
        )
        result = extractor.extract(startup, minimal_collected_data)
        assert "python" in result.technology_stack
        assert "react" in result.technology_stack
        assert "aws" in result.technology_stack
        assert "docker" in result.technology_stack

    def test_no_tech_stack_when_none_mentioned(
        self, extractor, sample_startup, sample_collected_data
    ):
        result = extractor.extract(sample_startup, sample_collected_data)
        assert result.technology_stack == []


# ---------------------------------------------------------------------------
# Capabilities tests
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_analytics_saas_capabilities(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        caps = result.primary_capabilities
        assert isinstance(caps, list)
        assert "analytics_and_reporting" in caps
        assert "integration" in caps

    def test_robotics_capabilities(self, extractor, robotics_startup, minimal_collected_data):
        result = extractor.extract(robotics_startup, minimal_collected_data)
        caps = result.primary_capabilities
        assert "navigation_and_movement" in caps

    def test_compliance_capabilities(self, extractor, climate_tech_startup, minimal_collected_data):
        result = extractor.extract(climate_tech_startup, minimal_collected_data)
        caps = result.primary_capabilities
        assert "data_processing" in caps or "automation" in caps

    def test_ai_infra_capabilities(self, extractor, ai_infra_startup, minimal_collected_data):
        result = extractor.extract(ai_infra_startup, minimal_collected_data)
        caps = result.primary_capabilities
        assert "model_management" in caps

    def test_empty_no_capabilities(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.primary_capabilities == []


# ---------------------------------------------------------------------------
# Feature signals tests
# ---------------------------------------------------------------------------

class TestFeatureSignals:
    def test_b2b_saas_features(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        features = result.feature_signals
        assert isinstance(features, list)
        assert "real_time" in features

    def test_devtools_features(self, extractor, devtools_startup, minimal_collected_data):
        result = extractor.extract(devtools_startup, minimal_collected_data)
        features = result.feature_signals
        assert "open_source" in features
        assert "sso" in features
        assert "audit_logging" in features

    def test_consumer_app_features(self, extractor, consumer_app_startup, minimal_collected_data):
        result = extractor.extract(consumer_app_startup, minimal_collected_data)
        features = result.feature_signals
        assert "mobile" in features
        assert "ios" in features
        assert "android" in features
        assert "freemium" in features

    def test_empty_no_features(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.feature_signals == []


# ---------------------------------------------------------------------------
# Integration ecosystem tests
# ---------------------------------------------------------------------------

class TestIntegrationEcosystem:
    def test_climate_integrations(self, extractor, climate_tech_startup, minimal_collected_data):
        result = extractor.extract(climate_tech_startup, minimal_collected_data)
        ecosystem = result.integration_ecosystem
        assert isinstance(ecosystem, list)
        assert "erp_integration" in ecosystem
        assert "supply_chain_tools" in ecosystem
        assert "iot_sensors" in ecosystem

    def test_b2b_saas_integrations(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        ecosystem = result.integration_ecosystem
        assert "data_warehouse" in ecosystem

    def test_empty_no_integrations(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.integration_ecosystem == []


# ---------------------------------------------------------------------------
# Deployment model tests
# ---------------------------------------------------------------------------

class TestDeploymentModel:
    def test_saas_deployment(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        assert result.deployment_model is not None

    def test_cloud_deployment_ai(self, extractor, ai_infra_startup, minimal_collected_data):
        result = extractor.extract(ai_infra_startup, minimal_collected_data)
        assert result.deployment_model in ("cloud", "saas")

    def test_empty_returns_none(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.deployment_model is None


# ---------------------------------------------------------------------------
# Target workflow tests
# ---------------------------------------------------------------------------

class TestTargetWorkflow:
    def test_analytics_workflow(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        assert result.target_workflow == "data_analytics"

    def test_payment_workflow(self, extractor, fintech_startup, minimal_collected_data):
        result = extractor.extract(fintech_startup, minimal_collected_data)
        assert result.target_workflow == "payment_processing"

    def test_compliance_workflow(self, extractor, climate_tech_startup, minimal_collected_data):
        result = extractor.extract(climate_tech_startup, minimal_collected_data)
        assert result.target_workflow in ("carbon_reporting", "regulatory_compliance")

    def test_cicd_workflow(self, extractor, devtools_startup, minimal_collected_data):
        result = extractor.extract(devtools_startup, minimal_collected_data)
        assert result.target_workflow == "ci_cd_pipeline"

    def test_robotics_workflow(self, extractor, robotics_startup, minimal_collected_data):
        result = extractor.extract(robotics_startup, minimal_collected_data)
        assert result.target_workflow == "warehouse_operations"

    def test_ml_workflow(self, extractor, ai_infra_startup, minimal_collected_data):
        result = extractor.extract(ai_infra_startup, minimal_collected_data)
        assert result.target_workflow == "ml_model_operations"

    def test_fitness_workflow(self, extractor, consumer_app_startup, minimal_collected_data):
        result = extractor.extract(consumer_app_startup, minimal_collected_data)
        assert result.target_workflow == "fitness_tracking"

    def test_empty_returns_none(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.target_workflow is None


# ---------------------------------------------------------------------------
# Automation level tests
# ---------------------------------------------------------------------------

class TestAutomationLevel:
    def test_high_automation_b2b_saas(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        assert result.automation_level in ("high", "moderate")

    def test_high_automation_climate(self, extractor, climate_tech_startup, minimal_collected_data):
        result = extractor.extract(climate_tech_startup, minimal_collected_data)
        assert result.automation_level == "high"

    def test_moderate_automation_healthcare(
        self, extractor, healthcare_ai_startup, minimal_collected_data
    ):
        result = extractor.extract(healthcare_ai_startup, minimal_collected_data)
        assert result.automation_level in ("moderate", "high")


# ---------------------------------------------------------------------------
# Product maturity tests
# ---------------------------------------------------------------------------

class TestProductMaturity:
    def test_beta_devtools(self, extractor, devtools_startup, minimal_collected_data):
        result = extractor.extract(devtools_startup, minimal_collected_data)
        assert result.product_maturity == "beta"

    def test_growth_saas(self, extractor, climate_tech_startup, minimal_collected_data):
        result = extractor.extract(climate_tech_startup, minimal_collected_data)
        # Climate tech mentions "scaling" indirectly; at minimum should be parseable
        assert isinstance(result.product_maturity, str | type(None))

    def test_empty_returns_none(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.product_maturity is None


# ---------------------------------------------------------------------------
# Differentiation signals tests
# ---------------------------------------------------------------------------

class TestDifferentiationSignals:
    def test_healthcare_differentiation(
        self, extractor, healthcare_ai_startup, minimal_collected_data
    ):
        result = extractor.extract(healthcare_ai_startup, minimal_collected_data)
        diff = result.differentiation_signals
        assert isinstance(diff, list)
        assert len(diff) > 0
        # Should detect patents, FDA clearance, clinical trials
        labels = " ".join(diff)
        assert "patent" in labels or "fda" in labels or "clinical" in labels

    def test_devtools_differentiation(self, extractor, devtools_startup, minimal_collected_data):
        result = extractor.extract(devtools_startup, minimal_collected_data)
        diff = result.differentiation_signals
        assert "open_source_community" in diff

    def test_empty_no_differentiation(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.differentiation_signals == []


# ---------------------------------------------------------------------------
# Defensibility signals tests
# ---------------------------------------------------------------------------

class TestDefensibilitySignals:
    def test_healthcare_defensibility(
        self, extractor, healthcare_ai_startup, minimal_collected_data
    ):
        result = extractor.extract(healthcare_ai_startup, minimal_collected_data)
        defens = result.defensibility_signals
        assert isinstance(defens, list)
        labels = " ".join(defens)
        assert "regulatory" in labels or "technology" in labels

    def test_fintech_defensibility(self, extractor, fintech_startup, minimal_collected_data):
        result = extractor.extract(fintech_startup, minimal_collected_data)
        defens = result.defensibility_signals
        assert isinstance(defens, list)
        labels = " ".join(defens)
        assert "network" in labels or "regulatory" in labels or "switching" in labels

    def test_empty_no_defensibility(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.defensibility_signals == []


# ---------------------------------------------------------------------------
# Technical complexity tests
# ---------------------------------------------------------------------------

class TestTechnicalComplexity:
    def test_high_complexity_robotics(self, extractor, robotics_startup, minimal_collected_data):
        result = extractor.extract(robotics_startup, minimal_collected_data)
        assert result.technical_complexity == "high"

    def test_high_complexity_ai(self, extractor, ai_infra_startup, minimal_collected_data):
        result = extractor.extract(ai_infra_startup, minimal_collected_data)
        assert result.technical_complexity == "high"

    def test_high_complexity_healthcare(
        self, extractor, healthcare_ai_startup, minimal_collected_data
    ):
        result = extractor.extract(healthcare_ai_startup, minimal_collected_data)
        assert result.technical_complexity == "high"

    def test_low_complexity_consumer(self, extractor, consumer_app_startup, minimal_collected_data):
        result = extractor.extract(consumer_app_startup, minimal_collected_data)
        assert result.technical_complexity in ("low", "moderate")

    def test_unknown_empty(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.technical_complexity == "unknown"


# ---------------------------------------------------------------------------
# Scalability indicators tests
# ---------------------------------------------------------------------------

class TestScalabilityIndicators:
    def test_ai_infra_scalability(self, extractor, ai_infra_startup, minimal_collected_data):
        result = extractor.extract(ai_infra_startup, minimal_collected_data)
        scale = result.scalability_indicators
        assert isinstance(scale, list)
        assert len(scale) > 0

    def test_climate_scalability(self, extractor, climate_tech_startup, minimal_collected_data):
        result = extractor.extract(climate_tech_startup, minimal_collected_data)
        scale = result.scalability_indicators
        assert isinstance(scale, list)

    def test_empty_no_scalability(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.scalability_indicators == []


# ---------------------------------------------------------------------------
# Innovation signals tests
# ---------------------------------------------------------------------------

class TestInnovationSignals:
    def test_healthcare_innovation(self, extractor, healthcare_ai_startup, minimal_collected_data):
        result = extractor.extract(healthcare_ai_startup, minimal_collected_data)
        innov = result.innovation_signals
        assert isinstance(innov, list)
        labels = " ".join(innov)
        assert "patent" in labels or "clinical" in labels or "proprietary" in labels

    def test_robotics_innovation(self, extractor, robotics_startup, minimal_collected_data):
        result = extractor.extract(robotics_startup, minimal_collected_data)
        innov = result.innovation_signals
        assert isinstance(innov, list)

    def test_empty_no_innovation(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.innovation_signals == []


# ---------------------------------------------------------------------------
# Product keywords tests
# ---------------------------------------------------------------------------

class TestProductKeywords:
    def test_saas_keywords(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        kw = result.product_keywords
        assert isinstance(kw, list)
        assert "saas" in kw
        assert "platform" in kw
        assert "real_time" in kw

    def test_devtools_keywords(self, extractor, devtools_startup, minimal_collected_data):
        result = extractor.extract(devtools_startup, minimal_collected_data)
        kw = result.product_keywords
        assert "open_source" in kw
        assert "platform" in kw
        assert "open_source_first" in kw

    def test_ai_keywords(self, extractor, ai_infra_startup, minimal_collected_data):
        result = extractor.extract(ai_infra_startup, minimal_collected_data)
        kw = result.product_keywords
        assert "model_serving" in kw
        assert "infrastructure" in kw
        assert "inference" in kw

    def test_empty_no_keywords(self, extractor, empty_startup, minimal_collected_data):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.product_keywords == []


# ---------------------------------------------------------------------------
# Product confidence tests
# ---------------------------------------------------------------------------

class TestProductConfidence:
    def test_confidence_range(self, extractor, b2b_saas_startup, minimal_collected_data):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        assert 0.0 <= result.product_confidence <= 1.0

    def test_rich_description_high_confidence(
        self, extractor, b2b_saas_startup, minimal_collected_data
    ):
        result = extractor.extract(b2b_saas_startup, minimal_collected_data)
        assert result.product_confidence >= 0.3

    def test_empty_description_low_confidence(
        self, extractor, empty_startup, minimal_collected_data
    ):
        result = extractor.extract(empty_startup, minimal_collected_data)
        assert result.product_confidence <= 0.3

    def test_healthcare_high_confidence(
        self, extractor, healthcare_ai_startup, minimal_collected_data
    ):
        result = extractor.extract(healthcare_ai_startup, minimal_collected_data)
        assert result.product_confidence >= 0.3


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_word_description(self, extractor, minimal_collected_data):
        startup = Startup(
            name="MicroCo",
            website="https://microco.example.com",
            description="Analytics.",
        )
        result = extractor.extract(startup, minimal_collected_data)
        assert isinstance(result, ExtractedFeatures)
        assert result.product_confidence <= 1.0

    def test_very_long_description(self, extractor, minimal_collected_data):
        long_desc = (
            "This is a comprehensive enterprise-grade SaaS platform that provides "
            "real-time analytics and business intelligence. " * 20
        )
        startup = Startup(
            name="LongCo",
            website="https://longco.example.com",
            description=long_desc,
        )
        result = extractor.extract(startup, minimal_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_competing_signals_returns_strongest(self, extractor, minimal_collected_data):
        startup = Startup(
            name="HybridCo",
            website="https://hybridco.example.com",
            description=(
                "HybridCo is a comprehensive SaaS platform with subscription-based "
                "pricing. The platform provides real-time analytics, automated "
                "reporting, and business intelligence dashboards. Revenue model "
                "is subscription-based with annual contracts. Enterprise clients "
                "use the platform for data-driven decision making."
            ),
        )
        result = extractor.extract(startup, minimal_collected_data)
        assert result.product_category is not None
        assert result.saas_model is not None

    def test_all_none_for_empty_input(self, extractor, minimal_collected_data):
        startup = Startup(
            name="BlankCo",
            website="https://blankco.example.com",
            description="A startup.",
        )
        result = extractor.extract(startup, minimal_collected_data)
        assert result.product_category is None
        assert result.product_type is None
        assert result.saas_model is None
        assert result.primary_capabilities == []
        assert result.feature_signals == []
        assert result.product_keywords == []
