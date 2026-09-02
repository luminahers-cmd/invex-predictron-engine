"""Deterministic benchmark startup cases for Predictron Engine evaluation.

Each case represents a realistic startup input across different industries
and business models. Cases are designed to exercise different extraction
paths, reasoning rules, and scoring dimensions.

Design principles:
  - No hardcoded "correct" venture decisions
  - No proprietary heuristics
  - Input data is deterministic and reproducible
  - Each case tests specific feature extraction and reasoning capabilities

Benchmark metadata:
  - industry_category: High-level industry grouping for coverage analysis
  - company_stage: Funding/company maturity stage
  - coverage_tags: Tags indicating which engine capabilities this case exercises
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CASE_ID_B2B_SAAS = "b2b_saas"
CASE_ID_HEALTHCARE_AI = "healthcare_ai"
CASE_ID_FINTECH = "fintech"
CASE_ID_DEVTOOLS = "devtools"
CASE_ID_MARKETPLACE = "marketplace"
CASE_ID_CONSUMER_APP = "consumer_app"
CASE_ID_CLIMATE_TECH = "climate_tech"
CASE_ID_ROBOTICS = "robotics"
CASE_ID_ENTERPRISE_SOFTWARE = "enterprise_software"
CASE_ID_AI_INFRASTRUCTURE = "ai_infrastructure"
CASE_ID_DEEP_TECH = "deep_tech"
CASE_ID_EDTECH = "edtech"
CASE_ID_HEALTHTECH_DEVICE = "healthtech_device"
CASE_ID_MRR_ONLY_DERIVATION = "mrr_only_derivation"
CASE_ID_CASH_BURN_RUNWAY = "cash_burn_runway"
CASE_ID_REVENUE_TEAM_EFFICIENCY = "revenue_team_efficiency"
CASE_ID_UNIT_ECONOMICS = "unit_economics"
CASE_ID_FUNDING_EFFICIENCY = "funding_efficiency"


INDUSTRY_CATEGORIES = [
    "AI",
    "SaaS",
    "FinTech",
    "Consumer",
    "Deep Tech",
    "Healthcare",
    "Robotics",
    "Marketplace",
    "Climate",
    "EdTech",
    "Enterprise",
    "Infrastructure",
]

COMPANY_STAGES = [
    "Idea",
    "Pre-Seed",
    "Seed",
    "Series A",
    "Series B",
    "Series C",
    "Growth",
]


@dataclass
class BenchmarkCaseMetadata:
    """Structural metadata for a benchmark case."""

    industry_category: str
    company_stage: str
    coverage_tags: list[str] = field(default_factory=list)


@dataclass
class ExpectedOutcomes:
    """Rich expected outcome specification for benchmark validation."""

    expected_industry: str | None = None
    expected_business_model: str | None = None
    expected_customer_type: str | None = None
    expected_has_revenue: bool | None = None
    expected_min_score: float = 0.0
    expected_max_score: float = 100.0
    expected_min_confidence: float = 0.0
    expected_max_confidence: float = 1.0
    expected_min_observations: int = 0
    expected_min_recommendations: int = 0
    expected_min_evidence: int = 0
    expected_score_dimensions: list[str] = field(default_factory=list)
    expected_recommendation_categories: list[str] = field(default_factory=list)
    expected_strength_signals: list[str] = field(default_factory=list)
    expected_weakness_signals: list[str] = field(default_factory=list)
    expected_risk_dimensions: list[str] = field(default_factory=list)


BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "id": CASE_ID_B2B_SAAS,
        "label": "B2B SaaS — Cloud analytics platform",
        "metadata": BenchmarkCaseMetadata(
            industry_category="SaaS",
            company_stage="Series A",
            coverage_tags=[
                "revenue_metrics", "nrr", "enterprise", "subscription",
                "founder_profiles", "pitch_deck",
            ],
        ),
        "request": {
            "startup_name": "Analytix Cloud",
            "website": "https://analytixcloud.example.com",
            "description": (
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
            "pitch_deck_url": "https://analytixcloud.example.com/investor-deck.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/analytix-ceo",
                "https://linkedin.com/in/analytix-cto",
            ],
        },
        "expected_features": {
            "industry": "enterprise_saas",
            "business_model": "saas",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 2,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="enterprise_saas",
            expected_business_model="saas",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=30.0,
            expected_max_score=90.0,
            expected_min_confidence=0.4,
            expected_max_confidence=1.0,
            expected_min_observations=3,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "market_opportunity",
                "product_strength",
                "founder_quality",
                "traction_signals",
                "business_model_viability",
            ],
            expected_strength_signals=[
                "arr_revenue",
                "retention_metric",
                "enterprise_clients",
            ],
            expected_risk_dimensions=["competitive_position"],
        ),
    },
    {
        "id": CASE_ID_HEALTHCARE_AI,
        "label": "Healthcare AI — Diagnostic imaging",
        "metadata": BenchmarkCaseMetadata(
            industry_category="Healthcare",
            company_stage="Series A",
            coverage_tags=[
                "regulatory", "patents", "clinical_trials", "ai_ml",
                "licensing_model", "healthcare",
            ],
        ),
        "request": {
            "startup_name": "MedVision AI",
            "website": "https://medvisionai.example.com",
            "description": (
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
            "pitch_deck_url": "https://medvisionai.example.com/series-a-deck.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/medvision-founder",
            ],
        },
        "expected_features": {
            "industry": "healthtech",
            "business_model": "licensing",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 1,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="healthtech",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=25.0,
            expected_max_score=85.0,
            expected_min_confidence=0.3,
            expected_max_confidence=1.0,
            expected_min_observations=3,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "market_opportunity",
                "product_strength",
                "founder_quality",
                "traction_signals",
            ],
            expected_strength_signals=[
                "regulatory_approval",
                "patents",
            ],
            expected_risk_dimensions=["regulatory"],
        ),
    },
    {
        "id": CASE_ID_FINTECH,
        "label": "FinTech — Embedded payments",
        "metadata": BenchmarkCaseMetadata(
            industry_category="FinTech",
            company_stage="Series B",
            coverage_tags=[
                "transactional", "global_scale", "multi_currency",
                "enterprise_clients", "marketplace_payments",
            ],
        ),
        "request": {
            "startup_name": "PayBridge",
            "website": "https://paybridge.example.com",
            "description": (
                "PayBridge provides embedded payment infrastructure for "
                "marketplace and platform businesses. The API-first solution "
                "handles split payments, escrow, KYC compliance, and multi-"
                "currency settlement across 35 countries. Processing $2.1B in "
                "annual payment volume with a take rate of 0.8%. The company "
                "serves 280 marketplace clients including 12 in the Fortune "
                "500. Founded in 2019, team of 120 across London, Singapore, "
                "and New York. Raised $45M Series B from Tier 1 fintech investors."
            ),
            "pitch_deck_url": "https://paybridge.example.com/deck.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/paybridge-ceo",
                "https://linkedin.com/in/paybridge-coo",
                "https://linkedin.com/in/paybridge-cto",
            ],
        },
        "expected_features": {
            "industry": "fintech",
            "business_model": "transactional",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 3,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="fintech",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=35.0,
            expected_max_score=95.0,
            expected_min_confidence=0.4,
            expected_max_confidence=1.0,
            expected_min_observations=3,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "market_opportunity",
                "business_model_viability",
                "traction_signals",
            ],
            expected_strength_signals=[
                "transaction_volume",
                "enterprise_clients",
            ],
            expected_risk_dimensions=["regulatory"],
        ),
    },
    {
        "id": CASE_ID_DEVTOOLS,
        "label": "DevTools — CI/CD platform",
        "metadata": BenchmarkCaseMetadata(
            industry_category="SaaS",
            company_stage="Pre-Seed",
            coverage_tags=[
                "open_source", "pre_revenue", "developer_community",
                "monetization_path", "no_pitch_deck",
            ],
        ),
        "request": {
            "startup_name": "ShipKit",
            "website": "https://shipkit.dev",
            "description": (
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
            "pitch_deck_url": None,
            "founder_linkedin_urls": [
                "https://linkedin.com/in/shipkit-founder-1",
                "https://linkedin.com/in/shipkit-founder-2",
            ],
        },
        "expected_features": {
            "industry": "enterprise_saas",
            "business_model": "saas",
            "customer_type": "b2b",
            "has_revenue": False,
            "has_pitch_deck": False,
            "founder_profile_count": 2,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="enterprise_saas",
            expected_customer_type="b2b",
            expected_has_revenue=False,
            expected_min_score=15.0,
            expected_max_score=75.0,
            expected_min_confidence=0.2,
            expected_max_confidence=0.9,
            expected_min_observations=2,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "product_strength",
                "founder_quality",
                "market_opportunity",
            ],
            expected_strength_signals=[
                "open_source_community",
                "founder_experience",
            ],
            expected_weakness_signals=[
                "pre_revenue",
            ],
            expected_risk_dimensions=["business_model_viability"],
        ),
    },
    {
        "id": CASE_ID_MARKETPLACE,
        "label": "Marketplace — B2B industrial supplies",
        "metadata": BenchmarkCaseMetadata(
            industry_category="Marketplace",
            company_stage="Series A",
            coverage_tags=[
                "two_sided", "gmv", "take_rate", "unit_economics",
                "ltv_cac", "b2b_marketplace",
            ],
        ),
        "request": {
            "startup_name": "SupplyHub",
            "website": "https://supplyhub.example.com",
            "description": (
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
            "pitch_deck_url": "https://supplyhub.example.com/investors.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/supplyhub-ceo",
            ],
        },
        "expected_features": {
            "industry": "marketplace",
            "business_model": "marketplace",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 1,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="marketplace",
            expected_business_model="marketplace",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=30.0,
            expected_max_score=90.0,
            expected_min_confidence=0.4,
            expected_max_confidence=1.0,
            expected_min_observations=3,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "market_opportunity",
                "business_model_viability",
                "traction_signals",
            ],
            expected_strength_signals=[
                "gmv",
                "unit_economics",
                "two_sided_marketplace",
            ],
        ),
    },
    {
        "id": CASE_ID_CONSUMER_APP,
        "label": "Consumer App — Fitness social network",
        "metadata": BenchmarkCaseMetadata(
            industry_category="Consumer",
            company_stage="Pre-Seed",
            coverage_tags=[
                "b2c", "freemium", "mobile", "retention",
                "no_pitch_deck", "no_founders", "pre_seed",
            ],
        ),
        "request": {
            "startup_name": "FitSocial",
            "website": "https://fitsocial.example.com",
            "description": (
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
            "pitch_deck_url": None,
            "founder_linkedin_urls": [],
        },
        "expected_features": {
            "industry": "consumer_tech",
            "business_model": "saas",
            "customer_type": "b2c",
            "has_revenue": True,
            "has_pitch_deck": False,
            "founder_profile_count": 0,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_customer_type="b2c",
            expected_has_revenue=True,
            expected_min_score=15.0,
            expected_max_score=70.0,
            expected_min_confidence=0.2,
            expected_max_confidence=0.9,
            expected_min_observations=2,
            expected_min_recommendations=1,
            expected_min_evidence=0,
            expected_score_dimensions=[
                "product_strength",
                "traction_signals",
            ],
            expected_weakness_signals=[
                "no_founder_data",
            ],
        ),
    },
    {
        "id": CASE_ID_CLIMATE_TECH,
        "label": "Climate Tech — Carbon measurement",
        "metadata": BenchmarkCaseMetadata(
            industry_category="Climate",
            company_stage="Seed",
            coverage_tags=[
                "regulatory_alignment", "enterprise", "saas",
                "esg", "ghg_protocol", "erp_integration",
            ],
        ),
        "request": {
            "startup_name": "CarbonLens",
            "website": "https://carbonlens.example.com",
            "description": (
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
            "pitch_deck_url": "https://carbonlens.example.com/seed-deck.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/carbonlens-ceo",
                "https://linkedin.com/in/carbonlens-cto",
            ],
        },
        "expected_features": {
            "industry": "climate_tech",
            "business_model": "saas",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 2,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="climate_tech",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=30.0,
            expected_max_score=85.0,
            expected_min_confidence=0.3,
            expected_max_confidence=1.0,
            expected_min_observations=3,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "market_opportunity",
                "product_strength",
                "traction_signals",
                "business_model_viability",
            ],
            expected_strength_signals=[
                "regulatory_alignment",
                "retention_metric",
            ],
        ),
    },
    {
        "id": CASE_ID_ROBOTICS,
        "label": "Robotics — Warehouse automation",
        "metadata": BenchmarkCaseMetadata(
            industry_category="Robotics",
            company_stage="Series B",
            coverage_tags=[
                "hardware", "raas", "fleet", "capital_intensive",
                "multi_site", "founder_profiles",
            ],
        ),
        "request": {
            "startup_name": "AutoWare Robotics",
            "website": "https://autoware.example.com",
            "description": (
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
            "pitch_deck_url": "https://autoware.example.com/series-b.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/autoware-ceo",
                "https://linkedin.com/in/autoware-cto",
                "https://linkedin.com/in/autoware-vp-eng",
            ],
        },
        "expected_features": {
            "industry": "hardware",
            "business_model": "paas",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 3,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=30.0,
            expected_max_score=85.0,
            expected_min_confidence=0.3,
            expected_max_confidence=1.0,
            expected_min_observations=3,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "product_strength",
                "market_opportunity",
                "founder_quality",
                "traction_signals",
            ],
            expected_strength_signals=[
                "fleet_deployment",
                "arr_revenue",
            ],
            expected_risk_dimensions=["business_model_viability"],
        ),
    },
    {
        "id": CASE_ID_ENTERPRISE_SOFTWARE,
        "label": "Enterprise Software — Compliance platform",
        "metadata": BenchmarkCaseMetadata(
            industry_category="Enterprise",
            company_stage="Series C",
            coverage_tags=[
                "enterprise", "compliance", "financial_services",
                "high_acv", "retention", "mature_startup",
            ],
        ),
        "request": {
            "startup_name": "ComplianceOS",
            "website": "https://complianceos.example.com",
            "description": (
                "ComplianceOS is an enterprise compliance management platform "
                "that automates regulatory tracking, policy management, and "
                "audit preparation for financial services companies. The "
                "platform covers SOC 2, PCI DSS, GDPR, and CCPA compliance "
                "workflows with automated evidence collection and control "
                "monitoring. Serving 95 financial institutions including 8 "
                "top-20 US banks. ACV of $120,000 with 98% gross retention. "
                "Series C stage with $85M raised. Team of 200 across New York, "
                "Charlotte, and remote. Founded in 2018."
            ),
            "pitch_deck_url": "https://complianceos.example.com/series-c.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/complianceos-ceo",
            ],
        },
        "expected_features": {
            "industry": "enterprise_saas",
            "business_model": "saas",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 1,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="enterprise_saas",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=35.0,
            expected_max_score=90.0,
            expected_min_confidence=0.4,
            expected_max_confidence=1.0,
            expected_min_observations=3,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "market_opportunity",
                "traction_signals",
                "business_model_viability",
            ],
            expected_strength_signals=[
                "enterprise_clients",
                "retention_metric",
                "high_acv",
            ],
        ),
    },
    {
        "id": CASE_ID_AI_INFRASTRUCTURE,
        "label": "AI Infrastructure — Model serving platform",
        "metadata": BenchmarkCaseMetadata(
            industry_category="Infrastructure",
            company_stage="Series A",
            coverage_tags=[
                "gpu", "ml_infrastructure", "usage_based",
                "enterprise", "high_growth", "ai_ml",
            ],
        ),
        "request": {
            "startup_name": "Inference Labs",
            "website": "https://inferencelabs.example.com",
            "description": (
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
            "pitch_deck_url": "https://inferencelabs.example.com/deck.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/inferencelabs-ceo",
                "https://linkedin.com/in/inferencelabs-cto",
            ],
        },
        "expected_features": {
            "industry": "ai_ml",
            "business_model": "paas",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 2,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="ai_ml",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=35.0,
            expected_max_score=90.0,
            expected_min_confidence=0.4,
            expected_max_confidence=1.0,
            expected_min_observations=3,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "market_opportunity",
                "product_strength",
                "founder_quality",
                "traction_signals",
            ],
            expected_strength_signals=[
                "enterprise_clients",
                "usage_volume",
            ],
            expected_risk_dimensions=["competitive_position"],
        ),
    },
    {
        "id": CASE_ID_DEEP_TECH,
        "label": "Deep Tech — Quantum error correction",
        "metadata": BenchmarkCaseMetadata(
            industry_category="Deep Tech",
            company_stage="Seed",
            coverage_tags=[
                "deep_tech", "research", "long_horizon",
                "high_uncertainty", "limited_revenue", "academic_founders",
            ],
        ),
        "request": {
            "startup_name": "QubitShield",
            "website": "https://qubitshield.example.com",
            "description": (
                "QubitShield develops quantum error correction software for "
                "superconducting quantum processors. The proprietary decoder "
                "algorithm achieves a 40% improvement in logical qubit fidelity "
                "compared to baseline surface codes. The company licenses "
                "software to quantum computing hardware manufacturers and "
                "cloud quantum providers. 3 peer-reviewed publications in Nature "
                "Physics and PRX Quantum. Pre-revenue, currently running pilot "
                "evaluations with 2 quantum hardware companies. Seed stage "
                "with $5M raised from deep-tech focused VCs. Founded in 2023 "
                "by a team of 3 quantum physicists from MIT. Based in Boston, "
                "team of 12."
            ),
            "pitch_deck_url": "https://qubitshield.example.com/seed.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/qubitshield-ceo",
                "https://linkedin.com/in/qubitshield-cto",
                "https://linkedin.com/in/qubitshield-scientist",
            ],
        },
        "expected_features": {
            "industry": "deep_tech",
            "business_model": "licensing",
            "customer_type": "b2b",
            "has_revenue": False,
            "has_pitch_deck": True,
            "founder_profile_count": 3,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_customer_type="b2b",
            expected_has_revenue=False,
            expected_min_score=15.0,
            expected_max_score=70.0,
            expected_min_confidence=0.15,
            expected_max_confidence=0.85,
            expected_min_observations=2,
            expected_min_recommendations=2,
            expected_min_evidence=0,
            expected_score_dimensions=[
                "product_strength",
                "founder_quality",
                "market_opportunity",
            ],
            expected_strength_signals=[
                "research_publications",
                "founder_expertise",
            ],
            expected_weakness_signals=[
                "pre_revenue",
                "long_commercialization_horizon",
            ],
            expected_risk_dimensions=["business_model_viability", "market_opportunity"],
        ),
    },
    {
        "id": CASE_ID_EDTECH,
        "label": "EdTech — Corporate training platform",
        "metadata": BenchmarkCaseMetadata(
            industry_category="EdTech",
            company_stage="Seed",
            coverage_tags=[
                "b2b_edtech", "enterprise_training", "saas",
                "retention", "content_platform",
            ],
        ),
        "request": {
            "startup_name": "SkillForge",
            "website": "https://skillforge.example.com",
            "description": (
                "SkillForge is a B2B EdTech platform providing AI-personalized "
                "corporate training programs for technology skills. The platform "
                "generates adaptive learning paths based on role requirements "
                "and individual skill gaps. Content library covers 200+ "
                "technology topics with hands-on labs. Serving 45 enterprise "
                "clients with 12,000 active learners. ACV of $35,000 with "
                "90% gross retention. Raised $6M seed round. Founded in 2022, "
                "team of 22 based in New York. Previously operated as a "
                "consulting firm for 3 years before pivoting to SaaS."
            ),
            "pitch_deck_url": "https://skillforge.example.com/seed.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/skillforge-ceo",
            ],
        },
        "expected_features": {
            "industry": "edtech",
            "business_model": "saas",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 1,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=25.0,
            expected_max_score=80.0,
            expected_min_confidence=0.3,
            expected_max_confidence=0.95,
            expected_min_observations=2,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "market_opportunity",
                "business_model_viability",
                "traction_signals",
            ],
            expected_strength_signals=[
                "retention_metric",
                "enterprise_clients",
            ],
            expected_risk_dimensions=["competitive_position"],
        ),
    },
    {
        "id": CASE_ID_HEALTHTECH_DEVICE,
        "label": "HealthTech Device — Wearable glucose monitor",
        "metadata": BenchmarkCaseMetadata(
            industry_category="Healthcare",
            company_stage="Series A",
            coverage_tags=[
                "hardware", "fda", "wearable", "dtc_plus_b2b",
                "clinical_data", "healthcare_device",
            ],
        ),
        "request": {
            "startup_name": "GlucoPulse",
            "website": "https://glucopulse.example.com",
            "description": (
                "GlucoPulse develops a non-invasive continuous glucose "
                "monitoring wearable using photoplethysmography sensors. The "
                "device provides real-time glucose trend data to a companion "
                "mobile app with AI-powered dietary recommendations. FDA 510(k) "
                "clearance obtained. Direct-to-consumer pricing at $299 device "
                "plus $19/month subscription. 8,000 units shipped in first "
                "quarter since launch. Raised $18M Series A from digital "
                "health investors. Based in San Diego, team of 45 including "
                "hardware engineers, clinical specialists, and data scientists. "
                "Founded in 2021."
            ),
            "pitch_deck_url": "https://glucopulse.example.com/series-a.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/glucopulse-ceo",
                "https://linkedin.com/in/glucopulse-vp-eng",
            ],
        },
        "expected_features": {
            "industry": "healthtech",
            "business_model": "saas",
            "customer_type": "b2c",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 2,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="healthtech",
            expected_customer_type="b2c",
            expected_has_revenue=True,
            expected_min_score=25.0,
            expected_max_score=85.0,
            expected_min_confidence=0.3,
            expected_max_confidence=0.95,
            expected_min_observations=3,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "product_strength",
                "market_opportunity",
                "traction_signals",
            ],
            expected_strength_signals=[
                "regulatory_approval",
                "units_shipped",
            ],
            expected_risk_dimensions=["regulatory", "competitive_position"],
        ),
    },
    # --- Derived Metrics Benchmark Cases (Sprint 17) ---
    {
        "id": CASE_ID_MRR_ONLY_DERIVATION,
        "label": "MRR-Only — ARR derivation from MRR",
        "metadata": BenchmarkCaseMetadata(
            industry_category="SaaS",
            company_stage="Seed",
            coverage_tags=[
                "derived_metrics", "mrr_to_arr", "saas",
                "subscription", "single_metric_derivation",
            ],
        ),
        "request": {
            "startup_name": "SubMetrics",
            "website": "https://submetrics.example.com",
            "description": (
                "SubMetrics provides subscription analytics for B2B SaaS "
                "companies. The platform tracks MRR, churn, and expansion "
                "revenue across billing systems. Currently generating $85K MRR "
                "from 120 subscription customers on a $499/month plan. "
                "Founded in 2023, team of 8 based in Denver. Raised $2M seed "
                "from angel investors. Pre-revenue on enterprise tier, "
                "currently focused on product-market fit."
            ),
            "pitch_deck_url": None,
            "founder_linkedin_urls": [
                "https://linkedin.com/in/submetrics-ceo",
            ],
        },
        "expected_features": {
            "industry": "enterprise_saas",
            "business_model": "saas",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": False,
            "founder_profile_count": 1,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="enterprise_saas",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=20.0,
            expected_max_score=75.0,
            expected_min_confidence=0.3,
            expected_max_confidence=0.9,
            expected_min_observations=2,
            expected_min_recommendations=1,
            expected_min_evidence=0,
            expected_score_dimensions=[
                "traction_signals",
                "product_strength",
            ],
        ),
    },
    {
        "id": CASE_ID_CASH_BURN_RUNWAY,
        "label": "Cash & Burn — Runway derivation",
        "metadata": BenchmarkCaseMetadata(
            industry_category="SaaS",
            company_stage="Series A",
            coverage_tags=[
                "derived_metrics", "runway_derivation", "burn_rate",
                "financial_health", "cash_management",
            ],
        ),
        "request": {
            "startup_name": "FlowState",
            "website": "https://flowstate.example.com",
            "description": (
                "FlowState builds workflow automation for operations teams. "
                "The platform integrates with Slack, Notion, and Jira to "
                "eliminate manual status updates and reporting. Currently "
                "burning $350K per month across engineering and go-to-market. "
                "Company has $6M in total funding from a Series A round. "
                "Revenue of $1.8M ARR from 45 enterprise customers. "
                "Team of 25 based in Austin, founded in 2022. "
                "Currently at 8 months of runway."
            ),
            "pitch_deck_url": "https://flowstate.example.com/deck.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/flowstate-ceo",
            ],
        },
        "expected_features": {
            "industry": "enterprise_saas",
            "business_model": "saas",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 1,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="enterprise_saas",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=25.0,
            expected_max_score=80.0,
            expected_min_confidence=0.3,
            expected_max_confidence=0.9,
            expected_min_observations=2,
            expected_min_recommendations=1,
            expected_min_evidence=0,
            expected_score_dimensions=[
                "traction_signals",
                "business_model_viability",
            ],
        ),
    },
    {
        "id": CASE_ID_REVENUE_TEAM_EFFICIENCY,
        "label": "Revenue per Employee — Team efficiency derivation",
        "metadata": BenchmarkCaseMetadata(
            industry_category="SaaS",
            company_stage="Series A",
            coverage_tags=[
                "derived_metrics", "revenue_per_employee", "team_efficiency",
                "capital_efficiency", "saas",
            ],
        ),
        "request": {
            "startup_name": "DataPipe",
            "website": "https://datapipe.example.com",
            "description": (
                "DataPipe provides automated data pipeline orchestration "
                "for analytics teams. The platform handles ETL, data "
                "quality monitoring, and warehouse optimization. Currently "
                "generating $6M ARR from 35 enterprise customers with "
                "average contract value of $171,000. Team of 15 engineers "
                "and 5 sales professionals based in New York. ARR per "
                "employee is exceptionally high at $300K. Raised $15M "
                "Series A. Founded in 2021."
            ),
            "pitch_deck_url": "https://datapipe.example.com/series-a.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/datapipe-ceo",
                "https://linkedin.com/in/datapipe-cto",
            ],
        },
        "expected_features": {
            "industry": "enterprise_saas",
            "business_model": "saas",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 2,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="enterprise_saas",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=30.0,
            expected_max_score=85.0,
            expected_min_confidence=0.4,
            expected_max_confidence=0.95,
            expected_min_observations=3,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "traction_signals",
                "product_strength",
                "business_model_viability",
            ],
            expected_strength_signals=[
                "arr_revenue",
                "enterprise_clients",
            ],
        ),
    },
    {
        "id": CASE_ID_UNIT_ECONOMICS,
        "label": "Unit Economics — LTV/CAC ratio derivation",
        "metadata": BenchmarkCaseMetadata(
            industry_category="SaaS",
            company_stage="Series A",
            coverage_tags=[
                "derived_metrics", "ltv_cac", "unit_economics",
                "saas", "customer_economics",
            ],
        ),
        "request": {
            "startup_name": "RetentionAI",
            "website": "https://retentionai.example.com",
            "description": (
                "RetentionAI provides an AI-powered SaaS platform for "
                "customer retention analytics, serving subscription-based "
                "businesses. The platform predicts churn risk and recommends "
                "intervention strategies. CAC is $2,400 with LTV of $18,000. "
                "Currently serving 80 subscription businesses with $2.4M ARR. "
                "Monthly burn rate of $200K with $4M in total funding. "
                "Team of 18 based in Chicago. Founded in 2022. "
                "Net revenue retention of 125%."
            ),
            "pitch_deck_url": "https://retentionai.example.com/deck.pdf",
            "founder_linkedin_urls": [
                "https://linkedin.com/in/retentionai-ceo",
            ],
        },
        "expected_features": {
            "industry": "enterprise_saas",
            "business_model": "saas",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": True,
            "founder_profile_count": 1,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="enterprise_saas",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=25.0,
            expected_max_score=85.0,
            expected_min_confidence=0.3,
            expected_max_confidence=0.95,
            expected_min_observations=3,
            expected_min_recommendations=2,
            expected_min_evidence=1,
            expected_score_dimensions=[
                "traction_signals",
                "business_model_viability",
            ],
            expected_strength_signals=[
                "retention_metric",
                "unit_economics",
            ],
        ),
    },
    {
        "id": CASE_ID_FUNDING_EFFICIENCY,
        "label": "Funding Efficiency — ARR/Funding ratio derivation",
        "metadata": BenchmarkCaseMetadata(
            industry_category="SaaS",
            company_stage="Seed",
            coverage_tags=[
                "derived_metrics", "funding_efficiency", "capital_efficiency",
                "saas", "bootstrapped_to_seed",
            ],
        ),
        "request": {
            "startup_name": "LeanStack",
            "website": "https://leanstack.example.com",
            "description": (
                "LeanStack is a SaaS platform providing serverless "
                "deployment tooling for startup engineering teams. "
                "The CLI-based platform automates infrastructure "
                "provisioning, CI/CD pipelines, and cost optimization "
                "across AWS and GCP. Currently generating $3.2M ARR from "
                "200 developer teams paying $1,333/month on average. Total "
                "funding of only $1.5M from a seed round. Team of 12 based "
                "in Portland. Founded in 2022 by two ex-AWS engineers. "
                "Monthly burn rate of $150K."
            ),
            "pitch_deck_url": None,
            "founder_linkedin_urls": [
                "https://linkedin.com/in/leanstack-ceo",
                "https://linkedin.com/in/leanstack-cto",
            ],
        },
        "expected_features": {
            "industry": "enterprise_saas",
            "business_model": "saas",
            "customer_type": "b2b",
            "has_revenue": True,
            "has_pitch_deck": False,
            "founder_profile_count": 2,
        },
        "expected_outcomes": ExpectedOutcomes(
            expected_industry="enterprise_saas",
            expected_customer_type="b2b",
            expected_has_revenue=True,
            expected_min_score=25.0,
            expected_max_score=85.0,
            expected_min_confidence=0.3,
            expected_max_confidence=0.9,
            expected_min_observations=2,
            expected_min_recommendations=1,
            expected_min_evidence=0,
            expected_score_dimensions=[
                "traction_signals",
                "product_strength",
                "business_model_viability",
            ],
            expected_strength_signals=[
                "arr_revenue",
                "capital_efficiency",
            ],
        ),
    },
]


def get_case_by_id(case_id: str) -> dict[str, Any] | None:
    """Retrieve a benchmark case by its unique identifier."""
    for case in BENCHMARK_CASES:
        if case["id"] == case_id:
            return case
    return None


def get_all_case_ids() -> list[str]:
    """Return all available benchmark case identifiers."""
    return [case["id"] for case in BENCHMARK_CASES]


def get_case_labels() -> dict[str, str]:
    """Return a mapping of case ID to human-readable label."""
    return {case["id"]: case["label"] for case in BENCHMARK_CASES}


def get_cases_by_industry(category: str) -> list[dict[str, Any]]:
    """Return all benchmark cases matching an industry category."""
    return [
        case for case in BENCHMARK_CASES
        if case["metadata"].industry_category == category
    ]


def get_cases_by_stage(stage: str) -> list[dict[str, Any]]:
    """Return all benchmark cases matching a company stage."""
    return [
        case for case in BENCHMARK_CASES
        if case["metadata"].company_stage == stage
    ]


def get_industry_coverage() -> dict[str, int]:
    """Return a count of cases per industry category."""
    coverage: dict[str, int] = {}
    for case in BENCHMARK_CASES:
        cat = case["metadata"].industry_category
        coverage[cat] = coverage.get(cat, 0) + 1
    return coverage


def get_stage_coverage() -> dict[str, int]:
    """Return a count of cases per company stage."""
    coverage: dict[str, int] = {}
    for case in BENCHMARK_CASES:
        stage = case["metadata"].company_stage
        coverage[stage] = coverage.get(stage, 0) + 1
    return coverage
