"""Deterministic benchmark startup cases for Predictron Engine evaluation.

Each case represents a realistic startup input across different industries
and business models. Cases are designed to exercise different extraction
paths, reasoning rules, and scoring dimensions.

Design principles:
  - No hardcoded "correct" venture decisions
  - No proprietary heuristics
  - Input data is deterministic and reproducible
  - Each case tests specific feature extraction and reasoning capabilities
"""

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


BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "id": CASE_ID_B2B_SAAS,
        "label": "B2B SaaS — Cloud analytics platform",
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
    },
    {
        "id": CASE_ID_HEALTHCARE_AI,
        "label": "Healthcare AI — Diagnostic imaging",
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
    },
    {
        "id": CASE_ID_FINTECH,
        "label": "FinTech — Embedded payments",
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
    },
    {
        "id": CASE_ID_DEVTOOLS,
        "label": "DevTools — CI/CD platform",
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
    },
    {
        "id": CASE_ID_MARKETPLACE,
        "label": "Marketplace — B2B industrial supplies",
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
    },
    {
        "id": CASE_ID_CONSUMER_APP,
        "label": "Consumer App — Fitness social network",
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
    },
    {
        "id": CASE_ID_CLIMATE_TECH,
        "label": "Climate Tech — Carbon measurement",
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
    },
    {
        "id": CASE_ID_ROBOTICS,
        "label": "Robotics — Warehouse automation",
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
    },
    {
        "id": CASE_ID_ENTERPRISE_SOFTWARE,
        "label": "Enterprise Software — Compliance platform",
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
    },
    {
        "id": CASE_ID_AI_INFRASTRUCTURE,
        "label": "AI Infrastructure — Model serving platform",
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
