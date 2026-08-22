"""Product extractor — rich deterministic product intelligence extraction.

Responsibilities:
  - Product category classification via weighted keyword scoring
  - Product type / form factor detection (platform, application, tool, API, infrastructure)
  - Delivery model classification (SaaS, API, infrastructure, marketplace, licensing)
  - AI orientation detection (AI-native, AI-enabled, non-AI)
  - Primary capability extraction
  - Feature signal detection
  - Integration ecosystem mapping
  - Deployment model classification
  - Target workflow inference
  - Automation level assessment
  - Product maturity stage detection
  - Differentiation signal extraction
  - Defensibility / moat signal detection
  - Technical complexity estimation
  - Scalability indicator extraction
  - Innovation signal detection
  - Product keyword extraction
  - Product extraction confidence scoring

Design principles:
  - Deterministic rule-based logic only (no LLMs, no ML)
  - Weighted keyword scoring with context-aware disambiguation
  - Every extracted field traceable to explicit input signals
  - Prefer "unknown" over incorrect classification
  - Modular rules that are easy to extend
  - Uses weighted evidence instead of first-match logic
  - Confidence thresholds to minimize false positives
"""

from __future__ import annotations

import re

from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

# ---------------------------------------------------------------------------
# Product category — weighted keyword scoring for product domain classification.
# Each keyword carries a weight (higher = stronger signal).
# ---------------------------------------------------------------------------

_PRODUCT_CATEGORY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "analytics": [
        ("analytics", 5.0), ("business intelligence", 5.0),
        ("data analysis", 4.5), ("dashboard", 4.0),
        ("reporting", 3.5), ("insight generation", 4.5),
        ("query optimization", 3.5), ("real-time analytics", 5.0),
        ("data visualization", 4.5), ("metric tracking", 3.5),
        ("data warehouse", 4.0), ("data pipeline", 3.5),
    ],
    "payments": [
        ("payment", 5.0), ("payments", 5.0), ("embedded payment", 5.0),
        ("split payment", 5.0), ("escrow", 4.5), ("settlement", 4.0),
        ("transaction processing", 4.5), ("payment infrastructure", 5.0),
        ("payment volume", 4.0), ("take rate", 3.5), ("multi-currency", 4.0),
        ("payment api", 5.0), ("checkout", 3.5),
    ],
    "diagnostics": [
        ("diagnostic", 5.0), ("diagnostic imaging", 5.0),
        ("medical imaging", 5.0), ("radiology", 5.0), ("x-ray", 4.0),
        ("mri", 4.0), ("ct scan", 4.0), ("anomaly detection", 3.5),
        ("detection sensitivity", 4.5), ("clinical", 3.0),
    ],
    "compliance": [
        ("compliance", 5.0), ("regulatory", 4.5), ("audit", 4.0),
        ("policy management", 4.5), ("regulatory tracking", 5.0),
        ("compliance management", 5.0), ("compliance platform", 5.0),
        ("soc 2", 4.5), ("pci dss", 4.5), ("gdpr", 4.0),
        ("ccpa", 4.0), ("audit preparation", 4.5),
        ("evidence collection", 3.5), ("control monitoring", 4.0),
    ],
    "ci_cd": [
        ("ci/cd", 5.0), ("ci cd", 5.0), ("continuous integration", 5.0),
        ("continuous deployment", 5.0), ("build pipeline", 4.5),
        ("deployment orchestration", 5.0), ("test parallelization", 4.5),
        ("incremental builds", 4.0), ("release management", 3.5),
    ],
    "marketplace": [
        ("marketplace", 5.0), ("two-sided", 4.5), ("buyer", 3.0),
        ("supplier", 3.5), ("vendor", 3.0), ("quoting", 3.5),
        ("procurement", 4.0), ("gmv", 4.0), ("transaction volume", 3.5),
        ("listing", 2.5),
    ],
    "fitness_social": [
        ("fitness", 5.0), ("workout", 4.5), ("exercise", 3.5),
        ("health tracking", 4.0), ("social network", 4.0),
        ("community challenge", 4.5), ("progress sharing", 3.5),
    ],
    "carbon_accounting": [
        ("carbon accounting", 5.0), ("carbon emissions", 5.0),
        ("emissions tracking", 5.0), ("scope 1", 4.5), ("scope 2", 4.5),
        ("scope 3", 4.5), ("ghg protocol", 5.0), ("sec climate", 4.5),
        ("climate disclosure", 4.5),
    ],
    "robotics": [
        ("autonomous mobile robot", 5.0), ("robotics", 5.0),
        ("warehouse automation", 5.0), ("lidar", 4.5), ("slam", 4.5),
        ("goods-to-person", 5.0), ("pallet transport", 4.5),
        ("fleet", 3.5), ("robot", 4.0),
    ],
    "model_serving": [
        ("model serving", 5.0), ("inference", 4.5), ("gpu", 4.0),
        ("model versioning", 4.5), ("auto-scaling", 3.5),
        ("gpu cluster", 5.0), ("inference request", 4.5),
        ("model deployment", 4.5), ("multi-cloud gpu", 5.0),
    ],
}

# ---------------------------------------------------------------------------
# Product type / form factor detection.
# ---------------------------------------------------------------------------

_PRODUCT_TYPE_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "platform": [
        ("platform", 5.0), ("saas platform", 5.0), ("cloud platform", 5.0),
        ("two-sided platform", 5.0), ("platform businesses", 4.0),
        ("platform providing", 4.0), ("developer platform", 5.0),
    ],
    "application": [
        ("application", 4.0), ("mobile app", 4.5), ("app", 2.0),
        ("consumer mobile", 4.5), ("consumer app", 5.0), ("web app", 4.0),
        ("tool", 3.0),
    ],
    "api": [
        ("api-first", 5.0), ("api first", 5.0), ("api-first solution", 5.0),
        ("api", 3.0), ("rest api", 4.5), ("graphql api", 4.5),
        ("payment api", 4.5), ("api solution", 4.0), ("programmatic", 3.5),
    ],
    "infrastructure": [
        ("infrastructure", 5.0), ("gpu-optimized model serving infrastructure", 5.0),
        ("payment infrastructure", 4.5), ("cloud infrastructure", 4.5),
        ("developer infrastructure", 4.5), ("backend infrastructure", 4.0),
    ],
    "tool": [
        ("developer tool", 5.0), ("open-source-first ci/cd platform", 4.5),
        ("tool", 2.5), ("cli", 3.0), ("developer tools", 5.0),
        ("diagnostic tool", 4.0),
    ],
}

# ---------------------------------------------------------------------------
# Delivery / SaaS model classification.
# ---------------------------------------------------------------------------

_SAAS_MODEL_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "saas": [
        ("saas", 5.0), ("b2b saas", 5.0), ("subscription", 4.0),
        ("subscription-based", 4.0), ("annual contract", 3.5),
        ("monthly subscription", 4.0), ("recurring revenue", 3.5),
        ("gross retention", 3.0), ("net revenue retention", 3.0),
        ("freemium", 3.0), ("premium subscription", 3.0),
    ],
    "api": [
        ("api-first", 5.0), ("api-first solution", 5.0), ("api", 3.0),
        ("per-call", 3.5), ("usage-based pricing", 3.5),
        ("per-study licensing", 3.0), ("payment api", 4.0),
    ],
    "infrastructure": [
        ("infrastructure", 4.0), ("gpu-optimized infrastructure", 5.0),
        ("model serving infrastructure", 5.0), ("payment infrastructure", 4.0),
        ("developer infrastructure", 4.0), ("cloud infrastructure", 4.0),
    ],
    "marketplace": [
        ("marketplace", 5.0), ("two-sided marketplace", 5.0),
        ("marketplace take rate", 5.0), ("transaction volume", 3.5),
        ("gmv", 4.0), ("take rate", 3.0),
    ],
    "licensing": [
        ("licensing", 4.5), ("per-study licensing", 5.0),
        ("enterprise licensing", 4.5), ("per-seat", 3.5),
        ("patent licensing", 4.0), ("license", 3.0),
    ],
    "paas": [
        ("platform as a service", 5.0), ("paas", 5.0),
        ("robotics-as-a-service", 5.0), ("raas", 5.0),
        ("per-robot monthly fees", 5.0), ("as-a-service", 3.5),
    ],
    "raas": [
        ("robotics-as-a-service", 5.0), ("raas", 5.0),
        ("per-robot", 5.0), ("fleet", 3.0),
    ],
}

# ---------------------------------------------------------------------------
# AI orientation detection.
# ---------------------------------------------------------------------------

_AI_NATIVE_KEYWORDS: list[tuple[str, float]] = [
    ("ai diagnostic", 5.0), ("ai-powered", 5.0), ("ai-driven", 5.0),
    ("machine learning", 5.0), ("deep learning", 5.0),
    ("neural network", 5.0), ("convolutional neural network", 5.0),
    ("large language model", 5.0), ("llm", 4.5),
    ("model serving", 4.5), ("inference", 4.0),
    ("training data", 4.0), ("pytorch", 4.5), ("tensorflow", 4.5),
    ("gpu cluster", 5.0), ("transformer", 4.0), ("nlp", 3.5),
    ("computer vision", 5.0), ("model deployment", 4.5),
    ("ai diagnostic tools", 5.0), ("artificial intelligence", 5.0),
]

_AI_ENABLED_KEYWORDS: list[tuple[str, float]] = [
    ("ai", 2.5), ("ml", 2.0), ("automated insight", 4.0),
    ("intelligent", 3.0), ("smart", 2.5), ("automated", 2.5),
    ("proprietary algorithm", 4.0), ("algorithm", 3.0),
    ("pattern recognition", 4.0), ("anomaly detection", 4.0),
    ("predictive", 3.0), ("automated detection", 3.5),
]

# ---------------------------------------------------------------------------
# Capability keywords — detect what the product does.
# ---------------------------------------------------------------------------

_CAPABILITY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "analytics_and_reporting": [
        ("analytics", 4.0), ("reporting", 3.5), ("dashboard", 3.5),
        ("insight generation", 4.0), ("data visualization", 4.0),
        ("business intelligence", 4.0),
    ],
    "data_processing": [
        ("data processing", 5.0), ("etl", 4.0), ("data pipeline", 4.5),
        ("data collection", 4.0), ("automated data collection", 4.5),
        ("data aggregation", 4.0),
    ],
    "automation": [
        ("automated", 3.5), ("automation", 4.0), ("automates", 4.0),
        ("orchestration", 3.5), ("auto-scaling", 3.5),
        ("automated evidence collection", 4.0),
    ],
    "monitoring": [
        ("monitoring", 4.0), ("tracking", 3.5), ("real-time", 3.0),
        ("control monitoring", 4.5), ("anomaly", 3.0),
    ],
    "integration": [
        ("integration", 3.5), ("integrates with", 4.0),
        ("connects", 2.5), ("ecosystem", 3.0), ("plug-in", 3.0),
    ],
    "compliance": [
        ("compliance", 4.0), ("regulatory", 3.5), ("audit", 3.0),
        ("policy management", 3.5), ("regulatory tracking", 4.0),
    ],
    "security": [
        ("security", 3.5), ("encryption", 3.0), ("authentication", 3.0),
        ("zero trust", 4.0), ("kyc compliance", 3.5),
    ],
    "navigation_and_movement": [
        ("navigation", 4.0), ("slam", 4.5), ("lidar", 4.0),
        ("picking", 3.0), ("pallet transport", 4.0),
    ],
    "model_management": [
        ("model versioning", 4.5), ("model serving", 5.0),
        ("model deployment", 4.5), ("inference", 4.0),
        ("a/b testing", 3.5), ("cost optimization", 3.0),
    ],
}

# ---------------------------------------------------------------------------
# Feature signals — specific product features mentioned.
# ---------------------------------------------------------------------------

_FEATURE_SIGNAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("real_time", re.compile(r"\breal[\s-]?time\b", re.I)),
    ("open_source", re.compile(r"\bopen[\s-]?source\b", re.I)),
    ("api_first", re.compile(r"\bapi[\s-]?first\b", re.I)),
    ("cloud_native", re.compile(r"\bcloud[\s-]?native\b", re.I)),
    ("multi_cloud", re.compile(r"\bmulti[\s-]?cloud\b", re.I)),
    ("auto_scaling", re.compile(r"\bauto[\s-]?scal\w*\b", re.I)),
    ("a_b_testing", re.compile(r"\ba/b\b", re.I)),
    ("sso", re.compile(r"\bsso\b", re.I)),
    ("audit_logging", re.compile(r"\baudit\s*log\w*\b", re.I)),
    ("multi_tenant", re.compile(r"\bmulti[\s-]?tenant\b", re.I)),
    ("freemium", re.compile(r"\bfreemium\b", re.I)),
    ("mobile", re.compile(r"\bmobile\b", re.I)),
    ("ios", re.compile(r"\bios\b", re.I)),
    ("android", re.compile(r"\bandroid\b", re.I)),
    ("web", re.compile(r"\bweb\b", re.I)),
    ("embedded", re.compile(r"\bembedded\b", re.I)),
    ("on_demand", re.compile(r"\bon[\s-]?demand\b", re.I)),
]

# ---------------------------------------------------------------------------
# Integration ecosystem — detectable integration points.
# ---------------------------------------------------------------------------

_INTEGRATION_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ("data_warehouse", re.compile(r"\bdata\s+warehou\w*\b", re.I)),
    ("erp_integration", re.compile(r"\berp\b", re.I)),
    ("crm_integration", re.compile(r"\bcrm\b", re.I)),
    ("supply_chain_tools", re.compile(r"\bsupply\s+chain\b", re.I)),
    ("iot_sensors", re.compile(r"\biot\s+sensor\w*\b", re.I)),
    ("cloud_platforms", re.compile(r"\b(aws|gcp|azure|multi[\s-]?cloud)\b", re.I)),
    ("payment_systems", re.compile(r"\bpayment\s+system\w*\b", re.I)),
    ("api_ecosystem", re.compile(r"\bapi\b", re.I)),
    ("github", re.compile(r"\bgithub\b", re.I)),
    ("docker", re.compile(r"\bdocker\b", re.I)),
    ("kubernetes", re.compile(r"\bkubernetes\b", re.I)),
    ("grafana", re.compile(r"\bgrafana\b", re.I)),
    ("slack", re.compile(r"\bslack\b", re.I)),
    ("jira", re.compile(r"\bjira\b", re.I)),
    ("confluence", re.compile(r"\bconfluence\b", re.I)),
]

# ---------------------------------------------------------------------------
# Deployment model keywords.
# ---------------------------------------------------------------------------

_DEPLOYMENT_MODEL_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "cloud": [
        ("cloud", 4.0), ("saas", 3.5), ("hosted", 3.0),
        ("cloud-native", 4.5), ("multi-cloud", 4.0), ("cloud platform", 3.5),
    ],
    "on_premise": [
        ("on-premise", 5.0), ("on premise", 5.0), ("self-hosted", 4.5),
        ("on-site", 3.5), ("private cloud", 4.0),
    ],
    "hybrid": [
        ("hybrid cloud", 5.0), ("hybrid deployment", 4.5),
        ("cloud and on-premise", 4.5),
    ],
    "edge": [
        ("edge", 4.0), ("edge computing", 5.0), ("embedded", 3.5),
        ("on-device", 4.0), ("firmware", 3.5),
    ],
    "saas": [
        ("saas", 5.0), ("subscription", 3.0), ("hosted", 3.5),
        ("cloud-delivered", 4.0),
    ],
    "ras": [
        ("robotics-as-a-service", 5.0), ("raas", 5.0),
        ("per-robot monthly fees", 5.0), ("fleet management", 4.0),
    ],
}

# ---------------------------------------------------------------------------
# Target workflow keywords.
# ---------------------------------------------------------------------------

_TARGET_WORKFLOW_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "data_analytics": [
        ("analytics", 4.0), ("reporting", 3.0), ("business intelligence", 4.5),
        ("insight generation", 4.0), ("query", 2.5),
    ],
    "payment_processing": [
        ("payment processing", 5.0), ("split payments", 5.0),
        ("escrow", 4.0), ("settlement", 4.0), ("checkout", 3.5),
    ],
    "regulatory_compliance": [
        ("compliance", 4.0), ("regulatory tracking", 5.0),
        ("audit preparation", 5.0), ("policy management", 4.5),
        ("evidence collection", 4.0),
    ],
    "ci_cd_pipeline": [
        ("ci/cd", 5.0), ("build pipeline", 4.5), ("deployment", 3.5),
        ("test parallelization", 4.5), ("release management", 3.5),
    ],
    "procurement": [
        ("procurement", 5.0), ("quoting", 4.0), ("purchasing", 3.5),
        ("mro supplies", 4.5),
    ],
    "medical_diagnosis": [
        ("diagnostic", 5.0), ("medical imaging", 5.0), ("radiology", 5.0),
        ("clinical trial", 4.5), ("diagnosis", 4.5),
    ],
    "fitness_tracking": [
        ("workout", 4.5), ("fitness tracking", 5.0),
        ("exercise logging", 4.5), ("progress", 2.5),
    ],
    "carbon_reporting": [
        ("carbon accounting", 5.0), ("emissions tracking", 5.0),
        ("scope 1", 4.5), ("ghg protocol", 5.0),
    ],
    "warehouse_operations": [
        ("warehouse", 4.0), ("fulfillment", 3.5), ("picking", 3.5),
        ("inventory scanning", 4.0), ("pallet transport", 4.5),
    ],
    "ml_model_operations": [
        ("model serving", 5.0), ("inference", 4.0), ("model versioning", 4.5),
        ("model deployment", 4.5), ("gpu cluster management", 4.5),
    ],
    "social_engagement": [
        ("social network", 5.0), ("community challenge", 4.5),
        ("sharing", 2.5), ("social platform", 4.0),
    ],
}

# ---------------------------------------------------------------------------
# Automation level keywords.
# ---------------------------------------------------------------------------

_AUTOMATION_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "high": [
        ("automated", 4.0), ("automates", 4.5), ("automation", 4.0),
        ("automate", 4.0), ("auto-scaling", 4.0), ("orchestration", 3.5),
        ("automated evidence collection", 5.0), ("automated detection", 4.5),
        ("automated regulatory", 4.5), ("automated data collection", 4.5),
        ("intelligent", 3.0),
    ],
    "moderate": [
        ("assists", 3.5), ("assist", 3.0), ("assist radiologists", 4.5),
        ("semi-automated", 4.0), ("intelligent", 2.5),
    ],
    "low": [
        ("manual", 3.0), ("hands-on", 3.0), ("human review", 3.5),
    ],
}

# ---------------------------------------------------------------------------
# Product maturity signals.
# ---------------------------------------------------------------------------

_MATURITY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "concept": [
        ("concept", 5.0), ("idea stage", 5.0), ("prototype", 4.0),
        ("proof of concept", 5.0), ("poc", 4.0),
    ],
    "beta": [
        ("beta", 5.0), ("beta launch", 5.0), ("alpha", 4.0),
        ("early access", 4.0), ("waitlist", 3.5),
    ],
    "growth": [
        ("scaling", 4.0), ("rapid growth", 5.0), ("growing", 3.0),
        ("expanding", 3.0), ("increasing", 2.5), ("traction", 2.5),
        ("product-market fit", 5.0),
    ],
    "mature": [
        ("mature", 5.0), ("established", 4.0), ("proven", 3.5),
        ("incumbent", 3.5), ("market leader", 4.5),
    ],
}

# ---------------------------------------------------------------------------
# Differentiation signals — detectable competitive advantages.
# ---------------------------------------------------------------------------

_DIFFERENTIATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("proprietary_technology", re.compile(r"\bproprietary\b", re.I)),
    ("patent_protection", re.compile(r"\bpatent\w*\b", re.I)),
    ("clinical_validation", re.compile(r"\bclinical\s+trial\w*\b", re.I)),
    ("fda_clearance", re.compile(r"\bfda[\s-]?cleared\b", re.I)),
    ("open_source_community", re.compile(
        r"\bopen[\s-]?source\b.*\b(github|star|contribut)\b", re.I
    )),
    ("regulatory_compliance", re.compile(
        r"\b(soc\s*2|pci\s*dss|gdpr|hipaa|ghg\s*protocol)\b", re.I
    )),
    ("first_mover", re.compile(r"\bfirst[\s-]?mover\b", re.I)),
    ("category_creator", re.compile(r"\bcategory[\s-]?creat\w*\b", re.I)),
    ("network_effects", re.compile(r"\bnetwork\s+effect\w*\b", re.I)),
    ("data_advantage", re.compile(r"\bproprietary\s+data\b", re.I)),
]

# ---------------------------------------------------------------------------
# Defensibility / moat signals.
# ---------------------------------------------------------------------------

_DEFENSIBILITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("regulatory_moat", re.compile(
        r"\b(regulat\w+|compliance|certification|hipaa|soc\s*2|fda|iso\s*27)\b",
        re.I,
    )),
    ("technology_moat", re.compile(
        r"\b(patent\w*|proprietary\s+(algorithm|technology|architecture)"
        r"|trade\s+secret)\b", re.I,
    )),
    ("data_moat", re.compile(
        r"\b(proprietary\s+data|training\s+data|data\s+advantage)\b", re.I
    )),
    ("network_effects", re.compile(
        r"\b(network\s+effect\w*|flywheel|two[\s-]sided)\b", re.I
    )),
    ("switching_costs", re.compile(
        r"\b(switching\s+cost\w*|lock[\s-]in|vendor\s+lock|deep\s+integrat\w+)\b",
        re.I,
    )),
    ("brand_recognition", re.compile(
        r"\b(fortune\s*500|top[\s-]20\s+us\s+banks)\b", re.I
    )),
    ("ecosystem_lock_in", re.compile(
        r"\b(ecosystem|platform\s+effect\w*|developer\s+ecosystem)\b", re.I
    )),
]

# ---------------------------------------------------------------------------
# Scalability indicators.
# ---------------------------------------------------------------------------

_SCALABILITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("auto_scaling", re.compile(r"\bauto[\s-]?scal\w*\b", re.I)),
    ("cloud_native", re.compile(r"\bcloud[\s-]?native\b", re.I)),
    ("multi_tenant", re.compile(r"\bmulti[\s-]?tenant\b", re.I)),
    ("multi_cloud", re.compile(r"\bmulti[\s-]?cloud\b", re.I)),
    ("api_first", re.compile(r"\bapi[\s-]?first\b", re.I)),
    ("global_deployment", re.compile(
        r"\b(glob\w*|worldwide|international|\d+\s*country\w*|multi[\s-]region)\b",
        re.I,
    )),
    ("high_volume", re.compile(
        r"\b(\d+\s*(?:billion|million)\s+(?:request\w*|transaction\w*|volume))\b",
        re.I,
    )),
    ("fleet_scaling", re.compile(r"\bfleet\s+of\s+\d+\b", re.I)),
    ("distributed_team", re.compile(r"\b(\d+\s*(?:engineer\w*|team\s+of)\s+\d+)\b", re.I)),
]

# ---------------------------------------------------------------------------
# Innovation signals.
# ---------------------------------------------------------------------------

_INNOVATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("novel_technology", re.compile(
        r"\b(novel|innovat\w+|breakthrough|first[\s-]of[\s-]its[\s-]kind)\b",
        re.I,
    )),
    ("patent_filed", re.compile(r"\bpatent\w*\b", re.I)),
    ("clinical_validation", re.compile(
        r"\bclinical\s+trial\w*\b", re.I
    )),
    ("proprietary_architecture", re.compile(
        r"\bproprietary\s+(algorithm|architecture|technolog\w+)\b", re.I
    )),
    ("ai_native_approach", re.compile(
        r"\b(ai[\s-]driven|ai[\s-]powered|deep\s+learning|neural\s+network)\b",
        re.I,
    )),
    ("open_source_innovation", re.compile(
        r"\bopen[\s-]?source\b.*\b(github|star|contribut)\b", re.I
    )),
    ("gpu_optimization", re.compile(r"\bgpu[\s-]?optimized\b", re.I)),
    ("novel_business_model", re.compile(
        r"\b(robotics[\s-]as[\s-]a[\s-]service|raas|as[\s-]a[\s-]service)\b",
        re.I,
    )),
]

# ---------------------------------------------------------------------------
# Product keywords — domain-specific terms to extract.
# ---------------------------------------------------------------------------

_PRODUCT_KEYWORD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("saas", re.compile(r"\bsaas\b", re.I)),
    ("api_first", re.compile(r"\bapi[\s-]?first\b", re.I)),
    ("cloud_native", re.compile(r"\bcloud[\s-]?native\b", re.I)),
    ("open_source", re.compile(r"\bopen[\s-]?source\b", re.I)),
    ("ai_powered", re.compile(r"\bai[\s-]?(?:powered|driven|enabled|native)\b", re.I)),
    ("real_time", re.compile(r"\breal[\s-]?time\b", re.I)),
    ("platform", re.compile(r"\bplatform\b", re.I)),
    ("marketplace", re.compile(r"\bmarketplace\b", re.I)),
    ("enterprise", re.compile(r"\benterprise\b", re.I)),
    ("automation", re.compile(r"\bautomat\w*\b", re.I)),
    ("analytics", re.compile(r"\banalytics\b", re.I)),
    ("compliance", re.compile(r"\bcompliance\b", re.I)),
    ("integration", re.compile(r"\bintegrat\w*\b", re.I)),
    ("infrastructure", re.compile(r"\binfrastructure\b", re.I)),
    ("developer_tools", re.compile(r"\bdeveloper\s+tool\w*\b", re.I)),
    ("subscription", re.compile(r"\bsubscription\b", re.I)),
    ("embedded", re.compile(r"\bembedded\b", re.I)),
    ("freemium", re.compile(r"\bfreemium\b", re.I)),
    ("open_source_first", re.compile(r"\bopen[\s-]?source[\s-]?first\b", re.I)),
    ("robotics", re.compile(r"\brobot\w*\b", re.I)),
    ("model_serving", re.compile(r"\bmodel\s+serving\b", re.I)),
    ("inference", re.compile(r"\binference\b", re.I)),
    ("gpu_cluster", re.compile(r"\bgpu\s+cluster\w*\b", re.I)),
    ("carbon_accounting", re.compile(r"\bcarbon\s+accounting\b", re.I)),
]


class ProductExtractor(BaseExtractor):
    """Extracts rich product intelligence from startup data.

    Produces:
      - product_category: primary product domain
      - product_type: form factor (platform, application, api, infrastructure, tool)
      - saas_model: delivery model (saas, api, infrastructure, marketplace, licensing, paas, raas)
      - ai_orientation: AI relationship (ai_native, ai_enabled, non_ai, unknown)
      - technology_stack: identified technologies (inherited from base)
      - primary_capabilities: core product capabilities
      - feature_signals: specific product features detected
      - integration_ecosystem: integration points detected
      - deployment_model: deployment approach
      - target_workflow: primary workflow addressed
      - automation_level: degree of automation
      - product_maturity: product lifecycle stage
      - differentiation_signals: competitive differentiators
      - defensibility_signals: moat indicators
      - technical_complexity: estimated complexity
      - scalability_indicators: growth enablers
      - innovation_signals: novelty indicators
      - product_keywords: domain-specific product terms
      - product_confidence: extraction confidence (0.0-1.0)
    """

    def extract(
        self,
        startup: Startup,
        data: CollectedData,
        evidence: EvidenceBundle | None = None,
    ) -> ExtractedFeatures:
        from predictron_engine.evidence.citation import build_citations
        from predictron_engine.extraction.evidence_agreement import (
            compute_agreement_ratio,
            detect_conflicts,
        )
        from predictron_engine.extraction.evidence_confidence import (
            compute_evidence_confidence,
        )
        from predictron_engine.extraction.evidence_retrieval import (
            ProductRetrievalStrategy,
        )

        strategy = ProductRetrievalStrategy()
        docs_filtered: list = []
        evidence_items: list = []
        citations: list = []

        if evidence is not None:
            [
                doc for doc in evidence.documents
                if doc.status.value == "success"
            ]
            docs_filtered = strategy.retrieve_documents(evidence)
            evidence_items = []  # Domain-specific items populated from reasoning layer
            citations = build_citations(evidence_items, docs_filtered)

        text = self._combined_text_from_docs(startup.description, docs_filtered) if docs_filtered else self._combined_text(startup.description, evidence)
        text_lower = text.lower()

        # Core classifications
        product_category = self._classify_product_category(text_lower)
        product_type = self._classify_product_type(text_lower)
        saas_model = self._classify_saas_model(text_lower)
        ai_orientation = self._classify_ai_orientation(text_lower, text)

        # Capabilities and features
        primary_capabilities = self._extract_capabilities(text_lower)
        feature_signals = self._extract_feature_signals(text)

        # Integration and deployment
        integration_ecosystem = self._extract_integration_ecosystem(text)
        deployment_model = self._classify_deployment_model(text_lower)

        # Workflow and automation
        target_workflow = self._infer_target_workflow(text_lower)
        automation_level = self._classify_automation_level(text_lower)

        # Maturity and quality
        product_maturity = self._classify_product_maturity(text_lower)

        # Competitive intelligence
        differentiation_signals = self._extract_differentiation_signals(text)
        defensibility_signals = self._extract_defensibility_signals(text)

        # Technical signals
        technical_complexity = self._estimate_technical_complexity(text_lower)
        scalability_indicators = self._extract_scalability_indicators(text)
        innovation_signals = self._extract_innovation_signals(text)

        # Keywords and confidence
        product_keywords = self._extract_product_keywords(text)
        tech_stack = self._extract_tech_terms(text)

        # Compute confidence based on signal strength
        product_confidence = self._compute_product_confidence(
            text_lower, product_category, product_type, saas_model,
            primary_capabilities, feature_signals,
        )

        features = ExtractedFeatures(
            product_category=product_category,
            product_type=product_type,
            saas_model=saas_model,
            ai_orientation=ai_orientation,
            technology_stack=tech_stack,
            primary_capabilities=primary_capabilities,
            feature_signals=feature_signals,
            integration_ecosystem=integration_ecosystem,
            deployment_model=deployment_model,
            target_workflow=target_workflow,
            automation_level=automation_level,
            product_maturity=product_maturity,
            differentiation_signals=differentiation_signals,
            defensibility_signals=defensibility_signals,
            technical_complexity=technical_complexity,
            scalability_indicators=scalability_indicators,
            innovation_signals=innovation_signals,
            product_keywords=product_keywords,
            product_confidence=product_confidence,
        )

        # --- Sprint 5C: Evidence-aware provenance ---
        if evidence is not None and docs_filtered:
            agreement = compute_agreement_ratio(evidence_items)
            conflicts = detect_conflicts(evidence_items)
            ev_confidence = compute_evidence_confidence(
                evidence_items=evidence_items,
                documents=docs_filtered,
                keywords_matched=self._count_keyword_matches(text_lower, strategy.get_keywords()),
                total_keywords=len(strategy.get_keywords()),
                description_length=len(text),
            )
            self._populate_evidence_provenance(
                features,
                documents=docs_filtered,
                evidence_items=evidence_items,
                citations=citations,
                evidence_confidence=ev_confidence,
                agreement_ratio=agreement,
                conflict_count=len(conflicts),
            )

        return features

    # ------------------------------------------------------------------
    # Product category classification
    # ------------------------------------------------------------------

    def _classify_product_category(self, text: str) -> str | None:
        """Classify primary product category using weighted keyword scoring."""
        scores: dict[str, float] = {}
        match_counts: dict[str, int] = {}

        for category, weighted_keywords in _PRODUCT_CATEGORY_KEYWORDS.items():
            total_score = 0.0
            matches = 0
            for keyword, weight in weighted_keywords:
                if keyword in text:
                    total_score += weight
                    matches += 1
            if total_score > 0:
                scores[category] = total_score
                match_counts[category] = matches

        if not scores:
            return None

        best_category = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_category]
        best_matches = match_counts[best_category]

        # Require minimum score for classification
        if best_score < 4.0:
            return None

        # Check for ambiguity: if second-best is too close, return None
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            margin = best_score - sorted_scores[1]
            if margin < 2.0 and best_matches < 3:
                return None

        return best_category

    # ------------------------------------------------------------------
    # Product type / form factor classification
    # ------------------------------------------------------------------

    def _classify_product_type(self, text: str) -> str | None:
        """Classify product form factor."""
        scores: dict[str, float] = {}

        for ptype, weighted_keywords in _PRODUCT_TYPE_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[ptype] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        if best_score < 3.0:
            return None

        return best

    # ------------------------------------------------------------------
    # Delivery / SaaS model classification
    # ------------------------------------------------------------------

    def _classify_saas_model(self, text: str) -> str | None:
        """Classify the product's delivery / business model from product perspective."""
        scores: dict[str, float] = {}

        for model, weighted_keywords in _SAAS_MODEL_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[model] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        if best_score < 3.0:
            return None

        return best

    # ------------------------------------------------------------------
    # AI orientation classification
    # ------------------------------------------------------------------

    def _classify_ai_orientation(self, text_lower: str, text_original: str) -> str:
        """Classify AI relationship: ai_native, ai_enabled, non_ai, or unknown."""
        native_score = sum(w for kw, w in _AI_NATIVE_KEYWORDS if kw in text_lower)
        enabled_score = sum(w for kw, w in _AI_ENABLED_KEYWORDS if kw in text_lower)

        if native_score >= 8.0:
            return "ai_native"
        if enabled_score >= 4.0:
            return "ai_enabled"
        if native_score >= 4.0 and enabled_score >= 2.0:
            return "ai_enabled"
        if native_score < 1.0 and enabled_score < 1.0:
            return "non_ai"
        return "unknown"

    # ------------------------------------------------------------------
    # Capability extraction
    # ------------------------------------------------------------------

    def _extract_capabilities(self, text: str) -> list[str]:
        """Extract core product capabilities from the description."""
        found: list[str] = []
        for capability, weighted_keywords in _CAPABILITY_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total >= 3.0:
                found.append(capability)
        return found

    # ------------------------------------------------------------------
    # Feature signal extraction
    # ------------------------------------------------------------------

    def _extract_feature_signals(self, text: str) -> list[str]:
        """Extract specific product features from the description."""
        found: list[str] = []
        for label, pattern in _FEATURE_SIGNAL_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Integration ecosystem extraction
    # ------------------------------------------------------------------

    def _extract_integration_ecosystem(self, text: str) -> list[str]:
        """Extract detected integration points from the description."""
        found: list[str] = []
        for label, pattern in _INTEGRATION_KEYWORDS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Deployment model classification
    # ------------------------------------------------------------------

    def _classify_deployment_model(self, text: str) -> str | None:
        """Classify deployment approach from description signals."""
        scores: dict[str, float] = {}

        for model, weighted_keywords in _DEPLOYMENT_MODEL_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[model] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        if best_score < 3.0:
            return None

        return best

    # ------------------------------------------------------------------
    # Target workflow inference
    # ------------------------------------------------------------------

    def _infer_target_workflow(self, text: str) -> str | None:
        """Infer the primary workflow the product addresses."""
        scores: dict[str, float] = {}

        for workflow, weighted_keywords in _TARGET_WORKFLOW_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[workflow] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        if best_score < 3.5:
            return None

        return best

    # ------------------------------------------------------------------
    # Automation level classification
    # ------------------------------------------------------------------

    def _classify_automation_level(self, text: str) -> str | None:
        """Classify the degree of automation."""
        scores: dict[str, float] = {}

        for level, weighted_keywords in _AUTOMATION_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[level] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        if best_score < 3.0:
            return None

        return best

    # ------------------------------------------------------------------
    # Product maturity classification
    # ------------------------------------------------------------------

    def _classify_product_maturity(self, text: str) -> str | None:
        """Classify product lifecycle stage."""
        scores: dict[str, float] = {}

        for stage, weighted_keywords in _MATURITY_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[stage] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] >= 3.0 else None

    # ------------------------------------------------------------------
    # Differentiation signal extraction
    # ------------------------------------------------------------------

    def _extract_differentiation_signals(self, text: str) -> list[str]:
        """Extract detected competitive differentiators."""
        found: list[str] = []
        for label, pattern in _DIFFERENTIATION_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Defensibility signal extraction
    # ------------------------------------------------------------------

    def _extract_defensibility_signals(self, text: str) -> list[str]:
        """Extract detected moat and defensibility indicators."""
        found: list[str] = []
        for label, pattern in _DEFENSIBILITY_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Technical complexity estimation
    # ------------------------------------------------------------------

    def _estimate_technical_complexity(self, text: str) -> str | None:
        """Estimate technical complexity from description signals."""
        high_signals = [
            "convolutional neural network", "slam", "lidar",
            "gpu cluster", "multi-cloud gpu", "gpu-optimized",
            "model serving infrastructure", "autonomous mobile robot",
            "deep learning", "neural network", "fda-cleared",
            "clinical trial", "embedded", "firmware", "robotics",
        ]
        moderate_signals = [
            "platform", "api", "integration", "compliance",
            "real-time", "auto-scaling", "infrastructure",
            "ci/cd", "pipeline", "orchestration",
        ]
        low_signals = [
            "mobile app", "social network", "tracking",
            "fitness", "freemium",
        ]

        high_count = sum(1 for s in high_signals if s in text)
        moderate_count = sum(1 for s in moderate_signals if s in text)
        low_count = sum(1 for s in low_signals if s in text)

        if high_count >= 2:
            return "high"
        if high_count >= 1 and moderate_count >= 2:
            return "high"
        if moderate_count >= 3:
            return "moderate"
        if high_count >= 1:
            return "moderate"
        if low_count >= 2:
            return "low"
        if moderate_count >= 1:
            return "moderate"
        return "unknown"

    # ------------------------------------------------------------------
    # Scalability indicator extraction
    # ------------------------------------------------------------------

    def _extract_scalability_indicators(self, text: str) -> list[str]:
        """Extract scalability and growth enablers."""
        found: list[str] = []
        for label, pattern in _SCALABILITY_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Innovation signal extraction
    # ------------------------------------------------------------------

    def _extract_innovation_signals(self, text: str) -> list[str]:
        """Extract innovation and novelty indicators."""
        found: list[str] = []
        for label, pattern in _INNOVATION_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Product keyword extraction
    # ------------------------------------------------------------------

    def _extract_product_keywords(self, text: str) -> list[str]:
        """Extract domain-specific product terms from the description."""
        found: list[str] = []
        for label, pattern in _PRODUCT_KEYWORD_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    def _compute_product_confidence(
        self,
        text: str,
        category: str | None,
        ptype: str | None,
        saas_model: str | None,
        capabilities: list[str],
        features: list[str],
    ) -> float:
        """Compute extraction confidence based on signal density.

        High confidence requires multiple strong signals across dimensions.
        """
        # Signal density component (0-0.4)
        word_count = max(len(text.split()), 1)
        signal_density = min(word_count / 50.0, 0.4)

        # Classification component (0-0.3): each known classification adds confidence
        classification_count = sum(1 for x in [category, ptype, saas_model] if x is not None)
        classification_component = min(classification_count / 3.0 * 0.3, 0.3)

        # Capability component (0-0.15)
        capability_component = min(len(capabilities) / 3.0 * 0.15, 0.15)

        # Feature component (0-0.15)
        feature_component = min(len(features) / 4.0 * 0.15, 0.15)

        combined = (
            signal_density
            + classification_component
            + capability_component
            + feature_component
        )

        return round(min(combined, 1.0), 2)
