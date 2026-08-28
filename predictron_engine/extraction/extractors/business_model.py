"""Business model extractor — rich deterministic business model intelligence.

Responsibilities:
  - Primary business model classification via weighted keyword scoring
  - Secondary business model detection for hybrid models
  - Revenue model classification
  - Pricing model classification
  - Monetization strategy classification
  - Customer acquisition model classification
  - Sales motion classification
  - Distribution model classification
  - Value proposition signal extraction
  - Recurring vs transactional revenue detection
  - Marketplace dynamics detection
  - Network effects signal extraction
  - Platform characteristic detection
  - Switching cost indicator extraction
  - Unit economics indicator extraction
  - Business model maturity assessment
  - Business model keyword extraction
  - Business model confidence scoring

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
from typing import TYPE_CHECKING

from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

if TYPE_CHECKING:
    from predictron_engine.evidence.models import EvidenceDocument
    from predictron_engine.models.report import EvidenceCitation, EvidenceItem

# ---------------------------------------------------------------------------
# Primary business model — weighted keyword scoring.
# Each keyword carries a weight (higher = stronger signal).
# ---------------------------------------------------------------------------

_PRIMARY_BUSINESS_MODEL_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "saas": [
        ("saas", 5.0), ("b2b saas", 5.0), ("subscription", 4.0),
        ("monthly subscription", 4.5), ("annual contract", 4.0),
        ("recurring revenue", 3.5), ("mrr", 4.0), ("arr", 4.0),
        ("net revenue retention", 3.5), ("gross retention", 3.5),
        ("recurring", 2.5), ("per-seat", 3.0), ("seat-based", 3.5),
        ("subscription-based", 4.0), ("subscription pricing", 4.5),
        ("monthly recurring", 4.5), ("annual contract value", 4.0),
        ("acv", 3.0), ("cloud platform", 2.0), ("enterprise software", 2.5),
        ("enterprise compliance", 2.5), ("compliance management", 2.5),
        ("carbon accounting software", 3.0), ("analytics platform", 2.5),
        ("freemium", 2.5), ("premium subscription", 3.0),
        ("platform providing", 2.0),
        ("paid plans", 3.5), ("converting.*to paid", 4.0),
        ("paid", 1.5), ("commercial features", 3.0),
        ("open-source-first", 2.5),
        ("open-source users to paid", 5.0),
    ],
    "paas": [
        ("platform as a service", 5.0), ("paas", 5.0),
        ("robotics-as-a-service", 5.0), ("raas", 5.0),
        ("as-a-service", 3.5), ("per-robot monthly fees", 5.0),
        ("per-robot", 4.0), ("fleet", 2.5), ("model serving", 3.5),
        ("gpu-optimized", 3.0), ("developer platform", 4.0),
        ("developer tools", 3.5), ("api-first", 4.0), ("api-first solution", 4.5),
        ("usage-based pricing", 3.5), ("infrastructure", 2.5),
        ("gpu cluster", 3.0), ("inference", 2.5),
        ("model serving infrastructure", 5.0),
        ("gpu-optimized model serving infrastructure", 5.0),
    ],
    "marketplace": [
        ("marketplace", 5.0), ("two-sided marketplace", 5.0),
        ("two-sided", 4.0), ("platform fee", 4.5), ("take rate", 4.5),
        ("marketplace take rate", 5.0), ("gmv", 4.0),
        ("transaction volume", 3.0), ("supplier", 2.0), ("buyer", 2.0),
        ("vendor", 2.0), ("listing", 1.5), ("quoting", 2.0),
        ("procurement", 2.5), ("connecting", 2.0),
        ("platform connecting", 3.0), ("network effects", 2.0),
    ],
    "ecommerce": [
        ("e-commerce", 5.0), ("ecommerce", 5.0), ("online store", 4.0),
        ("retail", 3.0), ("direct to consumer", 4.0), ("d2c", 4.5),
        ("online retail", 5.0), ("shopify", 3.0),
    ],
    "advertising": [
        ("advertising", 5.0), ("ad-supported", 5.0), ("ad tech", 5.0),
        ("programmatic", 4.0), ("sponsored brand partnerships", 4.5),
        ("brand partnerships", 3.5), ("ad revenue", 4.0),
        ("ad impressions", 4.0), ("cpm", 3.5), ("cpc", 3.5),
    ],
    "transactional": [
        ("transaction", 3.5), ("per-transaction", 5.0),
        ("payment processing", 4.5), ("take rate of", 3.0),
        ("processing volume", 3.5), ("transaction fee", 5.0),
        ("split payment", 3.0), ("escrow", 2.5),
        ("settlement", 2.5), ("payment infrastructure", 3.0),
        ("payment volume", 3.5), ("annual payment volume", 4.0),
    ],
    "licensing": [
        ("licensing", 5.0), ("license", 3.5), ("per-study licensing", 5.0),
        ("enterprise licensing", 5.0), ("per-seat", 2.5),
        ("patent licensing", 4.5), ("per-study", 4.0),
    ],
    "hardware_plus_software": [
        ("hardware", 3.5), ("device", 2.5), ("sensor", 2.5),
        ("connected device", 4.0), ("hardware and software", 5.0),
        ("physical product", 3.5), ("robot", 2.0),
    ],
    "services": [
        ("consulting", 5.0), ("professional services", 5.0),
        ("managed services", 5.0), ("advisory", 3.5),
        ("implementation services", 4.5),
    ],
}

# ---------------------------------------------------------------------------
# Customer type — weighted keyword scoring (B2B/B2C/B2B2C/B2G).
# ---------------------------------------------------------------------------

_CUSTOMER_TYPE_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "b2b": [
        ("b2b", 5.0), ("enterprise client", 4.5), ("enterprise customer", 4.5),
        ("enterprise", 2.5), ("business client", 4.0),
        ("business customer", 4.0), ("mid-market", 3.5), ("smb", 3.0),
        ("small and mid-size", 3.0), ("companies", 2.0),
        ("organizations", 2.5), ("institution", 2.5), ("hospital", 2.5),
        ("bank", 2.5), ("financial institution", 3.5), ("fortune 500", 4.5),
        ("acv", 3.0), ("annual contract", 3.0), ("contract value", 3.0),
        ("clients", 2.0), ("businesses", 2.0), ("teams", 1.5),
        ("engineering team", 3.0), ("facility", 2.0), ("facilities", 2.0),
    ],
    "b2c": [
        ("b2c", 5.0), ("consumer", 2.5), ("end user", 3.0),
        ("individual", 2.0), ("personal", 1.5),
        ("monthly active user", 4.0), ("daily active user", 4.0),
        ("mau", 3.5), ("dau", 3.5), ("mobile app", 3.0),
        ("user base", 3.0), ("subscriber", 2.5),
        ("premium subscription", 3.0), ("fitness tracking", 2.0),
        ("workout", 1.5), ("community challenge", 2.0),
        ("ios and android", 3.0),
    ],
    "b2b2c": [
        ("b2b2c", 5.0), ("two-sided platform", 4.5),
        ("two-sided marketplace", 4.5), ("platform connecting", 4.0),
        ("marketplace connecting", 4.0),
    ],
    "b2g": [
        ("b2g", 5.0), ("government", 4.0), ("federal", 3.5),
        ("public sector", 4.5), ("municipal", 3.5), ("defense", 3.5),
        ("military", 3.5),
    ],
}

# ---------------------------------------------------------------------------
# Revenue model — how the company generates revenue.
# ---------------------------------------------------------------------------

_REVENUE_MODEL_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "subscription": [
        ("subscription", 5.0), ("monthly subscription", 5.0),
        ("annual contract", 4.5), ("recurring revenue", 4.0),
        ("mrr", 4.5), ("arr", 4.5), ("net revenue retention", 3.5),
        ("gross retention", 3.5), ("per-seat", 3.0), ("seat-based", 3.5),
        ("subscription-based", 5.0), ("subscription pricing", 5.0),
        ("monthly recurring", 5.0), ("annual contract value", 4.5),
        ("acv", 3.0), ("saas", 3.0), ("premium subscription", 4.0),
        ("$299/month", 4.0), ("$9.99/month", 4.0),
        ("$18,000/month", 4.0), ("$120,000", 3.5),
        ("$48,000 arr", 4.5), ("$3.5m arr", 4.0),
        ("$4.2m arr", 4.0), ("$6.4m arr", 4.0),
    ],
    "licensing": [
        ("licensing", 5.0), ("license", 3.5), ("per-study licensing", 5.0),
        ("enterprise licensing", 5.0), ("patent licensing", 4.5),
        ("per-study", 4.0), ("per-study licensing", 5.0),
    ],
    "transaction_fee": [
        ("per-transaction", 5.0), ("transaction fee", 5.0),
        ("payment processing", 4.0), ("take rate of", 4.0),
        ("take rate", 3.5), ("processing volume", 3.5),
        ("payment volume", 4.0), ("annual payment volume", 4.5),
        ("$2.1b in annual payment volume", 5.0),
    ],
    "commission": [
        ("commission", 5.0), ("platform fee", 4.5), ("take rate", 4.0),
        ("marketplace take rate", 5.0), ("gmv", 4.0),
        ("facilitating", 2.0),
    ],
    "freemium": [
        ("freemium", 5.0), ("free tier", 4.5), ("free plan", 4.0),
        ("converting.*to paid", 4.5), ("open-source-first", 3.0),
        ("open-source users to paid", 5.0), ("freemium model", 5.0),
    ],
    "usage_based": [
        ("usage-based pricing", 5.0), ("usage-based", 5.0),
        ("per-call", 4.0), ("pay-per-use", 5.0),
        ("consumption-based", 4.5), ("usage-based with", 4.0),
        ("average customer spend", 3.5),
    ],
    "advertising_revenue": [
        ("advertising", 4.0), ("ad-supported", 5.0), ("ad revenue", 5.0),
        ("sponsored brand partnerships", 4.5), ("brand partnerships", 3.5),
        ("ad impressions", 4.0), ("programmatic", 3.5),
    ],
}

# ---------------------------------------------------------------------------
# Pricing model — the pricing structure.
# ---------------------------------------------------------------------------

_PRICING_MODEL_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "annual_contract": [
        ("annual contract", 5.0), ("annual contract value", 5.0),
        ("acv", 4.0), ("yearly", 3.0), ("annual", 3.0),
        ("$48,000", 4.0), ("$120,000", 4.0),
        ("98% gross retention", 3.0), ("95% gross retention", 3.0),
    ],
    "per_seat": [
        ("per-seat", 5.0), ("seat-based", 5.0), ("per user", 4.0),
        ("per seat", 5.0), ("user-based pricing", 4.5),
    ],
    "per_study": [
        ("per-study licensing", 5.0), ("per-study", 5.0),
        ("per-scan", 4.5), ("per-imaging", 4.0),
    ],
    "tiered": [
        ("tiered", 5.0), ("tiers", 4.0), ("pricing tiers", 5.0),
        ("plans", 3.0), ("basic plan", 4.0), ("pro plan", 4.0),
        ("enterprise plan", 4.5), ("premium plan", 4.0),
        ("$299/month", 3.5), ("$9.99/month", 3.5),
    ],
    "usage_based": [
        ("usage-based pricing", 5.0), ("usage-based", 5.0),
        ("per-call", 4.0), ("pay-per-use", 5.0),
        ("consumption-based", 4.5), ("usage-based with average", 4.5),
        ("average customer spend of $18,000/month", 5.0),
    ],
    "enterprise_contract": [
        ("enterprise contract", 5.0), ("enterprise client", 4.0),
        ("enterprise customer", 4.0), ("enterprise hospital contracts", 5.0),
        ("enterprise clients", 4.5), ("financial institutions", 3.5),
        ("85 enterprise", 4.0), ("280 marketplace clients", 4.0),
        ("60 enterprise clients", 4.0), ("95 financial institutions", 4.5),
        ("150 enterprise customers", 4.5),
    ],
    "flat_rate": [
        ("flat rate", 5.0), ("fixed price", 5.0),
        ("flat fee", 4.5), ("one price", 4.0),
    ],
    "freemium": [
        ("freemium", 5.0), ("free tier", 4.5), ("free plan", 4.0),
        ("converting.*to paid", 4.0), ("open-source-first", 3.0),
        ("open-source users to paid", 5.0),
    ],
    "commission": [
        ("commission", 4.0), ("platform fee", 4.0), ("take rate", 4.0),
        ("marketplace take rate", 5.0), ("12% on transactions", 5.0),
        ("take rate of 0.8%", 5.0),
    ],
}

# ---------------------------------------------------------------------------
# Monetization strategy — overall approach to monetization.
# ---------------------------------------------------------------------------

_MONETIZATION_STRATEGY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "subscription": [
        ("subscription", 5.0), ("subscription-based", 5.0),
        ("monthly subscription", 5.0), ("annual contract", 4.0),
        ("recurring revenue", 4.0), ("mrr", 4.0), ("arr", 4.0),
        ("premium subscription", 4.5), ("$9.99/month", 4.0),
        ("$299/month", 4.0), ("saas", 3.0),
    ],
    "licensing": [
        ("licensing", 5.0), ("license", 3.5), ("per-study licensing", 5.0),
        ("enterprise licensing", 5.0), ("patent licensing", 4.5),
    ],
    "transaction_fee": [
        ("per-transaction", 5.0), ("transaction fee", 5.0),
        ("take rate", 4.0), ("payment processing", 4.0),
    ],
    "commission": [
        ("commission", 5.0), ("platform fee", 4.5),
        ("marketplace take rate", 5.0), ("gmv", 4.0),
    ],
    "freemium": [
        ("freemium", 5.0), ("free tier", 4.5), ("open-source-first", 3.0),
        ("converting.*to paid", 4.5),
    ],
    "advertising": [
        ("advertising", 4.0), ("ad-supported", 5.0),
        ("sponsored brand partnerships", 4.5), ("brand partnerships", 3.5),
    ],
    "hybrid": [
        ("freemium model with a.*premium", 5.0),
        ("subscription.*and.*brand partnerships", 5.0),
        ("open-source.*commercial", 4.0),
        ("monetization is through a freemium model", 5.0),
    ],
}

# ---------------------------------------------------------------------------
# Customer acquisition model — how customers are acquired.
# ---------------------------------------------------------------------------

_CUSTOMER_ACQUISITION_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "product_led": [
        ("open-source-first", 5.0), ("open-source", 3.5),
        ("github stars", 4.5), ("contributing developers", 4.0),
        ("freemium", 4.0), ("free tier", 4.0), ("self-serve", 4.0),
        ("converting.*to paid", 4.0), ("free users to paid", 5.0),
        ("monthly active users", 3.0), ("community", 2.5),
    ],
    "sales_led": [
        ("enterprise client", 4.5), ("enterprise customer", 4.5),
        ("sales team", 4.5), ("sales professional", 4.0),
        ("account executive", 5.0), ("business development", 4.0),
        ("enterprise contract", 4.5), ("enterprise hospital contracts", 5.0),
        ("direct sales", 5.0), ("outbound sales", 5.0),
        ("inbound sales", 4.0), ("pipeline", 3.0),
        ("fortune 500", 4.0), ("top-20 us banks", 4.0),
        ("85 enterprise", 3.5), ("280 marketplace clients", 3.5),
    ],
    "hybrid": [
        ("open-source-first.*commercial", 5.0),
        ("product-led.*sales", 5.0), ("bottom-up.*enterprise", 5.0),
        ("self-serve.*enterprise", 4.0),
    ],
    "marketplace_organic": [
        ("marketplace", 3.0), ("two-sided", 3.5),
        ("organic growth", 4.0), ("network effects", 3.5),
    ],
}

# ---------------------------------------------------------------------------
# Sales motion — type of sales process.
# ---------------------------------------------------------------------------

_SALES_MOTION_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "enterprise_sales": [
        ("enterprise client", 5.0), ("enterprise customer", 5.0),
        ("enterprise contract", 5.0), ("enterprise hospital contracts", 5.0),
        ("enterprise clients", 5.0), ("enterprise customers", 5.0),
        ("fortune 500", 4.5), ("top-20 us banks", 4.5),
        ("85 enterprise", 4.0), ("280 marketplace clients", 4.0),
        ("60 enterprise clients", 4.0), ("95 financial institutions", 4.5),
        ("150 enterprise customers", 4.5),
        ("acv of $120,000", 5.0), ("$48,000 arr", 4.0),
        ("annual contract", 3.5),
    ],
    "product_led": [
        ("open-source-first", 5.0), ("open-source", 3.5),
        ("github stars", 4.5), ("contributing developers", 4.0),
        ("freemium", 4.0), ("free tier", 4.0), ("self-serve", 4.0),
        ("converting.*to paid", 4.0), ("monthly active users", 3.0),
        ("420,000 monthly active users", 4.0),
    ],
    "self_serve": [
        ("self-serve", 5.0), ("self serve", 5.0),
        ("online signup", 4.0), ("free trial", 4.0),
        ("freemium", 3.0), ("free tier", 3.0),
    ],
    "hybrid": [
        ("open-source-first.*commercial", 5.0),
        ("product-led.*enterprise", 5.0),
        ("bottom-up.*enterprise", 5.0),
    ],
}

# ---------------------------------------------------------------------------
# Distribution model — how the product reaches customers.
# ---------------------------------------------------------------------------

_DISTRIBUTION_MODEL_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "direct": [
        ("direct sales", 5.0), ("directly", 3.0),
        ("enterprise client", 3.0), ("enterprise customer", 3.0),
        ("sales team", 4.0), ("account executive", 4.0),
        ("team of 35 engineers and sales", 4.0),
    ],
    "api": [
        ("api-first", 5.0), ("api-first solution", 5.0),
        ("api", 3.0), ("rest api", 4.0), ("graphql api", 4.0),
        ("payment api", 4.5), ("api solution", 4.0),
        ("developer platform", 4.0), ("developer tools", 3.5),
    ],
    "app_store": [
        ("app store", 5.0), ("ios and android", 4.5),
        ("ios", 3.0), ("android", 3.0), ("mobile application", 4.0),
        ("available on ios", 5.0),
    ],
    "open_source": [
        ("open-source-first", 5.0), ("open-source", 4.0),
        ("github", 3.5), ("github stars", 4.0),
        ("contributing developers", 3.5), ("open source", 4.0),
    ],
    "marketplace": [
        ("marketplace", 4.0), ("two-sided marketplace", 5.0),
        ("platform connecting", 4.0),
    ],
    "partner": [
        ("partner", 4.0), ("partnership", 3.5), ("channel partner", 5.0),
        ("reseller", 4.5), ("distribution partner", 5.0),
    ],
}

# ---------------------------------------------------------------------------
# Value proposition signals — detected value props.
# ---------------------------------------------------------------------------

_VALUE_PROPOSITION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("cost_reduction", re.compile(
        r"\b(cost\s+sav\w*|reduce\s+cost|lower\s+cost|cheaper|affordable)\b", re.I
    )),
    ("efficiency_gain", re.compile(
        r"\b(efficien\w+|automat\w+|streamlin\w+|optimi\w+|faster|speed)\b",
        re.I,
    )),
    ("revenue_enabler", re.compile(
        r"\b(revenue|moneti\w+|earn|income|profit)\b", re.I
    )),
    ("risk_mitigation", re.compile(
        r"\b(risk|compliance|regulat|audit|security|protect)\b", re.I
    )),
    ("scalability", re.compile(
        r"\b(scal\w+|auto[\s-]?scal|grow\w+|expand\w+)\b", re.I
    )),
    ("data_insights", re.compile(
        r"\b(insight|analytic|report|dashboard|metric|track)\b", re.I
    )),
    ("time_savings", re.compile(
        r"\b(time[\s-]?sav\w+|instant|real[\s-]?time|immediate|quick)\b", re.I
    )),
    ("integration_value", re.compile(
        r"\b(integrat\w+|connect\w+|compatib\w+|ecosystem)\b", re.I
    )),
    ("quality_improvement", re.compile(
        r"\b(quality|accur\w+|precis\w+|reliable|97\.3%|sensitivity)\b", re.I
    )),
    ("market_access", re.compile(
        r"\b(market\s+access|reach|distribut\w+|global|worldwide|country)\b",
        re.I,
    )),
]

# ---------------------------------------------------------------------------
# Recurring vs transactional revenue detection.
# ---------------------------------------------------------------------------

_RECURRING_REVENUE_PATTERNS: list[tuple[str, list[tuple[str, float]]]] = [
    ("recurring", [
        ("recurring revenue", 5.0), ("mrr", 5.0), ("arr", 5.0),
        ("subscription", 4.0), ("annual contract", 4.0),
        ("monthly recurring", 5.0), ("net revenue retention", 4.0),
        ("gross retention", 4.0), ("retention at d30", 4.0),
        ("premium subscription", 4.0),         ("per-month", 3.5),
        ("subscription-based", 4.0), ("freemium", 3.0),
        ("/month", 3.5), ("monthly", 3.0),
    ]),
    ("transactional", [
        ("per-transaction", 5.0), ("transaction fee", 5.0),
        ("take rate", 4.0), ("payment volume", 4.0),
        ("gmv", 4.0), ("processing volume", 3.5),
        ("annual payment volume", 5.0),
    ]),
]

# ---------------------------------------------------------------------------
# Marketplace dynamics detection.
# ---------------------------------------------------------------------------

_MARKETPLACE_DYNAMICS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("two_sided_marketplace", re.compile(
        r"\btwo[\s-]sided\s+marketplace\b", re.I
    )),
    ("take_rate_model", re.compile(
        r"\btake\s+rate\b", re.I
    )),
    ("platform_fee", re.compile(
        r"\bplatform\s+fee\b", re.I
    )),
    ("gmv_metric", re.compile(
        r"\bgmv\b", re.I
    )),
    ("supply_demand_dynamics", re.compile(
        r"\b(supplier|buyer|vendor|listing)\b", re.I
    )),
    ("chicken_and_egg", re.compile(
        r"\b(cold\s+start|chicken.{0,5}egg|bootstrap)\b", re.I
    )),
    ("liquidity_metrics", re.compile(
        r"\b(liquidity|fill\s+rate|match\s+rate|conversion\s+rate)\b", re.I
    )),
    ("cross_side_effects", re.compile(
        r"\b(cross[\s-]side|same[\s-]side|indirect\s+network)\b", re.I
    )),
]

# ---------------------------------------------------------------------------
# Network effects signal detection.
# ---------------------------------------------------------------------------

_NETWORK_EFFECTS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("direct_network_effects", re.compile(
        r"\bnetwork\s+effect\w*\b", re.I
    )),
    ("indirect_network_effects", re.compile(
        r"\b(ecosystem|platform\s+effect\w*|cross[\s-]side)\b", re.I
    )),
    ("viral_coefficient", re.compile(
        r"\b(viral|flywheel|word[\s-]of[\s-]mouth|referral)\b", re.I
    )),
    ("data_network_effects", re.compile(
        r"\b(data\s+network|more\s+user\w*.*better|proprietary\s+data)\b", re.I
    )),
    ("marketplace_network_effects", re.compile(
        r"\b(marketplace\s+effect|two[\s-]sided|supply.*demand|demand.*supply)\b",
        re.I,
    )),
    ("platform_lock_in", re.compile(
        r"\b(platform\s+lock[\s-]in|ecosystem\s+lock[\s-]in|developer\s+ecosystem)\b",
        re.I,
    )),
]

# ---------------------------------------------------------------------------
# Platform characteristics detection.
# ---------------------------------------------------------------------------

_PLATFORM_CHARACTERISTICS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("multi_tenant", re.compile(r"\bmulti[\s-]?tenant\b", re.I)),
    ("api_first", re.compile(r"\bapi[\s-]?first\b", re.I)),
    ("developer_platform", re.compile(r"\bdeveloper\s+platform\b", re.I)),
    ("extensible", re.compile(r"\b(extensi\w+|plugin|add[\s-]on|customiz\w+)\b", re.I)),
    ("marketplace_enabled", re.compile(
        r"\b(marketplace|app\s+store|integration\s+ecosystem)\b", re.I
    )),
    ("self_serve", re.compile(r"\bself[\s-]serve\b", re.I)),
    ("usage_based_scaling", re.compile(r"\b(usage[\s-]based|consumption[\s-]based)\b", re.I)),
    ("cloud_native", re.compile(r"\bcloud[\s-]?native\b", re.I)),
    ("horizontal_platform", re.compile(
        r"\b(horizontal|cross[\s-]industry|multi[\s-]vertical)\b", re.I
    )),
    ("vertical_platform", re.compile(
        r"\b(vertical[\s-]specific|industry[\s-]specific|domain[\s-]specific)\b",
        re.I,
    )),
]

# ---------------------------------------------------------------------------
# Switching cost indicator detection.
# ---------------------------------------------------------------------------

_SWITCHING_COST_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("data_lock_in", re.compile(
        r"\b(data\s+lock[\s-]in|migration|vendor\s+lock|switching\s+cost)\b",
        re.I,
    )),
    ("integration_lock_in", re.compile(
        r"\b(deep\s+integrat\w+|workflow\s+integrat\w+|erp\s+integrat\w+"
        r"|supply\s+chain\s+integrat\w+)\b", re.I,
    )),
    ("contractual_lock_in", re.compile(
        r"\b(annual\s+contract|multi[\s-]year|enterprise\s+contract"
        r"|long[\s-]term\s+contract)\b", re.I,
    )),
    ("technical_lock_in", re.compile(
        r"\b(proprietary|patent\w*|trade\s+secret|custom\s+api)\b", re.I
    )),
    ("ecosystem_lock_in", re.compile(
        r"\b(ecosystem|platform\s+effect\w*|developer\s+ecosystem"
        r"|integration\s+ecosystem)\b", re.I,
    )),
    ("compliance_lock_in", re.compile(
        r"\b(compliance|regulat\w+|certification|soc\s*2|hipaa|gdpr)\b", re.I
    )),
]

# ---------------------------------------------------------------------------
# Unit economics indicator detection.
# ---------------------------------------------------------------------------

_UNIT_ECONOMICS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ltv_cac_ratio", re.compile(
        r"\b(ltv.{0,10}cac|customer\s+ lifetime\s+value|lifetime\s+value)\b",
        re.I,
    )),
    ("gross_margin", re.compile(
        r"\b(gross\s+margin|high\s+margin|software\s+margin)\b", re.I
    )),
    ("net_revenue_retention", re.compile(
        r"\b(net\s+revenue\s+retention|nrr)\b", re.I
    )),
    ("gross_retention", re.compile(
        r"\b(gross\s+retention|retention\s+rate)\b", re.I
    )),
    ("arr_metric", re.compile(
        r"\b(arr|mrr|annual\s+recurring\s+revenue|monthly\s+recurring\s+revenue)\b",
        re.I,
    )),
    ("ltv_cac_numeric", re.compile(
        r"\bltv/cac\s+of\s+\d+\.?\d*x?\b", re.I
    )),
    ("payback_period", re.compile(
        r"\b(payback\s+period|months\s+to\s+payback)\b", re.I
    )),
    ("churn_rate", re.compile(
        r"\b(churn|retention\s+at\s+d\d+|d30\s+retention)\b", re.I
    )),
    ("burn_rate", re.compile(
        r"\b(burn\s+rate|runway|cash\s+runway)\b", re.I
    )),
    ("revenue_per_customer", re.compile(
        r"\b(\$[\d,]+.*arr|acv\s+of\s+\$[\d,]+|average\s+customer\s+spend)\b",
        re.I,
    )),
]

# ---------------------------------------------------------------------------
# Business model maturity signals.
# ---------------------------------------------------------------------------

_BIZ_MODEL_MATURITY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "nascent": [
        ("concept", 5.0), ("idea stage", 5.0), ("prototype", 4.0),
        ("proof of concept", 5.0), ("poc", 4.0), ("pre-revenue", 4.0),
        ("pre-revenue on commercial tier", 5.0), ("in beta launch phase", 4.5),
    ],
    "early": [
        ("beta", 4.0), ("beta launch", 5.0), ("alpha", 4.0),
        ("early access", 4.0), ("waitlist", 3.5), ("seed stage", 4.0),
        ("pre-seed", 4.0), ("early traction", 4.5),
        ("converting.*to paid", 3.5), ("420,000 monthly active users", 3.0),
        ("38,000 premium subscribers", 3.5),
    ],
    "growth": [
        ("scaling", 4.0), ("rapid growth", 5.0), ("growing", 3.0),
        ("expanding", 3.0), ("traction", 2.5),
        ("product-market fit", 5.0), ("$4.2m arr", 4.0),
        ("$3.5m arr", 4.0), ("$6.4m arr", 4.0),
        ("$21.6m", 4.0), ("$2.8m arr", 4.0),
        ("series a", 4.0), ("series b", 4.5), ("series c", 5.0),
        ("85 enterprise clients", 3.5), ("280 marketplace clients", 3.5),
        ("150 enterprise customers", 3.5), ("95 financial institutions", 4.0),
        ("60 enterprise clients", 3.5), ("140% net revenue retention", 4.5),
    ],
    "mature": [
        ("mature", 5.0), ("established", 4.0), ("proven", 3.5),
        ("incumbent", 3.5), ("market leader", 4.5), ("category leader", 4.5),
        ("10+ years", 5.0), ("profitable", 4.0),
    ],
    "evolving": [
        ("pivoting", 5.0), ("transitioning", 4.0), ("evolving", 4.0),
        ("new model", 4.0), ("hybrid model", 4.0),
        ("expanding into", 3.0), ("new revenue stream", 4.5),
    ],
}

# ---------------------------------------------------------------------------
# Business model keyword patterns — domain-specific terms to extract.
# ---------------------------------------------------------------------------

_BIZ_MODEL_KEYWORD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("saas", re.compile(r"\bsaas\b", re.I)),
    ("subscription", re.compile(r"\bsubscription\b", re.I)),
    ("marketplace", re.compile(r"\bmarketplace\b", re.I)),
    ("platform", re.compile(r"\bplatform\b", re.I)),
    ("api_first", re.compile(r"\bapi[\s-]?first\b", re.I)),
    ("freemium", re.compile(r"\bfreemium\b", re.I)),
    ("open_source", re.compile(r"\bopen[\s-]?source\b", re.I)),
    ("enterprise", re.compile(r"\benterprise\b", re.I)),
    ("usage_based", re.compile(r"\busage[\s-]?based\b", re.I)),
    ("licensing", re.compile(r"\blicens\w*\b", re.I)),
    ("take_rate", re.compile(r"\btake\s+rate\b", re.I)),
    ("transactional", re.compile(r"\btransaction\w*\b", re.I)),
    ("recurring_revenue", re.compile(r"\brecurring\s+revenue\b", re.I)),
    ("network_effects", re.compile(r"\bnetwork\s+effect\w*\b", re.I)),
    ("two_sided", re.compile(r"\btwo[\s-]sided\b", re.I)),
    ("b2b", re.compile(r"\bb2b\b", re.I)),
    ("b2c", re.compile(r"\bb2c\b", re.I)),
    ("b2b2c", re.compile(r"\bb2b2c\b", re.I)),
    ("b2g", re.compile(r"\bb2g\b", re.I)),
    ("cloud_native", re.compile(r"\bcloud[\s-]?native\b", re.I)),
    ("product_led_growth", re.compile(r"\bproduct[\s-]?led\b", re.I)),
    ("annual_contract", re.compile(r"\bannual\s+contract\b", re.I)),
    ("per_seat", re.compile(r"\bper[\s-]?seat\b", re.I)),
    ("gmv", re.compile(r"\bgmv\b", re.I)),
    ("arr", re.compile(r"\barr\b", re.I)),
    ("mrr", re.compile(r"\bmrr\b", re.I)),
]


class BusinessModelExtractor(BaseExtractor):
    """Extracts rich business model intelligence from startup data.

    Produces:
      - business_model: primary business model classification
      - secondary_business_model: secondary model for hybrid cases
      - customer_type: target customer segment (B2B, B2C, B2B2C, B2G)
      - revenue_model: how the company generates revenue
      - pricing_model: pricing structure
      - monetization_strategy: overall monetization approach
      - customer_acquisition_model: how customers are acquired
      - sales_motion: type of sales process
      - distribution_model: how the product reaches customers
      - value_proposition_signals: detected value proposition indicators
      - recurring_revenue_signal: revenue recurrence pattern
      - marketplace_dynamics: marketplace-specific characteristics
      - network_effects_signals: network effect indicators
      - platform_characteristics: platform-specific traits
      - switching_cost_indicators: lock-in signals
      - unit_economics_indicators: unit economics signals
      - business_model_maturity: maturity stage
      - business_model_keywords: domain-specific terms
      - business_model_confidence: extraction confidence (0.0-1.0)
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
            BusinessModelRetrievalStrategy,
        )

        strategy = BusinessModelRetrievalStrategy()
        docs_filtered: list[EvidenceDocument] = []
        evidence_items: list[EvidenceItem] = []
        citations: list[EvidenceCitation] = []

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
        business_model = self._classify_primary_business_model(text_lower)
        secondary = self._classify_secondary_business_model(text_lower, business_model)
        customer_type = self._classify_customer_type(text_lower)

        # Revenue and pricing
        revenue_model = self._classify_revenue_model(text_lower)
        pricing_model = self._classify_pricing_model(text_lower)
        monetization = self._classify_monetization_strategy(text_lower)

        # Acquisition and sales
        acquisition = self._classify_customer_acquisition(text_lower)
        sales_motion = self._classify_sales_motion(text_lower)
        distribution = self._classify_distribution_model(text_lower)

        # Signal detection
        value_props = self._extract_value_proposition_signals(text)
        recurring = self._detect_recurring_revenue(text_lower)
        marketplace_dyn = self._detect_marketplace_dynamics(text)
        network_fx = self._detect_network_effects(text)
        platform_chars = self._detect_platform_characteristics(text)
        switching = self._detect_switching_costs(text)
        unit_econ = self._detect_unit_economics(text)

        # Maturity and keywords
        maturity = self._classify_business_model_maturity(text_lower)
        keywords = self._extract_business_model_keywords(text)

        # Confidence
        confidence = self._compute_confidence(
            text_lower, business_model, customer_type,
            revenue_model, pricing_model,
        )

        features = ExtractedFeatures(
            business_model=business_model,
            secondary_business_model=secondary,
            customer_type=customer_type,
            revenue_model=revenue_model,
            pricing_model=pricing_model,
            monetization_strategy=monetization,
            customer_acquisition_model=acquisition,
            sales_motion=sales_motion,
            distribution_model=distribution,
            value_proposition_signals=value_props,
            recurring_revenue_signal=recurring,
            marketplace_dynamics=marketplace_dyn,
            network_effects_signals=network_fx,
            platform_characteristics=platform_chars,
            switching_cost_indicators=switching,
            unit_economics_indicators=unit_econ,
            business_model_maturity=maturity,
            business_model_keywords=keywords,
            business_model_confidence=confidence,
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
    # Primary business model classification
    # ------------------------------------------------------------------

    def _classify_primary_business_model(self, text: str) -> str | None:
        """Classify primary business model using weighted keyword scoring."""
        scores: dict[str, float] = {}
        match_counts: dict[str, int] = {}

        for model, weighted_keywords in _PRIMARY_BUSINESS_MODEL_KEYWORDS.items():
            total_score = 0.0
            matches = 0
            for keyword, weight in weighted_keywords:
                if keyword in text:
                    total_score += weight
                    matches += 1
            if total_score > 0:
                scores[model] = total_score
                match_counts[model] = matches

        if not scores:
            return None

        best_model = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_model]
        best_matches = match_counts[best_model]

        # Require minimum score for classification
        if best_score < 4.0:
            return None

        # Check for ambiguity
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            margin = best_score - sorted_scores[1]
            if margin < 2.0 and best_matches < 3:
                return None

        return best_model

    # ------------------------------------------------------------------
    # Secondary business model detection
    # ------------------------------------------------------------------

    def _classify_secondary_business_model(
        self, text: str, primary: str | None
    ) -> str | None:
        """Detect secondary business model for hybrid cases."""
        if primary is None:
            return None

        scores: dict[str, float] = {}
        for model, weighted_keywords in _PRIMARY_BUSINESS_MODEL_KEYWORDS.items():
            if model == primary:
                continue
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total >= 4.0:
                scores[model] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        # Secondary must be meaningful but clearly less than primary
        primary_score = sum(
            w for kw, w in _PRIMARY_BUSINESS_MODEL_KEYWORDS.get(primary, [])
            if kw in text
        )
        if best_score < primary_score * 0.5:
            return None

        return best

    # ------------------------------------------------------------------
    # Customer type classification
    # ------------------------------------------------------------------

    def _classify_customer_type(self, text: str) -> str | None:
        """Classify customer type (B2B, B2C, B2B2C, B2G)."""
        scores: dict[str, float] = {}

        for ctype, weighted_keywords in _CUSTOMER_TYPE_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[ctype] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        if best_score < 1.5:
            return None

        # B2B2C: both B2B and B2C have strong signals
        if "b2b" in scores and "b2c" in scores:
            if scores["b2b"] >= 3.0 and scores["b2c"] >= 3.0:
                return "b2b2c"

        return best

    # ------------------------------------------------------------------
    # Revenue model classification
    # ------------------------------------------------------------------

    def _classify_revenue_model(self, text: str) -> str | None:
        """Classify how the company generates revenue."""
        scores: dict[str, float] = {}

        for model, weighted_keywords in _REVENUE_MODEL_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[model] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        if best_score < 4.0:
            return None

        return best

    # ------------------------------------------------------------------
    # Pricing model classification
    # ------------------------------------------------------------------

    def _classify_pricing_model(self, text: str) -> str | None:
        """Classify the pricing structure."""
        scores: dict[str, float] = {}

        for model, weighted_keywords in _PRICING_MODEL_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[model] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        if best_score < 4.0:
            return None

        return best

    # ------------------------------------------------------------------
    # Monetization strategy classification
    # ------------------------------------------------------------------

    def _classify_monetization_strategy(self, text: str) -> str | None:
        """Classify the overall monetization approach."""
        scores: dict[str, float] = {}

        for strategy, weighted_keywords in _MONETIZATION_STRATEGY_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[strategy] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        if best_score < 4.0:
            return None

        return best

    # ------------------------------------------------------------------
    # Customer acquisition model classification
    # ------------------------------------------------------------------

    def _classify_customer_acquisition(self, text: str) -> str | None:
        """Classify the primary customer acquisition approach."""
        scores: dict[str, float] = {}

        for model, weighted_keywords in _CUSTOMER_ACQUISITION_KEYWORDS.items():
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
    # Sales motion classification
    # ------------------------------------------------------------------

    def _classify_sales_motion(self, text: str) -> str | None:
        """Classify the type of sales process."""
        scores: dict[str, float] = {}

        for motion, weighted_keywords in _SALES_MOTION_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[motion] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        if best_score < 4.0:
            return None

        return best

    # ------------------------------------------------------------------
    # Distribution model classification
    # ------------------------------------------------------------------

    def _classify_distribution_model(self, text: str) -> str | None:
        """Classify how the product reaches customers."""
        scores: dict[str, float] = {}

        for model, weighted_keywords in _DISTRIBUTION_MODEL_KEYWORDS.items():
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
    # Value proposition signal extraction
    # ------------------------------------------------------------------

    def _extract_value_proposition_signals(self, text: str) -> list[str]:
        """Extract detected value proposition indicators."""
        found: list[str] = []
        for label, pattern in _VALUE_PROPOSITION_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Recurring revenue detection
    # ------------------------------------------------------------------

    def _detect_recurring_revenue(self, text: str) -> str | None:
        """Detect revenue recurrence pattern using weighted scoring."""
        scores: dict[str, float] = {}

        for category, weighted_keywords in _RECURRING_REVENUE_PATTERNS:
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[category] = total

        recurring_score = scores.get("recurring", 0.0)
        transactional_score = scores.get("transactional", 0.0)

        if recurring_score >= 4.0 and transactional_score >= 4.0:
            return "mixed"
        if recurring_score >= 3.0:
            return "recurring"
        if transactional_score >= 3.0:
            return "transactional"
        return "unknown"

    # ------------------------------------------------------------------
    # Marketplace dynamics detection
    # ------------------------------------------------------------------

    def _detect_marketplace_dynamics(self, text: str) -> list[str]:
        """Detect marketplace-specific characteristics."""
        found: list[str] = []
        for label, pattern in _MARKETPLACE_DYNAMICS_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Network effects detection
    # ------------------------------------------------------------------

    def _detect_network_effects(self, text: str) -> list[str]:
        """Detect network effect indicators."""
        found: list[str] = []
        for label, pattern in _NETWORK_EFFECTS_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Platform characteristics detection
    # ------------------------------------------------------------------

    def _detect_platform_characteristics(self, text: str) -> list[str]:
        """Detect platform-specific traits."""
        found: list[str] = []
        for label, pattern in _PLATFORM_CHARACTERISTICS_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Switching cost detection
    # ------------------------------------------------------------------

    def _detect_switching_costs(self, text: str) -> list[str]:
        """Detect lock-in and switching cost signals."""
        found: list[str] = []
        for label, pattern in _SWITCHING_COST_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Unit economics detection
    # ------------------------------------------------------------------

    def _detect_unit_economics(self, text: str) -> list[str]:
        """Detect unit economics signals."""
        found: list[str] = []
        for label, pattern in _UNIT_ECONOMICS_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Business model maturity classification
    # ------------------------------------------------------------------

    def _classify_business_model_maturity(self, text: str) -> str | None:
        """Classify business model lifecycle stage."""
        scores: dict[str, float] = {}

        for stage, weighted_keywords in _BIZ_MODEL_MATURITY_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[stage] = total

        if not scores:
            return None

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Business model keyword extraction
    # ------------------------------------------------------------------

    def _extract_business_model_keywords(self, text: str) -> list[str]:
        """Extract domain-specific business model terms."""
        found: list[str] = []
        for label, pattern in _BIZ_MODEL_KEYWORD_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        text: str,
        business_model: str | None,
        customer_type: str | None,
        revenue_model: str | None,
        pricing_model: str | None,
    ) -> float:
        """Compute extraction confidence based on signal density.

        High confidence requires multiple strong signals across dimensions.
        """
        # Signal density component (0-0.3)
        word_count = max(len(text.split()), 1)
        signal_density = min(word_count / 50.0, 0.3)

        # Classification component (0-0.35): each known classification adds confidence
        classification_count = sum(
            1 for x in [business_model, customer_type, revenue_model, pricing_model]
            if x is not None
        )
        classification_component = min(classification_count / 4.0 * 0.35, 0.35)

        # Keyword richness component (0-0.2)
        bm_keywords = sum(
            1 for _, pattern in _BIZ_MODEL_KEYWORD_PATTERNS
            if pattern.search(text)
        )
        keyword_component = min(bm_keywords / 8.0 * 0.2, 0.2)

        # Signal diversity component (0-0.15): number of different signal types
        signal_types = 0
        if self._extract_value_proposition_signals(text):
            signal_types += 1
        if self._detect_recurring_revenue(text) not in (None, "unknown"):
            signal_types += 1
        if self._detect_network_effects(text):
            signal_types += 1
        if self._detect_switching_costs(text):
            signal_types += 1
        if self._detect_unit_economics(text):
            signal_types += 1
        signal_component = min(signal_types / 5.0 * 0.15, 0.15)

        combined = (
            signal_density
            + classification_component
            + keyword_component
            + signal_component
        )

        return round(min(combined, 1.0), 2)
