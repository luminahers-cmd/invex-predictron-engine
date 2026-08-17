"""Market extractor — rich deterministic market intelligence extraction.

Responsibilities:
  - Primary industry classification via weighted multi-signal scoring
  - Sub-industry / secondary industry detection
  - Geographic market identification from description signals
  - Customer type and segment classification
  - Market maturity assessment
  - Market keyword extraction
  - Market signal detection
  - Market characteristic identification
  - Enterprise vs consumer orientation
  - Industry classification confidence scoring

Design principles:
  - Deterministic rule-based logic only (no LLMs, no ML)
  - Weighted keyword scoring with context-aware disambiguation
  - Every extracted field traceable to explicit input signals
  - Prefer "Unknown" over incorrect classification
  - Modular rules that are easy to extend
"""

from __future__ import annotations

import re

from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.extraction.quantitative import parse_market_size
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

# ---------------------------------------------------------------------------
# Weighted industry keywords — primary signal for industry classification.
# Each keyword carries a weight (higher = stronger signal). Phrases are
# checked as substrings in the lowercased description text.
# ---------------------------------------------------------------------------

_WEIGHTED_INDUSTRY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "fintech": [
        ("fintech", 5.0),
        ("financial technology", 5.0),
        ("payment processing", 4.0),
        ("embedded payment", 4.0),
        ("split payment", 3.5),
        ("escrow", 3.0),
        ("kyc", 3.0),
        ("compliance", 2.0),
        ("banking", 3.0),
        ("lending", 3.5),
        ("insurtech", 5.0),
        ("wealth management", 3.0),
        ("digital banking", 4.0),
        ("payment volume", 4.0),
        ("take rate", 3.0),
        ("multi-currency", 3.0),
        ("settlement", 3.0),
        ("financial services", 3.0),
    ],
    "healthtech": [
        ("healthtech", 5.0),
        ("health tech", 5.0),
        ("medical imaging", 5.0),
        ("diagnostic", 3.5),
        ("fda", 4.0),
        ("fda-cleared", 5.0),
        ("clinical trial", 5.0),
        ("hospital", 3.5),
        ("radiologist", 5.0),
        ("telehealth", 5.0),
        ("digital health", 5.0),
        ("patient", 2.5),
        ("healthcare", 4.0),
        ("medical", 3.0),
        ("pharmaceutical", 3.0),
        ("biomedical", 3.5),
        ("health", 2.0),
        ("diagnosis", 3.5),
        ("therapeutic", 3.0),
    ],
    "edtech": [
        ("edtech", 5.0),
        ("education technology", 5.0),
        ("online course", 4.0),
        ("learning management", 4.0),
        ("e-learning", 5.0),
        ("student", 2.5),
        ("curriculum", 3.5),
        ("training platform", 3.5),
        ("education", 2.5),
        ("learning", 2.0),
        ("teaching", 2.5),
        ("classroom", 2.5),
    ],
    "enterprise_saas": [
        ("enterprise saas", 5.0),
        ("b2b saas", 5.0),
        ("saas platform", 3.5),
        ("subscription-based", 3.0),
        ("crm", 3.5),
        ("erp", 3.5),
        ("enterprise software", 4.0),
        ("compliance management", 3.0),
        ("compliance platform", 3.0),
        ("audit", 2.5),
        ("workflow automation", 3.0),
        ("enterprise client", 3.5),
        ("enterprise customer", 3.5),
        ("enterprise", 2.0),
        ("saas", 2.5),
        ("b2b software", 4.0),
        ("annual contract", 3.0),
        ("recurring revenue", 2.5),
        ("gross retention", 3.0),
        ("net revenue retention", 3.0),
        ("soc 2", 3.5),
        ("pci dss", 3.5),
        ("gdpr", 2.5),
        ("policy management", 3.0),
        ("regulatory tracking", 3.0),
    ],
    "consumer_tech": [
        ("consumer app", 5.0),
        ("consumer platform", 5.0),
        ("mobile app", 4.0),
        ("consumer", 2.0),
        ("end user", 2.5),
        ("social network", 4.0),
        ("social platform", 4.0),
        ("consumer mobile", 4.0),
        ("monthly active user", 4.0),
        ("daily active user", 4.0),
        ("mau", 3.5),
        ("dau", 3.5),
        ("freemium", 3.0),
        ("premium subscription", 2.5),
        ("brand partnership", 2.5),
        ("fitness", 2.0),
        ("social", 1.5),
    ],
    "ecommerce": [
        ("e-commerce", 5.0),
        ("ecommerce", 5.0),
        ("online retail", 5.0),
        ("direct to consumer", 4.0),
        ("d2c", 4.5),
        ("online store", 4.0),
        ("shopify", 3.5),
        ("retail", 2.5),
        ("cart", 2.0),
        ("checkout", 2.0),
    ],
    "ai_ml": [
        ("artificial intelligence", 5.0),
        ("machine learning", 5.0),
        ("deep learning", 5.0),
        ("large language model", 5.0),
        ("llm", 4.5),
        ("neural network", 4.5),
        ("model serving", 4.5),
        ("inference", 3.5),
        ("gpu", 3.5),
        ("transformer", 3.5),
        ("nlp", 3.5),
        ("computer vision", 4.5),
        ("pytorch", 4.0),
        ("tensorflow", 4.0),
        (" ai ", 3.0),
        ("ml ", 2.5),
        ("ai-powered", 4.0),
        ("ai-driven", 4.0),
        ("model deployment", 4.0),
        ("training data", 3.0),
        ("auto-scaling", 2.5),
        ("multi-cloud", 2.5),
        ("gpu cluster", 4.0),
    ],
    "cybersecurity": [
        ("cybersecurity", 5.0),
        ("infosec", 5.0),
        ("threat detection", 5.0),
        ("security platform", 4.5),
        ("vulnerability", 3.5),
        ("penetration testing", 4.5),
        ("siem", 4.0),
        ("zero trust", 4.5),
        ("security", 2.0),
        ("encryption", 3.0),
    ],
    "climate_tech": [
        ("climate tech", 5.0),
        ("climate technology", 5.0),
        ("carbon accounting", 5.0),
        ("carbon emissions", 5.0),
        ("emissions tracking", 5.0),
        ("scope 1", 4.5),
        ("scope 2", 4.5),
        ("scope 3", 4.5),
        ("ghg protocol", 5.0),
        ("clean energy", 4.5),
        ("sustainability", 3.5),
        ("carbon", 3.0),
        ("emissions", 3.5),
        ("climate", 3.0),
        ("renewable", 3.5),
        ("green", 2.0),
        ("environmental", 2.5),
        ("sec climate", 4.5),
        ("disclosure", 2.0),
    ],
    "biotech": [
        ("biotech", 5.0),
        ("biotechnology", 5.0),
        ("drug discovery", 5.0),
        ("genomics", 5.0),
        ("protein", 3.0),
        ("therapeutic", 3.5),
        ("clinical", 2.5),
        ("pharmaceutical", 3.0),
        ("molecule", 3.5),
        ("assay", 4.0),
    ],
    "hardware": [
        ("robotics", 5.0),
        ("robot", 4.5),
        ("autonomous mobile robot", 5.0),
        ("lidar", 4.5),
        ("slam", 4.0),
        ("warehouse automation", 5.0),
        ("hardware", 3.5),
        ("iot", 3.5),
        ("embedded", 3.0),
        ("sensor", 3.0),
        ("semiconductor", 4.0),
        ("chip", 3.0),
        ("device", 2.0),
        ("physical", 2.0),
        ("fleet", 2.5),
        ("ruggedized", 3.5),
        ("electronics", 3.0),
    ],
    "marketplace": [
        ("two-sided marketplace", 5.0),
        ("marketplace", 3.5),
        ("two-sided", 4.0),
        ("network effects", 4.0),
        ("platform fee", 4.0),
        ("take rate", 3.0),
        ("supply and demand", 3.0),
        ("buyer", 2.0),
        ("supplier", 2.5),
        ("vendor", 2.0),
    ],
    "logistics": [
        ("logistics", 5.0),
        ("supply chain", 4.5),
        ("fulfillment", 4.0),
        ("freight", 4.0),
        ("last mile", 4.5),
        ("delivery", 2.5),
        ("warehousing", 3.5),
        ("distribution", 2.5),
    ],
    "gaming": [
        ("gaming", 4.0),
        ("esports", 5.0),
        ("virtual reality", 4.0),
        ("vr ", 3.0),
        ("ar ", 3.0),
        ("metaverse", 4.0),
        ("game engine", 5.0),
        ("game development", 4.5),
    ],
    "media_entertainment": [
        ("streaming", 3.5),
        ("content creation", 4.0),
        ("creator economy", 5.0),
        ("media platform", 4.5),
        ("entertainment", 3.0),
        ("video", 2.0),
        ("podcast", 3.0),
        ("content", 1.5),
    ],
}

# ---------------------------------------------------------------------------
# Sub-industry classification — applied after primary industry is determined.
# Maps (primary_industry, sub_keywords) to sub_industry labels.
# ---------------------------------------------------------------------------

_SUB_INDUSTRY_RULES: list[tuple[str, list[tuple[str, float]], str]] = [
    # Fintech sub-industries
    ("fintech", [("payments", 4.0), ("payment processing", 5.0)], "payments"),
    ("fintech", [("lending", 4.0), ("credit", 3.5), ("loan", 3.5)], "lending"),
    ("fintech", [("insurtech", 5.0), ("insurance", 3.5)], "insurance"),
    ("fintech", [("wealth", 3.0), ("investment", 3.0)], "wealth_management"),
    ("fintech", [("banking", 4.0), ("neobank", 5.0)], "banking"),
    ("fintech", [("embedded payment", 5.0), ("payment api", 5.0),
     ("payment infrastructure", 4.5)], "embedded_payments"),
    # Healthtech sub-industries
    ("healthtech", [("diagnostic imaging", 5.0), ("medical imaging", 5.0)],
     "diagnostic_imaging"),
    ("healthtech", [("telehealth", 5.0), ("telemedicine", 5.0)],
     "telehealth"),
    ("healthtech", [("ehr", 4.5), ("electronic health record", 5.0)],
     "health_records"),
    ("healthtech", [("drug discovery", 5.0), ("pharma", 3.5)],
     "drug_discovery"),
    # Enterprise SaaS sub-industries
    ("enterprise_saas", [("crm", 5.0), ("customer relationship", 4.0)],
     "crm"),
    ("enterprise_saas", [("erp", 5.0), ("enterprise resource", 4.0)],
     "erp"),
    ("enterprise_saas", [("compliance", 4.0), ("regulatory", 3.5)],
     "compliance"),
    ("enterprise_saas", [("devops", 4.5), ("ci/cd", 5.0), ("ci cd", 5.0)],
     "devops"),
    ("enterprise_saas", [("analytics", 4.0), ("business intelligence", 4.5)],
     "analytics"),
    ("enterprise_saas", [("hris", 5.0), ("human resource", 4.0)],
     "hr_tech"),
    # AI/ML sub-industries
    ("ai_ml", [("model serving", 5.0), ("inference", 4.0)],
     "ai_infrastructure"),
    ("ai_ml", [("nlp", 5.0), ("natural language", 5.0)], "nlp"),
    ("ai_ml", [("computer vision", 5.0), ("image recognition", 5.0)],
     "computer_vision"),
    ("ai_ml", [("mlops", 5.0), ("model training", 4.5)], "mlops"),
    # Hardware sub-industries
    ("hardware", [("robotics", 5.0), ("autonomous robot", 5.0),
     ("warehouse robot", 5.0), ("mobile robot", 4.5)], "robotics"),
    ("hardware", [("iot", 5.0), ("internet of things", 5.0)], "iot"),
    ("hardware", [("semiconductor", 5.0), ("chip design", 5.0)],
     "semiconductor"),
    # Consumer tech sub-industries
    ("consumer_tech", [("fitness", 4.0), ("health tracking", 3.5)],
     "fitness"),
    ("consumer_tech", [("social network", 5.0), ("social media", 4.5)],
     "social"),
    ("consumer_tech", [("productivity", 4.0), ("task management", 3.5)],
     "productivity"),
    # Climate tech sub-industries
    ("climate_tech", [("carbon accounting", 5.0), ("emissions tracking", 5.0)],
     "carbon_accounting"),
    ("climate_tech", [("renewable energy", 5.0), ("solar", 4.0)],
     "renewable_energy"),
    ("climate_tech", [("carbon capture", 5.0), ("carbon offset", 4.5)],
     "carbon_capture"),
]

# ---------------------------------------------------------------------------
# Geography detection — keywords mapped to geographic markets.
# ---------------------------------------------------------------------------

_GEOGRAPHY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "north_america": [
        ("north america", 5.0),
        ("united states", 5.0),
        (" u.s.", 4.0),
        (" usa", 4.5),
        ("silicon valley", 4.0),
        ("new york", 3.5),
        ("san francisco", 3.5),
        ("seattle", 3.5),
        ("los angeles", 3.5),
        ("chicago", 3.5),
        ("boston", 3.5),
        ("canada", 3.5),
        ("toronto", 3.5),
    ],
    "europe": [
        ("europe", 5.0),
        ("european", 4.5),
        ("united kingdom", 5.0),
        ("london", 4.0),
        ("berlin", 4.0),
        ("paris", 3.5),
        ("amsterdam", 3.5),
        ("munich", 3.5),
        ("uk ", 3.0),
    ],
    "asia_pacific": [
        ("asia pacific", 5.0),
        ("apac", 5.0),
        ("singapore", 4.0),
        ("india", 3.5),
        ("china", 3.5),
        ("japan", 3.5),
        ("korea", 3.5),
        ("australia", 3.5),
    ],
    "latin_america": [
        ("latin america", 5.0),
        ("latam", 5.0),
        ("brazil", 4.0),
        ("mexico", 3.5),
    ],
    "middle_east_africa": [
        ("middle east", 5.0),
        ("uae", 4.5),
        ("dubai", 4.0),
        ("africa", 4.0),
        ("nigeria", 3.5),
        ("kenya", 3.5),
    ],
    "global": [
        ("global", 4.0),
        ("worldwide", 4.5),
        ("international", 3.5),
        ("35 countries", 4.0),
        ("multiple countries", 3.5),
        ("cross-border", 3.5),
    ],
}

# ---------------------------------------------------------------------------
# Customer type signals — weighted keywords for B2B/B2C/B2B2C/B2G detection.
# ---------------------------------------------------------------------------

_CUSTOMER_TYPE_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "b2b": [
        ("b2b", 5.0),
        ("enterprise client", 4.5),
        ("enterprise customer", 4.5),
        ("enterprise", 2.5),
        ("business client", 4.0),
        ("business customer", 4.0),
        ("mid-market", 3.5),
        ("smb", 3.0),
        ("small and mid-size", 3.0),
        ("companies", 2.0),
        ("organizations", 2.5),
        ("institution", 2.5),
        ("hospital", 2.5),
        ("bank", 2.5),
        ("financial institution", 3.5),
        ("fortune 500", 4.5),
        ("acv", 3.0),
        ("annual contract", 3.0),
        ("contract value", 3.0),
        ("clients", 2.0),
        ("customer", 1.5),
        ("businesses", 2.0),
        ("teams", 1.5),
        ("engineering team", 3.0),
        ("facility", 2.0),
        ("facilities", 2.0),
    ],
    "b2c": [
        ("b2c", 5.0),
        ("consumer", 2.5),
        ("end user", 3.0),
        ("individual", 2.0),
        ("personal", 1.5),
        ("monthly active user", 4.0),
        ("daily active user", 4.0),
        ("mau", 3.5),
        ("dau", 3.5),
        ("mobile app", 3.0),
        ("user base", 3.0),
        ("subscriber", 2.5),
        ("premium subscription", 3.0),
    ],
    "b2b2c": [
        ("b2b2c", 5.0),
        ("two-sided platform", 4.5),
        ("two-sided marketplace", 4.5),
        ("platform connecting", 4.0),
        ("marketplace connecting", 4.0),
    ],
    "b2g": [
        ("b2g", 5.0),
        ("government", 4.0),
        ("federal", 3.5),
        ("public sector", 4.5),
        ("municipal", 3.5),
        ("defense", 3.5),
        ("military", 3.5),
    ],
}

# ---------------------------------------------------------------------------
# Customer segment signals — more specific than customer_type.
# ---------------------------------------------------------------------------

_CUSTOMER_SEGMENT_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "enterprise": [
        ("enterprise", 4.0),
        ("fortune 500", 5.0),
        ("large organization", 4.5),
        ("global corporation", 4.5),
        ("top-20", 3.5),
        ("top 20", 3.5),
        (" Fortune ", 4.0),
    ],
    "mid_market": [
        ("mid-market", 5.0),
        ("mid market", 5.0),
        ("midsize", 4.0),
        ("medium business", 4.0),
    ],
    "smb": [
        ("smb", 5.0),
        ("small business", 5.0),
        ("small and mid-size", 3.0),
        ("small and medium", 3.0),
    ],
    "government": [
        ("government", 4.0),
        ("federal", 3.5),
        ("public sector", 4.5),
        ("municipal", 3.5),
    ],
    "developer": [
        ("developer", 4.0),
        ("engineering team", 4.0),
        ("devops", 3.5),
        ("open-source", 3.0),
        ("github", 3.5),
    ],
}

# ---------------------------------------------------------------------------
# Market maturity signals — keywords mapped to maturity stages.
# ---------------------------------------------------------------------------

_MATURITY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "emerging": [
        ("emerging", 5.0),
        ("nascent", 5.0),
        ("early stage market", 5.0),
        ("new market", 4.0),
        ("first mover", 4.0),
        ("category creator", 4.5),
        ("unprecedented", 3.5),
        ("novel", 3.0),
    ],
    "growth": [
        ("rapid growth", 5.0),
        ("growing market", 5.0),
        ("high growth", 5.0),
        ("fast growing", 4.5),
        ("expanding", 3.5),
        ("increasing demand", 4.0),
        ("market adoption", 3.5),
        ("scaling", 2.5),
        (" traction ", 2.0),
    ],
    "mature": [
        ("mature", 5.0),
        ("established", 4.0),
        ("consolidated", 4.5),
        ("saturated", 5.0),
        ("well-established", 4.0),
        ("proven market", 4.0),
        ("incumbent", 3.5),
    ],
}

# ---------------------------------------------------------------------------
# Market signals — detectable indicators in description text.
# ---------------------------------------------------------------------------

_MARKET_SIGNAL_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("market_size_mentioned", "market_size", re.compile(
        r"\b(?:market|tam|sam|exceeds?\s+\$|billion|trillion)\b", re.I
    )),
    ("growth_rate_mentioned", "growth", re.compile(
        r"(?:\d+%\s*(?:growth|increase|yoy|year.over)|cagr|compound annual)", re.I
    )),
    ("regulatory_complexity", "regulatory", re.compile(
        r"(?:regulat|compliance|sec |fda|hipaa|gdpr|soc\s*2|pci\s*dss|ghg\s*protocol)", re.I
    )),
    ("competitive_dynamics", "competition", re.compile(
        r"(?:compet|incumbent|alternativ|disrupt|winner.take|fragmented)", re.I
    )),
    ("switching_costs", "switching", re.compile(
        r"(?:switching cost|lock.in|vendor lock|integration|migration)", re.I
    )),
    ("network_effects_present", "network", re.compile(
        r"(?:network effect|viral|flywheel|ecosystem|platform effect)", re.I
    )),
    ("recurring_revenue_signal", "revenue", re.compile(
        r"(?:recurring revenue|mrr|arr|subscription|annual contract|retention)", re.I
    )),
    ("scalability_signal", "scalability", re.compile(
        r"(?:scal|auto.?scal|cloud.?native|multi.?tenant|api.?first)", re.I
    )),
]

# ---------------------------------------------------------------------------
# Market characteristics — detectable properties.
# ---------------------------------------------------------------------------

_MARKET_CHARACTERISTIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("high_switching_costs", re.compile(
        r"(?:switching cost|lock.in|vendor lock|deep integrat|migration)", re.I
    )),
    ("network_effects", re.compile(
        r"(?:network effect|viral|flywheel|two.?sided|marketplace effect)", re.I
    )),
    ("recurring_revenue", re.compile(
        r"(?:recurring revenue|subscription|mrr|arr|annual contract)", re.I
    )),
    ("regulatory_moat", re.compile(
        r"(?:regulat|compliance|certification|hipaa|soc\s*2|fda|iso\s*27)", re.I
    )),
    ("technology_moat", re.compile(
        r"(?:proprietary|patent|trade secret|unique algorithm|secret sauce)", re.I
    )),
    ("data_moat", re.compile(
        r"(?:proprietary data|training data|data advantage|data network)", re.I
    )),
    ("platform_dynamics", re.compile(
        r"(?:platform|ecosystem|api|developer|integration)", re.I
    )),
    ("high_gross_margins", re.compile(
        r"(?:gross margin|high margin|software margin|70.*90%)", re.I
    )),
]

# ---------------------------------------------------------------------------
# Enterprise vs consumer orientation signals.
# ---------------------------------------------------------------------------

_ENTERPRISE_SIGNALS: list[tuple[str, float]] = [
    ("enterprise", 4.0),
    ("b2b", 3.5),
    ("annual contract", 3.5),
    ("recurring revenue", 2.5),
    ("gross retention", 3.5),
    ("net revenue retention", 3.5),
    ("acv", 3.0),
    ("contract", 2.0),
    ("compliance", 2.5),
    ("audit", 2.5),
    ("integration", 2.0),
    ("api", 2.0),
    ("developer", 2.5),
    ("saas", 2.0),
    ("subscription", 2.0),
    ("fortune 500", 4.0),
    ("institution", 2.5),
]

_CONSUMER_SIGNALS: list[tuple[str, float]] = [
    ("consumer", 4.0),
    ("b2c", 3.5),
    ("mobile app", 3.0),
    ("monthly active user", 4.0),
    ("daily active user", 4.0),
    ("mau", 3.5),
    ("dau", 3.5),
    ("freemium", 3.0),
    ("premium subscription", 2.5),
    ("user base", 3.0),
    ("social", 2.0),
    ("brand", 2.0),
    ("personal", 1.5),
]

# ---------------------------------------------------------------------------
# Market keywords — domain-specific terms to extract.
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("api_first", re.compile(r"\bapi[\s-]?first\b", re.I)),
    ("cloud_native", re.compile(r"\bcloud[\s-]?native\b", re.I)),
    ("saas", re.compile(r"\bsaas\b", re.I)),
    ("subscription", re.compile(r"\bsubscription\b", re.I)),
    ("platform", re.compile(r"\bplatform\b", re.I)),
    ("marketplace", re.compile(r"\bmarketplace\b", re.I)),
    ("open_source", re.compile(r"\bopen[\s-]?source\b", re.I)),
    ("ai_powered", re.compile(r"\bai[\s-]?(?:powered|driven|enabled)\b", re.I)),
    ("real_time", re.compile(r"\breal[\s-]?time\b", re.I)),
    ("enterprise", re.compile(r"\benterprise\b", re.I)),
    ("automation", re.compile(r"\bautomation\b", re.I)),
    ("analytics", re.compile(r"\banalytics\b", re.I)),
    ("compliance", re.compile(r"\bcompliance\b", re.I)),
    ("integration", re.compile(r"\bintegration\b", re.I)),
    ("scalability", re.compile(r"\bscalab\b", re.I)),
    ("security", re.compile(r"\bsecurity\b", re.I)),
    ("infrastructure", re.compile(r"\binfrastructure\b", re.I)),
    ("developer_tools", re.compile(r"\bdeveloper tool\b", re.I)),
    ("embedded", re.compile(r"\bembedded\b", re.I)),
    ("automation", re.compile(r"\bautomat\b", re.I)),
]


class MarketExtractor(BaseExtractor):
    """Extracts rich market intelligence from startup data.

    Produces:
      - industry: primary industry classification (weighted scoring)
      - sub_industry: secondary industry classification
      - geography: primary geographic market
      - customer_type: B2B/B2C/B2B2C/B2G
      - customer_segment: enterprise/mid_market/smb/government/developer
      - target_market: descriptive target market string
      - market_maturity: emerging/growth/mature/saturated
      - market_keywords: domain-specific terms
      - market_signals: detected market indicators
      - market_characteristics: detected market properties
      - enterprise_orientation: enterprise/consumer/hybrid
      - industry_confidence: classification confidence (0.0-1.0)
    """

    def extract(
        self,
        startup: Startup,
        data: CollectedData,
        evidence: EvidenceBundle | None = None,
    ) -> ExtractedFeatures:
        text = self._combined_text(startup.description, evidence)
        text_lower = text.lower()

        # Core classifications
        industry_result = self._classify_industry(text_lower)
        sub_industry = self._classify_sub_industry(
            industry_result["primary"], text_lower
        )
        geography = self._classify_geography(text_lower)
        customer_type = self._classify_customer_type(text_lower)
        customer_segment = self._classify_customer_segment(text_lower)
        enterprise_orientation = self._classify_orientation(text_lower)

        # Market intelligence
        target_market = self._infer_target_market(
            text_lower, industry_result["primary"], customer_type, customer_segment
        )
        market_maturity = self._classify_maturity(text_lower)
        market_keywords = self._extract_market_keywords(text)
        market_signals = self._detect_market_signals(text)
        market_characteristics = self._detect_market_characteristics(text)

        # --- Structured quantitative extraction (Sprint 14) ---
        market_size_usd = parse_market_size(text)

        return ExtractedFeatures(
            industry=industry_result["primary"],
            sub_industry=sub_industry,
            geography=geography,
            customer_type=customer_type,
            customer_segment=customer_segment,
            target_market=target_market,
            market_maturity=market_maturity,
            market_keywords=market_keywords,
            market_signals=market_signals,
            market_characteristics=market_characteristics,
            enterprise_orientation=enterprise_orientation,
            industry_confidence=industry_result["confidence"],
            market_size_usd=market_size_usd,
        )

    # ------------------------------------------------------------------
    # Industry classification — weighted multi-signal scoring
    # ------------------------------------------------------------------

    def _classify_industry(self, text: str) -> dict[str, object]:
        """Classify primary industry using weighted keyword scoring.

        Returns dict with 'primary' (str|None) and 'confidence' (float).
        """
        scores: dict[str, float] = {}
        match_counts: dict[str, int] = {}

        for industry, weighted_keywords in _WEIGHTED_INDUSTRY_KEYWORDS.items():
            total_score = 0.0
            matches = 0
            for keyword, weight in weighted_keywords:
                if keyword in text:
                    total_score += weight
                    matches += 1
            if total_score > 0:
                scores[industry] = total_score
                match_counts[industry] = matches

        if not scores:
            return {"primary": None, "confidence": 0.0}

        best_industry = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_industry]
        best_matches = match_counts[best_industry]

        # Confidence based on score margin and match count
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            margin = best_score - sorted_scores[1]
        else:
            margin = best_score

        confidence = self._compute_industry_confidence(best_score, margin, best_matches)

        return {"primary": best_industry, "confidence": round(confidence, 2)}

    def _compute_industry_confidence(
        self, score: float, margin: float, match_count: int
    ) -> float:
        """Compute confidence from score, margin, and match count.

        High score + high margin + many matches = high confidence.
        """
        # Score component (0-0.4): higher score = more confident
        score_component = min(score / 25.0, 0.4)

        # Margin component (0-0.35): larger margin = less ambiguity
        margin_component = min(margin / 15.0, 0.35)

        # Match count component (0-0.25): more keyword matches = stronger signal
        match_component = min(match_count / 6.0, 0.25)

        return min(score_component + margin_component + match_component, 1.0)

    # ------------------------------------------------------------------
    # Sub-industry classification
    # ------------------------------------------------------------------

    def _classify_sub_industry(
        self, primary: str | None, text: str
    ) -> str | None:
        """Classify sub-industry based on primary industry and text signals."""
        if primary is None:
            return None

        best_sub: str | None = None
        best_score = 0.0

        for rule_primary, weighted_keywords, sub_label in _SUB_INDUSTRY_RULES:
            if rule_primary != primary:
                continue
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > best_score:
                best_score = total
                best_sub = sub_label

        return best_sub if best_score >= 4.0 else None

    # ------------------------------------------------------------------
    # Geography classification
    # ------------------------------------------------------------------

    def _classify_geography(self, text: str) -> str | None:
        """Classify primary geographic market from description text."""
        scores: dict[str, float] = {}

        for region, weighted_keywords in _GEOGRAPHY_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[region] = total

        if not scores:
            return None

        return max(scores, key=scores.get)  # type: ignore[arg-type]

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

        # Require minimum score to classify (avoid false positives)
        if best_score < 1.5:
            return None

        # Check for B2B2C: if both B2B and B2C have strong signals
        if "b2b" in scores and "b2c" in scores:
            if scores["b2b"] >= 3.0 and scores["b2c"] >= 3.0:
                return "b2b2c"

        return best

    # ------------------------------------------------------------------
    # Customer segment classification
    # ------------------------------------------------------------------

    def _classify_customer_segment(self, text: str) -> str | None:
        """Classify specific customer segment."""
        scores: dict[str, float] = {}

        for segment, weighted_keywords in _CUSTOMER_SEGMENT_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[segment] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] >= 3.0 else None

    # ------------------------------------------------------------------
    # Enterprise vs consumer orientation
    # ------------------------------------------------------------------

    def _classify_orientation(self, text: str) -> str | None:
        """Classify enterprise vs consumer orientation."""
        enterprise_score = sum(w for kw, w in _ENTERPRISE_SIGNALS if kw in text)
        consumer_score = sum(w for kw, w in _CONSUMER_SIGNALS if kw in text)

        if enterprise_score > 5.0 and consumer_score > 5.0:
            return "hybrid"
        if enterprise_score > 5.0:
            return "enterprise"
        if consumer_score > 5.0:
            return "consumer"
        return None

    # ------------------------------------------------------------------
    # Target market inference
    # ------------------------------------------------------------------

    def _infer_target_market(
        self,
        text: str,
        industry: str | None,
        customer_type: str | None,
        segment: str | None,
    ) -> str | None:
        """Infer a descriptive target market string from available signals."""
        parts: list[str] = []

        # Segment prefix
        segment_labels = {
            "enterprise": "enterprise",
            "mid_market": "mid-market",
            "smb": "small and mid-size",
            "government": "government",
            "developer": "developer",
        }
        if segment and segment in segment_labels:
            parts.append(segment_labels[segment])

        # Customer type
        type_labels = {
            "b2b": "businesses",
            "b2c": "consumers",
            "b2b2c": "businesses and consumers",
            "b2g": "government agencies",
        }
        if customer_type and customer_type in type_labels:
            parts.append(type_labels[customer_type])

        # Industry context
        industry_labels = {
            "fintech": "in financial services",
            "healthtech": "in healthcare",
            "edtech": "in education",
            "enterprise_saas": "in enterprise software",
            "consumer_tech": "in consumer technology",
            "ecommerce": "in e-commerce",
            "ai_ml": "in AI/ML",
            "cybersecurity": "in cybersecurity",
            "climate_tech": "in climate technology",
            "biotech": "in biotechnology",
            "hardware": "in hardware/robotics",
            "marketplace": "in marketplace platforms",
            "logistics": "in logistics",
            "gaming": "in gaming",
            "media_entertainment": "in media/entertainment",
        }
        if industry and industry in industry_labels:
            parts.append(industry_labels[industry])

        if not parts:
            return None

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Market maturity classification
    # ------------------------------------------------------------------

    def _classify_maturity(self, text: str) -> str | None:
        """Classify market maturity from description signals."""
        scores: dict[str, float] = {}

        for stage, weighted_keywords in _MATURITY_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[stage] = total

        if not scores:
            return None

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Market keyword extraction
    # ------------------------------------------------------------------

    def _extract_market_keywords(self, text: str) -> list[str]:
        """Extract domain-specific market keywords from description."""
        found: list[str] = []
        for label, pattern in _DOMAIN_KEYWORD_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Market signal detection
    # ------------------------------------------------------------------

    def _detect_market_signals(self, text: str) -> list[str]:
        """Detect market signals present in the description."""
        found: list[str] = []
        for signal_label, _category, pattern in _MARKET_SIGNAL_PATTERNS:
            if pattern.search(text):
                found.append(signal_label)
        return found

    # ------------------------------------------------------------------
    # Market characteristic detection
    # ------------------------------------------------------------------

    def _detect_market_characteristics(self, text: str) -> list[str]:
        """Detect market characteristics from the description."""
        found: list[str] = []
        for label, pattern in _MARKET_CHARACTERISTIC_PATTERNS:
            if pattern.search(text):
                found.append(label)
        return found
