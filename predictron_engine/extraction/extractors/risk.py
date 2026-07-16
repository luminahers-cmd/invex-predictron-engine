"""Risk extractor — comprehensive deterministic risk intelligence extraction.

Responsibilities:
  - Market risk signal detection
  - Founder risk signal detection
  - Execution risk signal detection
  - Product risk signal detection
  - Technology risk signal detection
  - Business model risk signal detection
  - Traction risk signal detection
  - Competitive risk signal detection
  - Regulatory risk signal detection
  - Operational risk signal detection
  - Platform dependency risk detection
  - Customer concentration risk detection
  - Hiring risk signal detection
  - Funding risk signal detection
  - Scaling risk signal detection
  - Security risk signal detection
  - Compliance risk signal detection
  - Risk keyword extraction
  - Composite risk confidence scoring

Design principles:
  - Deterministic rule-based logic only (no LLMs, no ML)
  - Weighted keyword scoring with context-aware disambiguation
  - Every extracted field traceable to explicit input signals
  - Prefer "Unknown" / empty over incorrect classification
  - Modular rules that are easy to extend
"""

from __future__ import annotations

import re
from typing import Final

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

# ---------------------------------------------------------------------------
# Market risk patterns
# ---------------------------------------------------------------------------

_MARKET_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("uncertain_market", re.compile(
        r"\buncertain\s+(?:market|demand|outlook|future)\b", re.I,
    )),
    ("market_timing", re.compile(
        r"\b(?:too\s+early|ahead\s+of\s+its?\s+time|market\s+not\s+ready)\b",
        re.I,
    )),
    ("unproven_market", re.compile(
        r"\b(?:unproven|unvalidated|nascent|undeveloped)\s+"
        r"(?:market|demand|category)\b",
        re.I,
    )),
    ("declining_market", re.compile(
        r"\b(?:declining|shrinking|contracting|disrupting)\s+"
        r"(?:market|demand|industry)\b",
        re.I,
    )),
    ("market_saturated", re.compile(
        r"\b(?:saturated|oversaturated|mature|saturated)\s+market\b", re.I,
    )),
    ("niche_limitation", re.compile(
        r"\b(?:small\s+market|limited\s+addressable\s+market|"
        r"niche\s+only|niche\s+market)\b",
        re.I,
    )),
    ("cyclical_demand", re.compile(
        r"\b(?:cyclical|seasonal|volatile)\s+(?:demand|revenue|market)\b",
        re.I,
    )),
    ("low_awareness", re.compile(
        r"\b(?:low\s+(?:awareness|adoption|penetration)|"
        r"education\s+required|market\s+education)\b",
        re.I,
    )),
    ("fragmented_buyers", re.compile(
        r"\b(?:fragmented|scattered|dispersed)\s+buyer[s]?\b", re.I,
    )),
]

_MARKET_RISK_THRESHOLD: float = 3.0


# ---------------------------------------------------------------------------
# Founder risk patterns
# ---------------------------------------------------------------------------

_FOUNDER_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("solo_founder", re.compile(
        r"\b(?:solo\s+founder|single\s+founder|one\s+founder)\b", re.I,
    )),
    ("no_domain_expertise", re.compile(
        r"\b(?:no\s+(?:domain|industry)\s+experience|"
        r"first[\s-]time\s+founder|no\s+prior\s+(?:startup|founding))\b",
        re.I,
    )),
    ("key_person_dependency", re.compile(
        r"\b(?:key\s+person|risk|single\s+point\s+of\s+failure|"
        r"depends\s+on\s+(?:founder|ceo|cto))\b",
        re.I,
    )),
    ("founder_departure", re.compile(
        r"\b(?:founder\s+(?:departure|left|departed)|"
        r"co[\s-]founder\s+(?:left|departure|split))\b",
        re.I,
    )),
    ("inexperienced_team", re.compile(
        r"\b(?:inexperienced\s+(?:team|founder)|"
        r"first[\s-]time\s+(?:entrepreneur|founder|ceo))\b",
        re.I,
    )),
    ("no_technical_founder", re.compile(
        r"\b(?:no\s+technical\s+(?:co[\s-]?founder|founder)|"
        r"lacking\s+technical\s+leadership)\b",
        re.I,
    )),
    ("small_team", re.compile(
        r"\b(?:team\s+of\s+(?:[12345]|one|two|three|four|five)|"
        r"very\s+small\s+team)\b",
        re.I,
    )),
]

_FOUNDER_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Execution risk patterns
# ---------------------------------------------------------------------------

_EXECUTION_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("early_stage", re.compile(
        r"\b(?:pre[\s-]seed|idea\s+stage|concept\s+stage|"
        r"pre[\s-]product|pre[\s-]revenue|mvp)\b",
        re.I,
    )),
    ("pivot_history", re.compile(
        r"\b(?:pivoted|pivot(?:ing|ed)?|multiple\s+pivots|"
        r"changed\s+(?:direction|strategy|focus))\b",
        re.I,
    )),
    ("product_not_ready", re.compile(
        r"\b(?:not\s+yet\s+(?:launched|ready|built)|"
        r"still\s+(?:building|developing|in\s+development)|"
        r"work[\s-]in[\s-]progress)\b",
        re.I,
    )),
    ("no_revenue", re.compile(
        r"\b(?:pre[\s-]revenue|no\s+revenue|not\s+(?:yet\s+)?monetizing|"
        r"no\s+monetization)\b",
        re.I,
    )),
    ("slow_growth", re.compile(
        r"\b(?:slow\s+(?:growth|adoption|traction)|"
        r"below[\s-]target|behind\s+schedule)\b",
        re.I,
    )),
    ("scaling_challenges", re.compile(
        r"\b(?:scaling\s+(?:challenges?|difficulties?|issues?)|"
        r"growing\s+pains|operational\s+(?:complexity|bottleneck))\b",
        re.I,
    )),
]

_EXECUTION_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Product risk patterns
# ---------------------------------------------------------------------------

_PRODUCT_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("unproven_value", re.compile(
        r"\b(?:unproven\s+(?:value|proposition|demand)|"
        r"no\s+(?:product[\s-]market\s+fit|pmf|validated))\b",
        re.I,
    )),
    ("feature_factory", re.compile(
        r"\bfeature\s+factory\b", re.I,
    )),
    ("complex_product", re.compile(
        r"\b(?:highly\s+complex|complexity|complex\s+(?:product|system|"
        r"implementation|integration))\b",
        re.I,
    )),
    ("long_sales_cycle", re.compile(
        r"\b(?:long\s+sales\s+cycle|extended\s+sales|"
        r"months[\s-]long\s+(?:sales|cycle|process))\b",
        re.I,
    )),
    ("high_churn", re.compile(
        r"\b(?:high\s+churn|elevated\s+churn|churn\s+(?:rate|issue|problem)|"
        r"retention\s+(?:issues?|problems?|concerns?))\b",
        re.I,
    )),
    ("low_engagement", re.compile(
        r"\b(?:low\s+(?:engagement|usage|adoption|retention)|"
        r"poor\s+(?:engagement|adoption|retention))\b",
        re.I,
    )),
    ("beta_stage", re.compile(
        r"\b(?:in\s+beta|beta\s+(?:phase|version|launch)|"
        r"closed\s+beta|open\s+beta)\b",
        re.I,
    )),
]

_PRODUCT_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Technology risk patterns
# ---------------------------------------------------------------------------

_TECHNOLOGY_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("unproven_tech", re.compile(
        r"\b(?:unproven\s+(?:technology|tech|approach)|"
        r"experimental\s+(?:technology|tech|approach))\b",
        re.I,
    )),
    ("technical_debt", re.compile(
        r"\b(?:technical\s+debt|code\s+debt|legacy\s+code)\b", re.I,
    )),
    ("single_points_of_failure", re.compile(
        r"\b(?:single\s+point\s+of\s+failure|spof\b|"
        r"no\s+(?:redundancy|fallback|backup))\b",
        re.I,
    )),
    ("vendor_lock_in", re.compile(
        r"\b(?:vendor\s+lock[\s-]in|locked[\s-]in\s+to|"
        r"proprietary\s+(?:dependency|stack|platform))\b",
        re.I,
    )),
    ("scalability_concerns", re.compile(
        r"\b(?:scalability\s+(?:concerns?|issues?|challenges?)|"
        r"(?:does\s+not|doesn't)\s+scale|scaling\s+(?:limits?|issues?))\b",
        re.I,
    )),
    ("untested_infrastructure", re.compile(
        r"\b(?:untested\s+(?:infrastructure|system|architecture)|"
        r"not\s+(?:battle[\s-]tested|proven)\s+at\s+scale)\b",
        re.I,
    )),
    ("rapid_obsolescence", re.compile(
        r"\b(?:rapid\s+(?:obsolescence|change)|"
        r"fast[\s-]moving\s+(?:tech|technology|space)|"
        r"technology\s+(?:shift|turnover|replacement))\b",
        re.I,
    )),
]

_TECHNOLOGY_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Business model risk patterns
# ---------------------------------------------------------------------------

_BUSINESS_MODEL_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("unproven_monetization", re.compile(
        r"\b(?:unproven\s+(?:monetization|business\s+model|revenue\s+model)|"
        r"(?:no|lacking)\s+(?:clear|defined)\s+monetization)\b",
        re.I,
    )),
    ("low_margins", re.compile(
        r"\b(?:low\s+(?:margins?|margin)|"
        r"thin\s+margins?|negative\s+margins?)\b",
        re.I,
    )),
    ("dependency_on_single_revenue", re.compile(
        r"\b(?:single\s+revenue\s+(?:stream|source)|"
        r"one\s+revenue\s+(?:stream|source)|"
        r"solely\s+(?:reliant|dependent)\s+on)\b",
        re.I,
    )),
    ("advertising_dependency", re.compile(
        r"\b(?:ad[\s-]?supported|advertising[\s-]based|"
        r"revenue\s+from\s+ads|ad[\s-]driven)\b", re.I,
    )),
    ("negative_unit_economics", re.compile(
        r"\b(?:negative\s+unit\s+economics|"
        r"(?:ltv|clv)\s*[<≤]\s*cac|"
        r"unsustainable\s+unit\s+economics)\b",
        re.I,
    )),
    ("high_customer_acquisition_cost", re.compile(
        r"\b(?:high\s+(?:cac|customer\s+acquisition\s+cost)|"
        r"expensive\s+(?:acquisition|customer\s+acquisition))\b",
        re.I,
    )),
    ("low_ltv", re.compile(
        r"\b(?:low\s+(?:ltv|clv|lifetime\s+value)|"
        r"short\s+(?:customer\s+)?lifetime)\b",
        re.I,
    )),
]

_BUSINESS_MODEL_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Traction risk patterns
# ---------------------------------------------------------------------------

_TRACTION_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("no_traction", re.compile(
        r"\b(?:no\s+(?:traction|users|customers|revenue)|"
        r"zero\s+(?:traction|users|revenue)|"
        r"minimal\s+(?:traction|users|adoption))\b",
        re.I,
    )),
    ("declining_metrics", re.compile(
        r"\b(?:declining|decreasing|falling|dropping)\s+"
        r"(?:revenue|growth|users|retention|engagement|metrics)\b",
        re.I,
    )),
    ("low_retention", re.compile(
        r"\b(?:low\s+(?:retention|retention\s+rate|net\s+retention)|"
        r"(?:poor|weak)\s+retention|"
        r"(?:churn|churn\s+rate)\s+(?:exceeds?|above|high))\b",
        re.I,
    )),
    ("stagnant_growth", re.compile(
        r"\b(?:stagnant\s+(?:growth|revenue|user\s+base)|"
        r"flat\s+(?:growth|revenue|metrics)|"
        r"plateaued)\b",
        re.I,
    )),
    ("small_user_base", re.compile(
        r"\b(?:very\s+few\s+(?:users|customers)|"
        r"(?:less|fewer)\s+than\s+\d+\s+(?:users|customers)|"
        r"limited\s+(?:user|customer)\s+base)\b",
        re.I,
    )),
    ("no_clear_milestones", re.compile(
        r"\b(?:no\s+(?:clear|defined)\s+(?:milestones?|kpis?|metrics?)|"
        r"lacking\s+(?:traction|evidence|validation))\b",
        re.I,
    )),
]

_TRACTION_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Competitive risk patterns
# ---------------------------------------------------------------------------

_COMPETITIVE_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("dominant_competitor", re.compile(
        r"\b(?:dominant\s+(?:competitor|player|incumbent|leader)|"
        r"(?:well[\s-])?funded\s+competitor[s]?|"
        r"(?:deep|well)[\s-]funded)\b",
        re.I,
    )),
    ("race_to_bottom", re.compile(
        r"\b(?:race\s+to\s+the\s+bottom|"
        r"price\s+war|brutal\s+competition)\b",
        re.I,
    )),
    ("low_barrier_to_entry", re.compile(
        r"\b(?:low\s+(?:barrier|barriers)\s+to\s+entry|"
        r"easy\s+to\s+(?:replicate|copy|clone|duplicate)|"
        r"easy\s+barrier)\b",
        re.I,
    )),
    ("commoditization", re.compile(
        r"\b(?:commoditi[sz](?:ation|ed|ing)|"
        r"becoming\s+a\s+commodity|"
        r"table\s+stakes)\b",
        re.I,
    )),
    ("well_funded_rivals", re.compile(
        r"\b(?:well[\s-]funded\s+(?:rivals?|competitors?|players?)|"
        r"(?:backed|funded)\s+by\s+(?:tier[\s-]1|major|top)\s+vcs?)\b",
        re.I,
    )),
    ("big_tech_threat", re.compile(
        r"\b(?:big\s+tech|faang|gafam|"
        r"major\s+tech\s+(?:companies?|players?)|"
        r"(?:google|amazon|microsoft|apple|meta)\s+(?:could|might|may))\b",
        re.I,
    )),
]

_COMPETITIVE_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Regulatory risk patterns
# ---------------------------------------------------------------------------

_REGULATORY_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("pending_regulation", re.compile(
        r"\b(?:pending\s+(?:regulation|legislation|rules?|laws?)|"
        r"upcoming\s+(?:regulatory|compliance)|"
        r"regulatory\s+(?:changes?|uncertainty|overhaul))\b",
        re.I,
    )),
    ("heavy_regulation", re.compile(
        r"\b(?:heavily\s+regulated|strict\s+regulation|"
        r"complex\s+regulatory|regulatory\s+(?:burden|complexity|landscape))\b",
        re.I,
    )),
    ("compliance_requirement", re.compile(
        r"\b(?:must\s+comply|compliance\s+(?:requirements?|obligations?)|"
        r"(?:hipaa|gdpr|sox|pci[\s-]dss|bcp|bcr|ccpa|fedramp)\s+"
        r"(?:compliance|requirements?|certification))\b",
        re.I,
    )),
    ("legal_exposure", re.compile(
        r"\b(?:legal\s+(?:exposure|risk|liability|issues?)|"
        r"(?:lawsuit|litigation|regulatory\s+(?:action|investigation)))\b",
        re.I,
    )),
    ("data_privacy", re.compile(
        r"\b(?:data\s+(?:privacy|protection|breach|security)|"
        r"(?:gdpr|ccpa|pipeda)\s+(?:compliance|violation|risk))\b",
        re.I,
    )),
    ("geographic_restrictions", re.compile(
        r"\b(?:geographic\s+(?:restrictions?|limitations?)|"
        r"not\s+(?:available|permitted|allowed)\s+in|"
        r"(?:banned|blocked|restricted)\s+in)\b",
        re.I,
    )),
]

_REGULATORY_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Operational risk patterns
# ---------------------------------------------------------------------------

_OPERATIONAL_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("supply_chain", re.compile(
        r"\b(?:supply\s+chain\s+(?:risk|disruption|issues?)|"
        r"(?:hardware|component)\s+(?:shortage|dependency|risk))\b",
        re.I,
    )),
    ("geographic_concentration", re.compile(
        r"\b(?:geographic\s+(?:concentration|dependency|risk)|"
        r"(?:single|one)\s+(?:location|office|region))\b",
        re.I,
    )),
    ("single_point_failure", re.compile(
        r"\b(?:single\s+point\s+(?:of\s+)?failure|"
        r"no\s+(?:redundancy|backup|contingency))\b",
        re.I,
    )),
    ("operational_complexity", re.compile(
        r"\b(?:operational\s+(?:complexity|complex|challenges?)|"
        r"complex\s+operations?|logistically\s+(?:complex|challenging))\b",
        re.I,
    )),
    ("manual_processes", re.compile(
        r"\b(?:manual\s+(?:processes?|workflows?|operations?)|"
        r"not\s+(?:automated|scaled)|hand[\s-]rolled)\b",
        re.I,
    )),
]

_OPERATIONAL_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Platform dependency risk patterns
# ---------------------------------------------------------------------------

_PLATFORM_DEP_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("single_platform", re.compile(
        r"\b(?:single\s+platform|solely\s+on|only\s+on|"
        r"exclusively\s+(?:on|built))\b",
        re.I,
    )),
    ("platform_fee_risk", re.compile(
        r"\b(?:platform\s+(?:fees?|commission|tax|cut)|"
        r"(?:apple|google|shopify|salesforce|aws)\s+(?:fees?|commission))\b",
        re.I,
    )),
    ("app_store_dependency", re.compile(
        r"\b(?:app\s+store\s+(?:dependency|risk|rejection)|"
        r"(?:apple|google)\s+(?:review|approval|rejection))\b",
        re.I,
    )),
    ("api_dependency", re.compile(
        r"\b(?:api\s+(?:dependency|changes?|rate\s+limits?)|"
        r"api[\s-]key\s+(?:revocation|risk|dependency))\b",
        re.I,
    )),
    ("cloud_lock_in", re.compile(
        r"\b(?:cloud\s+(?:lock[\s-]in|dependency)|"
        r"(?:aws|gcp|azure)\s+(?:lock[\s-]in|dependency|costs?))\b",
        re.I,
    )),
]

_PLATFORM_DEP_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Customer concentration risk patterns
# ---------------------------------------------------------------------------

_CUSTOMER_CONC_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("few_customers", re.compile(
        r"\b(?:few\s+(?:customers?|clients?|accounts?)|"
        r"(?:less|fewer)\s+than\s+\d+\s+(?:customers?|clients?)|"
        r"limited\s+(?:customer|client)\s+base)\b",
        re.I,
    )),
    ("top_customer_dependency", re.compile(
        r"\b(?:top\s+(?:customer|client)\s+(?:concentration|dependency|risk)|"
        r"(?:single|one|major)\s+customer\s+(?:accounts?|represents?)|"
        r"single\s+customer)\b",
        re.I,
    )),
    ("concentration_risk", re.compile(
        r"\b(?:concentration\s+(?:risk|concern|issue)|"
        r"revenue\s+concentration)\b",
        re.I,
    )),
    ("few_large_contracts", re.compile(
        r"\b(?:few\s+large\s+contracts?|"
        r"(?:limited|small)\s+number\s+of\s+(?:contracts?|clients?)|"
        r"a\s+handful\s+of\s+(?:customers?|clients?))\b",
        re.I,
    )),
    ("dependent_on_enterprise", re.compile(
        r"\b(?:dependent\s+on\s+enterprise|"
        r"relies?\s+(?:heavily|solely|primarily)\s+on|"
        r"heavily\s+concentrated)\b",
        re.I,
    )),
]

_CUSTOMER_CONC_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Hiring risk patterns
# ---------------------------------------------------------------------------

_HIRING_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("talent_shortage", re.compile(
        r"\b(?:talent\s+(?:shortage|scarcity|gap)|"
        r"(?:difficulty|struggling)\s+(?:hiring|to\s+hire)|"
        r"hard\s+to\s+(?:hire|find|attract)\b)",
        re.I,
    )),
    ("key_role_vacant", re.compile(
        r"\b(?:key\s+(?:role|position)\s+(?:vacant|open|unfilled)|"
        r"still\s+(?:looking|searching)\s+for\s+(?:cto|ceo|vp))\b",
        re.I,
    )),
    ("high_turnover", re.compile(
        r"\b(?:high\s+(?:turnover|attrition|churn)\b|"
        r"(?:employee|team)\s+turnover)\b",
        re.I,
    )),
    ("competitive_hiring", re.compile(
        r"\b(?:competitive\s+(?:hiring|talent\s+market)|"
        r"(?:expensive|costly)\s+talent|"
        r"talent\s+(?:war|competition))\b",
        re.I,
    )),
    ("remote_only", re.compile(
        r"\b(?:fully\s+remote|distributed\s+team|"
        r"all[\s-]remote|remote[\s-]only)\b",
        re.I,
    )),
]

_HIRING_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Funding risk patterns
# ---------------------------------------------------------------------------

_FUNDING_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("unfunded", re.compile(
        r"\b(?:unfunded|bootstrapped|self[\s-]funded|"
        r"(?:no|without)\s+(?:external\s+)?(?:funding|investment))\b",
        re.I,
    )),
    ("runway_concern", re.compile(
        r"\b(?:short\s+runway|low\s+runway|"
        r"(?:less|fewer)\s+than\s+\d+\s+months?\s+of\s+runway|"
        r"burn\s+rate)\b",
        re.I,
    )),
    ("high_burn_rate", re.compile(
        r"\b(?:high\s+burn\s+rate|burning\s+cash|"
        r"(?:heavy|significant)\s+cash\s+burn)\b",
        re.I,
    )),
    ("down_round_risk", re.compile(
        r"\b(?:down\s+round|risk\s+of\s+down\s+round|"
        r"(?:at\s+)?lower\s+valuation)\b",
        re.I,
    )),
    ("revenue_not_funding", re.compile(
        r"\b(?:revenue[\s-]based|revenue[\s-]funded|"
        r"(?:not|no)\s+(?:yet\s+)?(?:profitable|cash[\s-]flow\s+positive))\b",
        re.I,
    )),
    ("capital_intensive", re.compile(
        r"\b(?:capital[\s-]intensive|high\s+capex|"
        r"significant\s+(?:capital|investment)\s+(?:required|needs?))\b",
        re.I,
    )),
    ("bridge_financing", re.compile(
        r"\b(?:bridge\s+(?:round|financing|note)|"
        r"convertible\s+note|safe\s+agreement)\b",
        re.I,
    )),
]

_FUNDING_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Scaling risk patterns
# ---------------------------------------------------------------------------

_SCALING_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("scaling_costs", re.compile(
        r"\b(?:scaling\s+(?:costs?|expenses?|infrastructure)|"
        r"(?:high|growing)\s+(?:operational|infrastructure)\s+costs?)\b",
        re.I,
    )),
    ("margin_pressure", re.compile(
        r"\b(?:margin\s+(?:compression|pressure|erosion)|"
        r"decreasing\s+margins?|shrinking\s+margins?)\b",
        re.I,
    )),
    ("geographic_scaling", re.compile(
        r"\b(?:geographic\s+(?:scaling|expansion\s+challenges?)|"
        r"(?:local|regional)\s+(?:scaling|expansion))\b",
        re.I,
    )),
    ("operational_scaling", re.compile(
        r"\b(?:operational\s+(?:scaling|challenges?)|"
        r"scaling\s+(?:operations|team|organization))\b",
        re.I,
    )),
    ("infrastructure_limits", re.compile(
        r"\b(?:infrastructure\s+(?:limits?|bottleneck|constraints?)|"
        r"infrastructure\s+(?:at|near)\s+capacity)\b",
        re.I,
    )),
]

_SCALING_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Security risk patterns
# ---------------------------------------------------------------------------

_SECURITY_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("data_breach_history", re.compile(
        r"\b(?:data\s+breach|security\s+(?:breach|incident|event)|"
        r"(?:cyber)?attack(?:ed|s)?|hacked|compromised)\b",
        re.I,
    )),
    ("no_security_mention", re.compile(
        r"\b(?:no\s+security|without\s+security|"
        r"not\s+mentioning?\s+security)\b",
        re.I,
    )),
    ("sensitive_data", re.compile(
        r"\b(?:sensitive\s+(?:data|information|customer\s+data)|"
        r"(?:pii|phi|hipaa|personal\s+(?:data|information))|"
        r"financial\s+data|payment\s+(?:data|information))\b",
        re.I,
    )),
    ("security_certifications_missing", re.compile(
        r"\b(?:no\s+(?:soc\s*2|iso\s+\d{4,5}|penetration|security\s+audit)|"
        r"(?:missing|lacking)\s+(?:security\s+certifications?|compliance))\b",
        re.I,
    )),
    ("encryption_gaps", re.compile(
        r"\b(?:unencrypted|not\s+(?:encrypted|encrypted)|"
        r"(?:data|information)\s+at\s+rest\s+without)\b",
        re.I,
    )),
    ("third_party_risk", re.compile(
        r"\b(?:third[\s-]party\s+(?:risk|dependency|vulnerability)|"
        r"(?:vendor|supplier)\s+(?:security|risk|exposure))\b",
        re.I,
    )),
]

_SECURITY_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Compliance risk patterns
# ---------------------------------------------------------------------------

_COMPLIANCE_RISK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("multi_framework", re.compile(
        r"\b(?:multiple\s+(?:compliance|regulatory)\s+frameworks?|"
        r"(?:soc\s*2|iso\s+\d{4,5}|hipaa|gdpr|pci[\s-]dss|fedramp|"
        r"sox|bcp|bcr)\s+(?:and|&|,\s*)\s*(?:soc\s*2|iso\s+\d{4,5}|"
        r"hipaa|gdpr|pci[\s-]dss|fedramp|sox))\b",
        re.I,
    )),
    ("audit_burden", re.compile(
        r"\b(?:audit\s+(?:burden|requirements?|frequency)|"
        r"(?:annual|regular)\s+audits?|"
        r"(?:compliance|regulatory)\s+audits?)\b",
        re.I,
    )),
    ("certification_required", re.compile(
        r"\b(?:certification\s+required|must\s+(?:obtain|maintain|achieve)|"
        r"(?:requires?|requiring)\s+(?:certification|accreditation|"
        r"authorization|approval))\b",
        re.I,
    )),
    ("cross_border_compliance", re.compile(
        r"\b(?:cross[\s-]border\s+(?:compliance|regulation|data)|"
        r"(?:multi[\s-]country|international)\s+compliance|"
        r"data\s+(?:transfer|residency)\s+(?:requirements?|restrictions?))\b",
        re.I,
    )),
    ("evolving_regulations", re.compile(
        r"\b(?:evolving\s+(?:regulations?|compliance|rules?)|"
        r"(?:new|changing|shifting)\s+(?:regulations?|requirements?)|"
        r"regulatory\s+(?:evolution|change|flux))\b",
        re.I,
    )),
]

_COMPLIANCE_RISK_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Risk keywords — domain-specific terms to extract
# ---------------------------------------------------------------------------

_RISK_KEYWORD_MAP: dict[str, list[str]] = {
    "financial_risk": [
        "burn rate", "runway", "cash flow", "unit economics",
        "negative margin", "down round", "bridge round",
        "capital intensive", "bootstrapped",
    ],
    "market_risk": [
        "market risk", "timing risk", "market education",
        "uncertain demand", "nascent market", "market saturation",
    ],
    "execution_risk": [
        "execution risk", "product-market fit", "scaling challenges",
        "key person dependency", "operational complexity",
    ],
    "technology_risk": [
        "technical debt", "single point of failure", "vendor lock-in",
        "scalability", "infrastructure risk", "unproven technology",
    ],
    "regulatory_risk": [
        "regulatory risk", "compliance requirements", "data privacy",
        "GDPR", "HIPAA", "legal exposure", "pending regulation",
    ],
    "competitive_risk": [
        "competitive risk", "low barriers to entry", "commoditization",
        "race to the bottom", "big tech threat", "price war",
    ],
    "team_risk": [
        "hiring challenges", "talent shortage", "key person",
        "solo founder", "small team", "turnover",
    ],
    "security_risk": [
        "data breach", "security incident", "sensitive data",
        "cyber attack", "third-party risk", "vulnerability",
    ],
}


# ---------------------------------------------------------------------------
# RiskExtractor
# ---------------------------------------------------------------------------


class RiskExtractor(BaseExtractor):
    """Extracts risk intelligence from startup data.

    Performs deterministic, explainable signal extraction across 19
    dimensions of venture risk assessment. Every signal is traceable
    to a specific text pattern or data field.

    Populates: market_risk, founder_risk, execution_risk, product_risk,
    technology_risk, business_model_risk, traction_risk, competitive_risk,
    regulatory_risk, operational_risk, platform_dependency_risk,
    customer_concentration_risk, hiring_risk, funding_risk, scaling_risk,
    security_risk, compliance_risk, risk_keywords, risk_confidence.
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        desc = startup.description
        text_lower = desc.lower()

        # Risk signal detection for each dimension
        market = self._detect_market_risk(desc)
        founder = self._detect_founder_risk(desc, data)
        execution = self._detect_execution_risk(desc)
        product = self._detect_product_risk(desc)
        technology = self._detect_technology_risk(desc)
        business = self._detect_business_model_risk(desc)
        traction = self._detect_traction_risk(desc, data)
        competitive = self._detect_competitive_risk(desc)
        regulatory = self._detect_regulatory_risk(desc)
        operational = self._detect_operational_risk(desc)
        platform_dep = self._detect_platform_dependency_risk(desc)
        customer_conc = self._detect_customer_concentration_risk(desc)
        hiring = self._detect_hiring_risk(desc)
        funding = self._detect_funding_risk(desc)
        scaling = self._detect_scaling_risk(desc)
        security = self._detect_security_risk(desc)
        compliance = self._detect_compliance_risk(desc)
        risk_kw = self._extract_risk_keywords(text_lower)

        # Composite confidence
        confidence = self._compute_risk_confidence(
            market=market,
            founder=founder,
            execution=execution,
            product=product,
            technology=technology,
            business=business,
            traction=traction,
            competitive=competitive,
            regulatory=regulatory,
            operational=operational,
            platform_dep=platform_dep,
            customer_conc=customer_conc,
            hiring=hiring,
            funding=funding,
            scaling=scaling,
            security=security,
            compliance=compliance,
            risk_kw=risk_kw,
        )

        return ExtractedFeatures(
            market_risk=market,
            founder_risk=founder,
            execution_risk=execution,
            product_risk=product,
            technology_risk=technology,
            business_model_risk=business,
            traction_risk=traction,
            competitive_risk=competitive,
            regulatory_risk=regulatory,
            operational_risk=operational,
            platform_dependency_risk=platform_dep,
            customer_concentration_risk=customer_conc,
            hiring_risk=hiring,
            funding_risk=funding,
            scaling_risk=scaling,
            security_risk=security,
            compliance_risk=compliance,
            risk_keywords=risk_kw,
            risk_confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Market risk detection
    # ------------------------------------------------------------------

    def _detect_market_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _MARKET_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Founder risk detection (uses data signals too)
    # ------------------------------------------------------------------

    def _detect_founder_risk(
        self, text: str, data: CollectedData
    ) -> list[str]:
        signals: list[str] = []
        for label, pattern in _FOUNDER_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        # Data-driven: single founder risk
        if data.founder_count == 1 and not signals:
            for label, pattern in _FOUNDER_RISK_PATTERNS:
                if label == "solo_founder":
                    signals.append("solo_founder: derived from founder_count=1")
                    break
        elif data.founder_count == 1:
            has_solo = any("solo_founder" in s for s in signals)
            if not has_solo:
                signals.append("solo_founder: derived from founder_count=1")
        # No founders at all
        if data.founder_count == 0:
            signals.append("no_founders: no founder profiles provided")
        return signals

    # ------------------------------------------------------------------
    # Execution risk detection
    # ------------------------------------------------------------------

    def _detect_execution_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _EXECUTION_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Product risk detection
    # ------------------------------------------------------------------

    def _detect_product_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _PRODUCT_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Technology risk detection
    # ------------------------------------------------------------------

    def _detect_technology_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _TECHNOLOGY_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Business model risk detection
    # ------------------------------------------------------------------

    def _detect_business_model_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _BUSINESS_MODEL_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Traction risk detection (uses data signals too)
    # ------------------------------------------------------------------

    def _detect_traction_risk(
        self, text: str, data: CollectedData
    ) -> list[str]:
        signals: list[str] = []
        for label, pattern in _TRACTION_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        # Data-driven: no revenue
        if "pre-revenue" in text.lower() or "pre revenue" in text.lower():
            if not any("no_revenue" in s for s in signals):
                signals.append("no_revenue: pre-revenue stage detected")
        return signals

    # ------------------------------------------------------------------
    # Competitive risk detection
    # ------------------------------------------------------------------

    def _detect_competitive_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _COMPETITIVE_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Regulatory risk detection
    # ------------------------------------------------------------------

    def _detect_regulatory_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _REGULATORY_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Operational risk detection
    # ------------------------------------------------------------------

    def _detect_operational_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _OPERATIONAL_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Platform dependency risk detection
    # ------------------------------------------------------------------

    def _detect_platform_dependency_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _PLATFORM_DEP_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Customer concentration risk detection
    # ------------------------------------------------------------------

    def _detect_customer_concentration_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _CUSTOMER_CONC_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Hiring risk detection
    # ------------------------------------------------------------------

    def _detect_hiring_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _HIRING_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Funding risk detection
    # ------------------------------------------------------------------

    def _detect_funding_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _FUNDING_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Scaling risk detection
    # ------------------------------------------------------------------

    def _detect_scaling_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _SCALING_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Security risk detection
    # ------------------------------------------------------------------

    def _detect_security_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _SECURITY_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Compliance risk detection
    # ------------------------------------------------------------------

    def _detect_compliance_risk(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _COMPLIANCE_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Risk keyword extraction
    # ------------------------------------------------------------------

    def _extract_risk_keywords(self, text: str) -> list[str]:
        keywords: list[str] = []
        for _category, kws in _RISK_KEYWORD_MAP.items():
            for kw in kws:
                if kw.lower() in text and kw not in keywords:
                    keywords.append(kw)
        return keywords

    # ------------------------------------------------------------------
    # Composite risk confidence scoring
    # ------------------------------------------------------------------

    def _compute_risk_confidence(
        self,
        *,
        market: list[str],
        founder: list[str],
        execution: list[str],
        product: list[str],
        technology: list[str],
        business: list[str],
        traction: list[str],
        competitive: list[str],
        regulatory: list[str],
        operational: list[str],
        platform_dep: list[str],
        customer_conc: list[str],
        hiring: list[str],
        funding: list[str],
        scaling: list[str],
        security: list[str],
        compliance: list[str],
        risk_kw: list[str],
    ) -> float:
        score = 0.0

        # Each dimension contributes proportionally to how many signals found
        # Weighted by importance: market, execution, founder, funding first
        score += min(len(market) * 0.08, 0.16)
        score += min(len(founder) * 0.07, 0.14)
        score += min(len(execution) * 0.08, 0.16)
        score += min(len(product) * 0.06, 0.12)
        score += min(len(technology) * 0.06, 0.12)
        score += min(len(business) * 0.06, 0.12)
        score += min(len(traction) * 0.06, 0.12)
        score += min(len(competitive) * 0.04, 0.08)
        score += min(len(regulatory) * 0.04, 0.08)
        score += min(len(operational) * 0.03, 0.06)
        score += min(len(platform_dep) * 0.03, 0.06)
        score += min(len(customer_conc) * 0.03, 0.06)
        score += min(len(hiring) * 0.03, 0.06)
        score += min(len(funding) * 0.05, 0.10)
        score += min(len(scaling) * 0.03, 0.06)
        score += min(len(security) * 0.03, 0.06)
        score += min(len(compliance) * 0.03, 0.06)

        # Risk keywords boost
        score += min(len(risk_kw) * 0.01, 0.05)

        return round(min(score, 1.0), 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_snippet(text: str, start: int, end: int, ctx: int) -> str:
    snippet_start = max(0, start - ctx)
    snippet_end = min(len(text), end + ctx)
    snippet = text[snippet_start:snippet_end].strip()
    if snippet_start > 0:
        snippet = "..." + snippet
    if snippet_end < len(text):
        snippet = snippet + "..."
    return snippet
