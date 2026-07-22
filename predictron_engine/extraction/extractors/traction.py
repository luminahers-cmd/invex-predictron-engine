"""Traction extractor — extracts traction-related factual attributes.

Deterministic, explainable extraction of market validation signals,
funding intelligence, revenue metrics, customer metrics, and growth
indicators from startup description text.

Responsibilities:
  - Funding stage classification with weighted evidence
  - Funding amount and investor signal detection
  - Grants and accelerator signal detection
  - Revenue signal detection (ARR/MRR/amounts)
  - GMV signal detection
  - Customer count, enterprise, pilot, and paying customer detection
  - Active user signal detection
  - Partnership signal detection
  - Retention signal detection
  - Engagement signal detection
  - Product adoption signal detection
  - Growth signal detection
  - Hiring growth signal detection
  - Expansion signal detection
  - Launch signal detection
  - Milestone signal detection
  - Awards and recognition detection
  - Traction keyword extraction
  - Composite traction confidence scoring
"""

from __future__ import annotations

import re
from typing import Final

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.extraction.quantitative import (
    parse_active_user_count,
    parse_arr,
    parse_burn_rate,
    parse_cac,
    parse_churn,
    parse_customer_count,
    parse_funding_amount,
    parse_gmv,
    parse_growth_rate,
    parse_ltv,
    parse_mrr,
    parse_nrr,
    parse_runway,
    parse_team_size,
    parse_valuation,
)
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

# ---------------------------------------------------------------------------
# Funding stage — weighted keyword scoring
# ---------------------------------------------------------------------------

_STAGE_WEIGHTED_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "pre_seed": [
        ("pre-seed", 7.0), ("pre-seed stage", 7.0),
        ("preseed", 5.0), ("bootstrapped", 3.0),
        ("idea stage", 3.0),
    ],
    "seed": [
        ("seed round", 5.0), ("seed stage", 5.0), ("seed-stage", 5.0),
        ("seed funding", 5.0), ("raised a seed", 6.0), ("seed from", 5.0),
        ("angel round", 4.0), ("angel investors", 3.0),
        ("friends and family", 3.0),
    ],
    "series_a": [
        ("series a", 5.0), ("series-a", 5.0), ("series a round", 5.0),
        ("series a funded", 6.0),
    ],
    "series_b": [
        ("series b", 5.0), ("series-b", 5.0), ("series b round", 5.0),
        ("series b raised", 6.0),
    ],
    "series_c": [
        ("series c", 5.0), ("series-c", 5.0), ("growth round", 3.0),
    ],
    "series_d": [
        ("series d", 5.0), ("series-d", 5.0),
    ],
    "series_e_plus": [
        ("series e", 5.0), ("series e+", 5.0), ("late stage", 3.0),
    ],
    "growth": [
        ("growth equity", 5.0), ("private equity", 4.0),
        ("growth stage", 3.0),
    ],
    "ipo_ready": [
        ("ipo", 5.0), ("going public", 5.0), ("public offering", 5.0),
        ("direct listing", 5.0),
    ],
}

# Minimum weighted score to classify a funding stage
_STAGE_CLASSIFICATION_THRESHOLD: float = 3.0

# Ambiguity margin — if top two stages are within this, classify as ambiguous
_STAGE_AMBIGUITY_MARGIN: float = 1.5


# ---------------------------------------------------------------------------
# Funding amount patterns
# ---------------------------------------------------------------------------

_FUNDING_AMOUNT_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("raised_amount", re.compile(
        r"\braised\s+\$[\d,.]+\s*(?:million|m|billion|b|thousand|k)?\b",
        re.I,
    )),
    ("funding_round_amount", re.compile(
        r"\$[\d,.]+\s*(?:million|m|billion|b)\s+"
        r"(?:series\s+[a-e]|seed|pre-seed|round)\b",
        re.I,
    )),
    ("total_funding", re.compile(
        r"\b(?:total|combined)\s+(?:funding|raised|capital)\s+"
        r"(?:of\s+)?\$[\d,.]+\s*(?:m|b|k)?\b",
        re.I,
    )),
    ("funded_through", re.compile(
        r"\$[\d,.]+\s*(?:million|m|billion|b)\s+in\s+total\s+funding\b",
        re.I,
    )),
    ("valuation_mention", re.compile(
        r"\b(?:valued|valuation)\s+(?:at\s+)?"
        r"\$[\d,.]+\s*(?:million|m|billion|b)\b",
        re.I,
    )),
]


# ---------------------------------------------------------------------------
# Investor signal patterns
# ---------------------------------------------------------------------------

_INVESTOR_SIGNAL_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("tier_1_vcs", re.compile(
        r"\b(?:tier[\s-]?1|top[\s-]?tier)\s+(?:vc|investor|fund)s?\b",
        re.I,
    )),
    ("vc_funded", re.compile(
        r"\b(?:vc|venture\s+capital)[\s-]+(?:backed|funded|investors?)\b",
        re.I,
    )),
    ("angel_investors", re.compile(r"\bangel\s+investor[s]?\b", re.I)),
    ("accelerator_backed", re.compile(
        r"\b(?:y[\s-]?combinator|techstars|500\s+startups"
        r"|plug\s+and\s+play|accelerator)\b",
        re.I,
    )),
    ("strategic_investor", re.compile(
        r"\bstrategic\s+(?:investor|partner|invest(?:ment)?)\b",
        re.I,
    )),
    ("corporate_venture", re.compile(
        r"\b(?:corporate\s+venture|cvc"
        r"|corporate\s+invest(?:ment)?)\b",
        re.I,
    )),
    ("healthcare_vc", re.compile(
        r"\bhealthcare[\s-]focused\s+vcs?\b", re.I,
    )),
    ("climate_vc", re.compile(
        r"\bclimate[\s-]focused\s+vcs?\b", re.I,
    )),
    ("fintech_investor", re.compile(
        r"\bfintech\s+investor[s]?\b", re.I,
    )),
]


# ---------------------------------------------------------------------------
# Grants and accelerator patterns
# ---------------------------------------------------------------------------

_GRANT_ACCELERATOR_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("sbir_grant", re.compile(r"\bsbir\b", re.I)),
    ("sttr_grant", re.compile(r"\bsttr\b", re.I)),
    ("nih_grant", re.compile(r"\bnih\s+(?:grant|funded|award)\b", re.I)),
    ("nsf_grant", re.compile(r"\bnsf\s+(?:grant|funded|award)\b", re.I)),
    ("doe_grant", re.compile(r"\bdoe\s+(?:grant|funded|award)\b", re.I)),
    ("eu_grant", re.compile(r"\beu\s+(?:grant|funded|horizon)\b", re.I)),
    ("yc_batch", re.compile(r"\by[\s-]?combinator\b", re.I)),
    ("techstars_batch", re.compile(r"\btechstars\b", re.I)),
    ("accelerator_program", re.compile(r"\baccelerator\s+program\b", re.I)),
    ("incubator", re.compile(r"\bincubator\b", re.I)),
    ("innovation_grant", re.compile(r"\binnovation\s+(?:grant|fund|award)\b", re.I)),
]


# ---------------------------------------------------------------------------
# Revenue amount patterns
# ---------------------------------------------------------------------------

_REVENUE_AMOUNT_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("arr_figure", re.compile(
        r"\barr\s+(?:of\s+|at\s+|reaching\s+|exceeding\s+)?"
        r"\$[\d,.]+\s*(?:million|m|billion|b|k)?\b",
        re.I,
    )),
    ("mrr_figure", re.compile(
        r"\bmrr\s+(?:of\s+|at\s+|reaching\s+)?"
        r"\$[\d,.]+\s*(?:thousand|k|million|m)?\b",
        re.I,
    )),
    ("arr_with_amount", re.compile(
        r"\$[\d,.]+\s*(?:million|m|billion|b)\s*(?:in\s+)?arr\b",
        re.I,
    )),
    ("mrr_with_amount", re.compile(
        r"\$[\d,.]+\s*(?:thousand|k|million|m)\s*mrr\b", re.I,
    )),
    ("revenue_figure", re.compile(
        r"\b(?:revenue|generating|earning)\s+"
        r"(?:of\s+|of\s+approximately\s+)?"
        r"\$[\d,.]+\s*(?:million|m|billion|b|k)?\b",
        re.I,
    )),
]


# ---------------------------------------------------------------------------
# Revenue signal keywords (weighted)
# ---------------------------------------------------------------------------

_REVENUE_WEIGHTED_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "recurring": [
        ("recurring revenue", 5.0), ("recurring", 3.0),
        ("subscription revenue", 4.0), ("annual contracts", 3.0),
        ("monthly contracts", 3.0),
    ],
    "transactional": [
        ("transaction fee", 4.0), ("take rate", 4.0),
        ("transaction volume", 3.0), ("per transaction", 3.0),
    ],
    "licensing": [
        ("licensing revenue", 5.0), ("per-study licensing", 5.0),
        ("license fee", 4.0),
    ],
    "freemium": [
        ("freemium", 4.0), ("free tier", 3.0),
        ("premium subscribers", 4.0), ("paid plan", 3.0),
    ],
    "commission": [
        ("commission", 4.0), ("take rate", 3.0),
    ],
    "usage_based": [
        ("usage-based", 4.0), ("usage based pricing", 4.0),
        ("pay per use", 3.0),
    ],
}


# ---------------------------------------------------------------------------
# GMV signal patterns
# ---------------------------------------------------------------------------

_GMV_SIGNAL_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("gmv_amount", re.compile(
        r"\$[\d,.]+\s*(?:million|m|billion|b)\s*(?:in\s+)?gmv\b",
        re.I,
    )),
    ("gmv_mention", re.compile(r"\bgmv\b", re.I)),
    ("gross_merchandise", re.compile(
        r"\bgross\s+merchandise\s+(?:volume|value)\b", re.I,
    )),
    ("transaction_volume", re.compile(
        r"\b(?:processing|processed|facilitating)\s+"
        r"\$[\d,.]+\s*(?:million|m|billion|b|trillion|t)?\s*"
        r"(?:in\s+)?(?:volume|payments?|transactions?)?\b",
        re.I,
    )),
    ("annual_volume", re.compile(
        r"\b\$[\d,.]+\s*(?:million|m|billion|b)\s+in\s+annual\b",
        re.I,
    )),
]


# ---------------------------------------------------------------------------
# Customer count patterns
# ---------------------------------------------------------------------------

_CUSTOMER_COUNT_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("enterprise_clients", re.compile(
        r"\b\d{2,}\s*enterprise\s+(?:clients?|customers?)\b", re.I
    )),
    ("client_count", re.compile(
        r"\b\d{2,}\s*(?:active\s+)?(?:clients?|customers?|accounts?)\b", re.I
    )),
    ("hospital_count", re.compile(
        r"\b\d{2,}\s*hospital[s]?\b", re.I
    )),
    ("institution_count", re.compile(
        r"\b\d{2,}\s*(?:financial\s+)?institutions?\b", re.I
    )),
    ("buyer_count", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s*(?:active\s+)?buyers?\b", re.I
    )),
    ("supplier_count", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s*supplier[s]?\b", re.I
    )),
    ("marketplace_sellers", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s*(?:sellers?|listings?)\b", re.I
    )),
    ("customer_mention_with_number", re.compile(
        r"\b\d{2,}\s+(?:\w+\s+){0,3}(?:customers?|clients?)\b", re.I
    )),
]


# ---------------------------------------------------------------------------
# Active user patterns
# ---------------------------------------------------------------------------

_ACTIVE_USER_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("mau", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s*(?:monthly\s+)?active\s+users?\b", re.I
    )),
    ("dau", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s*(?:daily\s+)?active\s+users?\b", re.I
    )),
    ("user_count", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s*(?:users?|downloads?|installs?)\b", re.I
    )),
    ("mau_mention", re.compile(r"\bmau\b", re.I)),
    ("dau_mention", re.compile(r"\bdau\b", re.I)),
    ("active_accounts", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s*active\s+accounts?\b", re.I
    )),
    ("fleet_size", re.compile(
        r"\b\d{2,}\s*deployed\s+(?:robots?|devices?|units?|machines?)\b", re.I
    )),
]


# ---------------------------------------------------------------------------
# Enterprise customer patterns
# ---------------------------------------------------------------------------

_ENTERPRISE_CUSTOMER_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("fortune_500", re.compile(
        r"\bfortune\s+(?:100|200|500|1000)\b", re.I,
    )),
    ("enterprise_contract", re.compile(
        r"\benterprise\s+(?:contract|deal|agreement|client|customer)\b",
        re.I,
    )),
    ("large_acv", re.compile(
        r"\b(?:acv|annual\s+contract\s+value)\s+"
        r"(?:of\s+|averaging\s+)?\$[\d,.]+\s*(?:k|million|m)?\b",
        re.I,
    )),
    ("top_banks", re.compile(
        r"\btop[\s-]?\d+\s+(?:us\s+)?banks?\b", re.I,
    )),
    ("global_enterprise", re.compile(
        r"\bglobal\s+enterprise[s]?\b", re.I,
    )),
]


# ---------------------------------------------------------------------------
# Pilot customer patterns
# ---------------------------------------------------------------------------

_PILOT_CUSTOMER_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("pilot_program", re.compile(
        r"\bpilot\s+(?:program|project|customer|partner|deployment)\b",
        re.I,
    )),
    ("trial_customer", re.compile(
        r"\b(?:free\s+)?trial\s+(?:customer|user|period|program)\b",
        re.I,
    )),
    ("beta_users", re.compile(
        r"\bbeta\s+(?:users?|customers?|program|phase)\b", re.I,
    )),
    ("early_adopter", re.compile(r"\bearly\s+adopter[s]?\b", re.I)),
    ("proof_of_concept", re.compile(
        r"\b(?:proof[\s-]of[\s-]concept|poc)\b", re.I,
    )),
    ("poc_deployment", re.compile(r"\bpoc\s+deployment\b", re.I)),
]


# ---------------------------------------------------------------------------
# Paying customer patterns
# ---------------------------------------------------------------------------

_PAYING_CUSTOMER_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("paying_customers", re.compile(
        r"\bpaying\s+(?:customers?|clients?|users?|subscribers?)\b",
        re.I,
    )),
    ("paid_subscribers", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s*(?:premium\s+)?subscribers?\b", re.I,
    )),
    ("converting_to_paid", re.compile(
        r"\bconverting\s+\d+\s+(?:open[\s-]source\s+)?users?\s+"
        r"to\s+paid\b",
        re.I,
    )),
    ("revenue_generating", re.compile(
        r"\brevenue[\s-]generating\b", re.I,
    )),
    ("monetized", re.compile(r"\bmonetized?\b", re.I)),
    ("paying", re.compile(r"\bpaying\b", re.I)),
]


# ---------------------------------------------------------------------------
# Partnership patterns
# ---------------------------------------------------------------------------

_PARTNERSHIP_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("strategic_partnership", re.compile(r"\bstrategic\s+partner(?:ship)?s?\b", re.I)),
    ("technology_partnership", re.compile(r"\btechnology\s+partner(?:ship)?s?\b", re.I)),
    ("distribution_partnership", re.compile(r"\bdistribution\s+partner(?:ship)?s?\b", re.I)),
    ("channel_partner", re.compile(r"\bchannel\s+partner[s]?\b", re.I)),
    ("integration_partner", re.compile(r"\bintegration\s+partner[s]?\b", re.I)),
    ("oem_partnership", re.compile(r"\boem\s+partner[s]?\b", re.I)),
    ("reseller", re.compile(r"\breseller[s]?\b", re.I)),
    ("partner_mention", re.compile(r"\bpartner(?:ed|ship|ing)?\b", re.I)),
]


# ---------------------------------------------------------------------------
# Retention patterns
# ---------------------------------------------------------------------------

_RETENTION_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("net_retention", re.compile(
        r"\b\d{2,}%?\s*net\s+(?:revenue\s+)?retention\b", re.I
    )),
    ("gross_retention", re.compile(
        r"\b\d{2,}%?\s*gross\s+(?:revenue\s+)?retention\b", re.I
    )),
    ("nrr_figure", re.compile(
        r"\bnrr\s+(?:of\s+)?\d{2,}%?\b", re.I
    )),
    ("retention_rate", re.compile(
        r"\bretention\s+(?:rate|at)\s+(?:of\s+)?\d{2,}%?\b", re.I
    )),
    ("churn_rate", re.compile(
        r"\b(?:churn|churned)\s+(?:rate|at)\s+(?:of\s+)?\d{1,2}%?\b", re.I
    )),
    ("low_churn", re.compile(r"\blow\s+churn\b", re.I)),
    ("high_retention", re.compile(r"\bhigh\s+retention\b", re.I)),
    ("d30_retention", re.compile(
        r"\bd30\s+(?:retention\s+(?:at\s+)?|is\s+)?\d{2,}%?\b", re.I,
    )),
    ("ltv_cac_ratio", re.compile(
        r"\bltv[/\s]cac\s+(?:of\s+)?[\d.]+x?\b", re.I
    )),
]


# ---------------------------------------------------------------------------
# Engagement patterns
# ---------------------------------------------------------------------------

_ENGAGEMENT_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("daily_usage", re.compile(
        r"\bdaily\s+(?:usage|active|sessions?|engagement)\b", re.I,
    )),
    ("session_depth", re.compile(
        r"\bsession\s+(?:depth|length|time|duration)\b", re.I,
    )),
    ("usage_frequency", re.compile(r"\busage\s+frequency\b", re.I)),
    ("api_calls", re.compile(
        r"\b\d[\d,.]*\s*(?:billion|million|thousand|k|b)?\s*"
        r"(?:api|inference|requests?)\s+(?:daily|monthly|per)\b",
        re.I,
    )),
    ("inference_volume", re.compile(
        r"\bprocessing\s+(?:over\s+)?\d[\d,.]*\s*"
        r"(?:billion|million|thousand|k)?\s*"
        r"(?:inference\s+)?requests?\b",
        re.I,
    )),
]


# ---------------------------------------------------------------------------
# Product adoption patterns
# ---------------------------------------------------------------------------

_PRODUCT_ADOPTION_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("github_stars", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s*github\s+stars?\b", re.I
    )),
    ("open_source_adoption", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s+"
        r"(?:contributing\s+)?"
        r"(?:open[\s-]source\s+)?(?:users?|developers?|contributors?)\b",
        re.I,
    )),
    ("beta_launch", re.compile(r"\bbeta\s+launch\b", re.I)),
    ("general_availability", re.compile(r"\b(?:general\s+availability|ga\s+launch)\b", re.I)),
    ("production_ready", re.compile(r"\bproduction[\s-]ready\b", re.I)),
    ("live_production", re.compile(r"\blive\s+in\s+production\b", re.I)),
]


# ---------------------------------------------------------------------------
# Growth signal patterns
# ---------------------------------------------------------------------------

_GROWTH_SIGNAL_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("mom_growth", re.compile(
        r"\b\d{2,}%?\s*(?:mom|month[\s-]over[\s-]month)\s+growth\b",
        re.I,
    )),
    ("yoy_growth", re.compile(
        r"\b\d{2,}%?\s*(?:yoy|year[\s-]over[\s-]year)\s+growth\b",
        re.I,
    )),
    ("revenue_growth", re.compile(r"\brevenue\s+growth\b", re.I)),
    ("rapid_growth", re.compile(
        r"\b(?:rapid|fast|strong|exponential)\s+(?:growth|scaling)\b",
        re.I,
    )),
    ("doubling", re.compile(
        r"\b(?:doubling|tripled|quadrupled)\s+"
        r"(?:revenue|users?|customers?|arr|growth)\b",
        re.I,
    )),
    ("growing", re.compile(
        r"\bgrowing\s+(?:rapidly|quickly|fast|exponentially)\b", re.I,
    )),
    ("scaling", re.compile(
        r"\bscaling\s+(?:rapidly|quickly|fast|operations?)\b", re.I,
    )),
]


# ---------------------------------------------------------------------------
# Hiring growth patterns
# ---------------------------------------------------------------------------

_HIRING_GROWTH_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("team_growth", re.compile(
        r"\bteam\s+of\s+\d{2,}\b", re.I
    )),
    ("headcount_growth", re.compile(
        r"\bgrowing\s+(?:team|headcount|rapidly)\b", re.I
    )),
    ("hiring_roles", re.compile(
        r"\b(?:hiring|recruiting|seeking)\s+(?:engineers?|developers?|sales|marketing)\b", re.I
    )),
    ("expanding_team", re.compile(r"\bexpanding\s+(?:the\s+)?(?:team|engineering|sales)\b", re.I)),
]


# ---------------------------------------------------------------------------
# Expansion signal patterns
# ---------------------------------------------------------------------------

_EXPANSION_SIGNAL_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("geographic_expansion", re.compile(
        r"\b(?:expanding|expanded|expansion)\s+(?:into\s+)?"
        r"(?:new\s+)?(?:markets?|regions?|countries?|geographies?)\b",
        re.I,
    )),
    ("new_market", re.compile(
        r"\bnew\s+market\s+(?:entry|expansion|opportunity)\b", re.I,
    )),
    ("international", re.compile(
        r"\binternational\s+(?:expansion|growth|market|operations?)\b",
        re.I,
    )),
    ("multi_country", re.compile(
        r"\bacross\s+\d{2,}\s*countries?\b", re.I,
    )),
    ("product_line_expansion", re.compile(
        r"\b(?:new\s+product|product\s+line|expanding\s+into)\b",
        re.I,
    )),
]


# ---------------------------------------------------------------------------
# Launch signal patterns
# ---------------------------------------------------------------------------

_LAUNCH_SIGNAL_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("product_launch", re.compile(
        r"\b(?:launched?|launch(?:ing)?)\s+(?:the\s+)?"
        r"(?:platform|product|app|solution)\b",
        re.I,
    )),
    ("market_launch", re.compile(
        r"\b(?:launch(?:ed|ing)?)\s+in\s+"
        r"(?:new\s+)?(?:market|region|country)\b",
        re.I,
    )),
    ("app_launch", re.compile(
        r"\bapp\s+(?:store|launch|release)\b", re.I,
    )),
    ("beta_launch", re.compile(r"\bbeta\s+launch(?:ed)?\b", re.I)),
    ("officially_launched", re.compile(
        r"\bofficially\s+launch(?:ed|ing)\b", re.I,
    )),
    ("went_live", re.compile(r"\bwent\s+live\b", re.I)),
]


# ---------------------------------------------------------------------------
# Milestone signal patterns
# ---------------------------------------------------------------------------

_MILESTONE_SIGNAL_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("first_enterprise", re.compile(
        r"\bfirst\s+enterprise\s+(?:customer|client|deal|contract)\b",
        re.I,
    )),
    ("hundredth_customer", re.compile(
        r"\b(?:100th|hundredth)\s+(?:customer|client)\b", re.I,
    )),
    ("millionth_user", re.compile(
        r"\b(?:millionth|1,000,000th)\s+(?:user|customer)\b", re.I,
    )),
    ("break_even", re.compile(r"\bbreak[\s-]even\b", re.I)),
    ("profitability", re.compile(r"\bprofitable?\b", re.I)),
    ("cash_flow_positive", re.compile(
        r"\bcash[\s-]flow\s+positive\b", re.I,
    )),
    ("first_patent", re.compile(r"\bfirst\s+patent\b", re.I)),
    ("fda_clearance", re.compile(r"\bfda[\s-]cleared?\b", re.I)),
    ("soc2_compliance", re.compile(r"\bsoc\s*2\b", re.I)),
    ("iso_certification", re.compile(r"\biso\s+\d{4,5}\b", re.I)),
]


# ---------------------------------------------------------------------------
# Awards and recognition patterns
# ---------------------------------------------------------------------------

_AWARDS_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("industry_award", re.compile(
        r"\b(?:award(?:ed)?|winner|winning)\s+(?:of\s+)?"
        r"(?:the\s+)?\w+\s+(?:award|prize)\b",
        re.I,
    )),
    ("best_in_class", re.compile(r"\bbest[\s-]in[\s-]class\b", re.I)),
    ("top_startup", re.compile(
        r"\btop\s+(?:\d+\s+)?(?:startup|company|innovation)\b", re.I,
    )),
    ("gartner", re.compile(r"\bgartner\b", re.I)),
    ("forbes", re.compile(r"\bforbes\b", re.I)),
    ("techcrunch", re.compile(r"\btechcrunch\b", re.I)),
    ("featured_in", re.compile(r"\bfeatured\s+in\b", re.I)),
    ("recognized", re.compile(r"\brecognized\s+(?:as|by)\b", re.I)),
]


# ---------------------------------------------------------------------------
# Traction keywords
# ---------------------------------------------------------------------------

_TRACTION_KEYWORD_MAP: dict[str, list[str]] = {
    "market_validation": [
        "product-market fit", "market validation", "customer validation",
        "problem-solution fit", "demand validated",
    ],
    "revenue_growth": [
        "revenue", "arr", "mrr", "monetized", "profitable",
        "cash flow", "sales", "paying customers",
    ],
    "user_growth": [
        "users", "customers", "clients", "downloads", "installs",
        "monthly active", "daily active",
    ],
    "retention": [
        "retention", "churn", "stickiness", "net revenue retention",
        "gross retention", "ltv", "recurring",
    ],
    "engagement": [
        "engagement", "usage", "sessions", "active", "daily usage",
        "api calls", "processing volume",
    ],
    "funding": [
        "raised", "funding", "investors", "venture", "capital",
        "series", "seed", "round",
    ],
    "partnerships": [
        "partner", "partnership", "integration", "channel",
        "distribution", "reseller", "strategic",
    ],
    "expansion": [
        "expansion", "international", "new markets", "scaling",
        "growing", "geographic",
    ],
    "launch": [
        "launched", "launch", "beta", "live", "production",
        "released", "general availability",
    ],
    "recognition": [
        "award", "featured", "recognized", "forbes", "gartner",
        "techcrunch", "best", "top",
    ],
}


class TractionExtractor(BaseExtractor):
    """Extracts traction attributes from startup data.

    Performs deterministic, explainable signal extraction across 22
    dimensions of traction and market validation assessment. Every signal
    is traceable to a specific text pattern or data field.

    Populates: funding_stage, has_revenue, funding_amount_signals,
    investor_signals, grants_accelerator_signals, revenue_amount_signals,
    arr_mrr_signals, gmv_signals, customer_count_signals,
    active_user_signals, enterprise_customer_signals, pilot_customer_signals,
    paying_customer_signals, partnership_signals, retention_signals,
    engagement_signals, product_adoption_signals, growth_signals,
    hiring_growth_signals, expansion_signals, launch_signals,
    milestone_signals, awards_recognition, traction_keywords,
    traction_confidence.
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        desc = startup.description
        text_lower = desc.lower()

        # Core existing fields
        funding_stage = self._classify_funding_stage(text_lower)
        has_revenue = self._detect_revenue_signal(text_lower)

        # Funding intelligence
        funding_amounts = self._detect_funding_amounts(desc)
        investor_signals = self._detect_investor_signals(desc)
        grant_signals = self._detect_grants_accelerators(desc)

        # Revenue intelligence
        revenue_amounts = self._detect_revenue_amounts(desc)
        arr_mrr = self._detect_arr_mrr(desc)

        # Volume / GMV signals
        gmv = self._detect_gmv_signals(desc)

        # Customer intelligence
        customer_count = self._detect_customer_count(desc)
        active_users = self._detect_active_users(desc)
        enterprise_customers = self._detect_enterprise_customers(desc)
        pilot_customers = self._detect_pilot_customers(desc)
        paying_customers = self._detect_paying_customers(desc)

        # Relationships
        partnerships = self._detect_partnerships(desc)

        # Retention & engagement
        retention = self._detect_retention_signals(desc)
        engagement = self._detect_engagement_signals(desc)

        # Adoption
        product_adoption = self._detect_product_adoption(desc)

        # Growth
        growth = self._detect_growth_signals(desc)
        hiring_growth = self._detect_hiring_growth(desc)

        # Expansion & launch
        expansion = self._detect_expansion_signals(desc)
        launch = self._detect_launch_signals(desc)

        # Milestones & recognition
        milestones = self._detect_milestone_signals(desc)
        awards = self._detect_awards_recognition(desc)

        # Traction keywords
        traction_kws = self._extract_traction_keywords(text_lower)

        # Confidence
        confidence = self._compute_traction_confidence(
            funding_stage=funding_stage,
            has_revenue=has_revenue,
            funding_amounts=funding_amounts,
            investor_signals=investor_signals,
            revenue_amounts=revenue_amounts,
            arr_mrr=arr_mrr,
            gmv=gmv,
            customer_count=customer_count,
            active_users=active_users,
            enterprise_customers=enterprise_customers,
            paying_customers=paying_customers,
            retention=retention,
            engagement=engagement,
            growth=growth,
            partnership=partnerships,
            launch=launch,
            milestones=milestones,
            awards=awards,
            traction_kws=traction_kws,
        )

        # --- Structured quantitative extraction (Sprint 14) ---
        funding_amount_usd = parse_funding_amount(desc)
        arr_usd = parse_arr(desc)
        mrr_usd = parse_mrr(desc)
        gmv_usd = parse_gmv(desc)
        customer_count_num = parse_customer_count(desc)
        active_user_count = parse_active_user_count(desc)
        growth_rate_pct = parse_growth_rate(desc)

        # Sprint 14.1 — shared parsers for NRR/churn/valuation
        nrr_pct = parse_nrr(desc)
        churn_rate_pct = parse_churn(desc)
        valuation_usd = parse_valuation(desc)

        # Sprint 14.1 — additional quantitative fields
        burn_rate_usd = parse_burn_rate(desc)
        runway_months = parse_runway(desc)
        cac_usd = parse_cac(desc)
        ltv_usd = parse_ltv(desc)
        team_size_num = parse_team_size(desc)

        return ExtractedFeatures(
            funding_stage=funding_stage,
            has_revenue=has_revenue,
            funding_amount_signals=funding_amounts,
            investor_signals=investor_signals,
            grants_accelerator_signals=grant_signals,
            revenue_amount_signals=revenue_amounts,
            arr_mrr_signals=arr_mrr,
            gmv_signals=gmv,
            customer_count_signals=customer_count,
            active_user_signals=active_users,
            enterprise_customer_signals=enterprise_customers,
            pilot_customer_signals=pilot_customers,
            paying_customer_signals=paying_customers,
            partnership_signals=partnerships,
            retention_signals=retention,
            engagement_signals=engagement,
            product_adoption_signals=product_adoption,
            growth_signals=growth,
            hiring_growth_signals=hiring_growth,
            expansion_signals=expansion,
            launch_signals=launch,
            milestone_signals=milestones,
            awards_recognition=awards,
            traction_keywords=traction_kws,
            traction_confidence=confidence,
            funding_amount_usd=funding_amount_usd,
            arr_usd=arr_usd,
            mrr_usd=mrr_usd,
            gmv_usd=gmv_usd,
            customer_count=customer_count_num,
            active_user_count=active_user_count,
            growth_rate_pct=growth_rate_pct,
            nrr_pct=nrr_pct,
            churn_rate_pct=churn_rate_pct,
            valuation_usd=valuation_usd,
            burn_rate_usd=burn_rate_usd,
            runway_months=runway_months,
            cac_usd=cac_usd,
            ltv_usd=ltv_usd,
            team_size_numeric=team_size_num,
        )

    # ------------------------------------------------------------------
    # Funding stage classification (weighted evidence)
    # ------------------------------------------------------------------

    def _classify_funding_stage(self, text: str) -> str | None:
        scores: dict[str, float] = {}
        for stage_name, weighted_kws in _STAGE_WEIGHTED_KEYWORDS.items():
            total = sum(w for kw, w in weighted_kws if kw in text)
            if total > 0:
                scores[stage_name] = total

        if not scores:
            return None

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_stage, best_score = sorted_scores[0]

        if best_score < _STAGE_CLASSIFICATION_THRESHOLD:
            return None

        if len(sorted_scores) > 1:
            margin = best_score - sorted_scores[1][1]
            if margin < _STAGE_AMBIGUITY_MARGIN and best_score < 8.0:
                return None

        return best_stage

    # ------------------------------------------------------------------
    # Revenue signal detection (simple boolean — backward compatible)
    # ------------------------------------------------------------------

    def _detect_revenue_signal(self, text: str) -> bool | None:
        # Exclude explicit pre-revenue / no-revenue signals
        if re.search(r"\bpre[\s-]revenue\b", text, re.I):
            return None
        if re.search(r"\bno\s+revenue\b", text, re.I):
            return None

        for category, weighted_kws in _REVENUE_WEIGHTED_KEYWORDS.items():
            score = sum(w for kw, w in weighted_kws if kw in text)
            if score >= 3.0:
                return True

        direct_keywords = [
            "revenue", "mrr", "arr", "recurring", "paying customers",
            "monetized", "profitable", "cash flow", "sales", "acv",
        ]
        for kw in direct_keywords:
            if kw in text:
                return True
        return None

    # ------------------------------------------------------------------
    # Funding amount detection
    # ------------------------------------------------------------------

    def _detect_funding_amounts(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _FUNDING_AMOUNT_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Investor signal detection
    # ------------------------------------------------------------------

    def _detect_investor_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _INVESTOR_SIGNAL_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Grants and accelerator detection
    # ------------------------------------------------------------------

    def _detect_grants_accelerators(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _GRANT_ACCELERATOR_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Revenue amount detection
    # ------------------------------------------------------------------

    def _detect_revenue_amounts(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _REVENUE_AMOUNT_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # ARR / MRR detection
    # ------------------------------------------------------------------

    def _detect_arr_mrr(self, text: str) -> list[str]:
        signals: list[str] = []
        arr_pattern = re.compile(
            r"\b\d[\d,.]*\s*(?:million|m|billion|b|thousand|k)?\s*arr\b",
            re.I,
        )
        mrr_pattern = re.compile(
            r"\b(?:mrr|monthly\s+recurring\s+revenue)\s+"
            r"(?:of\s+|at\s+|reaching\s+)?\$[\d,.]+",
            re.I,
        )
        mrr_number = re.compile(
            r"\b\d[\d,.]*\s*(?:thousand|k|million|m)\s*mrr\b",
            re.I,
        )
        arr_with = re.compile(
            r"\barr\s+(?:of\s+|at\s+|reaching\s+|exceeding\s+)?"
            r"\$[\d,.]+\s*(?:million|m|billion|b|k)?\b",
            re.I,
        )

        for pattern, label in [
            (arr_pattern, "arr_mention"),
            (mrr_pattern, "mrr_mention"),
            (mrr_number, "mrr_mention"),
            (arr_with, "arr_with_amount"),
        ]:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # GMV signal detection
    # ------------------------------------------------------------------

    def _detect_gmv_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _GMV_SIGNAL_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 40)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Customer count detection
    # ------------------------------------------------------------------

    def _detect_customer_count(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _CUSTOMER_COUNT_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Active user detection
    # ------------------------------------------------------------------

    def _detect_active_users(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _ACTIVE_USER_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 40)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Enterprise customer detection
    # ------------------------------------------------------------------

    def _detect_enterprise_customers(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _ENTERPRISE_CUSTOMER_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Pilot customer detection
    # ------------------------------------------------------------------

    def _detect_pilot_customers(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _PILOT_CUSTOMER_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Paying customer detection
    # ------------------------------------------------------------------

    def _detect_paying_customers(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _PAYING_CUSTOMER_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Partnership detection
    # ------------------------------------------------------------------

    def _detect_partnerships(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _PARTNERSHIP_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Retention signal detection
    # ------------------------------------------------------------------

    def _detect_retention_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _RETENTION_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Engagement signal detection
    # ------------------------------------------------------------------

    def _detect_engagement_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _ENGAGEMENT_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 40)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Product adoption detection
    # ------------------------------------------------------------------

    def _detect_product_adoption(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _PRODUCT_ADOPTION_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 40)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Growth signal detection
    # ------------------------------------------------------------------

    def _detect_growth_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _GROWTH_SIGNAL_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 40)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Hiring growth detection
    # ------------------------------------------------------------------

    def _detect_hiring_growth(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _HIRING_GROWTH_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 40)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Expansion signal detection
    # ------------------------------------------------------------------

    def _detect_expansion_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _EXPANSION_SIGNAL_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 40)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Launch signal detection
    # ------------------------------------------------------------------

    def _detect_launch_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _LAUNCH_SIGNAL_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 40)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Milestone signal detection
    # ------------------------------------------------------------------

    def _detect_milestone_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _MILESTONE_SIGNAL_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 40)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Awards and recognition detection
    # ------------------------------------------------------------------

    def _detect_awards_recognition(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _AWARDS_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Traction keyword extraction
    # ------------------------------------------------------------------

    def _extract_traction_keywords(self, text: str) -> list[str]:
        keywords: list[str] = []
        for category, kws in _TRACTION_KEYWORD_MAP.items():
            for kw in kws:
                if kw in text and kw not in keywords:
                    keywords.append(kw)
        return keywords

    # ------------------------------------------------------------------
    # Composite traction confidence scoring
    # ------------------------------------------------------------------

    def _compute_traction_confidence(
        self,
        *,
        funding_stage: str | None,
        has_revenue: bool | None,
        funding_amounts: list[str],
        investor_signals: list[str],
        revenue_amounts: list[str],
        arr_mrr: list[str],
        gmv: list[str],
        customer_count: list[str],
        active_users: list[str],
        enterprise_customers: list[str],
        paying_customers: list[str],
        retention: list[str],
        engagement: list[str],
        growth: list[str],
        partnership: list[str],
        launch: list[str],
        milestones: list[str],
        awards: list[str],
        traction_kws: list[str],
    ) -> float:
        score = 0.0

        if funding_stage is not None:
            score += 0.10
        if has_revenue is True:
            score += 0.12

        score += min(len(funding_amounts) * 0.04, 0.12)
        score += min(len(investor_signals) * 0.03, 0.09)
        score += min(len(revenue_amounts) * 0.04, 0.12)
        score += min(len(arr_mrr) * 0.05, 0.10)
        score += min(len(gmv) * 0.04, 0.08)
        score += min(len(customer_count) * 0.03, 0.09)
        score += min(len(active_users) * 0.03, 0.06)
        score += min(len(enterprise_customers) * 0.04, 0.08)
        score += min(len(paying_customers) * 0.04, 0.08)
        score += min(len(retention) * 0.04, 0.08)
        score += min(len(engagement) * 0.03, 0.06)
        score += min(len(growth) * 0.03, 0.06)
        score += min(len(partnership) * 0.02, 0.04)
        score += min(len(launch) * 0.02, 0.04)
        score += min(len(milestones) * 0.03, 0.06)
        score += min(len(awards) * 0.02, 0.04)
        score += min(len(traction_kws) * 0.01, 0.05)

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
