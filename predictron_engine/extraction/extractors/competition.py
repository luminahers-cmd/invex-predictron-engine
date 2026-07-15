"""Competition extractor — rich deterministic competitive intelligence extraction.

Responsibilities:
  - Direct competitor signal detection from description mentions
  - Indirect / adjacent competitor signal detection
  - Incumbent and legacy player signal detection
  - Market concentration classification (fragmented -> dominated)
  - Competitive density classification (sparse -> hyper-competitive)
  - Fragmentation signal detection
  - Winner-take-most dynamic detection
  - Network effect competitive dynamics classification
  - Switching cost and lock-in signal detection
  - Competitive differentiation signal detection
  - Competitive moat indicator detection
  - Barrier to entry detection
  - Substitute product signal detection
  - Platform dependency detection
  - Ecosystem dependency detection
  - Open-source competition signal detection
  - Regulatory competitive advantage detection
  - Geographic competition signal detection
  - Pricing pressure signal detection
  - Competitive keyword extraction
  - Composite competition confidence scoring

Design principles:
  - Deterministic rule-based logic only (no LLMs, no ML)
  - Weighted keyword scoring with context-aware disambiguation
  - Every extracted field traceable to explicit input signals
  - Prefer "Unknown" over incorrect classification
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
# Direct competitor signals — explicit mentions of competition
# ---------------------------------------------------------------------------

_DIRECT_COMPETITOR_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("competitor_mention", re.compile(
        r"\bcompet(?:e|ing|itor|ition)\b", re.I,
    )),
    ("versus", re.compile(
        r"\bversus\b|\bvs\.?\b", re.I,
    )),
    ("alternatives_to", re.compile(
        r"\balternative[s]?\s+to\b", re.I,
    )),
    ("competing_with", re.compile(
        r"\bcompeting\s+(?:with|against|in)\b", re.I,
    )),
    ("rival", re.compile(
        r"\brival[s]?\b", re.I,
    )),
    ("compete_directly", re.compile(
        r"\bcompete\s+directly\b", re.I,
    )),
    ("head_to_head", re.compile(
        r"\bhead[\s-]to[\s-]head\b", re.I,
    )),
    ("market_leader", re.compile(
        r"\bmarket\s+leader[s]?\b", re.I,
    )),
    ("incumbent_named", re.compile(
        r"\bincumbent[s]?\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Indirect competitor signals — adjacent or substitute solutions
# ---------------------------------------------------------------------------

_INDIRECT_COMPETITOR_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("adjacent_market", re.compile(
        r"\badjacent\s+(?:market|space|industry|sector)\b", re.I,
    )),
    ("indirect_competition", re.compile(
        r"\bindirect\s+(?:compet|threat|rival)\b", re.I,
    )),
    ("traditional_solution", re.compile(
        r"\btraditional\s+(?:solution|approach|method|way|process)\b", re.I,
    )),
    ("legacy_solution", re.compile(
        r"\blegacy\s+(?:system|solution|platform|tool|software|approach)\b",
        re.I,
    )),
    ("spreadsheets", re.compile(
        r"\bspreadsheet[s]?\b", re.I,
    )),
    ("manual_process", re.compile(
        r"\bmanual\s+(?:process|workflow|approach|method)\b", re.I,
    )),
    ("status_quo", re.compile(
        r"\bstatus[\s-]quo\b", re.I,
    )),
    ("homegrown", re.compile(
        r"\bhome[\s-]?grown\b", re.I,
    )),
    ("custom_built", re.compile(
        r"\bcustom[\s-]built\b|\bcustom[\s-]built\b", re.I,
    )),
    ("point_solution", re.compile(
        r"\bpoint[\s-]solution[s]?\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Incumbent signals — legacy players being displaced
# ---------------------------------------------------------------------------

_INCUMBENT_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("incumbent_disruption", re.compile(
        r"\bdisrupt(?:ing|ion|ed)?\s+(?:the\s+)?(?:incumbent|legacy|traditional)\b",
        re.I,
    )),
    ("replacing_incumbent", re.compile(
        r"\breplac(?:ing|es?)\s+(?:legacy|existing|incumbent)\b", re.I,
    )),
    ("legacy_system", re.compile(
        r"\blegacy\s+(?:system|platform|software|tool|infrastructure)\b",
        re.I,
    )),
    ("outdated_approach", re.compile(
        r"\boutdated\s+(?:approach|method|process|solution)\b", re.I,
    )),
    ("broken_process", re.compile(
        r"\bbroken\s+(?:process|workflow|system)\b", re.I,
    )),
    ("inefficient", re.compile(
        r"\binefficient\s+(?:process|workflow|system|tool)\b", re.I,
    )),
    ("replacing_manual", re.compile(
        r"\breplac(?:ing|es?)\s+manual\b", re.I,
    )),
    ("modern_alternative", re.compile(
        r"\bmodern\s+(?:alternative|replacement|approach|solution)\b", re.I,
    )),
    ("old_way", re.compile(
        r"\bold[\s-]fashioned\b|\bthe\s+old\s+way\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Market concentration — weighted keyword scoring
# ---------------------------------------------------------------------------

_CONCENTRATION_WEIGHTED_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "fragmented": [
        ("fragmented", 5.0), ("many players", 4.0),
        ("hundreds of", 3.0), ("numerous competitors", 4.0),
        ("crowded market", 3.5), ("long tail", 3.0),
        ("scattered", 2.5), ("dispersed", 2.5),
        ("no clear leader", 4.0), ("no dominant", 4.0),
        ("niche players", 3.0), ("many small", 3.0),
        ("dozens of", 3.0), ("thousands of", 3.0),
    ],
    "moderately_concentrated": [
        ("few major", 4.0), ("handful of", 4.0),
        ("select competitors", 3.0), ("several players", 3.0),
        ("top players", 3.5), ("leading companies", 3.0),
        ("tiered competition", 3.0), ("oligopoly", 4.0),
        ("three to five", 3.0), ("top three", 3.5),
        ("top five", 3.5), ("top ten", 3.0),
    ],
    "concentrated": [
        ("dominated by", 5.0), ("one or two", 4.0),
        ("duopoly", 5.0), ("monopoly", 5.0),
        ("single player", 4.0), ("dominant player", 5.0),
        ("market leader controls", 4.0), ("near monopoly", 5.0),
        ("two main", 4.0), ("two dominant", 5.0),
        ("oligarchic", 4.0),
    ],
    "dominated": [
        ("monopoly", 5.0), ("single dominant", 5.0),
        ("one company", 4.0), ("controlled by", 4.0),
        ("near-monopoly", 5.0), ("captures the majority", 5.0),
        ("market hegemony", 5.0), ("overwhelming share", 4.5),
        ("proprietary lock-in", 4.0),
    ],
}

_CONCENTRATION_THRESHOLD: float = 2.0

# ---------------------------------------------------------------------------
# Competitive density — weighted keyword scoring
# ---------------------------------------------------------------------------

_DENSITY_WEIGHTED_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "sparse": [
        ("untapped", 4.0), ("blue ocean", 5.0),
        ("white space", 4.0), ("nascent market", 4.0),
        ("emerging space", 3.5), ("new category", 4.0),
        ("first mover", 4.0), ("no competitors", 5.0),
        ("uncontested", 4.0), ("no direct competition", 5.0),
        ("category creator", 4.5), ("pioneering", 3.0),
    ],
    "moderate": [
        ("growing competition", 3.5), ("increasing competition", 3.5),
        ("emerging competitors", 3.5), ("few competitors", 3.0),
        ("selective competition", 3.0), ("limited competition", 3.5),
        ("niche competition", 3.0), ("developing market", 3.0),
    ],
    "dense": [
        ("highly competitive", 5.0), ("intensely competitive", 5.0),
        ("fierce competition", 5.0), ("crowded", 3.5),
        ("saturated", 4.0), ("intense competition", 5.0),
        ("red ocean", 5.0), ("battleground", 4.0),
        ("competitive landscape", 3.0), ("many competitors", 4.0),
        ("well-established competitors", 4.0),
    ],
    "hyper_competitive": [
        ("hyper-competitive", 5.0), ("cutthroat", 5.0),
        ("brutal competition", 5.0), ("price war", 5.0),
        ("race to the bottom", 4.0), ("overcrowded", 5.0),
        ("extremely crowded", 5.0), ("fierce rivalry", 5.0),
        ("dozens of funded", 4.0), ("arms race", 4.0),
    ],
}

_DENSITY_THRESHOLD: float = 2.0

# ---------------------------------------------------------------------------
# Fragmentation signals
# ---------------------------------------------------------------------------

_FRAGMENTATION_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("many_small_players", re.compile(
        r"\b(?:many|numerous|dozens|hundreds)\s+of\s+"
        r"(?:small|tiny|niche|local)\s+(?:players?|companies?|vendors?)\b",
        re.I,
    )),
    ("long_tail", re.compile(
        r"\blong[\s-]tail\b", re.I,
    )),
    ("fragmented_space", re.compile(
        r"\bfragmented\s+(?:space|market|industry|landscape)\b", re.I,
    )),
    ("no_clear_winner", re.compile(
        r"\bno\s+clear\s+(?:winner|leader|dominant)\b", re.I,
    )),
    ("consolidation_opportunity", re.compile(
        r"\bconsolidat(?:ion|ing)\s+(?:opportunity|play|strategy)\b", re.I,
    )),
    ("acquisition_target", re.compile(
        r"\bacquisition\s+target\b", re.I,
    )),
    ("scattered_vendors", re.compile(
        r"\bscattered\s+(?:vendors?|providers?|solutions?)\b", re.I,
    )),
    ("lack_of_standardization", re.compile(
        r"\black\s+of\s+standardization\b", re.I,
    )),
    ("patchwork", re.compile(
        r"\bpatchwork\s+(?:of|approach|solution)\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Winner-take-most signals
# ---------------------------------------------------------------------------

_WINNER_TAKE_MOST_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("winner_take_all", re.compile(
        r"\bwinner[\s-]take[\s-]all\b", re.I,
    )),
    ("winner_take_most", re.compile(
        r"\bwinner[\s-]take[\s-]most\b", re.I,
    )),
    ("network_effects_dominance", re.compile(
        r"\bnetwork\s+effect[s]?\s+(?:create|drive|enable|build)\s+"
        r"(?:a\s+)?(?:moat|dominance|winner)\b",
        re.I,
    )),
    ("scale_advantage", re.compile(
        r"\bscale\s+(?:advantage|economies|benefit)\b", re.I,
    )),
    ("winner_dynamics", re.compile(
        r"\b(?:winner|dominant|leading)\s+(?:dynamic|outcome)\b", re.I,
    )),
    ("platform_lock_in", re.compile(
        r"\bplatform[\s-](?:lock[\s-]in|lockin)\b", re.I,
    )),
    ("market_consolidation", re.compile(
        r"\bmarket\s+consolidat(?:ion|ing)\b", re.I,
    )),
    ("flywheel_effect", re.compile(
        r"\bflywheel\s+(?:effect|dynamic|advantage)\b", re.I,
    )),
    ("data_network_effect", re.compile(
        r"\bdata\s+network\s+effect[s]?\b", re.I,
    )),
    ("critical_mass", re.compile(
        r"\bcritical\s+mass\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Network effect competitive dynamics
# ---------------------------------------------------------------------------

_NETWORK_EFFECT_COMPETITION_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "strong_network_effects": [
        ("network effect", 5.0), ("network effects", 5.0),
        ("viral coefficient", 5.0), ("viral growth", 4.5),
        ("flywheel", 4.5), ("two-sided marketplace", 4.0),
        ("more users attract", 5.0), ("value increases with", 4.5),
        ("data network effect", 5.0), ("cross-side effect", 5.0),
        ("same-side effect", 4.5), ("demand-side economies", 5.0),
        ("platform effect", 4.0), ("marketplace effect", 4.0),
    ],
    "moderate_network_effects": [
        ("ecosystem", 3.0), ("community", 2.5),
        ("user-generated content", 3.0), ("social proof", 2.5),
        ("word of mouth", 2.5), ("referral", 2.0),
        ("sticky", 2.0), ("engagement loop", 3.0),
    ],
    "no_network_effects": [
        ("standalone", 2.0), ("single player", 3.0),
        ("individual use", 2.0), ("no network effect", 5.0),
        ("linear value", 3.0), ("independent", 2.0),
    ],
}

_NETWORK_EFFECT_THRESHOLD: float = 2.0

# ---------------------------------------------------------------------------
# Switching cost signals
# ---------------------------------------------------------------------------

_SWITCHING_COST_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("high_switching_cost", re.compile(
        r"\bhigh\s+switching\s+cost[s]?\b", re.I,
    )),
    ("vendor_lock_in", re.compile(
        r"\bvendor[\s-]lock[\s-]in\b|\block[\s-]in\b", re.I,
    )),
    ("deep_integration", re.compile(
        r"\bdeep\s+integrat(?:ion|ed)\b", re.I,
    )),
    ("migration_complexity", re.compile(
        r"\bmigrat(?:ion|ing)\s+(?:is\s+)?(?:complex|difficult|costly)\b",
        re.I,
    )),
    ("switching_barrier", re.compile(
        r"\bswitching\s+barrier[s]?\b", re.I,
    )),
    ("sticky_workflow", re.compile(
        r"\bsticky\s+(?:workflow|process|data|platform)\b", re.I,
    )),
    ("data_lock_in", re.compile(
        r"\bdata[\s-]lock[\s-]in\b|\bdata\s+portability\b", re.I,
    )),
    ("integration_depth", re.compile(
        r"\bdeeply\s+(?:integrated|embedded)\b", re.I,
    )),
    ("workflow_dependency", re.compile(
        r"\bworkflow\s+dependen(?:cy|cies)\b", re.I,
    )),
    ("enterprise_integration", re.compile(
        r"\benterprise[\s-]grade\s+integrat(?:ion|ed)\b", re.I,
    )),
    ("custom_integration", re.compile(
        r"\bcustom\s+integrat(?:ion|ed)\b", re.I,
    )),
    ("api_dependency", re.compile(
        r"\bapi[\s-]dependen(?:cy|cies)\b", re.I,
    )),
    ("multi_year_contract", re.compile(
        r"\bmulti[\s-]year\s+contract[s]?\b", re.I,
    )),
    ("long_term_agreement", re.compile(
        r"\blong[\s-]term\s+(?:agreement|contract|deal)\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Differentiation signals
# ---------------------------------------------------------------------------

differential_signal_patterns: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("proprietary_technology", re.compile(
        r"\bproprietary\s+(?:technology|algorithm|engine|platform|system)\b",
        re.I,
    )),
    ("unique_approach", re.compile(
        r"\bunique\s+(?:approach|method|technology|solution|architecture)\b",
        re.I,
    )),
    ("patent_protection", re.compile(
        r"\bpatent[s]?\b", re.I,
    )),
    ("ai_differentiation", re.compile(
        r"\b(?:ai|ml|machine\s+learning)[\s-]+(?:powered|driven|enabled|first)\b",
        re.I,
    )),
    ("first_principles", re.compile(
        r"\bfirst[\s-]principles\b", re.I,
    )),
    ("innovative", re.compile(
        r"\binnovat(?:ive|ion)\b", re.I,
    )),
    ("unprecedented", re.compile(
        r"\bunprecedented\b", re.I,
    )),
    ("unlike_any", re.compile(
        r"\bunlike\s+any\b", re.I,
    )),
    ("purpose_built", re.compile(
        r"\bpurpose[\s-]built\b", re.I,
    )),
    ("only_platform", re.compile(
        r"\bonly\s+(?:platform|solution|tool|product)\b", re.I,
    )),
    ("category_first", re.compile(
        r"\bfirst[\s-](?:of[\s-]its[\s-]kind|to[\s-]market|in[\s-]class)\b",
        re.I,
    )),
    ("differentiated", re.compile(
        r"\bdifferentiat(?:ed|ing)\b", re.I,
    )),
    ("secret_sauce", re.compile(
        r"\bsecret\s+sauce\b", re.I,
    )),
    ("trade_secret", re.compile(
        r"\btrade\s+secret[s]?\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Competitive moat indicators
# ---------------------------------------------------------------------------

_MOAT_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("data_moat", re.compile(
        r"\bdata\s+(?:moat|advantage|network|flywheel|asset)\b", re.I,
    )),
    ("proprietary_data", re.compile(
        r"\bproprietary\s+data\b", re.I,
    )),
    ("network_effect_moat", re.compile(
        r"\bnetwork\s+effect[s]?\s+(?:moat|advantage|defen[sc])\b", re.I,
    )),
    ("brand_moat", re.compile(
        r"\bbrand\s+(?:moat|advantage|equity|recognition)\b", re.I,
    )),
    ("regulatory_moat", re.compile(
        r"\bregulatory\s+(?:moat|advantage|barrier|approval)\b", re.I,
    )),
    ("technology_moat", re.compile(
        r"\b(?:technology|technical|patent)\s+(?:moat|advantage|barrier)\b",
        re.I,
    )),
    ("scale_moat", re.compile(
        r"\bscale\s+(?:moat|advantage|economies)\b", re.I,
    )),
    ("ecosystem_moat", re.compile(
        r"\becosystem\s+(?:moat|advantage|lock[\s-]in)\b", re.I,
    )),
    ("switching_cost_moat", re.compile(
        r"\bswitching\s+cost[s]?\s+(?:moat|advantage|barrier)\b", re.I,
    )),
    ("defensible", re.compile(
        r"\bdefensible\s+(?:advantage|position|moat)\b", re.I,
    )),
    ("hard_to_replicate", re.compile(
        r"\bhard\s+to\s+(?:replicate|copy|duplicate|reproduce)\b", re.I,
    )),
    ("durable_advantage", re.compile(
        r"\bdurable\s+(?:advantage|moat|position)\b", re.I,
    )),
    ("compounding_advantage", re.compile(
        r"\bcompounding\s+(?:advantage|moat|returns)\b", re.I,
    )),
    ("flywheel_moat", re.compile(
        r"\bflywheel\s+(?:moat|advantage|defen[sc])\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Barrier to entry signals
# ---------------------------------------------------------------------------

_BARRIER_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("regulatory_barrier", re.compile(
        r"\bregulatory\s+(?:barriers?|requirement|approval|clearance|burden)\b",
        re.I,
    )),
    ("fda_approval", re.compile(
        r"\bfda[\s-](?:cleared?|approved|510\(k\)|de[\s-]novo)\b", re.I,
    )),
    ("compliance_certification", re.compile(
        r"\b(?:soc\s*2|iso\s+\d{4,5}|hipaa|gdpr|pci[\s-]dss|fedramp)\b",
        re.I,
    )),
    ("capital_intensity", re.compile(
        r"\bcapital[\s-]intensive\b|\bhigh\s+capex\b|\bmillion[s]?\s+"
        r"in\s+(?:investment|funding|capital)\b",
        re.I,
    )),
    ("technical_complexity", re.compile(
        r"\btechnical\s+complexity\b|\b(?:deep|hard|complex)\s+"
        r"(?:technical|engineering|science)\b",
        re.I,
    )),
    ("talent_barrier", re.compile(
        r"\bscarce\s+(?:talent|expertise|skill)\b|\bhighly\s+specialized\b",
        re.I,
    )),
    ("ip_barrier", re.compile(
        r"\bintellectual\s+property\b|\bpatent[\s-]+protected\b", re.I,
    )),
    ("data_barrier", re.compile(
        r"\bdata\s+(?:advantage|moat|barrier|requirement)\b|\bneed\s+"
        r"extensive\s+data\b",
        re.I,
    )),
    ("infrastructure_barrier", re.compile(
        r"\binfrastructure\s+(?:barrier|investment|requirement)\b|\b"
        r"heavy\s+infrastructure\b",
        re.I,
    )),
    ("time_to_build", re.compile(
        r"\b(?:years?|long)\s+(?:to\s+build|of\s+development)\b", re.I,
    )),
    ("high_r_and_d", re.compile(
        r"\b(?:significant|high|heavy)\s+(?:r&d|r\s*&\s*d|research)\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Substitute product signals
# ---------------------------------------------------------------------------

_SUBSTITUTE_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("instead_of", re.compile(
        r"\binstead\s+of\b", re.I,
    )),
    ("replace", re.compile(
        r"\breplac(?:e|ing|es|ement)\s+(?:existing|legacy|manual|traditional)\b",
        re.I,
    )),
    ("alternative_to", re.compile(
        r"\balternative\s+to\b", re.I,
    )),
    ("instead_of_sheets", re.compile(
        r"\binstead\s+of\s+(?:spreadsheet|excel|email)\b", re.I,
    )),
    ("instead_of_email", re.compile(
        r"\binstead\s+of\s+email\b", re.I,
    )),
    ("替代品", re.compile(
        r"\bsubstitut(?:e|ion|es)\b", re.I,
    )),
    ("取代", re.compile(
        r"\bdo away with\b|\beliminate\s+the\s+need\s+for\b", re.I,
    )),
    ("replaces_spreadsheets", re.compile(
        r"\breplac(?:e|ing|es)\s+spreadsheet[s]?\b", re.I,
    )),
    ("replaces_email", re.compile(
        r"\breplac(?:e|ing|es)\s+email\b", re.I,
    )),
    ("replaces_manual", re.compile(
        r"\breplac(?:e|ing|es)\s+manual\b", re.I,
    )),
    ("replaces_sheets", re.compile(
        r"\breplac(?:e|ing|es)\s+(?:excel|google\s+sheets?)\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Platform dependency signals
# ---------------------------------------------------------------------------

_PLATFORM_DEPENDENCY_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("aws_dependency", re.compile(
        r"\baws[\s-]dependent\b|\bbuilt\s+on\s+aws\b|\baws[\s-]native\b",
        re.I,
    )),
    ("gcp_dependency", re.compile(
        r"\bgcp[\s-]dependent\b|\bbuilt\s+on\s+gcp\b|\bgcp[\s-]native\b",
        re.I,
    )),
    ("azure_dependency", re.compile(
        r"\bazure[\s-]dependent\b|\bbuilt\s+on\s+azure\b|\bazure[\s-]native\b",
        re.I,
    )),
    ("shopify_dependency", re.compile(
        r"\bshopify[\s-](?:app|partner|built)\b", re.I,
    )),
    ("salesforce_dependency", re.compile(
        r"\bsalesforce[\s-](?:app|partner|built)\b|\bsfdc\b", re.I,
    )),
    ("ios_dependency", re.compile(
        r"\bios[\s-](?:only|exclusive|dependent)\b|\bapp\s+store[\s-]only\b",
        re.I,
    )),
    ("android_dependency", re.compile(
        r"\bandroid[\s-](?:only|exclusive|dependent)\b|\bgoogle\s+play[\s-]only\b",
        re.I,
    )),
    ("platform_built_on", re.compile(
        r"\bbuilt\s+(?:on|upon|with)\s+(?:aws|gcp|azure|shopify|salesforce)\b",
        re.I,
    )),
    ("plugin_extension", re.compile(
        r"\b(?:plugin|extension|add[\s-]on)\s+(?:for|to)\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Ecosystem dependency signals
# ---------------------------------------------------------------------------

_ECOSYSTEM_DEPENDENCY_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("ecosystem_play", re.compile(
        r"\becosystem\s+(?:play|strategy|partner|position)\b", re.I,
    )),
    ("platform_ecosystem", re.compile(
        r"\bplatform\s+ecosystem\b", re.I,
    )),
    ("partner_ecosystem", re.compile(
        r"\bpartner\s+ecosystem\b", re.I,
    )),
    ("integrates_with", re.compile(
        r"\bintegrat(?:es?|ing|ion)\s+with\b", re.I,
    )),
    ("compatible_with", re.compile(
        r"\bcompatible\s+with\b", re.I,
    )),
    ("works_with", re.compile(
        r"\bworks\s+with\b", re.I,
    )),
    ("connector", re.compile(
        r"\bconnector[s]?\b", re.I,
    )),
    ("marketplace_listing", re.compile(
        r"\bmarketplace\s+listing\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Open source competition signals
# ---------------------------------------------------------------------------

_OPEN_SOURCE_COMPETITION_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("open_source_alternative", re.compile(
        r"\bopen[\s-]source\s+(?:alternatives?|competitor|option|version)\b",
        re.I,
    )),
    ("oss_competition", re.compile(
        r"\bopen[\s-]source\s+competition\b", re.I,
    )),
    ("github_stars_competition", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s+github\s+stars?\b", re.I,
    )),
    ("community_edition", re.compile(
        r"\bcommunity\s+edition\b", re.I,
    )),
    ("oss_core", re.compile(
        r"\bopen[\s-]source[\s-]+first\b|\bopen[\s-]source\s+core\b", re.I,
    )),
    ("foss_threat", re.compile(
        r"\b(?:foss|free\s+open[\s-]source)\s+(?:threat|risk|alternative)\b",
        re.I,
    )),
    ("open_source_moat", re.compile(
        r"\bopen[\s-]source\s+(?:moat|advantage|defen[sc]|strategy)\b", re.I,
    )),
    ("contributing_devs", re.compile(
        r"\b\d{2,}(?:,\d{3})*\s+(?:contributing\s+)?(?:developers?|contributors?)\b",
        re.I,
    )),
    ("fork_risk", re.compile(
        r"\bfork(?:ing|ed)?\s+(?:risk|threat|concern)\b", re.I,
    )),
    ("open_vs_premium", re.compile(
        r"\bopen[\s-]source\s+(?:vs\.?|versus|compared)\s+(?:premium|paid|commercial)\b",
        re.I,
    )),
]

# ---------------------------------------------------------------------------
# Regulatory competition signals
# ---------------------------------------------------------------------------

_REGULATORY_COMPETITION_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("regulatory_advantage", re.compile(
        r"\bregulatory\s+(?:advantage|moat|barrier|head\s+start)\b", re.I,
    )),
    ("compliance_first", re.compile(
        r"\bcompliance[\s-]first\b|\bbuilt\s+for\s+compliance\b", re.I,
    )),
    ("regulatory_expertise", re.compile(
        r"\bregulatory\s+(?:expertise|knowledge|capability)\b", re.I,
    )),
    ("licensed_certified", re.compile(
        r"\b(?:licensed|certified|accredited|authorized)\s+by\b", re.I,
    )),
    ("regulatory_approval", re.compile(
        r"\bregulatory\s+(?:approval|clearance|authorization)\b", re.I,
    )),
    ("compliance_platform", re.compile(
        r"\bcompliance\s+platform\b", re.I,
    )),
    ("regulatory_moat", re.compile(
        r"\bregulatory\s+moat\b", re.I,
    )),
    ("certification_barrier", re.compile(
        r"\bcertification\s+barrier\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Geographic competition signals
# ---------------------------------------------------------------------------

_GEOGRAPHIC_COMPETITION_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("global_competition", re.compile(
        r"\bglobal\s+(?:compet|landscape|market|rival)\b", re.I,
    )),
    ("regional_leader", re.compile(
        r"\bregional\s+(?:leader|dominant|champion)\b", re.I,
    )),
    ("local_competition", re.compile(
        r"\blocal\s+(?:competitor|rival|competition)\b", re.I,
    )),
    ("international_expansion", re.compile(
        r"\binternational\s+(?:expansion|growth|market|competition)\b", re.I,
    )),
    ("cross_border", re.compile(
        r"\bcross[\s-]border\b", re.I,
    )),
    ("multi_country", re.compile(
        r"\bacross\s+\d{2,}\s*countries?\b", re.I,
    )),
    ("geographic_expansion", re.compile(
        r"\b(?:expanding|expanded)\s+(?:into\s+)?"
        r"(?:new\s+)?(?:markets?|regions?|countries?)\b",
        re.I,
    )),
    ("us_europe", re.compile(
        r"\b(?:us|united\s+states)\s+(?:and|&|,\s*)\s*(?:europe|eu)\b", re.I,
    )),
    ("asia_expansion", re.compile(
        r"\b(?:asia|apac|asia[\s-]pacific)\s+(?:market|expansion|growth)\b",
        re.I,
    )),
]

# ---------------------------------------------------------------------------
# Pricing pressure signals
# ---------------------------------------------------------------------------

_PRICING_PRESSURE_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("price_competition", re.compile(
        r"\bprice\s+competition\b|\bcompeting\s+on\s+price\b", re.I,
    )),
    ("price_war", re.compile(
        r"\bprice\s+war\b", re.I,
    )),
    ("race_to_bottom", re.compile(
        r"\brace\s+to\s+the\s+bottom\b", re.I,
    )),
    ("cost_pressure", re.compile(
        r"\bcost\s+(?:pressure|pressure|optimization)\b", re.I,
    )),
    ("pricing_disruption", re.compile(
        r"\bpricing\s+(?:disruption|innovation|model)\b", re.I,
    )),
    ("undercutting", re.compile(
        r"\bundercut(?:ting|s)?\b", re.I,
    )),
    ("margin_pressure", re.compile(
        r"\bmargin\s+pressure\b", re.I,
    )),
    ("price_sensitivity", re.compile(
        r"\bprice[\s-]sensitive\b|\bprice\s+sensitivity\b", re.I,
    )),
    ("discount_pressure", re.compile(
        r"\bdiscount\s+(?:pressure|war|strategy)\b", re.I,
    )),
    ("affordable", re.compile(
        r"\baffordable\s+(?:alternative|option|solution)\b", re.I,
    )),
    ("low_cost", re.compile(
        r"\blow[\s-]cost\s+(?:alternative|solution|option|competitor)\b", re.I,
    )),
    ("free_tier_competition", re.compile(
        r"\bfree\s+tier\b", re.I,
    )),
    ("freemium_competition", re.compile(
        r"\bfreemium\b", re.I,
    )),
]

# ---------------------------------------------------------------------------
# Competitive keywords — domain-specific terms to extract
# ---------------------------------------------------------------------------

_COMPETITIVE_KEYWORD_MAP: dict[str, list[str]] = {
    "market_dynamics": [
        "competitive landscape", "market dynamics", "competitive dynamics",
        "market structure", "industry structure", "competitive intensity",
    ],
    "positioning": [
        "positioning", "value proposition", "differentiation",
        "unique selling proposition", "usp", "competitive advantage",
    ],
    "disruption": [
        "disruption", "disruptive", "disrupting", "displacing",
        "unseating", "overtaking",
    ],
    "moat": [
        "moat", "defensible", "barriers", "switching cost",
        "lock-in", "network effect", "flywheel",
    ],
    "competition_level": [
        "blue ocean", "red ocean", "white space", "crowded",
        "saturated", "fragmented", "consolidated",
    ],
    "incumbent": [
        "incumbent", "legacy", "traditional", "established",
        "old guard", "status quo",
    ],
    "substitute": [
        "alternative", "substitute", "instead of", "replacing",
        "displacing", "do away with",
    ],
    "open_source": [
        "open-source", "open source", "foss", "community edition",
        "github stars", "contributing developers",
    ],
}


class CompetitionExtractor(BaseExtractor):
    """Extracts competitive landscape intelligence from startup data.

    Performs deterministic, explainable signal extraction across 22
    dimensions of competitive landscape assessment. Every signal
    is traceable to a specific text pattern or data field.

    Populates: direct_competitor_signals, indirect_competitor_signals,
    incumbent_signals, market_concentration, competitive_density,
    fragmentation_signals, winner_take_most_signals,
    network_effect_competition, switching_cost_signals,
    differentiation_signals, competitive_moat_indicators,
    barriers_to_entry, substitute_product_signals,
    platform_dependency, ecosystem_dependency,
    open_source_competition, regulatory_competition,
    geographic_competition, pricing_pressure,
    competitive_keywords, competition_confidence.
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        desc = startup.description
        text_lower = desc.lower()

        # Core competitive signals
        direct_competitors = self._detect_direct_competitors(desc)
        indirect_competitors = self._detect_indirect_competitors(desc)
        incumbent_signals = self._detect_incumbent_signals(desc)

        # Market structure
        concentration = self._classify_concentration(text_lower)
        density = self._classify_density(text_lower)
        fragmentation = self._detect_fragmentation_signals(desc)
        winner_take_most = self._detect_winner_take_most(desc)

        # Network effects and switching costs
        network_effect_comp = self._classify_network_effect_competition(
            text_lower
        )
        switching_costs = self._detect_switching_costs(desc)

        # Differentiation and moats
        differentiation = self._detect_differentiation_signals(desc)
        moat_indicators = self._detect_moat_indicators(desc)

        # Barriers and substitutes
        barriers = self._detect_barriers_to_entry(desc)
        substitutes = self._detect_substitute_products(desc)

        # Dependencies
        platform_dep = self._detect_platform_dependency(desc)
        ecosystem_dep = self._detect_ecosystem_dependency(desc)

        # External competition dimensions
        open_source = self._detect_open_source_competition(desc)
        regulatory = self._detect_regulatory_competition(desc)
        geographic = self._detect_geographic_competition(desc)
        pricing = self._detect_pricing_pressure(desc)

        # Keywords
        comp_keywords = self._extract_competitive_keywords(text_lower)

        # Confidence
        confidence = self._compute_competition_confidence(
            direct_competitors=direct_competitors,
            indirect_competitors=indirect_competitors,
            incumbent_signals=incumbent_signals,
            concentration=concentration,
            density=density,
            fragmentation=fragmentation,
            winner_take_most=winner_take_most,
            network_effect=network_effect_comp,
            switching_costs=switching_costs,
            differentiation=differentiation,
            moat_indicators=moat_indicators,
            barriers=barriers,
            substitutes=substitutes,
            platform_dep=platform_dep,
            ecosystem_dep=ecosystem_dep,
            open_source=open_source,
            regulatory=regulatory,
            geographic=geographic,
            pricing=pricing,
            comp_keywords=comp_keywords,
        )

        return ExtractedFeatures(
            direct_competitor_signals=direct_competitors,
            indirect_competitor_signals=indirect_competitors,
            incumbent_signals=incumbent_signals,
            market_concentration=concentration,
            competitive_density=density,
            fragmentation_signals=fragmentation,
            winner_take_most_signals=winner_take_most,
            network_effect_competition=network_effect_comp,
            switching_cost_signals=switching_costs,
            differentiation_signals=differentiation,
            competitive_moat_indicators=moat_indicators,
            barriers_to_entry=barriers,
            substitute_product_signals=substitutes,
            platform_dependency=platform_dep,
            ecosystem_dependency=ecosystem_dep,
            open_source_competition=open_source,
            regulatory_competition=regulatory,
            geographic_competition=geographic,
            pricing_pressure=pricing,
            competitive_keywords=comp_keywords,
            competition_confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Direct competitor detection
    # ------------------------------------------------------------------

    def _detect_direct_competitors(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _DIRECT_COMPETITOR_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Indirect competitor detection
    # ------------------------------------------------------------------

    def _detect_indirect_competitors(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _INDIRECT_COMPETITOR_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Incumbent signal detection
    # ------------------------------------------------------------------

    def _detect_incumbent_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _INCUMBENT_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Market concentration classification (weighted evidence)
    # ------------------------------------------------------------------

    def _classify_concentration(self, text: str) -> str | None:
        scores: dict[str, float] = {}
        for level, weighted_kws in _CONCENTRATION_WEIGHTED_KEYWORDS.items():
            total = sum(w for kw, w in weighted_kws if kw in text)
            if total > 0:
                scores[level] = total

        if not scores:
            return None

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_level, best_score = sorted_scores[0]

        if best_score < _CONCENTRATION_THRESHOLD:
            return None

        return best_level

    # ------------------------------------------------------------------
    # Competitive density classification (weighted evidence)
    # ------------------------------------------------------------------

    def _classify_density(self, text: str) -> str | None:
        scores: dict[str, float] = {}
        for level, weighted_kws in _DENSITY_WEIGHTED_KEYWORDS.items():
            total = sum(w for kw, w in weighted_kws if kw in text)
            if total > 0:
                scores[level] = total

        if not scores:
            return None

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_level, best_score = sorted_scores[0]

        if best_score < _DENSITY_THRESHOLD:
            return None

        return best_level

    # ------------------------------------------------------------------
    # Fragmentation signal detection
    # ------------------------------------------------------------------

    def _detect_fragmentation_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _FRAGMENTATION_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Winner-take-most detection
    # ------------------------------------------------------------------

    def _detect_winner_take_most(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _WINNER_TAKE_MOST_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Network effect competition classification
    # ------------------------------------------------------------------

    def _classify_network_effect_competition(self, text: str) -> str | None:
        scores: dict[str, float] = {}
        for level, weighted_kws in _NETWORK_EFFECT_COMPETITION_KEYWORDS.items():
            total = sum(w for kw, w in weighted_kws if kw in text)
            if total > 0:
                scores[level] = total

        if not scores:
            return None

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_level, best_score = sorted_scores[0]

        if best_score < _NETWORK_EFFECT_THRESHOLD:
            return None

        return best_level

    # ------------------------------------------------------------------
    # Switching cost detection
    # ------------------------------------------------------------------

    def _detect_switching_costs(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _SWITCHING_COST_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Differentiation signal detection
    # ------------------------------------------------------------------

    def _detect_differentiation_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in differential_signal_patterns:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Moat indicator detection
    # ------------------------------------------------------------------

    def _detect_moat_indicators(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _MOAT_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Barrier to entry detection
    # ------------------------------------------------------------------

    def _detect_barriers_to_entry(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _BARRIER_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Substitute product detection
    # ------------------------------------------------------------------

    def _detect_substitute_products(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _SUBSTITUTE_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Platform dependency detection
    # ------------------------------------------------------------------

    def _detect_platform_dependency(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _PLATFORM_DEPENDENCY_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Ecosystem dependency detection
    # ------------------------------------------------------------------

    def _detect_ecosystem_dependency(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _ECOSYSTEM_DEPENDENCY_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Open-source competition detection
    # ------------------------------------------------------------------

    def _detect_open_source_competition(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _OPEN_SOURCE_COMPETITION_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Regulatory competition detection
    # ------------------------------------------------------------------

    def _detect_regulatory_competition(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _REGULATORY_COMPETITION_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Geographic competition detection
    # ------------------------------------------------------------------

    def _detect_geographic_competition(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _GEOGRAPHIC_COMPETITION_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Pricing pressure detection
    # ------------------------------------------------------------------

    def _detect_pricing_pressure(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _PRICING_PRESSURE_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Competitive keyword extraction
    # ------------------------------------------------------------------

    def _extract_competitive_keywords(self, text: str) -> list[str]:
        keywords: list[str] = []
        for _category, kws in _COMPETITIVE_KEYWORD_MAP.items():
            for kw in kws:
                if kw in text and kw not in keywords:
                    keywords.append(kw)
        return keywords

    # ------------------------------------------------------------------
    # Composite competition confidence scoring
    # ------------------------------------------------------------------

    def _compute_competition_confidence(
        self,
        *,
        direct_competitors: list[str],
        indirect_competitors: list[str],
        incumbent_signals: list[str],
        concentration: str | None,
        density: str | None,
        fragmentation: list[str],
        winner_take_most: list[str],
        network_effect: str | None,
        switching_costs: list[str],
        differentiation: list[str],
        moat_indicators: list[str],
        barriers: list[str],
        substitutes: list[str],
        platform_dep: list[str],
        ecosystem_dep: list[str],
        open_source: list[str],
        regulatory: list[str],
        geographic: list[str],
        pricing: list[str],
        comp_keywords: list[str],
    ) -> float:
        score = 0.0

        # Direct competitor signals (strongest signal)
        score += min(len(direct_competitors) * 0.06, 0.18)

        # Indirect competitor signals
        score += min(len(indirect_competitors) * 0.05, 0.15)

        # Incumbent signals
        score += min(len(incumbent_signals) * 0.04, 0.12)

        # Market structure classification
        if concentration is not None:
            score += 0.08
        if density is not None:
            score += 0.06

        # Fragmentation signals
        score += min(len(fragmentation) * 0.03, 0.09)

        # Winner-take-most dynamics
        score += min(len(winner_take_most) * 0.04, 0.08)

        # Network effects
        if network_effect is not None:
            score += 0.06

        # Switching costs
        score += min(len(switching_costs) * 0.04, 0.12)

        # Differentiation
        score += min(len(differentiation) * 0.03, 0.09)

        # Moat indicators
        score += min(len(moat_indicators) * 0.04, 0.12)

        # Barriers to entry
        score += min(len(barriers) * 0.03, 0.09)

        # Substitute products
        score += min(len(substitutes) * 0.03, 0.09)

        # Platform and ecosystem dependency
        score += min(len(platform_dep) * 0.03, 0.06)
        score += min(len(ecosystem_dep) * 0.02, 0.06)

        # External competition dimensions
        score += min(len(open_source) * 0.03, 0.06)
        score += min(len(regulatory) * 0.03, 0.06)
        score += min(len(geographic) * 0.03, 0.06)
        score += min(len(pricing) * 0.03, 0.09)

        # Competitive keywords
        score += min(len(comp_keywords) * 0.01, 0.05)

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
