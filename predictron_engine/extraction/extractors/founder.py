"""Founder extractor — extracts founder and team intelligence signals.

Deterministic, explainable extraction of founder quality indicators,
team composition signals, and execution capability evidence.

Responsibilities:
  - Founder count propagation from collected data
  - Team size indicator detection from description text
  - Technical vs business founder classification
  - Domain expertise signal detection
  - Serial founder indicator detection
  - Leadership role extraction from LinkedIn URL slugs
  - Hiring and team growth signal detection
  - Advisor and board mention detection
  - Engineering strength assessment
  - Product strength assessment
  - Founder-market fit signal detection
  - Execution and traction signal detection
  - Composite founder confidence scoring
"""

from __future__ import annotations

import re
from typing import Final

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

# ---------------------------------------------------------------------------
# Team size detection
# ---------------------------------------------------------------------------

_TEAM_SIZE_NUMERIC: list[tuple[str, int, int]] = [
    ("1-10", 1, 10),
    ("11-50", 11, 50),
    ("51-200", 51, 200),
    ("200+", 201, 999_999),
]

_TEAM_SIZE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:team of|employees?|people|staff|members)"
    r"\s+(?:about\s+|approximately\s+|roughly\s+|over\s+|around\s+)?"
    r"(\d{1,6})\b",
    re.I,
)

# Also match "N-person team", "N engineer team", etc.
_TEAM_COMPOSITION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(\d{1,4})\s*[-\s]?\s*(?:person|engineer|developer|member)s?\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Leadership role keywords (extracted from LinkedIn URL slugs)
# ---------------------------------------------------------------------------

_ROLE_KEYWORDS: dict[str, list[str]] = {
    "CEO": ["ceo", "chief-executive", "founder"],
    "CTO": ["cto", "chief-technical", "chief-technology"],
    "COO": ["coo", "chief-operating"],
    "CFO": ["cfo", "chief-financial"],
    "VP-Engineering": ["vp-eng", "vp-engineering", "vice-president-engineering"],
    "VP-Product": ["vp-product", "vice-president-product"],
    "VP-Sales": ["vp-sales", "vice-president-sales"],
    "Head of Engineering": ["head-of-engineering", "head-engineering"],
    "Head of Product": ["head-of-product", "head-product"],
    "Co-Founder": ["co-founder", "cofounder"],
}

# ---------------------------------------------------------------------------
# Technical vs business classification signals
# ---------------------------------------------------------------------------

_TECHNICAL_SIGNALS: Final[list[str]] = [
    "engineers", "engineering", "technical", "developers", "code",
    "architecture", "infrastructure", "platform", "api", "open-source",
    "github", "software", "data science", "machine learning", "ai",
    "algorithm", "systems", "backend", "frontend", "full-stack",
    "devops", "cloud", "security", "database", "ml", "gpu",
    "build", "deploy", "microservices", "kubernetes", "docker",
]

_BUSINESS_SIGNALS: Final[list[str]] = [
    "sales", "marketing", "revenue", "customers", "partnerships",
    "business development", "accounts", "enterprise", "commercial",
    "gtm", "growth", "pipeline", "channel", "reseller",
    "strategy", "operations", "finance", "legal", "compliance",
    "hr", "recruiting", "talent", "onboarding", "accounting",
]

# ---------------------------------------------------------------------------
# Serial founder indicators
# ---------------------------------------------------------------------------

_SERIAL_FOUNDER_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\b(?:founded|co-founded|co-founded|started)\b", re.I),
    re.compile(r"\bserial\s+(?:entrepreneur|founder|ceo)\b", re.I),
    re.compile(r"\bprevious(?:ly)?\s+(?:founded|started)\b", re.I),
    re.compile(r"\bex[-\s](?:founder|entrepreneur|ceo)\b", re.I),
    re.compile(r"\by(?:et)?\s+another\b.*\b(?:startup|company|venture)\b", re.I),
    re.compile(r"\bfounding\s+(?:team|member|ceo|cto)\b", re.I),
    re.compile(r"\bmultiple\s+(?:startups|companies|ventures)\b", re.I),
]

# ---------------------------------------------------------------------------
# Domain expertise signals
# ---------------------------------------------------------------------------

_DOMAIN_EXPERTISE_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("healthcare domain expertise", re.compile(r"\b(?:healthcare|medical|clinical|biotech|pharma|diagnostic)\b.*(?:expert|specialist|background|experience|domain)", re.I)),
    ("fintech domain expertise", re.compile(r"\b(?:fintech|financial|banking|payments|lending|insurance)\b.*(?:expert|specialist|background|experience|domain)", re.I)),
    ("enterprise domain expertise", re.compile(r"\b(?:enterprise|b2b|smb|corporate)\b.*(?:expert|specialist|background|experience|domain)", re.I)),
    ("AI/ML domain expertise", re.compile(r"\b(?:ai|ml|machine learning|deep learning|nlp|computer vision|neural)\b.*(?:expert|specialist|background|experience|domain|research)", re.I)),
    ("hardware domain expertise", re.compile(r"\b(?:hardware|robotics|iot|embedded|sensor|lidar|automation)\b.*(?:expert|specialist|background|experience|domain)", re.I)),
    ("climate domain expertise", re.compile(r"\b(?:climate|sustainability|carbon|emissions|renewable|clean.?tech|energy)\b.*(?:expert|specialist|background|experience|domain)", re.I)),
    ("enterprise compliance expertise", re.compile(r"\b(?:compliance|regulatory|soc.?2|pci|gdpr|ccpa|audit)\b.*(?:expert|specialist|background|experience|domain)", re.I)),
    ("cloud infrastructure expertise", re.compile(r"\b(?:cloud|infrastructure|devops|sre|platform)\b.*(?:expert|specialist|background|experience|domain)", re.I)),
]

# ---------------------------------------------------------------------------
# Hiring signals
# ---------------------------------------------------------------------------

_HIRING_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\b(?:hiring|recruiting|seeking|looking for)\b", re.I),
    re.compile(r"\bteam\s+of\s+\d{2,}", re.I),
    re.compile(r"\bgrowing\s+(?:team|rapidly|quickly)\b", re.I),
    re.compile(r"\bjoin(?:ing)?\s+(?:our|the|a)\s+team\b", re.I),
    re.compile(r"\bopen\s+(?:roles|positions|headcount)\b", re.I),
    re.compile(r"\b\d{2,}\s+(?:engineers?|developers?|employees?)\b", re.I),
]

# ---------------------------------------------------------------------------
# Advisor patterns
# ---------------------------------------------------------------------------

_ADVISOR_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\badvisor[s]?\b", re.I),
    re.compile(r"\badvisory\s+(?:board|panel|committee)\b", re.I),
    re.compile(r"\bboard\s+of\s+directors?\b", re.I),
    re.compile(r"\bmentor[s]?\b", re.I),
    re.compile(r"\bsenior\s+advisor[s]?\b", re.I),
]

# ---------------------------------------------------------------------------
# Engineering strength signals
# ---------------------------------------------------------------------------

_ENG_STRONG_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\b\d{2,}\s+engineers?\b", re.I),
    re.compile(r"\bengineering\s+team\s+of\s+\d{2,}\b", re.I),
    re.compile(r"\bopen[-\s]source\b.*\b\d{3,}\s*(?:github|star|contrib)", re.I),
    re.compile(r"\b\d{3,}\s*github\s+stars?\b", re.I),
    re.compile(r"\bdistributed\s+engineering\b", re.I),
    re.compile(r"\bproprietary\s+(?:technology|platform|engine|architecture)\b", re.I),
]

_ENG_MODERATE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\bengineering\s+team\b", re.I),
    re.compile(r"\btechnical\s+team\b", re.I),
    re.compile(r"\bsoftware\s+(?:platform|product|solution)\b", re.I),
    re.compile(r"\bapi[-\s]first\b", re.I),
    re.compile(r"\bmicroservices?\b", re.I),
    re.compile(r"\bcloud[-\s]native\b", re.I),
]

# ---------------------------------------------------------------------------
# Product strength signals
# ---------------------------------------------------------------------------

_PRODUCT_STRONG_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\bpatent[sd]?\b", re.I),
    re.compile(r"\bproprietary\s+(?:technology|algorithm|platform|engine)\b", re.I),
    re.compile(r"\b\d{3,}\s*(?:users?|customers?|clients?)\b", re.I),
    re.compile(r"\b\d{1,}\s*(?:million|billion)\b.*\b(?:users?|customers?|requests?)\b", re.I),
    re.compile(r"\b(?:series\s+[a-d]|raised\s+\$\d+[mb])\b", re.I),
    re.compile(r"\bregulatory\s+(?:clearance|approval|certification)\b", re.I),
    re.compile(r"\bfda[-\s]cleared?\b", re.I),
]

_PRODUCT_MODERATE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\bplatform\b", re.I),
    re.compile(r"\bproduct\b", re.I),
    re.compile(r"\blaunch(?:ed)?\b", re.I),
    re.compile(r"\bbeta\b", re.I),
    re.compile(r"\bmobile\s+app\b", re.I),
    re.compile(r"\bsaas\b", re.I),
]

# ---------------------------------------------------------------------------
# Execution/traction signals
# ---------------------------------------------------------------------------

_EXECUTION_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("arr_revenue", re.compile(r"\$\d[\d,.]*\s*(?:m|mm|b|bn)?\s*arr\b", re.I)),
    ("mrr_revenue", re.compile(r"\$\d[\d,.]*\s*(?:k|m)?\s*mrr\b", re.I)),
    ("revenue_mention", re.compile(r"\b(?:revenue|generating|earning|profitable)\b", re.I)),
    ("customer_count", re.compile(r"\b\d{2,}\s*(?:enterprise\s+)?(?:customers?|clients?|accounts?|hospitals?)\b", re.I)),
    ("gmv", re.compile(r"\$\d[\d,.]*\s*(?:m|b)?\s*gmv\b", re.I)),
    ("user_count", re.compile(r"\b\d{2,}(?:,\d{3})*\s*(?:users?|mau|dau|downloads?)\b", re.I)),
    ("retention_metric", re.compile(r"\b\d{2,}%\s*(?:net\s+)?(?:revenue\s+)?retention\b", re.I)),
    ("ltv_cac", re.compile(r"\bltv[/\s]cac\b", re.I)),
    ("funding_raised", re.compile(r"\braised\s+\$\d[\d,.]*\s*(?:m|b)?\b", re.I)),
    ("funding_stage", re.compile(r"\b(?:seed|pre-seed|series\s+[a-d]|ipo)\s+(?:stage|round|funded)?\b", re.I)),
    ("arr_figure", re.compile(r"\barr\s+(?:of\s+)?\$\d[\d,.]*\s*(?:m|b)?\b", re.I)),
    ("processing_volume", re.compile(r"\bprocessing\s+\$[\d,.]*\s*(?:m|b|t)?\b", re.I)),
    ("deployed_scale", re.compile(r"\b\d{2,}\s*(?:deployed|active|live)\b", re.I)),
    ("patent_count", re.compile(r"\b\d+\s+patent[s]?\b", re.I)),
    ("founding_year", re.compile(r"\bfounded\s+in\s+\d{4}\b", re.I)),
]

# ---------------------------------------------------------------------------
# Founder-market fit patterns
# ---------------------------------------------------------------------------

_FOUNDER_MARKET_FIT_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\b(?:former|ex|previously)\s+.*(?:engineer|scientist|architect|lead)\b.*\b(?:at|from)\b.*(?:google|meta|amazon|apple|microsoft|netflix|uber|stripe|airbnb|github|openai|deepmind)", re.I),
    re.compile(r"\b(?:infrastructure|platform)\s+leads?\s+from\s+major\b", re.I),
    re.compile(r"\bex[-\s](?:github|google|meta|amazon|apple|microsoft|netflix|uber|stripe|airbnb|openai)\b", re.I),
    re.compile(r"\bformer(?:ly)?\s+at\s+(?:google|meta|amazon|apple|microsoft|netflix|uber|stripe|airbnb|github|openai)\b", re.I),
    re.compile(r"\bfrom\s+(?:major|leading|top)\s+(?:ai|tech|cloud)\s+labs?\b", re.I),
    re.compile(r"\bex[-\s]?(?:faang|big\s+tech)\b", re.I),
]


class FounderExtractor(BaseExtractor):
    """Extracts founder and team intelligence from startup data.

    Performs deterministic, explainable signal extraction across 13
    dimensions of founder/team quality assessment. Every signal is
    traceable to a specific text pattern or data field.
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        desc = startup.description
        linkedin_urls = startup.founder_linkedin_urls
        founder_count = data.founder_count

        team_size = self._detect_team_size(desc)
        team_size_numeric = self._detect_team_size_numeric(desc)
        team_type = self._classify_team_type(desc, linkedin_urls)
        domain_expertise = self._detect_domain_expertise(desc)
        serial_indicators = self._detect_serial_founder(desc)
        leadership = self._extract_leadership_roles(linkedin_urls)
        hiring = self._detect_hiring_signals(desc)
        advisors = self._detect_advisor_mentions(desc)
        eng_strength = self._assess_engineering_strength(desc)
        prod_strength = self._assess_product_strength(desc)
        market_fit = self._detect_founder_market_fit(desc, linkedin_urls)
        execution = self._detect_execution_signals(desc)
        confidence = self._compute_founder_confidence(
            founder_count=founder_count,
            team_type=team_type,
            domain_expertise=domain_expertise,
            serial_indicators=serial_indicators,
            leadership=leadership,
            eng_strength=eng_strength,
            prod_strength=prod_strength,
            execution=execution,
            market_fit=market_fit,
        )

        return ExtractedFeatures(
            founder_profile_count=founder_count,
            team_size_indicator=team_size,
            team_size_numeric=team_size_numeric,
            founder_team_type=team_type,
            domain_expertise_signals=domain_expertise,
            serial_founder_indicators=serial_indicators,
            leadership_roles=leadership,
            hiring_signals=hiring,
            advisor_mentions=advisors,
            engineering_strength=eng_strength,
            product_strength=prod_strength,
            founder_market_fit_signals=market_fit,
            execution_signals=execution,
            founder_confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Team size detection
    # ------------------------------------------------------------------

    def _detect_team_size(self, text: str) -> str | None:
        match = _TEAM_SIZE_PATTERN.search(text)
        if match:
            count = int(match.group(1))
            return _label_for_count(count)

        comp_match = _TEAM_COMPOSITION_PATTERN.search(text)
        if comp_match:
            count = int(comp_match.group(1))
            if count >= 3:
                return _label_for_count(count)

        return None

    def _detect_team_size_numeric(self, text: str) -> int | None:
        """Extract team size as a numeric integer value."""
        match = _TEAM_SIZE_PATTERN.search(text)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, TypeError):
                pass

        comp_match = _TEAM_COMPOSITION_PATTERN.search(text)
        if comp_match:
            try:
                count = int(comp_match.group(1))
                if count >= 3:
                    return count
            except (ValueError, TypeError):
                pass

        return None

    # ------------------------------------------------------------------
    # Technical vs business classification
    # ------------------------------------------------------------------

    def _classify_team_type(
        self, description: str, linkedin_urls: list[str]
    ) -> str | None:
        text_lower = description.lower()

        tech_score = sum(1 for s in _TECHNICAL_SIGNALS if s in text_lower)
        biz_score = sum(1 for s in _BUSINESS_SIGNALS if s in text_lower)

        for url in linkedin_urls:
            slug = url.split("/")[-1].lower() if "/" in url else ""
            if any(k in slug for k in ["cto", "vp-eng", "engineer", "architect", "head-of-engineering"]):
                tech_score += 3
            if any(k in slug for k in ["ceo", "coo", "cfo", "vp-sales", "head-of-business"]):
                biz_score += 2

        if tech_score > biz_score and tech_score >= 2:
            return "technical"
        if biz_score > tech_score and biz_score >= 2:
            return "business"
        if tech_score >= 1 and biz_score >= 1:
            return "mixed"
        if tech_score >= 1:
            return "technical"
        if biz_score >= 1:
            return "business"
        return None

    # ------------------------------------------------------------------
    # Domain expertise detection
    # ------------------------------------------------------------------

    def _detect_domain_expertise(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _DOMAIN_EXPERTISE_PATTERNS:
            if pattern.search(text):
                signals.append(label)
        return signals

    # ------------------------------------------------------------------
    # Serial founder indicators
    # ------------------------------------------------------------------

    def _detect_serial_founder(self, text: str) -> list[str]:
        indicators: list[str] = []
        for pattern in _SERIAL_FOUNDER_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 40)
                if snippet not in indicators:
                    indicators.append(snippet)
        return indicators

    # ------------------------------------------------------------------
    # Leadership role extraction
    # ------------------------------------------------------------------

    def _extract_leadership_roles(self, linkedin_urls: list[str]) -> list[str]:
        roles: list[str] = []
        for url in linkedin_urls:
            slug = url.rstrip("/").split("/")[-1].lower() if "/" in url else ""
            for role_name, keywords in _ROLE_KEYWORDS.items():
                if any(kw in slug for kw in keywords):
                    if role_name not in roles:
                        roles.append(role_name)
        return roles

    # ------------------------------------------------------------------
    # Hiring signals
    # ------------------------------------------------------------------

    def _detect_hiring_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for pattern in _HIRING_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                if snippet not in signals:
                    signals.append(snippet)
        return signals

    # ------------------------------------------------------------------
    # Advisor mentions
    # ------------------------------------------------------------------

    def _detect_advisor_mentions(self, text: str) -> list[str]:
        mentions: list[str] = []
        for pattern in _ADVISOR_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 30)
                if snippet not in mentions:
                    mentions.append(snippet)
        return mentions

    # ------------------------------------------------------------------
    # Engineering strength assessment
    # ------------------------------------------------------------------

    def _assess_engineering_strength(self, text: str) -> str | None:
        strong = sum(1 for p in _ENG_STRONG_PATTERNS if p.search(text))
        moderate = sum(1 for p in _ENG_MODERATE_PATTERNS if p.search(text))

        if strong >= 2:
            return "strong"
        if strong >= 1 and moderate >= 2:
            return "strong"
        if moderate >= 4:
            return "strong"
        if moderate >= 1:
            return "moderate"
        return None

    # ------------------------------------------------------------------
    # Product strength assessment
    # ------------------------------------------------------------------

    def _assess_product_strength(self, text: str) -> str | None:
        strong = sum(1 for p in _PRODUCT_STRONG_PATTERNS if p.search(text))
        moderate = sum(1 for p in _PRODUCT_MODERATE_PATTERNS if p.search(text))

        if strong >= 2:
            return "strong"
        if strong >= 1 and moderate >= 2:
            return "strong"
        if moderate >= 5:
            return "strong"
        if moderate >= 1:
            return "moderate"
        return None

    # ------------------------------------------------------------------
    # Founder-market fit detection
    # ------------------------------------------------------------------

    def _detect_founder_market_fit(
        self, text: str, linkedin_urls: list[str]
    ) -> list[str]:
        signals: list[str] = []
        combined = text + " " + " ".join(linkedin_urls)
        for pattern in _FOUNDER_MARKET_FIT_PATTERNS:
            match = pattern.search(combined)
            if match:
                snippet = _extract_snippet(combined, match.start(), match.end(), 50)
                if snippet not in signals:
                    signals.append(snippet)
        return signals

    # ------------------------------------------------------------------
    # Execution/traction signal detection
    # ------------------------------------------------------------------

    def _detect_execution_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        for label, pattern in _EXECUTION_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end(), 40)
                signals.append(f"{label}: {snippet}")
        return signals

    # ------------------------------------------------------------------
    # Composite founder confidence scoring
    # ------------------------------------------------------------------

    def _compute_founder_confidence(
        self,
        *,
        founder_count: int,
        team_type: str | None,
        domain_expertise: list[str],
        serial_indicators: list[str],
        leadership: list[str],
        eng_strength: str | None,
        prod_strength: str | None,
        execution: list[str],
        market_fit: list[str],
    ) -> float:
        score = 0.0

        if founder_count >= 2:
            score += 0.15
        elif founder_count == 1:
            score += 0.10

        if team_type in ("technical", "mixed"):
            score += 0.08
        elif team_type == "business":
            score += 0.05

        score += min(len(domain_expertise) * 0.06, 0.18)
        score += min(len(serial_indicators) * 0.05, 0.10)
        score += min(len(leadership) * 0.03, 0.12)

        if eng_strength == "strong":
            score += 0.12
        elif eng_strength == "moderate":
            score += 0.06

        if prod_strength == "strong":
            score += 0.10
        elif prod_strength == "moderate":
            score += 0.05

        score += min(len(execution) * 0.03, 0.15)
        score += min(len(market_fit) * 0.05, 0.10)

        return round(min(score, 1.0), 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_for_count(count: int) -> str:
    for label, lo, hi in _TEAM_SIZE_NUMERIC:
        if lo <= count <= hi:
            return label
    return "200+"


def _extract_snippet(text: str, start: int, end: int, ctx: int) -> str:
    snippet_start = max(0, start - ctx)
    snippet_end = min(len(text), end + ctx)
    snippet = text[snippet_start:snippet_end].strip()
    if snippet_start > 0:
        snippet = "..." + snippet
    if snippet_end < len(text):
        snippet = snippet + "..."
    return snippet
