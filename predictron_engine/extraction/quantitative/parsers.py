"""Reusable numeric extraction parsers from text.

All functions are deterministic, pure, and return None when no valid
match is found. They never raise exceptions.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Multiplier suffix mapping
# ---------------------------------------------------------------------------

_MULTIPLIERS: dict[str, float] = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
}

# ---------------------------------------------------------------------------
# Core regex patterns
# ---------------------------------------------------------------------------

_DOLLAR_PATTERN = re.compile(
    r"\$\s*([\d][\d,]*\.?\d*)\s*(k|m|b|t|thousand|million|billion|trillion)?\b",
    re.I,
)

_PERCENTAGE_PATTERN = re.compile(
    r"([\d][\d,]*\.?\d*)\s*%",
)

_INTEGER_PATTERN = re.compile(
    r"\b([\d][\d,]*)\b",
)

_YEAR_PATTERN = re.compile(
    r"(?:founded|established|started|incorporated|est\.?|launched|founding\s+year)"
    r"\s+(?:in\s+)?(\d{4})",
    re.I,
)

# Standalone year pattern (4 digits between 1900-2100)
_STANDALONE_YEAR = re.compile(
    r"\b(19\d{2}|20[0-2]\d|2100)\b",
)

# Growth rate patterns
_GROWTH_MOM_PATTERN = re.compile(
    r"([\d][\d,]*\.?\d*)\s*%?\s*(?:mom|month[\s-]over[\s-]month)\s+growth",
    re.I,
)

_GROWTH_YOY_PATTERN = re.compile(
    r"([\d][\d,]*\.?\d*)\s*%?\s*(?:yoy|year[\s-]over[\s-]year)\s+growth",
    re.I,
)

_GROWTH_GENERIC_PATTERN = re.compile(
    r"(?:growth|growing|grew)\s+(?:of\s+|at\s+|rate\s+of\s+)?"
    r"([\d][\d,]*\.?\d*)\s*%",
    re.I,
)

# Customer / user count patterns
_CUSTOMER_COUNT_PATTERN = re.compile(
    r"([\d][\d,]*\.?\d*)\s*(k|m|b|thousand|million|billion)?\s*"
    r"(?:active\s+)?(?:customers?|clients?|accounts?|enterprises?|hospitals?|"
    r"institutions?|buyers?|suppliers?|sellers?)",
    re.I,
)

_ACTIVE_USER_PATTERN = re.compile(
    r"([\d][\d,]*\.?\d*)\s*(k|m|b|thousand|million|billion)?\s*"
    r"(?:monthly|daily)?\s*active\s+users?\b",
    re.I,
)

_ACTIVE_USER_REVERSED = re.compile(
    r"(?:mau|dau|monthly\s+active|daily\s+active)\s*"
    r"(?:of\s+|at\s+|reaching\s+)?([\d][\d,]*\.?\d*)\s*"
    r"(k|m|b|thousand|million|billion)?",
    re.I,
)

_USER_DOWNLOAD_PATTERN = re.compile(
    r"([\d][\d,]*\.?\d*)\s*(k|m|b|thousand|million|billion)?\s*"
    r"(?:users?|downloads?|installs?)",
    re.I,
)


# ---------------------------------------------------------------------------
# Primitive parsers
# ---------------------------------------------------------------------------


def _normalize_comma_number(raw: str) -> float:
    """Strip commas from a number string and return float."""
    return float(raw.replace(",", ""))


def _apply_multiplier(value: float, suffix: str | None) -> float:
    """Apply a multiplier suffix to a numeric value."""
    if suffix is None:
        return value
    mult = _MULTIPLIERS.get(suffix.lower())
    if mult is not None:
        return value * mult
    return value


def parse_dollar_amount(text: str) -> float | None:
    """Extract the first dollar amount from text, normalized to USD.

    Examples:
        "$12M" -> 12_000_000.0
        "$45.5 million" -> 45_500_000.0
        "$1.2B" -> 1_200_000_000.0
        "$500" -> 500.0
        "$50K" -> 50_000.0
    """
    match = _DOLLAR_PATTERN.search(text)
    if not match:
        return None
    try:
        value = _normalize_comma_number(match.group(1))
        suffix = match.group(2)
        return _apply_multiplier(value, suffix)
    except (ValueError, TypeError):
        return None


def parse_percentage(text: str) -> float | None:
    """Extract the first percentage from text.

    Returns value on 0-100 scale (e.g. "42%" -> 42.0).
    """
    match = _PERCENTAGE_PATTERN.search(text)
    if not match:
        return None
    try:
        return _normalize_comma_number(match.group(1))
    except (ValueError, TypeError):
        return None


def parse_integer(text: str) -> int | None:
    """Extract the first integer from text.

    Strips commas from numbers. Returns None if no integer found.
    """
    match = _INTEGER_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(_normalize_comma_number(match.group(1)))
    except (ValueError, TypeError):
        return None


def parse_year(text: str) -> int | None:
    """Extract a founding year from text.

    Prefers explicit 'founded in YYYY' patterns. Falls back to
    standalone 4-digit year if no explicit context found.
    """
    match = _YEAR_PATTERN.search(text)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= 2100:
            return year

    standalone = _STANDALONE_YEAR.search(text)
    if standalone:
        year = int(standalone.group(1))
        if 1900 <= year <= 2100:
            return year
    return None


# ---------------------------------------------------------------------------
# Domain-specific parsers
# ---------------------------------------------------------------------------


def parse_funding_amount(text: str) -> float | None:
    """Extract a funding amount from text.

    Matches patterns like "$12M raised", "$45M Series B", "raised $1.2B".
    Returns the first dollar amount found, normalized to USD.
    """
    patterns = [
        re.compile(
            r"\braised\s+\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\b",
            re.I,
        ),
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s+raised\b",
            re.I,
        ),
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s+"
            r"(?:raised\s+)?(?:in\s+)?"
            r"(?:series\s+[a-e]|seed|pre-seed|round)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:total|combined)\s+(?:funding|raised|capital)\s+"
            r"(?:of\s+)?\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\b",
            re.I,
        ),
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s+in\s+total\s+funding\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = _normalize_comma_number(match.group(1))
                suffix = match.group(2)
                return _apply_multiplier(value, suffix)
            except (ValueError, TypeError):
                continue
    return None


def parse_arr(text: str) -> float | None:
    """Extract ARR (Annual Recurring Revenue) from text.

    Matches patterns like "$4.2M ARR", "ARR of $1.2B", "$500K in ARR",
    "ARR has reached $5M", "ARR stands at $3M",
    "Annual recurring revenue of $10M".
    """
    patterns = [
        re.compile(
            r"\barr\s+(?:has\s+)?(?:reached|surpassed|exceeded|grown\s+to)\s+"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\b",
            re.I,
        ),
        re.compile(
            r"\barr\s+(?:of\s+|at\s+|stands\s+at\s+|reaching\s+|exceeding\s+)?"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\b",
            re.I,
        ),
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s*(?:in\s+)?arr\b",
            re.I,
        ),
        re.compile(
            r"\b(?:annual\s+recurring\s+revenue)\s+"
            r"(?:of\s+|at\s+|reaching\s+)?"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = _normalize_comma_number(match.group(1))
                suffix = match.group(2)
                return _apply_multiplier(value, suffix)
            except (ValueError, TypeError):
                continue
    return None


def parse_mrr(text: str) -> float | None:
    """Extract MRR (Monthly Recurring Revenue) from text.

    Matches patterns like "$50K MRR", "MRR of $120,000".
    """
    patterns = [
        re.compile(
            r"\bmrr\s+(?:of\s+|at\s+|reaching\s+)?"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\b",
            re.I,
        ),
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s+mrr\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = _normalize_comma_number(match.group(1))
                suffix = match.group(2)
                return _apply_multiplier(value, suffix)
            except (ValueError, TypeError):
                continue
    return None


def parse_gmv(text: str) -> float | None:
    """Extract GMV (Gross Merchandise Volume) from text.

    Matches patterns like "$500M GMV", "$1.2B in GMV".
    """
    patterns = [
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s*(?:in\s+)?gmv\b",
            re.I,
        ),
        re.compile(
            r"\bgmv\s+(?:of\s+)?\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = _normalize_comma_number(match.group(1))
                suffix = match.group(2)
                return _apply_multiplier(value, suffix)
            except (ValueError, TypeError):
                continue
    return None


def parse_customer_count(text: str) -> int | None:
    """Extract customer count from text.

    Matches patterns like "500 enterprise customers", "1,000 clients".
    """
    match = _CUSTOMER_COUNT_PATTERN.search(text)
    if not match:
        return None
    try:
        value = _normalize_comma_number(match.group(1))
        suffix = match.group(2)
        final = _apply_multiplier(value, suffix)
        return int(final)
    except (ValueError, TypeError):
        return None


def parse_active_user_count(text: str) -> int | None:
    """Extract active user count (MAU/DAU) from text.

    Matches patterns like "10,000 MAU", "500K monthly active users".
    """
    for pattern in (_ACTIVE_USER_PATTERN, _ACTIVE_USER_REVERSED, _USER_DOWNLOAD_PATTERN):
        match = pattern.search(text)
        if match:
            try:
                value = _normalize_comma_number(match.group(1))
                suffix = match.group(2)
                final = _apply_multiplier(value, suffix)
                return int(final)
            except (ValueError, TypeError):
                continue
    return None


def parse_growth_rate(text: str) -> float | None:
    """Extract a growth rate percentage from text.

    Matches MoM and YoY growth rates. Returns percentage on 0-100 scale.
    """
    for pattern in (_GROWTH_MOM_PATTERN, _GROWTH_YOY_PATTERN, _GROWTH_GENERIC_PATTERN):
        match = pattern.search(text)
        if match:
            try:
                return _normalize_comma_number(match.group(1))
            except (ValueError, TypeError):
                continue
    return None


def parse_market_size(text: str) -> float | None:
    """Extract market size (TAM/SAM/SOM) from text.

    Matches patterns like "$50 billion TAM", "$1T market", "$200M SAM",
    "TAM of $10B", "$5B market", "$200M addressable market".
    """
    patterns = [
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s*"
            r"(?:total\s+)?(?:addressable|serviceable|obtainable)?\s*"
            r"(?:market|tam|sam|som)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:tam|sam|som|market\s+size)\s+(?:of\s+)?"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\b",
            re.I,
        ),
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s+"
            r"(?:addressable\s+)?market\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = _normalize_comma_number(match.group(1))
                suffix = match.group(2)
                return _apply_multiplier(value, suffix)
            except (ValueError, TypeError):
                continue
    return None


# ---------------------------------------------------------------------------
# Domain-specific parsers — Sprint 14.1 additions
# ---------------------------------------------------------------------------

def parse_valuation(text: str) -> float | None:
    """Extract company valuation from text.

    Matches patterns like:
        "$220M valuation", "valued at $1.2B",
        "$42M raised at a $220M valuation",
        "pre-money valuation of $500M",
        "post-money valuation $300M".

    Returns the valuation amount (not the funding amount) in USD.
    """
    patterns = [
        # "$220M valuation" or "$220M pre-money valuation"
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s+"
            r"(?:pre[\s-]money|post[\s-]money)?\s*valuation\b",
            re.I,
        ),
        # "valued at $1.2B"
        re.compile(
            r"\bvalued\s+at\s+"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\b",
            re.I,
        ),
        # "valuation of $500M"
        re.compile(
            r"\bvaluation\s+(?:of\s+|at\s+)?"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\b",
            re.I,
        ),
        # "$42M raised at a $220M valuation"
        re.compile(
            r"\$\s*[\d][\d,]*\.?\d*\s*"
            r"(?:k|m|b|t|thousand|million|billion|trillion)?\s+"
            r"raised\s+(?:at\s+(?:a|an)?\s+)?"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s+valuation\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = _normalize_comma_number(match.group(1))
                suffix = match.group(2)
                return _apply_multiplier(value, suffix)
            except (ValueError, TypeError):
                continue
    return None


def parse_nrr(text: str) -> float | None:
    """Extract Net Revenue Retention percentage from text.

    Matches patterns like "NRR of 120%", "130% net revenue retention",
    "Net revenue retention is 115%", "NRR stands at 125%".
    Returns percentage on 0-100+ scale.
    """
    patterns = [
        re.compile(
            r"\bnrr\s+(?:of\s+|at\s+|stands\s+at\s+|is\s+)?"
            r"([\d][\d,]*\.?\d*)\s*%",
            re.I,
        ),
        re.compile(
            r"\bnrr\s+(?:of\s+|at\s+|stands\s+at\s+|is\s+)?"
            r"([\d][\d,]*\.?\d*)\b",
            re.I,
        ),
        re.compile(
            r"\b([\d][\d,]*\.?\d*)\s*%?\s*net\s+(?:revenue\s+)?retention\b",
            re.I,
        ),
        re.compile(
            r"\bnet\s+(?:revenue\s+)?retention\s+"
            r"(?:stands\s+at\s+|is\s+|of\s+|at\s+)?"
            r"([\d][\d,]*\.?\d*)\s*%",
            re.I,
        ),
        re.compile(
            r"\bnet\s+(?:revenue\s+)?retention\s+"
            r"(?:stands\s+at\s+|is\s+|of\s+|at\s+)?"
            r"([\d][\d,]*\.?\d*)\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return _normalize_comma_number(match.group(1))
            except (ValueError, TypeError):
                continue
    return None


def parse_churn(text: str) -> float | None:
    """Extract churn rate percentage from text.

    Matches patterns like "churn is 5%", "churn rate of 3.2%",
    "monthly churn of 2%", "churn is below 4%",
    "churned at a rate of 8%".
    Returns percentage on 0-100 scale.
    """
    patterns = [
        re.compile(
            r"\bchurn(?:ed)?\s+(?:is\s+(?:below\s+|under\s+|at\s+)?)?"
            r"(\d{1,2}(?:\.\d+)?)\s*%",
            re.I,
        ),
        re.compile(
            r"\bchurn\s+rate\s+(?:of\s+|at\s+|is\s+)?"
            r"(\d{1,2}(?:\.\d+)?)\s*%",
            re.I,
        ),
        re.compile(
            r"\b(?:monthly\s+)?churn\s+(?:of\s+|at\s+|is\s+)?"
            r"(\d{1,2}(?:\.\d+)?)\s*%",
            re.I,
        ),
        re.compile(
            r"\bchurned?\s+(?:at\s+(?:a\s+)?)?rate\s+(?:of\s+|at\s+)?"
            r"(\d{1,2}(?:\.\d+)?)\s*%",
            re.I,
        ),
        re.compile(
            r"\b(\d{1,2}(?:\.\d+)?)\s*%\s*(?:monthly\s+)?churn(?:ed)?\s+rate\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return _normalize_comma_number(match.group(1))
            except (ValueError, TypeError):
                continue
    return None


def parse_burn_rate(text: str) -> float | None:
    """Extract monthly burn rate from text.

    Matches patterns like "$500K monthly burn", "burn rate of $1.2M",
    "burning $200K per month".
    Returns USD amount.
    """
    patterns = [
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s*"
            r"(?:monthly\s+)?burn(?:\s+rate)?\b",
            re.I,
        ),
        re.compile(
            r"\bburn\s+rate\s+(?:of\s+|at\s+|is\s+)?"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\b",
            re.I,
        ),
        re.compile(
            r"\bburning\s+"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|b|t|thousand|million|billion|trillion)?\s*"
            r"(?:per\s+)?month(?:ly)?\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = _normalize_comma_number(match.group(1))
                suffix = match.group(2)
                return _apply_multiplier(value, suffix)
            except (ValueError, TypeError):
                continue
    return None


def parse_runway(text: str) -> int | None:
    """Extract runway in months from text.

    Matches patterns like "18 months runway", "runway of 24 months",
    "12-month runway", "24 months of runway".
    Returns integer months.
    """
    patterns = [
        re.compile(
            r"([\d][\d,]*)\s*[\-]?\s*months?\s+(?:of\s+)?runway\b",
            re.I,
        ),
        re.compile(
            r"\brunway\s+(?:of\s+|at\s+|is\s+)?"
            r"([\d][\d,]*)\s*[\-]?\s*months?\b",
            re.I,
        ),
        re.compile(
            r"([\d][\d,]*)\s*[\-]?\s*months?\s+runway\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return int(_normalize_comma_number(match.group(1)))
            except (ValueError, TypeError):
                continue
    return None


def parse_cac(text: str) -> float | None:
    """Extract Customer Acquisition Cost from text.

    Matches patterns like "CAC of $500", "$1,200 CAC",
    "customer acquisition cost is $800".
    Returns USD amount.
    """
    patterns = [
        re.compile(
            r"\bcac\s+(?:of\s+|at\s+|is\s+|averaging\s+)?"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|thousand|million)?\b",
            re.I,
        ),
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|thousand|million)?\s*(?:in\s+)?cac\b",
            re.I,
        ),
        re.compile(
            r"\bcustomer\s+acquisition\s+cost\s+"
            r"(?:of\s+|is\s+|at\s+)?"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|thousand|million)?\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = _normalize_comma_number(match.group(1))
                suffix = match.group(2)
                return _apply_multiplier(value, suffix)
            except (ValueError, TypeError):
                continue
    return None


def parse_ltv(text: str) -> float | None:
    """Extract Customer Lifetime Value from text.

    Matches patterns like "LTV of $5,000", "$12K LTV",
    "lifetime value is $8,000".
    Returns USD amount.
    """
    patterns = [
        re.compile(
            r"\bltv\s+(?:of\s+|at\s+|is\s+)?"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|thousand|million)?\b",
            re.I,
        ),
        re.compile(
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|thousand|million)?\s*(?:in\s+)?ltv\b",
            re.I,
        ),
        re.compile(
            r"\blifetime\s+value\s+"
            r"(?:of\s+|is\s+|at\s+)?"
            r"\$\s*([\d][\d,]*\.?\d*)\s*"
            r"(k|m|thousand|million)?\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = _normalize_comma_number(match.group(1))
                suffix = match.group(2)
                return _apply_multiplier(value, suffix)
            except (ValueError, TypeError):
                continue
    return None


def parse_team_size(text: str) -> int | None:
    """Extract team size from text.

    Matches patterns like "team of 42", "50-person team",
    "200 employees", "headcount of 75".
    Returns integer count.
    """
    patterns = [
        re.compile(
            r"\bteam\s+of\s+([\d][\d,]*)\b",
            re.I,
        ),
        re.compile(
            r"\b([\d][\d,]*)\s*[\-]?\s*person\s+team\b",
            re.I,
        ),
        re.compile(
            r"\b([\d][\d,]*)\s+employees?\b",
            re.I,
        ),
        re.compile(
            r"\bheadcount\s+(?:of\s+|is\s+|at\s+)?([\d][\d,]*)\b",
            re.I,
        ),
        re.compile(
            r"\b([\d][\d,]*)\s*[\-]?\s*person\s+(?:company|startup|team|org)\b",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return int(_normalize_comma_number(match.group(1)))
            except (ValueError, TypeError):
                continue
    return None
