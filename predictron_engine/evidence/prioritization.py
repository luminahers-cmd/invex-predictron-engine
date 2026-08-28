"""Official website identification and evidence prioritization.

This module sits *downstream* of :mod:`~predictron_engine.evidence.ranking`
and upstream of the fetch layer.  It receives ranked URLs and produces
prioritized page candidates with quality scores, evidence types, and
selection decisions.

Responsibilities
----------------
* **Official Website Identification** — determine the most likely official
  company website from ranked search results using multiple heuristic
  signals (domain similarity, name matching, path structure, ranking
  position, trusted domain avoidance).
* **Evidence Quality Scoring** — classify each candidate URL by its
  evidence type (Homepage, About, Pricing, Documentation, etc.) and
  assign a quality score.
* **Page Prioritization & Selection** — select only the highest-value
  pages for downstream fetching, respecting configurable limits.

All functions are pure and deterministic: identical inputs always produce
identical outputs.  No LLMs, no network calls, no side effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from predictron_engine.evidence.ranking import RankedUrl
from predictron_engine.evidence.url_utils import extract_host

# ---------------------------------------------------------------------------
# Evidence types
# ---------------------------------------------------------------------------


class EvidenceType(str):
    """Classification of a candidate URL's evidence value."""

    OFFICIAL_HOMEPAGE = "official_homepage"
    ABOUT = "about"
    PRODUCT = "product"
    PRICING = "pricing"
    DOCUMENTATION = "documentation"
    BLOG = "blog"
    CAREERS = "careers"
    NEWS = "news"
    DIRECTORY = "directory"
    SOCIAL = "social"
    REPOSITORY = "repository"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class PrioritizationSettings(BaseModel):
    """Configuration for evidence prioritization and page selection."""

    min_official_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to declare an official website",
    )
    max_fetch_pages: int = Field(
        default=10,
        ge=1,
        description="Maximum number of pages to select for fetching",
    )


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfficialWebsiteResult:
    """Output of official website identification."""

    url: str | None = None
    confidence: float = 0.0
    factors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrioritizedPage:
    """A candidate URL with its quality score, evidence type, and reasons."""

    url: str
    quality_score: float = field(compare=False)
    evidence_type: str = field(compare=False)
    priority: int = field(compare=False)
    reasons: tuple[str, ...] = field(default=(), compare=False)

    def __lt__(self, other: PrioritizedPage) -> bool:
        if not isinstance(other, PrioritizedPage):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.quality_score != other.quality_score:
            return self.quality_score > other.quality_score
        return self.url < other.url


# ---------------------------------------------------------------------------
# Internal signal helpers — pure functions
# ---------------------------------------------------------------------------

# Domain similarity: normalised Levenshtein-like overlap
_SIMILAR_DOMAINS: frozenset[str] = frozenset({
    "github.com",
    "gitlab.com",
    "crunchbase.com",
    "ycombinator.com",
    "wellfound.com",
    "angel.co",
    "linkedin.com",
    "producthunt.com",
    "g2.com",
    "trustradius.com",
    "techcrunch.com",
    "forbes.com",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "nytimes.com",
    "businessinsider.com",
    "venturebeat.com",
    "theverge.com",
    "arstechnica.com",
    "wired.com",
    "fastcompany.com",
    "inc.com",
    "entrepreneur.com",
    "saastr.com",
    "medium.com",
    "stackoverflow.com",
})

_THIRD_PARTY_EVIDENCE: dict[str, tuple[str, float, str]] = {
    "github.com": (EvidenceType.REPOSITORY, 0.10, "GitHub repository"),
    "gitlab.com": (EvidenceType.REPOSITORY, 0.08, "GitLab repository"),
    "crunchbase.com": (EvidenceType.DIRECTORY, 0.12, "Crunchbase profile"),
    "ycombinator.com": (EvidenceType.DIRECTORY, 0.12, "Y Combinator profile"),
    "wellfound.com": (EvidenceType.DIRECTORY, 0.10, "Wellfound/AngelList profile"),
    "angel.co": (EvidenceType.DIRECTORY, 0.10, "AngelList profile"),
    "linkedin.com": (EvidenceType.SOCIAL, 0.10, "LinkedIn profile"),
    "producthunt.com": (EvidenceType.DIRECTORY, 0.10, "Product Hunt listing"),
    "g2.com": (EvidenceType.DIRECTORY, 0.08, "G2 reviews"),
    "techcrunch.com": (EvidenceType.NEWS, 0.08, "TechCrunch article"),
    "forbes.com": (EvidenceType.NEWS, 0.06, "Forbes article"),
    "bloomberg.com": (EvidenceType.NEWS, 0.06, "Bloomberg article"),
    "reuters.com": (EvidenceType.NEWS, 0.06, "Reuters article"),
    "venturebeat.com": (EvidenceType.NEWS, 0.06, "VentureBeat article"),
    "medium.com": (EvidenceType.BLOG, 0.04, "Medium article"),
}

_PATH_TYPE_SIGNALS: tuple[tuple[str, str, float, str], ...] = (
    ("/about", EvidenceType.ABOUT, 0.15, "about/company page"),
    ("/company", EvidenceType.ABOUT, 0.15, "company page"),
    ("/pricing", EvidenceType.PRICING, 0.14, "pricing page"),
    ("/plans", EvidenceType.PRICING, 0.14, "pricing page"),
    ("/docs", EvidenceType.DOCUMENTATION, 0.12, "documentation page"),
    ("/documentation", EvidenceType.DOCUMENTATION, 0.12, "documentation page"),
    ("/api", EvidenceType.DOCUMENTATION, 0.12, "API documentation"),
    ("/blog", EvidenceType.BLOG, 0.10, "blog page"),
    ("/news", EvidenceType.NEWS, 0.08, "news page"),
    ("/product", EvidenceType.PRODUCT, 0.12, "product page"),
    ("/features", EvidenceType.PRODUCT, 0.12, "features page"),
    ("/platform", EvidenceType.PRODUCT, 0.12, "platform page"),
    ("/careers", EvidenceType.CAREERS, 0.06, "careers page"),
    ("/jobs", EvidenceType.CAREERS, 0.06, "jobs page"),
    ("/customers", EvidenceType.ABOUT, 0.06, "customers page"),
    ("/case-studies", EvidenceType.ABOUT, 0.06, "case studies"),
)


def _name_in_domain(name: str, host: str) -> float:
    """Return a similarity score in [0, 1] between *name* and *host*.

    Uses a simple token-overlap heuristic: the fraction of name tokens
    that appear in the domain (after stripping TLD and common prefixes).
    """
    if not name or not host:
        return 0.0
    name_tokens = set(re.split(r"[\s_\-]+", name.lower()))
    name_tokens.discard("")
    if not name_tokens:
        return 0.0
    # Strip common TLDs and prefixes from host for comparison
    clean = host.lower()
    for suffix in (".com", ".io", ".ai", ".co", ".org", ".net", ".dev"):
        clean = clean.removesuffix(suffix)
    host_tokens = set(re.split(r"[\s_\-\.]+", clean))
    if not host_tokens:
        return 0.0
    overlap = name_tokens & host_tokens
    return len(overlap) / len(name_tokens)


def _is_third_party(host: str) -> str | None:
    """Return the evidence type if *host* is a known third-party domain."""
    for domain, (etype, _bonus, _reason) in _THIRD_PARTY_EVIDENCE.items():
        if host == domain or host.endswith(f".{domain}"):
            return etype
    return None


def _classify_path(url: str) -> tuple[str, float, str]:
    """Classify a URL's path into an evidence type."""
    try:
        path = urlparse(url).path.lower().rstrip("/")
    except Exception:  # noqa: BLE001
        return EvidenceType.UNKNOWN, 0.0, ""
    for pattern, etype, bonus, reason in _PATH_TYPE_SIGNALS:
        if pattern in path:
            return etype, bonus, reason
    return EvidenceType.UNKNOWN, 0.0, ""


def _is_homepage(url: str) -> bool:
    """Return True if *url* points to a root homepage."""
    try:
        path = urlparse(url).path.rstrip("/")
        return path == "" or path == "/"
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Public API — Official Website Identification
# ---------------------------------------------------------------------------


def identify_official_website(
    ranked: list[RankedUrl],
    *,
    startup_name: str = "",
    known_website_host: str | None = None,
) -> OfficialWebsiteResult:
    """Identify the most likely official company website from ranked results.

    Uses multiple heuristic signals:
    - Exact match with known website host
    - Domain-name similarity to startup name
    - Homepage path preference
    - Search ranking position
    - Third-party domain avoidance
    - URL simplicity

    Parameters
    ----------
    ranked:
        Ranked URLs from :func:`~predictron_engine.evidence.ranking.rank_urls`.
    startup_name:
        Company name for domain-similarity matching.
    known_website_host:
        Lowercased hostname of the known official website (if any).

    Returns
    -------
    :class:`OfficialWebsiteResult` with URL, confidence, and factors.
    """
    if not ranked:
        return OfficialWebsiteResult(url=None, confidence=0.0, factors=())

    best_url: str | None = None
    best_confidence = 0.0
    best_factors: list[str] = []

    for idx, item in enumerate(ranked):
        host = extract_host(item.url)
        if not host:
            continue

        confidence = 0.0
        factors: list[str] = []

        # Signal 1: exact match with known website
        if known_website_host:
            clean_known = known_website_host.lower().removeprefix("www.")
            if host == clean_known:
                confidence = 1.0
                factors.append("exact match with known website")
            elif host.endswith(f".{clean_known}"):
                confidence = 0.85
                factors.append("subdomain of known website")

        # Signal 2: domain-name similarity
        name_sim = _name_in_domain(startup_name, host)
        if name_sim > 0:
            boost = 0.3 * name_sim
            confidence = max(confidence, confidence + boost)
            if confidence < boost:
                confidence = boost
            factors.append(f"name similarity {name_sim:.2f}")

        # Signal 3: homepage preference
        if _is_homepage(item.url):
            confidence = max(confidence, confidence + 0.10)
            if confidence < 0.10:
                confidence = 0.10
            factors.append("homepage path")

        # Signal 4: ranking position bonus (first 3 results)
        if idx < 3:
            position_boost = 0.15 - (idx * 0.04)
            confidence = max(confidence, confidence + position_boost)
            if confidence < position_boost:
                confidence = position_boost
            factors.append(f"rank position {idx + 1}")

        # Signal 5: third-party domain penalty
        if _is_third_party(host):
            confidence *= 0.2
            factors.append("third-party domain penalty")

        # Signal 6: URL simplicity (shorter paths are more likely official)
        try:
            path_len = len(urlparse(item.url).path)
        except Exception:  # noqa: BLE001
            path_len = 100
        if path_len < 10:
            confidence = max(confidence, confidence + 0.05)
            factors.append("short URL path")

        confidence = max(0.0, min(1.0, confidence))

        if confidence > best_confidence:
            best_confidence = confidence
            best_url = item.url
            best_factors = list(factors)

    if best_url is None:
        return OfficialWebsiteResult(url=None, confidence=0.0, factors=())

    return OfficialWebsiteResult(
        url=best_url,
        confidence=round(best_confidence, 4),
        factors=tuple(best_factors),
    )


# ---------------------------------------------------------------------------
# Public API — Evidence Quality Scoring
# ---------------------------------------------------------------------------


def score_evidence_quality(
    url: str,
    *,
    title: str = "",
    snippet: str = "",
    rank_position: int = 0,
    official_host: str | None = None,
) -> tuple[float, str, tuple[str, ...]]:
    """Score the evidence quality of a single URL.

    Returns a quality score in [0, 1], an evidence type, and a tuple of
    human-readable reason strings.

    Parameters
    ----------
    url:
        The candidate URL.
    title:
        Page title from search results.
    snippet:
        Page snippet from search results.
    rank_position:
        0-indexed position in the ranked list.
    official_host:
        Lowercased hostname of the identified official website.
    """
    host = extract_host(url)
    quality = 0.5
    reasons: list[str] = []

    # Classify evidence type
    etype, path_bonus, path_reason = _classify_path(url)
    if path_reason:
        quality += path_bonus
        reasons.append(path_reason)

    # Homepage detection
    if _is_homepage(url):
        etype = EvidenceType.OFFICIAL_HOMEPAGE
        quality += 0.15
        reasons.append("root homepage")

    # Official website match
    if official_host and host:
        clean = official_host.lower().removeprefix("www.")
        if host == clean:
            quality += 0.20
            reasons.append("official domain match")
        elif host.endswith(f".{clean}"):
            quality += 0.10
            reasons.append("official subdomain")

    # Third-party classification
    third_party = _is_third_party(host)
    if third_party:
        etype = third_party
        for tp_domain, (_tp_type, tp_bonus, tp_reason) in _THIRD_PARTY_EVIDENCE.items():
            if host == tp_domain or host.endswith(f".{tp_domain}"):
                quality += tp_bonus
                reasons.append(tp_reason)
                break

    # Ranking position signal
    if rank_position < 5:
        position_bonus = 0.08 - (rank_position * 0.015)
        quality += max(0.0, position_bonus)
        reasons.append(f"rank position {rank_position + 1}")

    # Title contains brand signals
    title_lower = title.lower()
    if any(w in title_lower for w in ("official", "homepage", "home")):
        quality += 0.05
        reasons.append("branded title signal")

    quality = max(0.0, min(1.0, quality))
    return quality, etype, tuple(reasons)


# ---------------------------------------------------------------------------
# Public API — Page Prioritization & Selection
# ---------------------------------------------------------------------------


def prioritise_pages(
    ranked: list[RankedUrl],
    *,
    search_results: dict[str, tuple[str, str]] | None = None,
    official_host: str | None = None,
    settings: PrioritizationSettings | None = None,
) -> list[PrioritizedPage]:
    """Score, classify, and prioritise ranked URLs for evidence collection.

    Parameters
    ----------
    ranked:
        Ranked URLs from the ranking module.
    search_results:
        Optional mapping of ``url → (title, snippet)`` from search results.
    official_host:
        Lowercased hostname of the identified official website.
    settings:
        Optional configuration for thresholds and limits.

    Returns
    -------
    A deterministic, sorted list of :class:`PrioritizedPage` objects.
    """
    settings = settings or PrioritizationSettings()
    search_results = search_results or {}

    pages: list[PrioritizedPage] = []
    for idx, item in enumerate(ranked):
        title, snippet = search_results.get(item.url, ("", ""))
        quality, etype, reasons = score_evidence_quality(
            item.url,
            title=title,
            snippet=snippet,
            rank_position=idx,
            official_host=official_host,
        )
        # Priority: lower number = higher priority
        # Official homepage gets priority 1, then by quality
        if etype == EvidenceType.OFFICIAL_HOMEPAGE:
            priority = 1
        elif etype in (EvidenceType.ABOUT, EvidenceType.PRODUCT):
            priority = 2
        elif etype in (EvidenceType.PRICING, EvidenceType.DOCUMENTATION):
            priority = 3
        elif etype in (EvidenceType.BLOG, EvidenceType.NEWS):
            priority = 4
        elif etype in (EvidenceType.DIRECTORY, EvidenceType.SOCIAL, EvidenceType.REPOSITORY):
            priority = 5
        else:
            priority = 6

        pages.append(
            PrioritizedPage(
                url=item.url,
                quality_score=round(quality, 4),
                evidence_type=etype,
                priority=priority,
                reasons=reasons,
            )
        )

    pages.sort()
    return pages


def select_pages_for_fetch(
    prioritised: list[PrioritizedPage],
    *,
    settings: PrioritizationSettings | None = None,
) -> list[PrioritizedPage]:
    """Select the highest-value pages for downstream fetching.

    Applies the ``max_fetch_pages`` limit and skips low-value pages.

    Parameters
    ----------
    prioritised:
        Sorted list of prioritised pages.
    settings:
        Optional configuration.

    Returns
    -------
    A filtered list of pages to fetch, respecting the configured limit.
    """
    settings = settings or PrioritizationSettings()
    return prioritised[: settings.max_fetch_pages]
