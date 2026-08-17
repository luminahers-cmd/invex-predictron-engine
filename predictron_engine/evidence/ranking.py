"""Deterministic URL scoring and ranking for the Search Discovery Layer.

All ranking logic is isolated here so that:
- The search provider stays thin and delegation-focused.
- Backends remain stateless (no ranking logic leaks into them).
- Future ML / learned-ranking models can replace these heuristics
  without touching the provider or orchestrator.

Ranking is *deterministic*: identical inputs always produce identical
outputs, and ties are broken lexicographically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from predictron_engine.evidence.search_interfaces import SearchResult
from predictron_engine.evidence.url_utils import (
    extract_host,
    has_valid_scheme,
    normalise_url_for_dedup,
)

# ---------------------------------------------------------------------------
# Public value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=False)
class RankedUrl:
    """A candidate URL with its composite quality score."""

    url: str = field(compare=False)
    score: float = field(compare=False)
    reasons: tuple[str, ...] = field(default=(), compare=False)

    def __lt__(self, other: RankedUrl) -> bool:  # type: ignore[override]
        if not isinstance(other, RankedUrl):
            return NotImplemented
        if self.score != other.score:
            return self.score > other.score  # higher is better
        return self.url < other.url  # lexicographic tie-break


# ---------------------------------------------------------------------------
# Internal helpers — pure functions, no side effects
# ---------------------------------------------------------------------------

_REPUTABLE_DOMAINS: frozenset[str] = frozenset({
    "techcrunch.com",
    "crunchbase.com",
    "ycombinator.com",
    "wellfound.com",
    "angel.co",
    "linkedin.com",
    "github.com",
    "gitlab.com",
    "stackoverflow.com",
    "producthunt.com",
    "g2.com",
    "trustradius.com",
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
})

_PATH_BOOSTS: tuple[tuple[str, float, str], ...] = (
    ("about", 0.15, "about/company page"),
    ("company", 0.15, "company page"),
    ("pricing", 0.12, "pricing page"),
    ("plans", 0.12, "pricing page"),
    ("docs", 0.10, "documentation"),
    ("documentation", 0.10, "documentation"),
    ("api", 0.10, "API docs"),
    ("blog", 0.08, "blog"),
    ("news", 0.08, "news page"),
    ("product", 0.10, "product page"),
    ("features", 0.10, "features page"),
    ("customers", 0.06, "customers page"),
    ("case-studies", 0.06, "case studies"),
    ("careers", 0.04, "careers page"),
    ("jobs", 0.04, "jobs page"),
)

_LOW_VALUE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"/login", "login page"),
    (r"/signin", "sign-in page"),
    (r"/sign-in", "sign-in page"),
    (r"/auth", "authentication page"),
    (r"/register", "registration page"),
    (r"/signup", "signup page"),
    (r"/sign-up", "signup page"),
    (r"/cart", "shopping cart"),
    (r"/checkout", "checkout page"),
    (r"/privacy", "privacy policy"),
    (r"/terms", "terms of service"),
    (r"/cookie", "cookie policy"),
    (r"/legal", "legal page"),
    (r"/sitemap", "sitemap"),
    (r"/feed", "RSS/feed"),
    (r"/rss", "RSS/feed"),
    (r"\.xml$", "XML feed"),
    (r"\.rss$", "RSS feed"),
    (r"/tag/", "tag archive"),
    (r"/category/", "category archive"),
    (r"/author/", "author archive"),
    (r"/page/\d+", "paginated archive"),
    (r"/search", "search page"),
    (r"/404", "error page"),
    (r"/thank-you", "thank you page"),
    (r"/unsubscribe", "unsubscribe page"),
)


def _is_low_value(url: str) -> str | None:
    """Return the reason string if *url* is low-value, else ``None``."""
    lower = url.lower()
    for pattern, reason in _LOW_VALUE_PATTERNS:
        if re.search(pattern, lower):
            return reason
    return None


def _score_path_signals(path_lower: str) -> tuple[float, str]:
    """Return (bonus, reason) from path-based heuristics."""
    for keyword, bonus, reason in _PATH_BOOSTS:
        if keyword in path_lower:
            return bonus, reason
    return 0.0, ""


def _score_domain_authority(host: str) -> tuple[float, str]:
    """Return (bonus, reason) if *host* is a recognised reputable domain."""
    if host in _REPUTABLE_DOMAINS:
        return 0.15, f"reputable domain ({host})"
    for rd in _REPUTABLE_DOMAINS:
        if host.endswith(f".{rd}"):
            return 0.12, f"reputable domain subdomain ({host})"
    return 0.0, ""


def _score_social_platform(host: str) -> tuple[float, str]:
    """Return (bonus, reason) for major social / professional platforms."""
    social: dict[str, tuple[float, str]] = {
        "github.com": (0.15, "GitHub profile/repo"),
        "gitlab.com": (0.12, "GitLab profile/repo"),
        "crunchbase.com": (0.12, "Crunchbase profile"),
        "ycombinator.com": (0.12, "Y Combinator profile"),
        "wellfound.com": (0.12, "Wellfound/AngelList profile"),
        "angel.co": (0.12, "AngelList profile"),
        "linkedin.com": (0.10, "LinkedIn profile/company page"),
        "producthunt.com": (0.10, "Product Hunt listing"),
        "g2.com": (0.08, "G2 reviews"),
    }
    for domain, (bonus, reason) in social.items():
        if host == domain or host.endswith(f".{domain}"):
            return bonus, reason
    return 0.0, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_MIN_SCORE: float = 0.05


def score_url(
    url: str,
    *,
    website_host: str | None = None,
    backend_score: float = 0.0,
) -> tuple[float, tuple[str, ...]]:
    """Compute a deterministic composite quality score for *url*.

    Parameters
    ----------
    url:
        The raw URL string to score.
    website_host:
        Lowercased hostname of the company's official website (if known).
        Matching this provides the strongest signal.
    backend_score:
        The search backend's native relevance score in [0, 1].

    Returns
    -------
    (score, reasons)
        A tuple of (composite score in [0, 1], tuple of human-readable reason strings).
    """
    host = extract_host(url)
    try:
        path_lower = urlparse(url).path.lower()
    except Exception:  # noqa: BLE001
        path_lower = ""

    total = 0.5 * max(0.0, min(1.0, backend_score))
    reasons: list[str] = []
    if backend_score > 0:
        reasons.append(f"backend score {backend_score:.2f}")

    # Official website match — strongest signal
    if website_host and host:
        clean_website = website_host.lower().removeprefix("www.")
        if host == clean_website:
            total += 0.30
            reasons.append("official website match")
        elif host.endswith(f".{clean_website}"):
            total += 0.20
            reasons.append("official website subdomain")

    # Social / professional platform boosts
    social_bonus, social_reason = _score_social_platform(host)
    if social_bonus > 0:
        total += social_bonus
        reasons.append(social_reason)

    # Domain authority
    auth_bonus, auth_reason = _score_domain_authority(host)
    if auth_bonus > 0:
        total += auth_bonus
        reasons.append(auth_reason)

    # Path-level signals
    path_bonus, path_reason = _score_path_signals(path_lower)
    if path_bonus > 0:
        total += path_bonus
        reasons.append(path_reason)

    total = max(0.0, min(1.0, total))
    return total, tuple(reasons)


def rank_urls(
    results: list[SearchResult],
    *,
    website_host: str | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[RankedUrl]:
    """Score, deduplicate, filter, and rank a list of search results.

    This function is fully deterministic:
    1. Each URL is normalised for deduplication.
    2. For duplicate normalised forms, the result with the highest
       backend score wins (ties broken by shortest raw URL).
    3. Each unique URL is scored.
    4. URLs below *min_score* are removed.
    5. Remaining URLs are sorted by score descending, then
       lexicographically for determinism.

    Parameters
    ----------
    results:
        Raw search results from any backend.
    website_host:
        The company's official website host for boosting.
    min_score:
        Minimum composite score to keep (inclusive).

    Returns
    -------
    A deterministic, deduplicated, filtered, and sorted list of
    :class:`RankedUrl` objects.
    """
    # Deduplicate: keep the best result per normalised URL
    best_by_norm: dict[str, SearchResult] = {}
    for r in results:
        try:
            norm = normalise_url_for_dedup(r.url)
        except Exception:  # noqa: BLE001
            norm = r.url.strip().lower()
        if norm not in best_by_norm or r.score > best_by_norm[norm].score:
            best_by_norm[norm] = r
        elif (
            r.score == best_by_norm[norm].score
            and len(r.url) < len(best_by_norm[norm].url)
        ):
            best_by_norm[norm] = r

    # Score each unique URL
    ranked: list[RankedUrl] = []
    for _norm, result in best_by_norm.items():
        if not has_valid_scheme(result.url):
            continue
        low_value = _is_low_value(result.url)
        if low_value:
            continue
        score, reasons = score_url(
            result.url,
            website_host=website_host,
            backend_score=result.score,
        )
        if score < min_score:
            continue
        ranked.append(RankedUrl(url=result.url, score=score, reasons=reasons))

    # Deterministic sort: score descending, URL lexicographic for ties
    ranked.sort()
    return ranked
