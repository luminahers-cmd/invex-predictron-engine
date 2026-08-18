"""Document Intelligence — classification, quality analysis, duplicates, and authority.

Sits downstream of HTML cleaning and upstream of extraction.  Receives
:class:`~predictron_engine.evidence.models.EvidenceDocument` objects and
enriches them with deterministic metadata — document type, quality
metrics, authority score, and duplicate resolution — without touching any
extractor interfaces.

Every function in this module is **pure and deterministic**: identical
inputs always produce identical outputs.  No LLMs, no network calls, no
side effects.

Design
------
* :class:`DocumentMetadata` is a separate model attached as an optional
  field on :class:`EvidenceDocument`.  This keeps provenance (how a
  document was collected) separate from intelligence (what we know about
  it) and avoids bloating the core model.
* Duplicate detection uses SHA-256 of normalised text and URL
  normalisation to identify canonical documents deterministically.
* Authority is scored independently of search ranking using page-type
  heuristics, domain signals, and content quality.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlparse

from predictron_engine.evidence.models import (
    DocumentMetadata,
    DocumentStatus,
    DocumentType,
    EvidenceDocument,
    IntelligenceSummary,
)
from predictron_engine.evidence.url_utils import extract_host, normalise_url_for_dedup

# ────────────────────────────────────────────────────────────────────
# Part 1 — Document Classification
# ────────────────────────────────────────────────────────────────────


# Path → (DocumentType, match priority). Higher priority wins ties.
_PATH_SIGNALS: tuple[tuple[str, DocumentType, int], ...] = (
    ("/pricing", DocumentType.PRICING, 10),
    ("/plans", DocumentType.PRICING, 10),
    ("/api/docs", DocumentType.API_DOCS, 11),
    ("/api-reference", DocumentType.API_DOCS, 11),
    ("/developer", DocumentType.API_DOCS, 9),
    ("/docs", DocumentType.DOCUMENTATION, 10),
    ("/documentation", DocumentType.DOCUMENTATION, 10),
    ("/guide", DocumentType.DOCUMENTATION, 8),
    ("/tutorial", DocumentType.DOCUMENTATION, 8),
    ("/blog", DocumentType.BLOG, 10),
    ("/news", DocumentType.NEWS, 10),
    ("/press", DocumentType.PRESS_RELEASE, 10),
    ("/careers", DocumentType.CAREERS, 10),
    ("/jobs", DocumentType.CAREERS, 10),
    ("/hiring", DocumentType.CAREERS, 8),
    ("/about", DocumentType.ABOUT, 10),
    ("/company", DocumentType.ABOUT, 9),
    ("/team", DocumentType.ABOUT, 8),
    ("/product", DocumentType.PRODUCT, 10),
    ("/products", DocumentType.PRODUCT, 10),
    ("/features", DocumentType.PRODUCT, 9),
    ("/platform", DocumentType.PRODUCT, 9),
    ("/security", DocumentType.SECURITY, 10),
    ("/privacy", DocumentType.PRIVACY, 10),
    ("/privacy-policy", DocumentType.PRIVACY, 10),
    ("/terms", DocumentType.TERMS, 10),
    ("/terms-of-service", DocumentType.TERMS, 10),
    ("/legal", DocumentType.TERMS, 8),
    ("/faq", DocumentType.FAQ, 10),
    ("/help", DocumentType.FAQ, 8),
    ("/support", DocumentType.FAQ, 8),
    ("/contact", DocumentType.CONTACT, 10),
    ("/investor", DocumentType.INVESTOR, 10),
    ("/investors", DocumentType.INVESTOR, 10),
    ("/ir", DocumentType.INVESTOR, 8),
)

_TITLE_KEYWORDS: tuple[tuple[str, DocumentType, int], ...] = (
    ("pricing", DocumentType.PRICING, 8),
    ("price", DocumentType.PRICING, 7),
    ("plan", DocumentType.PRICING, 5),
    ("documentation", DocumentType.DOCUMENTATION, 9),
    ("api reference", DocumentType.API_DOCS, 10),
    ("api docs", DocumentType.API_DOCS, 10),
    ("developer", DocumentType.API_DOCS, 6),
    ("blog", DocumentType.BLOG, 8),
    ("careers", DocumentType.CAREERS, 8),
    ("join us", DocumentType.CAREERS, 7),
    ("we're hiring", DocumentType.CAREERS, 8),
    ("about us", DocumentType.ABOUT, 8),
    ("about", DocumentType.ABOUT, 6),
    ("our company", DocumentType.ABOUT, 7),
    ("security", DocumentType.SECURITY, 8),
    ("privacy", DocumentType.PRIVACY, 8),
    ("terms of service", DocumentType.TERMS, 9),
    ("terms and conditions", DocumentType.TERMS, 9),
    ("faq", DocumentType.FAQ, 8),
    ("frequently asked", DocumentType.FAQ, 8),
    ("contact us", DocumentType.CONTACT, 8),
    ("get in touch", DocumentType.CONTACT, 7),
    ("news", DocumentType.NEWS, 6),
    ("press release", DocumentType.PRESS_RELEASE, 9),
    ("investor", DocumentType.INVESTOR, 8),
    ("github", DocumentType.REPOSITORY, 8),
    ("repository", DocumentType.REPOSITORY, 8),
)

_HEADING_KEYWORDS: tuple[tuple[str, DocumentType, int], ...] = (
    ("pricing", DocumentType.PRICING, 5),
    ("documentation", DocumentType.DOCUMENTATION, 5),
    ("careers", DocumentType.CAREERS, 4),
    ("about us", DocumentType.ABOUT, 5),
    ("contact", DocumentType.CONTACT, 4),
    ("security", DocumentType.SECURITY, 5),
    ("privacy", DocumentType.PRIVACY, 5),
    ("terms", DocumentType.TERMS, 4),
    ("faq", DocumentType.FAQ, 5),
)

_THIRD_PARTY_TYPES: dict[str, DocumentType] = {
    "github.com": DocumentType.REPOSITORY,
    "gitlab.com": DocumentType.REPOSITORY,
}


def classify_document(
    url: str,
    *,
    title: str = "",
    headings: list[str] | None = None,
) -> DocumentType:
    """Classify an evidence document by its content type.

    Uses URL path, title keywords, and heading keywords to assign a
    :class:`DocumentType`.  Falls back to HOMEPAGE for root paths and
    UNKNOWN when no signal matches.

    Parameters
    ----------
    url:
        The document URL.
    title:
        Page title from the cleaned HTML.
    headings:
        Extracted heading strings from the cleaned HTML.
    """
    headings = headings or []

    # Signal 1: third-party domain
    host = extract_host(url)
    for domain, dtype in _THIRD_PARTY_TYPES.items():
        if host == domain or host.endswith(f".{domain}"):
            return dtype

    # Signal 2: URL path classification
    try:
        path = urlparse(url).path.lower().rstrip("/")
    except Exception:  # noqa: BLE001
        return DocumentType.UNKNOWN

    best_type: DocumentType | None = None
    best_priority = -1

    for pattern, dtype, priority in _PATH_SIGNALS:
        if pattern in path:
            if priority > best_priority:
                best_priority = priority
                best_type = dtype

    # Homepage: root path with no path signals
    if best_type is None and (path == "" or path == "/") and url:
        return DocumentType.HOMEPAGE

    if best_type is not None:
        return best_type

    # Signal 3: title keywords
    title_lower = title.lower()
    best_priority = -1
    for keyword, dtype, priority in _TITLE_KEYWORDS:
        if keyword in title_lower:
            if priority > best_priority:
                best_priority = priority
                best_type = dtype

    if best_type is not None:
        return best_type

    # Signal 4: heading keywords (first 5 headings only)
    heading_text = " ".join(h.lower() for h in headings[:5])
    best_priority = -1
    for keyword, dtype, priority in _HEADING_KEYWORDS:
        if keyword in heading_text:
            if priority > best_priority:
                best_priority = priority
                best_type = dtype

    if best_type is not None:
        return best_type

    return DocumentType.UNKNOWN


# ────────────────────────────────────────────────────────────────────
# Part 2 — Evidence Metadata
# ────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────
# Part 3 — Quality Analysis
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QualityMetrics:
    """Deterministic quality signals derived from document content."""

    text_density: float = 0.0
    navigation_ratio: float = 0.0
    boilerplate_ratio: float = 0.0
    heading_quality: float = 0.0
    duplicate_heading_ratio: float = 0.0
    content_completeness: float = 0.0


_NAVIGATION_PATTERNS = re.compile(
    r"(^|\s)(home|menu|skip to|breadcrumb|login|sign up|sign in|"
    r"register|subscribe|newsletter|back to|next|previous|page \d+|"
    r"©|all rights reserved|cookie|privacy policy|terms)(\s|$)",
    re.IGNORECASE,
)

_BOILERPLATE_PATTERNS = re.compile(
    r"(^|\s)(copyright|all rights reserved|cookie|privacy|terms of|"
    r"powered by|built with|subscribe|newsletter|follow us|"
    r"share|tweet|like us|connect with)(\s|$)",
    re.IGNORECASE,
)


def _compute_text_density(words: list[str]) -> float:
    """Estimate content word density using stopword ratio heuristic."""
    if not words:
        return 0.0
    content_words = sum(1 for w in words if len(w) > 3)
    return min(1.0, content_words / len(words))


def _compute_navigation_ratio(text: str) -> float:
    """Estimate the fraction of text that is navigation."""
    if not text:
        return 0.0
    matches = _NAVIGATION_PATTERNS.findall(text)
    nav_chars = sum(len(m[0] + m[1] + m[2]) for m in matches)
    return min(1.0, nav_chars / max(1, len(text)))


def _compute_boilerplate_ratio(text: str) -> float:
    """Estimate the fraction of text that is boilerplate."""
    if not text:
        return 0.0
    matches = _BOILERPLATE_PATTERNS.findall(text)
    bp_chars = sum(len(m[0] + m[1] + m[2]) for m in matches)
    return min(1.0, bp_chars / max(1, len(text)))


def _compute_heading_quality(headings: list[str]) -> float:
    """Score heading quality: hierarchical, descriptive, unique."""
    if not headings:
        return 0.0
    score = 0.0
    # Has at least one heading
    score += 0.3
    # Has multiple headings (structure)
    if len(headings) >= 3:
        score += 0.2
    # Headings vary in length (not all identical)
    lengths = [len(h) for h in headings]
    if len(set(lengths)) > 1:
        score += 0.2
    # Headings contain meaningful content (avg length > 5 chars)
    avg_len = sum(lengths) / len(lengths)
    if avg_len > 5:
        score += 0.15
    # No extremely short headings
    if min(lengths) >= 3:
        score += 0.15
    return min(1.0, score)


def _compute_duplicate_heading_ratio(headings: list[str]) -> float:
    """Fraction of headings that are duplicates."""
    if not headings:
        return 0.0
    seen: dict[str, int] = {}
    for h in headings:
        key = h.lower().strip()
        seen[key] = seen.get(key, 0) + 1
    duplicates = sum(v - 1 for v in seen.values())
    return duplicates / len(headings)


def _compute_content_completeness(
    word_count: int,
    heading_count: int,
    paragraph_count: int,
) -> float:
    """Estimate how complete the page content is."""
    score = 0.0
    # Word count signals
    if word_count >= 100:
        score += 0.3
    elif word_count >= 30:
        score += 0.15
    # Has headings
    if heading_count >= 1:
        score += 0.2
    # Has paragraphs
    if paragraph_count >= 2:
        score += 0.25
    elif paragraph_count >= 1:
        score += 0.1
    # Longer content suggests completeness
    if word_count >= 300:
        score += 0.25
    elif word_count >= 150:
        score += 0.15
    return min(1.0, score)


def compute_quality_metrics(
    doc: EvidenceDocument,
    *,
    headings: list[str] | None = None,
    paragraphs: list[str] | None = None,
    list_items: list[str] | None = None,
    table_rows: list[str] | None = None,
) -> QualityMetrics:
    """Compute deterministic quality metrics for a document.

    Parameters
    ----------
    doc:
        The evidence document.
    headings, paragraphs, list_items, table_rows:
        Structured content from the cleaner (if available).
    """
    headings = headings or []
    paragraphs = paragraphs or []
    list_items = list_items or []
    table_rows = table_rows or []

    words = doc.text.split()

    return QualityMetrics(
        text_density=_compute_text_density(words),
        navigation_ratio=_compute_navigation_ratio(doc.text),
        boilerplate_ratio=_compute_boilerplate_ratio(doc.text),
        heading_quality=_compute_heading_quality(headings),
        duplicate_heading_ratio=_compute_duplicate_heading_ratio(headings),
        content_completeness=_compute_content_completeness(
            len(words),
            len(headings),
            len(paragraphs),
        ),
    )


# ────────────────────────────────────────────────────────────────────
# Part 5 — Authority Estimation
# ────────────────────────────────────────────────────────────────────


# Domain → base authority bonus
_AUTHORITY_DOMAINS: dict[str, float] = {
    "github.com": 0.15,
    "gitlab.com": 0.12,
    "crunchbase.com": 0.10,
    "linkedin.com": 0.08,
}

# Document type → base authority
_TYPE_AUTHORITY: dict[DocumentType, float] = {
    DocumentType.HOMEPAGE: 0.85,
    DocumentType.ABOUT: 0.70,
    DocumentType.PRODUCT: 0.75,
    DocumentType.PRICING: 0.70,
    DocumentType.DOCUMENTATION: 0.80,
    DocumentType.API_DOCS: 0.80,
    DocumentType.BLOG: 0.55,
    DocumentType.CAREERS: 0.50,
    DocumentType.SECURITY: 0.65,
    DocumentType.PRIVACY: 0.60,
    DocumentType.TERMS: 0.60,
    DocumentType.FAQ: 0.55,
    DocumentType.CONTACT: 0.55,
    DocumentType.NEWS: 0.50,
    DocumentType.PRESS_RELEASE: 0.55,
    DocumentType.INVESTOR: 0.60,
    DocumentType.REPOSITORY: 0.65,
    DocumentType.UNKNOWN: 0.40,
}


def estimate_authority(
    doc: EvidenceDocument,
    *,
    document_type: DocumentType = DocumentType.UNKNOWN,
    official_host: str | None = None,
    quality_score: float = 0.0,
) -> float:
    """Compute an authority confidence score for a document.

    Combines document type, domain signals, official website match, and
    content quality into a single [0, 1] score.

    Parameters
    ----------
    doc:
        The evidence document.
    document_type:
        The classified document type.
    official_host:
        Lowercased hostname of the identified official website.
    quality_score:
        The document's quality score (0–1).
    """
    host = extract_host(str(doc.url))
    score = _TYPE_AUTHORITY.get(document_type, 0.40)

    # Official domain match bonus
    if official_host and host:
        clean = official_host.lower().removeprefix("www.")
        if host == clean:
            score += 0.15
        elif host.endswith(f".{clean}"):
            score += 0.08

    # Third-party domain bonus
    for domain, bonus in _AUTHORITY_DOMAINS.items():
        if host == domain or host.endswith(f".{domain}"):
            score += bonus
            break

    # Quality bonus
    score += quality_score * 0.10

    # Successful fetch bonus
    if doc.status == DocumentStatus.SUCCESS:
        score += 0.05

    return round(max(0.0, min(1.0, score)), 4)


# ────────────────────────────────────────────────────────────────────
# Part 4 — Duplicate Detection
# ────────────────────────────────────────────────────────────────────


def _normalise_text_for_hash(text: str) -> str:
    """Normalise text for deterministic content hashing."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_content_hash(text: str) -> str:
    """Return a deterministic SHA-256 hex digest of normalised text."""
    normalised = _normalise_text_for_hash(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _url_sort_key(url_str: str) -> str:
    """Deterministic sort key: shorter path wins, then lexicographic."""
    try:
        parsed = urlparse(url_str)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        path = parsed.path.rstrip("/")
        return f"{len(path):04d}:{host}{path}"
    except Exception:  # noqa: BLE001
        return url_str


@dataclass
class DuplicateGroup:
    """A set of document ids that are duplicates of each other."""

    canonical_id: str = ""
    duplicate_ids: tuple[str, ...] = ()
    content_hash: str = ""
    reason: str = ""


def detect_duplicates(
    documents: list[EvidenceDocument],
) -> tuple[list[EvidenceDocument], list[DuplicateGroup]]:
    """Detect and resolve duplicate documents.

    Returns a deduplicated list (keeping canonical documents) and a list
    of :class:`DuplicateGroup` records for traceability.

    Duplicate detection uses, in priority order:
    1. Same content hash (identical normalised text)
    2. Same canonical URL (normalised form)

    The canonical document is chosen deterministically:
    - Shortest URL path
    - Lexicographic tie-break
    """
    if not documents:
        return [], []

    # Only consider successful documents for dedup
    successful = [d for d in documents if d.status == DocumentStatus.SUCCESS]
    failed = [d for d in documents if d.status != DocumentStatus.SUCCESS]

    # Build content hash → list of document ids
    hash_groups: dict[str, list[str]] = {}
    url_groups: dict[str, list[str]] = {}
    doc_map: dict[str, EvidenceDocument] = {}

    for doc in successful:
        doc_map[doc.id] = doc
        # Content hash grouping
        c_hash = compute_content_hash(doc.text)
        hash_groups.setdefault(c_hash, []).append(doc.id)
        # URL normalised grouping
        n_url = normalise_url_for_dedup(str(doc.url))
        url_groups.setdefault(n_url, []).append(doc.id)

    # Merge groups: union-find approach via hash→canonical mapping
    id_to_canonical: dict[str, str] = {}
    groups: list[DuplicateGroup] = []

    # Process content hash groups first (stronger signal)
    for c_hash, ids in sorted(hash_groups.items()):
        if len(ids) < 2:
            continue
        sorted_ids = sorted(ids, key=lambda i: _url_sort_key(str(doc_map[i].url)))
        canonical = sorted_ids[0]
        for doc_id in sorted_ids:
            id_to_canonical[doc_id] = canonical
        groups.append(
            DuplicateGroup(
                canonical_id=canonical,
                duplicate_ids=tuple(sorted_ids[1:]),
                content_hash=c_hash,
                reason="identical content hash",
            )
        )

    # Process URL groups (catch redirects / trailing-slash variants)
    for n_url, ids in sorted(url_groups.items()):
        if len(ids) < 2:
            continue
        # Only add if not already fully resolved
        unresolved = [i for i in ids if i not in id_to_canonical]
        if len(unresolved) < 2:
            continue
        sorted_ids = sorted(
            unresolved, key=lambda i: _url_sort_key(str(doc_map[i].url)),
        )
        canonical = sorted_ids[0]
        for doc_id in sorted_ids:
            id_to_canonical.setdefault(doc_id, canonical)
        groups.append(
            DuplicateGroup(
                canonical_id=canonical,
                duplicate_ids=tuple(sorted_ids[1:]),
                content_hash="",
                reason="normalised URL match",
            )
        )

    # Build output: canonical docs with metadata, duplicates removed
    seen_canonicals: set[str] = set()
    result: list[EvidenceDocument] = []

    for doc in successful:
        if doc.id in id_to_canonical:
            canonical_id = id_to_canonical[doc.id]
            if doc.id == canonical_id and doc.id not in seen_canonicals:
                seen_canonicals.add(doc.id)
                result.append(doc)
        else:
            result.append(doc)

    result.extend(failed)
    result.sort(key=lambda d: str(d.original_url))

    return result, groups


# ────────────────────────────────────────────────────────────────────
# Part 6 — Integration
# ────────────────────────────────────────────────────────────────────


def _priority_for_type(dtype: DocumentType) -> int:
    """Map a document type to its priority tier (1 = highest)."""
    if dtype == DocumentType.HOMEPAGE:
        return 1
    if dtype in (DocumentType.ABOUT, DocumentType.PRODUCT):
        return 2
    if dtype in (DocumentType.PRICING, DocumentType.DOCUMENTATION, DocumentType.API_DOCS):
        return 3
    if dtype in (DocumentType.BLOG, DocumentType.NEWS, DocumentType.PRESS_RELEASE):
        return 4
    if dtype in (DocumentType.REPOSITORY, DocumentType.CAREERS):
        return 5
    return 6


def _trust_level(
    doc: EvidenceDocument,
    official_host: str | None,
) -> str:
    """Determine the trust level of a document."""
    host = extract_host(str(doc.url))
    if official_host and host:
        clean = official_host.lower().removeprefix("www.")
        if host == clean or host.endswith(f".{clean}"):
            return "official"
    for domain in _AUTHORITY_DOMAINS:
        if host == domain or host.endswith(f".{domain}"):
            return "third_party"
    return "unknown"


def _detect_language(text: str) -> str:
    """Simple heuristic language detection using common English words."""
    if not text:
        return "en"
    words = text.lower().split()
    if not words:
        return "en"
    english_signal = sum(
        1 for w in words[:200]
        if w in _ENGLISH_COMMON
    )
    ratio = english_signal / min(len(words), 200)
    return "en" if ratio > 0.1 else "und"


_ENGLISH_COMMON: frozenset[str] = frozenset({
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
    "in", "with", "to", "for", "of", "not", "no", "can", "had", "has",
    "was", "were", "are", "be", "been", "being", "have", "from", "this",
    "that", "it", "its", "as", "by", "at", "we", "our", "you", "your",
    "they", "their", "what", "when", "where", "how", "who", "all", "each",
    "than", "them", "then", "these", "those", "will", "more", "most",
    "other", "some", "such", "only", "very", "just", "about", "also",
    "new", "one", "two", "use", "used", "using", "get", "make", "made",
    "way", "well", "back", "any", "may", "much", "go", "see", "now",
    "through", "after", "before", "between", "under", "over", "into",
})


def enrich_documents(
    documents: list[EvidenceDocument],
    *,
    official_host: str | None = None,
    source_provider: str = "",
    structured_content: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[list[EvidenceDocument], IntelligenceSummary]:
    """Run the full Document Intelligence pipeline on a list of documents.

    Performs classification, quality analysis, authority estimation,
    duplicate detection, and metadata enrichment in a single pass.

    Parameters
    ----------
    documents:
        Cleaned evidence documents from the collection stage.
    official_host:
        Lowercased hostname of the identified official website.
    source_provider:
        Name of the provider that collected these documents.
    structured_content:
        Optional mapping of ``doc_id → {"headings": [...], "paragraphs": [...], ...}``
        with structured content from the HTML cleaner.

    Returns
    -------
    A tuple of (enriched documents, diagnostics summary).
    """
    started = time.monotonic()
    structured_content = structured_content or {}
    input_count = len(documents)

    # Step 1: Duplicate detection
    deduplicated, duplicate_groups = detect_duplicates(documents)
    removed_count = input_count - len(deduplicated)

    # Build canonical lookup
    canonical_of: dict[str, str] = {}
    for group in duplicate_groups:
        for dup_id in group.duplicate_ids:
            canonical_of[dup_id] = group.canonical_id

    # Step 2: Classify, score, and enrich each document
    type_counts: Counter[str] = Counter()
    authority_scores: list[float] = []
    quality_scores: list[float] = []
    classified_count = 0

    enriched: list[EvidenceDocument] = []
    for doc in deduplicated:
        # Skip failed/empty documents — no intelligence for broken pages
        if doc.status != DocumentStatus.SUCCESS:
            enriched.append(doc)
            continue

        # Get structured content if available
        sc = structured_content.get(doc.id, {})
        headings = sc.get("headings", [])
        paragraphs = sc.get("paragraphs", [])
        list_items = sc.get("list_items", [])
        table_rows = sc.get("table_rows", [])

        # Part 1: Classify
        dtype = classify_document(
            str(doc.url),
            title=doc.title,
            headings=headings,
        )
        if dtype != DocumentType.UNKNOWN:
            classified_count += 1
        type_counts[dtype.value] += 1

        # Part 3: Quality metrics
        quality = compute_quality_metrics(
            doc,
            headings=headings,
            paragraphs=paragraphs,
            list_items=list_items,
            table_rows=table_rows,
        )
        quality_score = round(
            (quality.text_density * 0.25
             + (1.0 - quality.navigation_ratio) * 0.15
             + (1.0 - quality.boilerplate_ratio) * 0.15
             + quality.heading_quality * 0.20
             + quality.content_completeness * 0.25),
            4,
        )
        quality_score = max(0.0, min(1.0, quality_score))
        quality_scores.append(quality_score)

        # Part 5: Authority
        authority = estimate_authority(
            doc,
            document_type=dtype,
            official_host=official_host,
            quality_score=quality_score,
        )
        authority_scores.append(authority)

        # Build metadata
        content_hash = compute_content_hash(doc.text)
        words = doc.text.split()
        priority = _priority_for_type(dtype)

        is_dup = doc.id in canonical_of
        metadata = DocumentMetadata(
            document_type=dtype,
            authority_score=authority,
            quality_score=quality_score,
            priority=priority,
            canonical_url=str(doc.url),
            language=_detect_language(doc.text),
            word_count=len(words),
            heading_count=len(headings),
            table_count=len(table_rows),
            list_count=len(list_items),
            content_hash=content_hash,
            is_duplicate=is_dup,
            duplicate_of=canonical_of.get(doc.id),
            trust_level=_trust_level(doc, official_host),
            source_provider=source_provider,
        )

        enriched.append(
            doc.model_copy(update={"metadata": metadata})
        )

    enriched.sort(key=lambda d: str(d.original_url))

    duration_ms = int((time.monotonic() - started) * 1000)

    avg_auth = (
        round(sum(authority_scores) / len(authority_scores), 4)
        if authority_scores else 0.0
    )
    avg_qual = (
        round(sum(quality_scores) / len(quality_scores), 4)
        if quality_scores else 0.0
    )
    conf = (
        round(classified_count / len(deduplicated), 4)
        if deduplicated else 0.0
    )

    summary = IntelligenceSummary(
        documents_input=input_count,
        documents_classified=classified_count,
        duplicates_removed=removed_count,
        average_authority=avg_auth,
        average_quality=avg_qual,
        document_type_distribution=dict(sorted(type_counts.items())),
        processing_duration_ms=duration_ms,
        classification_confidence=conf,
    )

    return enriched, summary


class DocumentIntelligence:
    """Orchestrator for the Document Intelligence pipeline.

    Wraps :func:`enrich_documents` with an instance-based interface
    compatible with the evidence provider pattern.  Stateless — all
    configuration is passed at construction time.
    """

    name = "document_intelligence"

    def __init__(
        self,
        *,
        official_host: str | None = None,
        source_provider: str = "",
    ) -> None:
        self._official_host = official_host
        self._source_provider = source_provider

    def enrich(
        self,
        documents: list[EvidenceDocument],
        *,
        structured_content: dict[str, dict[str, list[str]]] | None = None,
    ) -> tuple[list[EvidenceDocument], IntelligenceSummary]:
        """Enrich documents with intelligence metadata.

        Delegates to :func:`enrich_documents` with the configured
        official host and source provider.
        """
        return enrich_documents(
            documents,
            official_host=self._official_host,
            source_provider=self._source_provider,
            structured_content=structured_content,
        )
