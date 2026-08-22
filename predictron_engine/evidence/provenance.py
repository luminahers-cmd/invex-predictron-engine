"""Provenance & Trust models for evidence documents.

Provides deterministic, multi-factor trust scoring and provenance
records that link every document and extracted feature back to its
source.  All functions are pure: identical inputs always produce
identical outputs.  No LLMs, no network calls, no side effects.

Design
------
* :class:`TrustScore` decomposes trust into weighted factors so every
  component is inspectable and auditable.
* :class:`ProvenanceRecord` creates a chain-of-custody from any
  downstream artifact back to the original evidence document.
* :class:`TrustSummary` provides bundle-level aggregate diagnostics
  without exposing per-document internals.
* :func:`compute_document_trust` is the single scoring entry point.
  Its signature accepts only primitive scalars — no domain objects —
  guaranteeing testability and determinism.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────────────
# Enums
# ────────────────────────────────────────────────────────────────────


class SourceTrustLevel(str, Enum):
    """Formal trust classification replacing the plain string.

    Replaces the free-form ``trust_level`` string with a deterministic
    enum derived from source-type scoring thresholds.
    """

    OFFICIAL = "official"
    THIRD_PARTY_CREDIBLE = "third_party_credible"
    THIRD_PARTY_UNVERIFIED = "third_party_unverified"
    UNKNOWN = "unknown"


# ────────────────────────────────────────────────────────────────────
# Value objects
# ────────────────────────────────────────────────────────────────────


class TrustFactor(BaseModel):
    """A single component of the trust score with its weight and value.

    Each factor is independently inspectable so downstream consumers
    can understand *why* a document received its trust score.
    """

    name: str = Field(..., description="Factor identifier")
    weight: float = Field(..., ge=0.0, le=1.0, description="Relative weight in [0, 1]")
    value: float = Field(..., ge=0.0, le=1.0, description="Factor value in [0, 1]")
    explanation: str = Field(
        default="", description="Human-readable explanation of the factor value"
    )


class TrustScore(BaseModel):
    """Deterministic trust score for an evidence document.

    ``overall`` is the weighted sum of all factors, clamped to [0, 1].
    Every score is fully decomposable via its ``factors`` list so
    consumers can inspect individual contributions.
    """

    overall: float = Field(..., ge=0.0, le=1.0, description="Composite trust score in [0, 1]")
    factors: list[TrustFactor] = Field(
        default_factory=list,
        description="Individual trust factors with weights",
    )
    source_trust_level: SourceTrustLevel = Field(
        default=SourceTrustLevel.UNKNOWN,
        description="Formal trust classification of the source",
    )
    freshness_days: int = Field(
        default=0,
        description="Days elapsed since the document was fetched",
    )


class ProvenanceRecord(BaseModel):
    """Chain-of-custody record linking an artifact to its source document.

    Stored on ``EvidenceDocument``, ``EvidenceItem``, and
    ``ExtractedFeatures`` for full traceability.  Every record
    captures enough information to reconstruct the document's trust
    assessment independently.
    """

    document_id: str = Field(..., description="UUIDv5 document id")
    url: str = Field(..., description="Original URL that was requested")
    final_url: str = Field(default="", description="Final URL after redirects")
    document_type: str = Field(default="", description="Classified document type value")
    source_provider: str = Field(default="", description="Provider that collected the document")
    trust_level: str = Field(default="unknown", description="Trust classification string")
    trust_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Trust score in [0, 1]",
    )
    fetched_at: datetime | None = Field(
        default=None, description="UTC timestamp of retrieval"
    )
    is_official: bool = Field(
        default=False,
        description="True when document is from the official website",
    )
    is_duplicate: bool = Field(default=False, description="True when document is a duplicate")
    duplicate_of: str | None = Field(
        default=None, description="Document id of the canonical document when duplicate"
    )
    freshness_days: int = Field(default=0, description="Days elapsed since fetched_at")


class TrustSummary(BaseModel):
    """Aggregate trust diagnostics for an :class:`EvidenceBundle`.

    Provides a high-level overview of trust distribution without
    exposing per-document internals.  Always optional — absent when
    no trust scoring has been performed.
    """

    average_trust_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Mean trust score across all scored documents",
    )
    high_trust_count: int = Field(
        default=0, ge=0,
        description="Documents with trust score >= 0.7",
    )
    medium_trust_count: int = Field(
        default=0, ge=0,
        description="Documents with trust score in [0.4, 0.7)",
    )
    low_trust_count: int = Field(
        default=0, ge=0,
        description="Documents with trust score in (0.0, 0.4)",
    )
    unknown_trust_count: int = Field(
        default=0, ge=0,
        description="Documents with trust score == 0.0",
    )
    total_documents: int = Field(default=0, ge=0, description="Total scored documents")
    source_trust_levels: dict[str, int] = Field(
        default_factory=dict,
        description="Count of documents per SourceTrustLevel value",
    )
    trust_score_range: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="(min, max) trust scores",
    )


# ────────────────────────────────────────────────────────────────────
# Trust Scoring Constants
# ────────────────────────────────────────────────────────────────────

_TRUST_WEIGHTS: dict[str, float] = {
    "authority": 0.30,
    "freshness": 0.15,
    "quality": 0.25,
    "source_type": 0.20,
    "duplicate_penalty": 0.10,
}

_FRESHNESS_HALF_LIFE_DAYS: int = 180

_SOURCE_TYPE_SCORES: dict[str, float] = {
    "official": 1.0,
    "third_party_credible": 0.7,
    "third_party_unverified": 0.4,
    "unknown": 0.2,
}


# ────────────────────────────────────────────────────────────────────
# Pure scoring functions
# ────────────────────────────────────────────────────────────────────


def _compute_freshness_score(fetched_at: datetime, now: datetime) -> tuple[float, int]:
    """Exponential-decay freshness score with a 6-month half-life.

    Returns ``(score, days_elapsed)``.
    """
    delta_seconds = (now - fetched_at).total_seconds()
    days = max(0, int(delta_seconds / 86400))
    decay = 0.5 ** (days / _FRESHNESS_HALF_LIFE_DAYS)
    return max(0.0, min(1.0, decay)), days


def _compute_source_type_score(
    is_official: bool,
    is_third_party_credible: bool,
) -> float:
    """Map source type flags to a trust score in [0, 1]."""
    if is_official:
        return _SOURCE_TYPE_SCORES["official"]
    if is_third_party_credible:
        return _SOURCE_TYPE_SCORES["third_party_credible"]
    return _SOURCE_TYPE_SCORES["unknown"]


def _source_trust_level_from_score(score: float) -> SourceTrustLevel:
    """Map a source-type score to its trust level enum."""
    if score >= 0.9:
        return SourceTrustLevel.OFFICIAL
    if score >= 0.6:
        return SourceTrustLevel.THIRD_PARTY_CREDIBLE
    if score >= 0.3:
        return SourceTrustLevel.THIRD_PARTY_UNVERIFIED
    return SourceTrustLevel.UNKNOWN


# ────────────────────────────────────────────────────────────────────
# Public API — Trust Scoring
# ────────────────────────────────────────────────────────────────────


def compute_document_trust(
    *,
    authority_score: float,
    quality_score: float,
    fetched_at: datetime,
    is_official: bool = False,
    is_third_party_credible: bool = False,
    is_duplicate: bool = False,
    now: datetime | None = None,
) -> TrustScore:
    """Compute a deterministic trust score for a single document.

    All parameters are scalars — no domain objects, no I/O, no side
    effects.  Identical inputs always produce identical outputs.

    Parameters
    ----------
    authority_score:
        Authority confidence in [0, 1] from :func:`estimate_authority`.
    quality_score:
        Content quality score in [0, 1] from quality analysis.
    fetched_at:
        UTC timestamp of when the document was retrieved.
    is_official:
        True when the document is hosted on the official website.
    is_third_party_credible:
        True when the document is from a recognized third-party domain
        (e.g. GitHub, Crunchbase, LinkedIn).
    is_duplicate:
        True when the document is a duplicate of another.
    now:
        Current UTC timestamp for freshness calculation.  Defaults to
        ``datetime.now(UTC)`` — pass explicitly for deterministic tests.
    """
    now = now or datetime.now(UTC)

    freshness_val, freshness_days = _compute_freshness_score(fetched_at, now)
    source_val = _compute_source_type_score(is_official, is_third_party_credible)
    dup_val = 0.0 if is_duplicate else 1.0

    factors = [
        TrustFactor(
            name="authority",
            weight=_TRUST_WEIGHTS["authority"],
            value=round(authority_score, 4),
        ),
        TrustFactor(
            name="freshness",
            weight=_TRUST_WEIGHTS["freshness"],
            value=round(freshness_val, 4),
            explanation=f"{freshness_days}d old",
        ),
        TrustFactor(
            name="quality",
            weight=_TRUST_WEIGHTS["quality"],
            value=round(quality_score, 4),
        ),
        TrustFactor(
            name="source_type",
            weight=_TRUST_WEIGHTS["source_type"],
            value=round(source_val, 4),
        ),
        TrustFactor(
            name="duplicate_penalty",
            weight=_TRUST_WEIGHTS["duplicate_penalty"],
            value=round(dup_val, 4),
        ),
    ]

    overall = sum(f.weight * f.value for f in factors)
    overall = round(max(0.0, min(1.0, overall)), 4)

    source_level = _source_trust_level_from_score(source_val)

    return TrustScore(
        overall=overall,
        factors=factors,
        source_trust_level=source_level,
        freshness_days=freshness_days,
    )


# ────────────────────────────────────────────────────────────────────
# Public API — Provenance Record Construction
# ────────────────────────────────────────────────────────────────────


def build_provenance_record(
    *,
    document_id: str,
    url: str,
    final_url: str,
    document_type: str,
    source_provider: str,
    trust_level: str,
    trust_score: float,
    fetched_at: datetime | None,
    is_official: bool,
    is_duplicate: bool,
    duplicate_of: str | None,
    freshness_days: int,
) -> ProvenanceRecord:
    """Construct a :class:`ProvenanceRecord` from document metadata.

    Pure factory function — no I/O, no side effects.
    """
    return ProvenanceRecord(
        document_id=document_id,
        url=url,
        final_url=final_url,
        document_type=document_type,
        source_provider=source_provider,
        trust_level=trust_level,
        trust_score=trust_score,
        fetched_at=fetched_at,
        is_official=is_official,
        is_duplicate=is_duplicate,
        duplicate_of=duplicate_of,
        freshness_days=freshness_days,
    )


# ────────────────────────────────────────────────────────────────────
# Public API — Trust Summary Aggregation
# ────────────────────────────────────────────────────────────────────


def compute_trust_summary(scores: list[TrustScore]) -> TrustSummary:
    """Aggregate a list of trust scores into a summary.

    Pure function — identical inputs produce identical summaries.

    Parameters
    ----------
    scores:
        One :class:`TrustScore` per document.  Empty list returns a
        zero-valued summary.
    """
    if not scores:
        return TrustSummary()

    values = [s.overall for s in scores]
    avg = round(sum(values) / len(values), 4)

    levels: dict[str, int] = {}
    high = med = low = unknown = 0

    for s in scores:
        lvl = s.source_trust_level.value
        levels[lvl] = levels.get(lvl, 0) + 1
        if s.overall >= 0.7:
            high += 1
        elif s.overall >= 0.4:
            med += 1
        elif s.overall > 0.0:
            low += 1
        else:
            unknown += 1

    return TrustSummary(
        average_trust_score=avg,
        high_trust_count=high,
        medium_trust_count=med,
        low_trust_count=low,
        unknown_trust_count=unknown,
        total_documents=len(scores),
        source_trust_levels=levels,
        trust_score_range=(round(min(values), 4), round(max(values), 4)),
    )
