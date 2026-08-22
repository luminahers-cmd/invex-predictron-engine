"""Deterministic reasoning confidence.

Sprint 6A computes a reproducible confidence score for the reasoning
layer from five evidence-aware factors:

* **extraction confidence** — Sprint 5C composite evidence-aware score
  recorded on the extracted features.
* **evidence trust** — average deterministic trust score across the
  collected evidence documents.
* **evidence agreement** — fraction of evidence items corroborated by
  multiple independent sources.
* **citation diversity** — number of distinct citation providers,
  saturated at three.
* **feature completeness** — fraction of populated feature fields.

The final score is a fixed-weight linear combination, rounded to four
decimals and clamped to [0, 1].  No ML, no LLMs, no randomness —
identical inputs always produce identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from predictron_engine.reasoning.context import ReasoningContext

# Fixed, documented weights — part of the public contract.
WEIGHTS: dict[str, float] = {
    "extraction_confidence": 0.25,
    "evidence_trust": 0.25,
    "evidence_agreement": 0.20,
    "citation_diversity": 0.15,
    "feature_completeness": 0.15,
}

_CITATION_DIVERSITY_SATURATION = 3


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Fully decomposed reasoning confidence for auditability."""

    overall: float
    extraction_confidence: float
    evidence_trust: float
    evidence_agreement: float
    citation_diversity: float
    feature_completeness: float


def compute_reasoning_confidence(
    *,
    extraction_confidence: float,
    evidence_trust: float,
    evidence_agreement: float,
    citation_diversity: float,
    feature_completeness: float,
) -> float:
    """Compute deterministic reasoning confidence from five scalar factors.

    All inputs must be in [0, 1]; values outside the range are clamped.
    Returns the weighted combination rounded to four decimals.
    """
    factors = {
        "extraction_confidence": _clamp(extraction_confidence),
        "evidence_trust": _clamp(evidence_trust),
        "evidence_agreement": _clamp(evidence_agreement),
        "citation_diversity": _clamp(citation_diversity),
        "feature_completeness": _clamp(feature_completeness),
    }
    combined = sum(
        factors[name] * WEIGHTS[name] for name in WEIGHTS
    )
    return round(_clamp(combined), 4)


def compute_reasoning_confidence_breakdown(
    *,
    extraction_confidence: float,
    evidence_trust: float,
    evidence_agreement: float,
    citation_diversity: float,
    feature_completeness: float,
) -> ConfidenceBreakdown:
    """Same computation as :func:`compute_reasoning_confidence` but returns
    the full factor decomposition for explainability."""
    extraction_confidence = _clamp(extraction_confidence)
    evidence_trust = _clamp(evidence_trust)
    evidence_agreement = _clamp(evidence_agreement)
    citation_diversity = _clamp(citation_diversity)
    feature_completeness = _clamp(feature_completeness)

    overall = round(
        (
            extraction_confidence * WEIGHTS["extraction_confidence"]
            + evidence_trust * WEIGHTS["evidence_trust"]
            + evidence_agreement * WEIGHTS["evidence_agreement"]
            + citation_diversity * WEIGHTS["citation_diversity"]
            + feature_completeness * WEIGHTS["feature_completeness"]
        ),
        4,
    )

    return ConfidenceBreakdown(
        overall=_clamp(overall),
        extraction_confidence=extraction_confidence,
        evidence_trust=evidence_trust,
        evidence_agreement=evidence_agreement,
        citation_diversity=citation_diversity,
        feature_completeness=feature_completeness,
    )


def compute_context_confidence(
    context: ReasoningContext,
) -> ConfidenceBreakdown:
    """Derive the five confidence factors from a ReasoningContext.

    Every factor is computed deterministically from context state:

    * extraction confidence ← ``features.evidence_confidence``
    * evidence trust ← mean document trust score
    * evidence agreement ← corroboration ratio over evidence items,
      falling back to the extraction-recorded ratio
    * citation diversity ← distinct citation providers (saturating at 3),
      falling back to distinct evidence item sources
    * feature completeness ← ``features.data_completeness``
    """
    features = context.features

    extraction_confidence = float(
        getattr(features, "evidence_confidence", 0.0) or 0.0
    )
    evidence_trust = context.average_trust

    agreement = _agreement_from_items(context)
    if agreement is None:
        agreement = float(
            getattr(features, "evidence_agreement_ratio", 0.0) or 0.0
        )

    diversity = _diversity_from_citations(context)

    completeness = float(getattr(features, "data_completeness", 0.0) or 0.0)

    return compute_reasoning_confidence_breakdown(
        extraction_confidence=extraction_confidence,
        evidence_trust=evidence_trust,
        evidence_agreement=agreement,
        citation_diversity=diversity,
        feature_completeness=completeness,
    )


# ────────────────────────────────────────────────────────────────────
# Private helpers
# ────────────────────────────────────────────────────────────────────


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _agreement_from_items(context: ReasoningContext) -> float | None:
    """Fraction of evidence items corroborated by multiple sources.

    Returns None when there are no evidence items so callers can fall
    back to the extraction-recorded ratio.
    """
    items = context.evidence_items
    if not items:
        return None

    groups: dict[tuple[str, str], set[str]] = {}
    for item in items:
        groups.setdefault((item.domain, item.category), set()).add(item.source)

    corroborated = sum(
        1
        for item in items
        if len(groups[(item.domain, item.category)]) >= 2
    )
    return round(corroborated / len(items), 4)


def _diversity_from_citations(context: ReasoningContext) -> float:
    """Distinct citation providers, saturated at three.

    Falls back to distinct evidence item sources when no citations exist.
    """
    providers: set[str] = set()
    for item in context.evidence_items:
        for citation in item.citations:
            if citation.provider:
                providers.add(citation.provider)

    if not providers:
        providers = {
            item.source for item in context.evidence_items if item.source
        }

    if not providers:
        return 0.0

    return round(min(len(providers) / _CITATION_DIVERSITY_SATURATION, 1.0), 4)
