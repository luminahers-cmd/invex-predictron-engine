"""Unified opportunity register aggregation (Sprint 6C).

Aggregates opportunities from already-produced artifacts:
  - readiness key strengths
  - high dimension scores
  - reinforcing cross-signal relationships
  - positive quantitative metrics (mirroring readiness thresholds)
  - strong evidence observations (high trust and high confidence)

Deduplicates by normalized label, ranks by impact then dimension,
carries citations, provenance, and propagated (minimum) confidence.
"""

from __future__ import annotations

import re

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    DimensionAssessment,
    EvidenceCitation,
    InvestmentReadiness,
    Observation,
    OpportunityItem,
    ScoreResult,
    SynthesisSeverity,
    severity_order,
)

# Quantitative opportunity thresholds mirror the investment-readiness
# strength collection conventions.
_STRONG_ARR_USD = 10_000_000.0
_STRONG_NRR_PCT = 110.0
_STRONG_GROWTH_PCT = 50.0
_STRONG_LTV_CAC = 5.0

_HIGH_SCORE = 80.0
_NOTABLE_SCORE = 65.0

# Observation must clear both bars to count as "strong evidence".
_STRONG_TRUST = 0.7
_STRONG_OBSERVATION_CONFIDENCE = 0.6

_MAX_STATEMENTS_PER_ITEM = 5
_MAX_OPPORTUNITIES = 8

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_label(label: str) -> str:
    return _WHITESPACE_RE.sub(" ", label.strip().lower()).rstrip(".")


def _impact_for_score(score: float) -> SynthesisSeverity:
    if score >= _HIGH_SCORE:
        return SynthesisSeverity.HIGH
    if score >= _NOTABLE_SCORE:
        return SynthesisSeverity.MODERATE
    return SynthesisSeverity.LOW


def _citations_from(observations: list[Observation]) -> list[EvidenceCitation]:
    citations: list[EvidenceCitation] = []
    seen: set[tuple[str, str, str]] = set()
    for obs in observations:
        for citation in obs.citations:
            key = (citation.claim, citation.domain, citation.category)
            if key not in seen:
                seen.add(key)
                citations.append(citation)
    return citations


def _provenance_from(observations: list[Observation]) -> list[str]:
    ids: list[str] = []
    for obs in observations:
        for doc_id in obs.provenance_document_ids:
            if doc_id not in ids:
                ids.append(doc_id)
    return ids


class _Register:
    """Label-keyed accumulator with deterministic merge semantics."""

    def __init__(self) -> None:
        self._items: dict[str, OpportunityItem] = {}

    def add(
        self,
        *,
        label: str,
        dimension: str,
        impact: SynthesisSeverity,
        source: str,
        statements: list[str],
        observations: list[Observation],
        confidence: float,
    ) -> None:
        statements = [s for s in statements if s][:_MAX_STATEMENTS_PER_ITEM]
        key = _normalize_label(label)
        existing = self._items.get(key)
        if existing is None:
            self._items[key] = OpportunityItem(
                label=label.strip(),
                dimension=dimension,
                impact=impact,
                sources=[source],
                statements=statements,
                citations=_citations_from(observations),
                provenance_document_ids=_provenance_from(observations),
                confidence=round(min([confidence, *(o.confidence for o in observations)]), 4),
            )
            return
        merged_statements = list(existing.statements)
        for statement in statements:
            within_cap = len(merged_statements) < _MAX_STATEMENTS_PER_ITEM
            if statement not in merged_statements and within_cap:
                merged_statements.append(statement)
        merged_citations = list(existing.citations)
        seen_claims = {
            (c.claim, c.domain, c.category) for c in merged_citations
        }
        for citation in _citations_from(observations):
            claim_key = (citation.claim, citation.domain, citation.category)
            if claim_key not in seen_claims:
                seen_claims.add(claim_key)
                merged_citations.append(citation)
        merged_provenance = list(existing.provenance_document_ids)
        for doc_id in _provenance_from(observations):
            if doc_id not in merged_provenance:
                merged_provenance.append(doc_id)
        new_confidence = min(
            [existing.confidence, confidence, *(o.confidence for o in observations)]
        )
        self._items[key] = OpportunityItem(
            label=existing.label,
            dimension=existing.dimension or dimension,
            impact=(
                impact
                if severity_order(impact) < severity_order(existing.impact)
                else existing.impact
            ),
            sources=[*existing.sources, source]
            if source not in existing.sources
            else existing.sources,
            statements=merged_statements,
            citations=merged_citations,
            provenance_document_ids=merged_provenance,
            confidence=round(new_confidence, 4),
        )

    def ranked(self, max_items: int) -> list[OpportunityItem]:
        items = sorted(
            self._items.values(),
            key=lambda item: (severity_order(item.impact), item.dimension, item.label),
        )
        return items[:max_items]


def aggregate_opportunities(
    features: ExtractedFeatures,
    scores: list[ScoreResult],
    assessments: list[DimensionAssessment],
    observations: list[Observation],
    readiness: InvestmentReadiness | None,
    *,
    max_opportunities: int = _MAX_OPPORTUNITIES,
) -> list[OpportunityItem]:
    """Build the unified, deduplicated, impact-ranked opportunity register."""
    register = _Register()

    # --- Source 1: high dimension scores ----------------------------------
    assessment_confidence = {
        assessment.dimension: assessment.confidence for assessment in assessments
    }
    for score in sorted(scores, key=lambda s: (-s.score, s.dimension)):
        if score.score < _NOTABLE_SCORE:
            continue
        dim_observations = [obs for obs in observations if obs.dimension == score.dimension]
        register.add(
            label=f"strong {score.dimension.replace('_', ' ')}",
            dimension=score.dimension,
            impact=_impact_for_score(score.score),
            source=f"score:{score.dimension}",
            statements=[score.rationale]
            + [obs.statement for obs in dim_observations[:2]],
            observations=dim_observations,
            confidence=min([1.0, assessment_confidence.get(score.dimension, 1.0)]),
        )

    # --- Source 2: readiness key strengths ---------------------------------
    if readiness is not None:
        for statement in readiness.key_strengths:
            matched_dimension = ""
            for dimension in assessment_confidence:
                keyword = dimension.split("_")[0]
                if keyword and keyword in statement.lower():
                    matched_dimension = dimension
                    break
            register.add(
                label=f"readiness strength ({statement.split(':')[0][:40].strip().lower()})",
                dimension=matched_dimension,
                impact=SynthesisSeverity.MODERATE,
                source="readiness:key_strength",
                statements=[statement],
                observations=[
                    obs
                    for obs in observations
                    if not matched_dimension or obs.dimension == matched_dimension
                ],
                confidence=assessment_confidence.get(matched_dimension, 1.0),
            )

        # --- Source 3: reinforcing signal relationships --------------------
        reinforcing_by_dimension: dict[str, list[str]] = {}
        for relationship in readiness.signal_relationships:
            if relationship.relationship_type != "reinforcing":
                continue
            description = relationship.description
            reinforcing_by_dimension.setdefault(
                relationship.source_dimension, []
            ).append(description)
            reinforcing_by_dimension.setdefault(
                relationship.target_dimension, []
            ).append(description)
        for dimension, descriptions in sorted(reinforcing_by_dimension.items()):
            impact = (
                SynthesisSeverity.HIGH if len(descriptions) >= 3 else SynthesisSeverity.MODERATE
            )
            register.add(
                label=f"{dimension.replace('_', ' ')} reinforcing signals",
                dimension=dimension,
                impact=impact,
                source="readiness:reinforcing_relationship",
                statements=descriptions,
                observations=[obs for obs in observations if obs.dimension == dimension],
                confidence=1.0,
            )

    # --- Source 4: positive quantitative metrics ---------------------------
    if features.arr_usd is not None and features.arr_usd >= _STRONG_ARR_USD:
        register.add(
            label="strong annual recurring revenue",
            dimension="traction_signals",
            impact=SynthesisSeverity.HIGH,
            source="feature:arr_usd",
            statements=[f"ARR of ${features.arr_usd:,.0f} clears the "
                        f"${_STRONG_ARR_USD:,.0f} strength threshold"],
            observations=[obs for obs in observations if obs.dimension == "traction_signals"],
            confidence=1.0,
        )
    if features.nrr_pct is not None and features.nrr_pct >= _STRONG_NRR_PCT:
        register.add(
            label="expansion-grade net revenue retention",
            dimension="traction_signals",
            impact=SynthesisSeverity.HIGH,
            source="feature:nrr_pct",
            statements=[f"NRR of {features.nrr_pct:.0f}% is at or above "
                        f"{_STRONG_NRR_PCT:.0f}%"],
            observations=[obs for obs in observations if obs.dimension == "traction_signals"],
            confidence=1.0,
        )
    if features.growth_rate_pct is not None and features.growth_rate_pct >= _STRONG_GROWTH_PCT:
        register.add(
            label="high growth rate",
            dimension="traction_signals",
            impact=SynthesisSeverity.MODERATE,
            source="feature:growth_rate_pct",
            statements=[f"Growth rate of {features.growth_rate_pct:.0f}% exceeds "
                        f"{_STRONG_GROWTH_PCT:.0f}%"],
            observations=[obs for obs in observations if obs.dimension == "traction_signals"],
            confidence=1.0,
        )
    if features.ltv_cac_ratio is not None and features.ltv_cac_ratio >= _STRONG_LTV_CAC:
        register.add(
            label="strong unit economics",
            dimension="business_model_viability",
            impact=SynthesisSeverity.HIGH,
            source="feature:ltv_cac_ratio",
            statements=[f"LTV/CAC ratio of {features.ltv_cac_ratio:.2f} is at or above "
                        f"{_STRONG_LTV_CAC:.1f}"],
            observations=[
                obs for obs in observations
                if obs.dimension == "business_model_viability"
            ],
            confidence=1.0,
        )

    # --- Source 5: strong-evidence observations ----------------------------
    for obs in observations:
        if obs.trust_score < _STRONG_TRUST or obs.confidence < _STRONG_OBSERVATION_CONFIDENCE:
            continue
        register.add(
            label=f"strongly evidenced {obs.dimension.replace('_', ' ')} signal",
            dimension=obs.dimension,
            impact=SynthesisSeverity.LOW,
            source="observation:strong_evidence",
            statements=[obs.statement],
            observations=[obs],
            confidence=obs.confidence,
        )

    return register.ranked(max_opportunities)
