"""Unified risk register aggregation (Sprint 6C).

Aggregates risks from already-produced artifacts:
  - feature risks (ExtractedFeatures *_risk lists)
  - critical quantitative metrics (mirroring scorer thresholds)
  - consistency conflicts and unsupported observations
  - low-confidence assessments
  - assessed dimensions without observations (missing evidence)
  - weakening calibration factors
  - conflicting signal relationships

Deduplicates by normalized label, ranks by severity then dimension,
carries citations, provenance, and propagated (minimum) confidence.
"""

from __future__ import annotations

import re

from predictron_engine.decision.models import DecisionConfidence
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    DimensionAssessment,
    EvidenceCitation,
    InvestmentReadiness,
    Observation,
    RiskItem,
    SynthesisSeverity,
    severity_order,
)

# Quantitative risk thresholds mirror the traction scorer / readiness
# assessment conventions so no new judgment scale is introduced.
_CRITICAL_RUNWAY_MONTHS = 6
_ELEVATED_RUNWAY_MONTHS = 12
_NRR_FLOOR_PCT = 90.0
_CHURN_CEILING_PCT = 10.0
_LTV_CAC_FLOOR = 1.0

# A dimension assessment below this confidence becomes a risk entry.
_LOW_ASSESSMENT_CONFIDENCE = 0.35

_MAX_STATEMENTS_PER_ITEM = 5
_MAX_RISKS = 10

_WHITESPACE_RE = re.compile(r"\s+")

_FEATURE_RISK_FIELDS: tuple[tuple[str, str], ...] = (
    ("market_risk", "market"),
    ("founder_risk", "founder_quality"),
    ("execution_risk", "team_execution"),
    ("product_risk", "product_strength"),
    ("technology_risk", "product_strength"),
    ("business_model_risk", "business_model_viability"),
    ("traction_risk", "traction_signals"),
    ("competitive_risk", "competitive_position"),
    ("regulatory_risk", "market_opportunity"),
    ("operational_risk", "team_execution"),
    ("platform_dependency_risk", "business_model_viability"),
    ("customer_concentration_risk", "traction_signals"),
    ("hiring_risk", "team_execution"),
    ("funding_risk", "business_model_viability"),
    ("scaling_risk", "traction_signals"),
    ("security_risk", "product_strength"),
    ("compliance_risk", "market_opportunity"),
)


def _normalize_label(label: str) -> str:
    return _WHITESPACE_RE.sub(" ", label.strip().lower()).rstrip(".")


def _severity_for_count(count: int) -> SynthesisSeverity:
    if count >= 8:
        return SynthesisSeverity.HIGH
    if count >= 4:
        return SynthesisSeverity.MODERATE
    return SynthesisSeverity.LOW


def _collect_observations(
    observations: list[Observation],
    dimension: str | None = None,
) -> list[Observation]:
    if dimension is None:
        return list(observations)
    return [obs for obs in observations if obs.dimension == dimension]


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
        self._items: dict[str, RiskItem] = {}

    def add(
        self,
        *,
        label: str,
        dimension: str,
        severity: SynthesisSeverity,
        source: str,
        statements: list[str],
        observations: list[Observation],
        confidence: float,
    ) -> None:
        statements = [s for s in statements if s][:_MAX_STATEMENTS_PER_ITEM]
        key = _normalize_label(label)
        existing = self._items.get(key)
        if existing is None:
            self._items[key] = RiskItem(
                label=label.strip(),
                dimension=dimension,
                severity=severity,
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
        self._items[key] = RiskItem(
            label=existing.label,
            dimension=existing.dimension or dimension,
            severity=(
                severity
                if severity_order(severity) < severity_order(existing.severity)
                else existing.severity
            ),
            sources=[*existing.sources, source]
            if source not in existing.sources
            else existing.sources,
            statements=merged_statements,
            citations=merged_citations,
            provenance_document_ids=merged_provenance,
            confidence=round(new_confidence, 4),
        )

    def ranked(self, max_items: int) -> list[RiskItem]:
        items = sorted(
            self._items.values(),
            key=lambda item: (severity_order(item.severity), item.dimension, item.label),
        )
        return items[:max_items]


def aggregate_risks(
    features: ExtractedFeatures,
    observations: list[Observation],
    assessments: list[DimensionAssessment],
    readiness: InvestmentReadiness | None,
    decision_confidence: DecisionConfidence | None,
    consistency: object | None = None,
    contradiction_graph: object | None = None,
    *,
    max_risks: int = _MAX_RISKS,
) -> list[RiskItem]:
    """Build the unified, deduplicated, severity-ranked risk register."""
    register = _Register()

    obs_by_dim: dict[str, list[Observation]] = {}
    for obs in observations:
        obs_by_dim.setdefault(obs.dimension, []).append(obs)

    # --- Source 1: feature-level risk flags ------------------------------
    risk_confidence = float(getattr(features, "risk_confidence", 0.5))
    for field_name, dimension in _FEATURE_RISK_FIELDS:
        flagged = getattr(features, field_name, []) or []
        if not flagged:
            continue
        register.add(
            label=f"{field_name.replace('_', ' ')} flags",
            dimension=dimension,
            severity=_severity_for_count(len(flagged)),
            source=f"feature:{field_name}",
            statements=[str(item) for item in flagged],
            observations=obs_by_dim.get(dimension, []),
            confidence=risk_confidence,
        )

    # --- Source 2: critical quantitative metrics -------------------------
    runway = features.runway_months
    if runway is not None and runway < _CRITICAL_RUNWAY_MONTHS:
        register.add(
            label="critical runway",
            dimension="business_model_viability",
            severity=SynthesisSeverity.CRITICAL,
            source="feature:runway_months",
            statements=[
                f"Runway of {runway} months is below the "
                f"{_CRITICAL_RUNWAY_MONTHS}-month critical threshold"
            ],
            observations=obs_by_dim.get("business_model_viability", []),
            confidence=1.0,
        )
    elif (
        runway is not None
        and runway < _ELEVATED_RUNWAY_MONTHS
        and (features.burn_rate_usd or 0) > 0
    ):
        register.add(
            label="elevated burn relative to runway",
            dimension="business_model_viability",
            severity=SynthesisSeverity.HIGH,
            source="feature:runway_months",
            statements=[
                f"Runway of {runway} months with active monthly burn"
            ],
            observations=obs_by_dim.get("business_model_viability", []),
            confidence=1.0,
        )

    nrr = features.nrr_pct
    if nrr is not None and nrr < _NRR_FLOOR_PCT:
        register.add(
            label="weak net revenue retention",
            dimension="traction_signals",
            severity=SynthesisSeverity.HIGH,
            source="feature:nrr_pct",
            statements=[f"Net revenue retention of {nrr:.0f}% is below {_NRR_FLOOR_PCT:.0f}%"],
            observations=obs_by_dim.get("traction_signals", []),
            confidence=1.0,
        )

    churn = features.churn_rate_pct
    if churn is not None and churn > _CHURN_CEILING_PCT:
        register.add(
            label="high customer churn",
            dimension="traction_signals",
            severity=SynthesisSeverity.HIGH,
            source="feature:churn_rate_pct",
            statements=[f"Churn rate of {churn:.1f}% exceeds {_CHURN_CEILING_PCT:.0f}%"],
            observations=obs_by_dim.get("traction_signals", []),
            confidence=1.0,
        )

    ltv_cac = features.ltv_cac_ratio
    if ltv_cac is not None and ltv_cac < _LTV_CAC_FLOOR:
        register.add(
            label="unit economics below break-even",
            dimension="business_model_viability",
            severity=SynthesisSeverity.HIGH,
            source="feature:ltv_cac_ratio",
            statements=[f"LTV/CAC ratio of {ltv_cac:.2f} is below {_LTV_CAC_FLOOR:.1f}"],
            observations=obs_by_dim.get("business_model_viability", []),
            confidence=1.0,
        )

    # --- Source 3: consistency conflicts ---------------------------------
    if consistency is not None:
        contradictions = getattr(consistency, "contradictions", []) or []
        conflict_severity = (
            SynthesisSeverity.HIGH if len(contradictions) > 3 else SynthesisSeverity.MODERATE
        )
        for finding in contradictions:
            register.add(
                label="conflicting internal signals",
                dimension="",
                severity=conflict_severity,
                source="consistency:contradiction",
                statements=[str(finding.reason)],
                observations=[],
                confidence=1.0,
            )
        unsupported = getattr(consistency, "unsupported_observations", []) or []
        for finding in unsupported:
            register.add(
                label="unsupported conclusion",
                dimension="",
                severity=SynthesisSeverity.MODERATE,
                source="consistency:unsupported_observation",
                statements=[str(finding.statement)],
                observations=[],
                confidence=1.0,
            )

    # --- Source 4: conflicting cross-signal relationships -----------------
    if readiness is not None:
        conflicts_by_dimension: dict[str, list[str]] = {}
        for relationship in readiness.signal_relationships:
            if relationship.relationship_type != "conflicting":
                continue
            description = relationship.description
            conflicts_by_dimension.setdefault(relationship.source_dimension, []).append(description)
            conflicts_by_dimension.setdefault(relationship.target_dimension, []).append(description)
        for dimension, descriptions in sorted(conflicts_by_dimension.items()):
            severity = (
                SynthesisSeverity.HIGH if len(descriptions) >= 3 else SynthesisSeverity.MODERATE
            )
            register.add(
                label=f"{dimension.replace('_', ' ')} signal conflict",
                dimension=dimension,
                severity=severity,
                source="readiness:conflicting_relationship",
                statements=descriptions,
                observations=obs_by_dim.get(dimension, []),
                confidence=1.0,
            )

    # --- Source 4b: contradiction graph conflicts (Sprint P8D) ------------
    # Reuses the richer O(n²) pairwise contradiction graph when the
    # reasoning layer produced one; falls back gracefully when absent.
    if contradiction_graph is not None:
        graph_edges = getattr(contradiction_graph, "edges", ()) or ()
        columns: dict[str, list[str]] = {}
        for edge in graph_edges:
            if not getattr(edge, "is_conflict", False):
                continue
            dim = edge.dimension or ""
            columns.setdefault(dim, []).append(getattr(edge, "reason", "") or "")
        for dim, reasons in sorted(columns.items()):
            severity = (
                SynthesisSeverity.HIGH if len(reasons) >= 3 else SynthesisSeverity.MODERATE
            )
            register.add(
                label=f"{dim.replace(':', ' vs ').replace('_', ' ')} contradiction",
                dimension=dim,
                severity=severity,
                source="contradiction_graph:edge",
                statements=reasons[:_MAX_STATEMENTS_PER_ITEM],
                observations=obs_by_dim.get(dim, []),
                confidence=1.0,
            )
        dominant = getattr(contradiction_graph, "dominant_conflict", None)
        if dominant is not None:
            register.add(
                label="dominant contradiction",
                dimension=getattr(dominant, "dimension", "") or "",
                severity=SynthesisSeverity.HIGH,
                source="contradiction_graph:dominant_conflict",
                statements=[getattr(dominant, "explanation", "") or ""],
                observations=[],
                confidence=1.0,
            )

    # --- Source 5: low-confidence assessments -----------------------------
    for assessment in assessments:
        if assessment.confidence >= _LOW_ASSESSMENT_CONFIDENCE:
            continue
        register.add(
            label=f"low confidence {assessment.dimension.replace('_', ' ')} assessment",
            dimension=assessment.dimension,
            severity=SynthesisSeverity.MODERATE,
            source="evaluation:low_confidence_assessment",
            statements=[
                f"{assessment.dimension.replace('_', ' ').title()} assessment "
                f"confidence is only {assessment.confidence:.2f}"
            ],
            observations=assessment.supporting_observations,
            confidence=assessment.confidence,
        )

    # --- Source 6: assessed dimensions without observations ---------------
    observed_dimensions = {obs.dimension for obs in observations}
    for assessment in assessments:
        if assessment.dimension in observed_dimensions:
            continue
        register.add(
            label=f"{assessment.dimension.replace('_', ' ')} evidence gap",
            dimension=assessment.dimension,
            severity=SynthesisSeverity.LOW,
            source="evidence:missing_dimension_evidence",
            statements=[
                f"No reasoning observations support the "
                f"{assessment.dimension.replace('_', ' ')} assessment"
            ],
            observations=assessment.supporting_observations,
            confidence=min(assessment.confidence, 0.5),
        )

    # --- Source 7: weakening calibration factors ---------------------------
    if decision_confidence is not None and decision_confidence.weakening_factors:
        register.add(
            label="weakening calibration factors",
            dimension="",
            severity=SynthesisSeverity.LOW,
            source="calibration:weakening_factors",
            statements=list(decision_confidence.weakening_factors),
            observations=[],
            confidence=decision_confidence.confidence,
        )

    return register.ranked(max_risks)
