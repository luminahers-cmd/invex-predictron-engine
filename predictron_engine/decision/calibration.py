"""Decision Calibration — deterministic confidence & uncertainty (Sprint 6B).

Calibrates investment decisions by combining outputs that already exist
in the pipeline.  This module introduces **no** new evidence gathering,
trust scoring, provenance, citation logic, or pipeline stages — every
input is an existing pipeline artifact:

* evidence trust          -> ``EvidenceBundle.trust_summary`` /
                             ``EvidenceDocument.metadata.trust_score``
* extraction confidence   -> ``IntelligenceSummary`` quality metrics
* evidence agreement      -> ``Observation.evidence_agreement_ratio``
* reasoning confidence    -> ``Observation.confidence``
* evaluator agreement     -> ``DimensionAssessment.confidence`` spread
* feature completeness    -> ``ExtractedFeatures.data_completeness``
* evidence diversity      -> distinct ``source_provider`` values

Design
------
* Every function is pure: identical inputs always produce identical
  outputs.  No I/O, no global state, no randomness, no external
  dependencies.
* All weights and thresholds are module-level named constants.
* Uncertainty is computed independently from confidence so the two can
  be inspected side by side (a decision can be uncertain because
  evidence is missing even when existing evidence is trusted).

Public API
----------
* :func:`compute_decision_confidence` — main entry point.
* :func:`classify_confidence_level` — deterministic band mapping.
* :func:`action_for_confidence_level` — deterministic action guidance.
* :func:`build_calibration_summary` — compact digest for reports.
* :func:`apply_recommendation_risk` — expose per-recommendation risk.

Ownership
---------
This module owns **decision-level** calibration: a
:class:`~predictron_engine.decision.models.DecisionConfidence` (combined
confidence *and* uncertainty) for a single investment decision, built
exclusively from existing pipeline artifacts.  It is distinct from two
other confidence implementations in the codebase:

* :mod:`predictron_engine.reasoning.confidence` — reasoning-layer
  confidence for the observation set.
* :mod:`predictron_engine.confidence.confidence_engine` — per-dimension
  :class:`ConfidenceAssessment` for each scored dimension.

These three compute confidence for different pipeline artifacts at
different stages and are intentionally NOT consolidated.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from predictron_engine.decision.models import (
    CalibrationSummary,
    ConfidenceBreakdown,
    ConfidenceFactor,
    ConfidenceLevel,
    DecisionConfidence,
    UncertaintyBreakdown,
)

if TYPE_CHECKING:
    from predictron_engine.evidence.models import EvidenceBundle
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import (
        DimensionAssessment,
        Observation,
        Recommendation,
        ScoreResult,
    )
    from predictron_engine.reasoning.contradiction_graph import (
        ContradictionGraph,
    )

# ---------------------------------------------------------------------------
# Confidence factor weights — sum to 1.0.  Deterministic by design.
# ---------------------------------------------------------------------------

_WEIGHT_EVIDENCE_TRUST: float = 0.25
_WEIGHT_EVIDENCE_CONFIDENCE: float = 0.15
_WEIGHT_EVIDENCE_AGREEMENT: float = 0.15
_WEIGHT_REASONING_CONFIDENCE: float = 0.20
_WEIGHT_EVALUATOR_AGREEMENT: float = 0.10
_WEIGHT_FEATURE_COMPLETENESS: float = 0.10
_WEIGHT_EVIDENCE_DIVERSITY: float = 0.05

# ---------------------------------------------------------------------------
# Uncertainty driver weights — sum to 1.0.  Deterministic by design.
# ---------------------------------------------------------------------------

_WEIGHT_MISSING_EVIDENCE: float = 0.25
_WEIGHT_CONFLICTING_EVIDENCE: float = 0.25
_WEIGHT_LOW_TRUST: float = 0.20
_WEIGHT_LOW_COVERAGE: float = 0.15
_WEIGHT_EVALUATOR_DISAGREEMENT: float = 0.15

# ---------------------------------------------------------------------------
# Component thresholds and normalization constants.
# ---------------------------------------------------------------------------

# Confidence bands: score >= threshold => level (mirrors conviction style).
_CONFIDENCE_THRESHOLDS: list[tuple[float, ConfidenceLevel]] = [
    (0.80, ConfidenceLevel.VERY_HIGH),
    (0.65, ConfidenceLevel.HIGH),
    (0.45, ConfidenceLevel.MEDIUM),
    (0.25, ConfidenceLevel.LOW),
    (0.00, ConfidenceLevel.VERY_LOW),
]

# A factor is "strong" (supports confidence) at or above this value...
_STRONG_FACTOR_THRESHOLD: float = 0.65
# ...and "weak" (undermines confidence) at or below this value.
_WEAK_FACTOR_THRESHOLD: float = 0.35

# Diversity saturates at this many distinct source providers (same
# normalization convention as extraction-time source diversity).
_DIVERSITY_SATURATION_PROVIDERS: int = 3

# Reasoning coverage saturates at this many observations per dimension
# (mirrors the confidence engine's full-confidence observation count).
_FULL_COVERAGE_OBS_PER_DIMENSION: int = 3

# Evaluator confidences spread by this population std-dev or more are
# treated as full disagreement.
_MAX_DISAGREEMENT_STD: float = 0.25

# Official-document support threshold for explanation generation.
_OFFICIAL_SUPPORT_TRUST_THRESHOLD: float = 0.70

# Maximum statements kept per explanation list.
_MAX_EXPLANATIONS: int = 8

# Per-recommendation risk shaping constants (deterministic).
_REC_FLOOR_BLEND: float = 0.60  # weight on decision confidence floor
_REC_UNCERTAINTY_ADJUSTMENT: float = 0.10  # penalty for unsupported recs


# ---------------------------------------------------------------------------
# Public API — calibration entry point
# ---------------------------------------------------------------------------


def compute_decision_confidence(
    *,
    bundle: EvidenceBundle | None,
    observations: list[Observation],
    assessments: list[DimensionAssessment],
    features: ExtractedFeatures,
    scores: list[ScoreResult] | None = None,
    contradiction_graph: ContradictionGraph | None = None,
) -> DecisionConfidence:
    """Compute calibrated decision confidence from existing pipeline outputs.

    Parameters
    ----------
    bundle:
        The collected ``EvidenceBundle`` (may be empty or None).
    observations:
        Reasoning-layer observations (already evidence-enriched).
    assessments:
        Evaluator dimension assessments.
    features:
        Extracted factual features.
    scores:
        Optional scoring results used only to identify assessed
        dimensions when computing missing-evidence uncertainty.
    contradiction_graph:
        Optional pre-built contradiction graph.  When supplied the
        ``conflicting_evidence`` uncertainty driver reuses the richer
        O(n²) pairwise conflict count instead of only the coarse
        per-observation ``evidence_conflict_count`` proxy.
    """
    score_list = scores or []
    evidence_trust = compute_evidence_trust(bundle)
    breakdown = _compute_confidence_breakdown(
        bundle=bundle,
        observations=observations,
        assessments=assessments,
        features=features,
        evidence_trust=evidence_trust,
    )
    uncertainty_breakdown = _compute_uncertainty_breakdown(
        bundle=bundle,
        observations=observations,
        assessments=assessments,
        features=features,
        scores=score_list,
        evidence_trust=evidence_trust,
        contradiction_graph=contradiction_graph,
    )
    assessed_dimensions = (
        {a.dimension for a in assessments}
        | {s.dimension for s in score_list}
    )

    confidence = breakdown.composite
    uncertainty = uncertainty_breakdown.composite

    supporting = _collect_supporting_factors(
        breakdown=breakdown,
        uncertainty=uncertainty_breakdown,
        observations=observations,
        features=features,
    )
    weakening = _collect_weakening_factors(
        breakdown=breakdown,
        uncertainty=uncertainty_breakdown,
        observations=observations,
        features=features,
        assessed_dimensions=assessed_dimensions,
        contradiction_graph=contradiction_graph,
    )

    return DecisionConfidence(
        confidence=confidence,
        level=classify_confidence_level(confidence),
        uncertainty_score=uncertainty,
        breakdown=breakdown,
        uncertainty_breakdown=uncertainty_breakdown,
        supporting_factors=supporting,
        weakening_factors=weakening,
    )


def classify_confidence_level(confidence: float) -> ConfidenceLevel:
    """Map a confidence value in [0, 1] to a deterministic band."""
    clamped = max(0.0, min(1.0, confidence))
    for threshold, level in _CONFIDENCE_THRESHOLDS:
        if clamped >= threshold:
            return level
    return ConfidenceLevel.VERY_LOW


def action_for_confidence_level(level: ConfidenceLevel) -> str:
    """Return deterministic action guidance for a confidence band."""
    return {
        ConfidenceLevel.VERY_HIGH: (
            "Proceed with standard due diligence"
        ),
        ConfidenceLevel.HIGH: (
            "Proceed with due diligence; validate identified gaps"
        ),
        ConfidenceLevel.MEDIUM: (
            "Proceed only with enhanced due diligence; request missing evidence"
        ),
        ConfidenceLevel.LOW: (
            "Manual review required before acting on this analysis"
        ),
        ConfidenceLevel.VERY_LOW: (
            "Do not act on this analysis; collect substantially more evidence"
        ),
    }[level]


def build_calibration_summary(
    decision_confidence: DecisionConfidence,
    *,
    observation_count: int,
    assessed_dimension_count: int,
    evidence_document_count: int,
    top_n: int = 3,
) -> CalibrationSummary:
    """Build a compact report-facing digest of a decision calibration."""
    return CalibrationSummary(
        confidence=decision_confidence.confidence,
        level=decision_confidence.level,
        uncertainty_score=decision_confidence.uncertainty_score,
        recommended_action=action_for_confidence_level(
            decision_confidence.level,
        ),
        strong_factor_count=len(decision_confidence.supporting_factors),
        weak_factor_count=len(decision_confidence.weakening_factors),
        top_supporting_factors=(
            decision_confidence.supporting_factors[:top_n]
        ),
        top_weakening_factors=(
            decision_confidence.weakening_factors[:top_n]
        ),
        observation_count=max(0, observation_count),
        assessed_dimension_count=max(0, assessed_dimension_count),
        evidence_document_count=max(0, evidence_document_count),
    )


def apply_recommendation_risk(
    recommendations: list[Recommendation],
    decision_confidence: DecisionConfidence,
) -> list[Recommendation]:
    """Expose expected confidence, uncertainty, and action per recommendation.

    Returns new recommendation objects (the inputs are not mutated) with
    three additional risk fields populated deterministically:

    * ``expected_confidence`` — decision confidence blended toward a
      floor based on how well-supported the individual recommendation is.
    * ``expected_uncertainty`` — decision uncertainty plus a bounded
      penalty when the recommendation itself is weakly supported.
    * ``recommended_action`` — guidance for the calibrated level.
    """
    annotated: list[Recommendation] = []
    for rec in recommendations:
        rec_conf = max(0.0, min(1.0, rec.confidence))
        expected_confidence = round(
            decision_confidence.confidence
            * (_REC_FLOOR_BLEND + (1.0 - _REC_FLOOR_BLEND) * rec_conf),
            4,
        )
        expected_uncertainty = round(
            min(
                1.0,
                decision_confidence.uncertainty_score
                + (1.0 - rec_conf) * _REC_UNCERTAINTY_ADJUSTMENT,
            ),
            4,
        )
        annotated.append(
            rec.model_copy(
                update={
                    "expected_confidence": expected_confidence,
                    "expected_uncertainty": expected_uncertainty,
                    "recommended_action": action_for_confidence_level(
                        decision_confidence.level,
                    ),
                },
            ),
        )
    return annotated


# ---------------------------------------------------------------------------
# Confidence components — each derived from ONE existing pipeline output
# ---------------------------------------------------------------------------


def _compute_confidence_breakdown(
    *,
    bundle: EvidenceBundle | None,
    observations: list[Observation],
    assessments: list[DimensionAssessment],
    features: ExtractedFeatures,
    evidence_trust: float | None = None,
) -> ConfidenceBreakdown:
    """Compute the weighted confidence decomposition."""
    if evidence_trust is None:
        evidence_trust = compute_evidence_trust(bundle)
    values: dict[str, float] = {
        "evidence_trust": evidence_trust,
        "evidence_confidence": compute_evidence_confidence(bundle),
        "evidence_agreement": compute_evidence_agreement(observations),
        "reasoning_confidence": compute_reasoning_confidence(observations),
        "evaluator_agreement": compute_evaluator_agreement(assessments),
        "feature_completeness": round(
            max(0.0, min(1.0, features.data_completeness)),
            4,
        ),
        "evidence_diversity": compute_evidence_diversity(bundle),
    }
    weights: dict[str, float] = {
        "evidence_trust": _WEIGHT_EVIDENCE_TRUST,
        "evidence_confidence": _WEIGHT_EVIDENCE_CONFIDENCE,
        "evidence_agreement": _WEIGHT_EVIDENCE_AGREEMENT,
        "reasoning_confidence": _WEIGHT_REASONING_CONFIDENCE,
        "evaluator_agreement": _WEIGHT_EVALUATOR_AGREEMENT,
        "feature_completeness": _WEIGHT_FEATURE_COMPLETENESS,
        "evidence_diversity": _WEIGHT_EVIDENCE_DIVERSITY,
    }

    factors = [
        ConfidenceFactor(
            name=name,
            value=values[name],
            weight=weights[name],
            contribution=round(values[name] * weights[name], 4),
        )
        for name in weights  # dict order is insertion order — deterministic
    ]
    composite = round(sum(f.contribution for f in factors), 4)
    composite = max(0.0, min(1.0, composite))

    return ConfidenceBreakdown(factors=factors, composite=composite)


def compute_evidence_trust(bundle: EvidenceBundle | None) -> float:
    """Reuse bundle-level trust diagnostics (no re-scoring).

    Prefers the pre-computed ``TrustSummary.average_trust_score``;
    falls back to averaging per-document ``TrustScore.overall`` values.
    Returns 0.0 when no trust information exists.
    """
    if bundle is None:
        return 0.0
    summary = bundle.trust_summary
    if summary is not None and summary.total_documents > 0:
        return round(summary.average_trust_score, 4)

    trust_values = [
        doc.metadata.trust_score.overall
        for doc in bundle.documents
        if doc.metadata is not None and doc.metadata.trust_score is not None
    ]
    if not trust_values:
        return 0.0
    return round(sum(trust_values) / len(trust_values), 4)


def compute_evidence_confidence(bundle: EvidenceBundle | None) -> float:
    """Derive bundle-level extraction/evidence confidence.

    Combines existing Document Intelligence outputs (average content
    quality, classification confidence) with retrieval success ratio.
    When intelligence diagnostics are absent, only retrieval success is
    used.  Deterministic throughout.
    """
    if bundle is None:
        return 0.0

    retrieval_ratio = _retrieval_success_ratio(bundle)

    intelligence = bundle.intelligence
    if intelligence is None:
        return round(retrieval_ratio, 4)

    combined = (
        intelligence.average_quality * 0.50
        + intelligence.classification_confidence * 0.20
        + retrieval_ratio * 0.30
    )
    return round(max(0.0, min(1.0, combined)), 4)


def compute_evidence_agreement(observations: list[Observation]) -> float:
    """Mean corroboration ratio across observations (existing metadata)."""
    if not observations:
        return 0.0
    total = sum(o.evidence_agreement_ratio for o in observations)
    return round(max(0.0, min(1.0, total / len(observations))), 4)


def compute_reasoning_confidence(observations: list[Observation]) -> float:
    """Mean observation confidence produced by the reasoning layer."""
    if not observations:
        return 0.0
    total = sum(o.confidence for o in observations)
    return round(max(0.0, min(1.0, total / len(observations))), 4)


def compute_evaluator_agreement(
    assessments: list[DimensionAssessment],
) -> float:
    """Agreement across evaluators from existing assessment confidences.

    Agreement is the mean assessment confidence penalized by the
    normalized dispersion (population std-dev) of those confidences.
    Evaluators that agree strongly reinforce confidence; evaluators
    that diverge reduce it.
    """
    if not assessments:
        return 0.0

    confidences = [a.confidence for a in assessments]
    mean = sum(confidences) / len(confidences)
    variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
    std = math.sqrt(variance)
    disagreement = min(std / _MAX_DISAGREEMENT_STD, 1.0)

    agreement = mean * (1.0 - disagreement)
    return round(max(0.0, min(1.0, agreement)), 4)


def compute_evidence_diversity(bundle: EvidenceBundle | None) -> float:
    """Distinct source-provider count, saturated like extraction diversity."""
    if bundle is None or not bundle.documents:
        return 0.0

    providers = {
        doc.metadata.source_provider
        for doc in bundle.documents
        if doc.metadata is not None and doc.metadata.source_provider
    }
    if not providers:
        return 0.0
    return round(
        min(len(providers) / _DIVERSITY_SATURATION_PROVIDERS, 1.0),
        4,
    )


def _retrieval_success_ratio(bundle: EvidenceBundle) -> float:
    """Fraction of recorded retrieval attempts that succeeded."""
    if not bundle.sources:
        return 0.0
    successful = sum(1 for s in bundle.sources if s.success)
    return successful / len(bundle.sources)


# ---------------------------------------------------------------------------
# Uncertainty drivers
# ---------------------------------------------------------------------------


def _compute_uncertainty_breakdown(
    *,
    bundle: EvidenceBundle | None,
    observations: list[Observation],
    assessments: list[DimensionAssessment],
    features: ExtractedFeatures,
    scores: list[ScoreResult],
    evidence_trust: float | None = None,
    contradiction_graph: ContradictionGraph | None = None,
) -> UncertaintyBreakdown:
    """Compute the weighted uncertainty decomposition.

    When a ``contradiction_graph`` is supplied, the
    ``conflicting_evidence`` driver reuses the richer O(n²) pairwise
    conflict count instead of only the coarse per-observation proxy.
    """
    observed_dimensions = {o.dimension for o in observations}
    assessed_dimensions = {
        a.dimension for a in assessments
    } | {s.dimension for s in scores}

    if evidence_trust is None:
        evidence_trust = compute_evidence_trust(bundle)
    values: dict[str, float] = {
        "missing_evidence": _missing_evidence_factor(
            assessed_dimensions, observed_dimensions, bool(observations),
        ),
        "conflicting_evidence": _conflicting_evidence_factor(
            observations, contradiction_graph=contradiction_graph,
        ),
        "low_trust": round(1.0 - evidence_trust, 4),
        "low_coverage": _low_coverage_factor(
            observations, assessed_dimensions,
        ),
        "evaluator_disagreement": _evaluator_disagreement_factor(assessments),
    }
    weights: dict[str, float] = {
        "missing_evidence": _WEIGHT_MISSING_EVIDENCE,
        "conflicting_evidence": _WEIGHT_CONFLICTING_EVIDENCE,
        "low_trust": _WEIGHT_LOW_TRUST,
        "low_coverage": _WEIGHT_LOW_COVERAGE,
        "evaluator_disagreement": _WEIGHT_EVALUATOR_DISAGREEMENT,
    }

    factors = [
        ConfidenceFactor(
            name=name,
            value=max(0.0, min(1.0, values[name])),
            weight=weights[name],
            contribution=round(
                max(0.0, min(1.0, values[name])) * weights[name], 4,
            ),
        )
        for name in weights  # insertion order — deterministic
    ]
    composite = round(sum(f.contribution for f in factors), 4)
    composite = max(0.0, min(1.0, composite))

    return UncertaintyBreakdown(factors=factors, composite=composite)


def _missing_evidence_factor(
    assessed_dimensions: set[str],
    observed_dimensions: set[str],
    has_observations: bool,
) -> float:
    """Fraction of assessed dimensions with no supporting observation.

    With no assessed dimensions at all, absence of observations means
    nothing is known (1.0); observations without assessments imply full
    dimensional coverage of what was asked (0.0).
    """
    if not assessed_dimensions:
        return 0.0 if has_observations else 1.0
    missing = assessed_dimensions - observed_dimensions
    return round(len(missing) / len(assessed_dimensions), 4)


def _negative_signal_categories() -> frozenset[str]:
    """Shared negative-signal category set (lazy to avoid import cycles)."""
    from predictron_engine.models.report import NEGATIVE_SIGNAL_CATEGORIES

    return NEGATIVE_SIGNAL_CATEGORIES


def _conflicting_evidence_factor(
    observations: list[Observation],
    *,
    contradiction_graph: ContradictionGraph | None = None,
) -> float:
    """Conflict density across observations using existing metadata.

    Counts both per-observation conflict tallies and observations whose
    category belongs to the shared negative-signal set (produced by the
    reasoning layer).  When a ``contradiction_graph`` is available its
    richer O(n²) ``conflicting_count`` replaces the coarse
    per-observation tally.
    """
    if not observations:
        return 0.0
    if contradiction_graph is not None and contradiction_graph.conflicting_count > 0:
        conflicts = contradiction_graph.conflicting_count
    else:
        conflicts = sum(o.evidence_conflict_count for o in observations)
    conflicts += sum(
        1
        for o in observations
        if o.category in _negative_signal_categories()
    )
    return round(min(conflicts / len(observations), 1.0), 4)


def _low_coverage_factor(
    observations: list[Observation],
    assessed_dimensions: set[str],
) -> float:
    """Observation sparsity relative to assessed dimension count."""
    dimension_count = max(len(assessed_dimensions), 1)
    target = _FULL_COVERAGE_OBS_PER_DIMENSION * dimension_count
    coverage = min(len(observations) / target, 1.0)
    return round(1.0 - coverage, 4)


def _evaluator_disagreement_factor(
    assessments: list[DimensionAssessment],
) -> float:
    """Normalized dispersion of evaluator confidences."""
    if len(assessments) < 2:
        return 0.0
    confidences = [a.confidence for a in assessments]
    mean = sum(confidences) / len(confidences)
    variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
    std = math.sqrt(variance)
    return round(min(std / _MAX_DISAGREEMENT_STD, 1.0), 4)


# ---------------------------------------------------------------------------
# Structured explanations — deterministic template rules
# ---------------------------------------------------------------------------


def _collect_supporting_factors(
    *,
    breakdown: ConfidenceBreakdown,
    uncertainty: UncertaintyBreakdown,
    observations: list[Observation],
    features: ExtractedFeatures,
) -> list[str]:
    """Deterministic statements explaining why confidence is high."""
    statements: list[str] = []

    def _value(name: str) -> float:
        return breakdown.value_of(name) or 0.0

    trust = _value("evidence_trust")
    if trust >= _STRONG_FACTOR_THRESHOLD:
        statements.append(
            f"multiple trusted sources support the analysis "
            f"(evidence trust {trust:.2f})",
        )

    if _has_official_support(observations):
        statements.append("official website supports claims")

    agreement = _value("evidence_agreement")
    if agreement >= _STRONG_FACTOR_THRESHOLD:
        statements.append(
            f"reasoning is corroborated by multiple sources "
            f"(agreement {agreement:.2f})",
        )

    reasoning = _value("reasoning_confidence")
    if reasoning >= _STRONG_FACTOR_THRESHOLD:
        statements.append(
            f"reasoning observations are strong ({reasoning:.2f})",
        )

    evaluator = _value("evaluator_agreement")
    if evaluator >= _STRONG_FACTOR_THRESHOLD:
        statements.append(
            f"evaluators agree on dimension assessments ({evaluator:.2f})",
        )

    completeness = _value("feature_completeness")
    if completeness >= _STRONG_FACTOR_THRESHOLD:
        statements.append(
            f"strong feature completeness ({completeness:.0%})",
        )

    diversity = _value("evidence_diversity")
    if diversity >= 2.0 / _DIVERSITY_SATURATION_PROVIDERS:
        statements.append(
            f"evidence drawn from multiple distinct sources "
            f"(diversity {diversity:.2f})",
        )

    conflict_value = uncertainty.value_of("conflicting_evidence") or 0.0
    if observations and conflict_value == 0.0:
        statements.append("no major contradictions detected")

    return statements[:_MAX_EXPLANATIONS]


def _collect_weakening_factors(
    *,
    breakdown: ConfidenceBreakdown,
    uncertainty: UncertaintyBreakdown,
    observations: list[Observation],
    features: ExtractedFeatures,
    assessed_dimensions: set[str],
    contradiction_graph: ContradictionGraph | None = None,
) -> list[str]:
    """Deterministic statements explaining why confidence is low."""
    statements: list[str] = []

    def _value(name: str) -> float:
        return breakdown.value_of(name) or 0.0

    def _driver(name: str) -> float:
        return uncertainty.value_of(name) or 0.0

    if features.founder_profile_count == 0:
        statements.append("founder information missing")

    if not features.technology_stack:
        statements.append("technology evidence weak")

    conflicts = _count_conflicts(observations, contradiction_graph=contradiction_graph)
    if conflicts > 0:
        statements.append(
            f"{conflicts} conflicting evidence signal(s) detected",
        )

    if (
        contradiction_graph is not None
        and contradiction_graph.dominant_conflict is not None
    ):
        dc = contradiction_graph.dominant_conflict
        statements.append(
            f"dominant contradiction in {_label_dim(dc.dimension)}: "
            f"{dc.explanation}",
        )

    trust = _value("evidence_trust")
    if trust <= _WEAK_FACTOR_THRESHOLD:
        statements.append(f"evidence trust is low ({trust:.2f})")

    missing_dims = len(assessed_dimensions - {o.dimension for o in observations})
    if missing_dims > 0:
        statements.append(
            f"evidence missing for {missing_dims} assessed dimension(s)",
        )

    coverage = _driver("low_coverage")
    if coverage >= 0.60:
        statements.append("reasoning coverage is sparse")

    disagreement = _driver("evaluator_disagreement")
    if disagreement > 0.0:
        statements.append(
            f"evaluators disagree on dimension confidence "
            f"(disagreement {disagreement:.2f})",
        )

    completeness = _value("feature_completeness")
    if completeness <= _WEAK_FACTOR_THRESHOLD:
        statements.append(
            f"data completeness is limited ({completeness:.0%})",
        )

    if not observations:
        statements.append("no reasoning observations available")

    return statements[:_MAX_EXPLANATIONS]


def _has_official_support(observations: list[Observation]) -> bool:
    """True when any observation cites an officially-trusted document."""
    return any(
        o.trust_score >= _OFFICIAL_SUPPORT_TRUST_THRESHOLD
        for o in observations
    )


def _count_conflicts(
    observations: list[Observation],
    *,
    contradiction_graph: ContradictionGraph | None = None,
) -> int:
    """Total conflicting signals using existing observation metadata.

    When a ``contradiction_graph`` is available its richer conflict
    count is used instead of only the per-observation proxy.
    """
    if (
        contradiction_graph is not None
        and contradiction_graph.conflicting_count > 0
    ):
        conflicts = contradiction_graph.conflicting_count
    else:
        conflicts = sum(o.evidence_conflict_count for o in observations)
    conflicts += sum(
        1
        for o in observations
        if o.category in _negative_signal_categories()
    )
    return conflicts


def _label_dim(dimension: str) -> str:
    """Produce a human-friendly label for a dimension string."""
    return dimension.replace("_", " ") if dimension else "cross-dimension"
