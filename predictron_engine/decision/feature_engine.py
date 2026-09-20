"""Decision Feature Engine — consumes Feature Store outputs for decision intelligence.

Normalizes feature values, computes weighted contributions, and preserves
evidence provenance. Never recomputes raw features — only consumes existing
feature snapshots.

Key principles:
  - Pure consumption of Feature Store snapshots
  - Deterministic normalization with named constants
  - Evidence provenance preserved on every output
  - No external API calls, no randomness, no I/O
"""

from __future__ import annotations

import math
from typing import Any

from predictron_engine.decision.intelligence_models import (
    Contribution,
    ContributionType,
)
from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    FeatureCategory,
    FeatureSnapshot,
    FeatureStatus,
)

# ---------------------------------------------------------------------------
# Normalization constants — named, deterministic thresholds.
# ---------------------------------------------------------------------------

# Feature categories and their default weights for decision contributions.
_CATEGORY_WEIGHTS: dict[FeatureCategory, float] = {
    FeatureCategory.COMPANY: 0.12,
    FeatureCategory.GROWTH: 0.20,
    FeatureCategory.FOUNDER: 0.18,
    FeatureCategory.FUNDING: 0.18,
    FeatureCategory.KNOWLEDGE_GRAPH: 0.10,
    FeatureCategory.SIGNALS: 0.12,
    FeatureCategory.BENCHMARK: 0.10,
}

# Value range constants for normalization.
_FLOAT_MIN = 0.0
_FLOAT_MAX = 1.0
_SCORE_MIN = 0.0
_SCORE_MAX = 100.0
_PERCENTAGE_MIN = 0.0
_PERCENTAGE_MAX = 100.0
_BOOL_POSITIVE = 1.0
_BOOL_NEGATIVE = -1.0
_BOOL_NEUTRAL = 0.0

# Thresholds for contribution classification.
_POSITIVE_THRESHOLD = 0.1
_NEGATIVE_THRESHOLD = -0.1
_CONFIDENCE_BOOST_THRESHOLD = 0.6
_CONFIDENCE_THRESHOLD = 0.3

# Maximum features per category for contribution ranking.
_MAX_FEATURES_PER_CATEGORY = 10


class DecisionFeatureEngine:
    """Consumes Feature Store snapshots and produces normalized decision features.

    Responsibilities:
      - Consume Feature Store outputs (never recompute raw features)
      - Normalize feature values to [-1, 1] range
      - Compute weighted feature contributions
      - Preserve evidence provenance
      - Deterministic ordering of outputs
    """

    def __init__(
        self,
        category_weights: dict[FeatureCategory, float] | None = None,
    ) -> None:
        self._category_weights = category_weights or dict(_CATEGORY_WEIGHTS)

    @property
    def category_weights(self) -> dict[FeatureCategory, float]:
        """Deterministic category weight map."""
        return dict(self._category_weights)

    def normalize_feature(self, snapshot: FeatureSnapshot) -> float:
        """Normalize a feature value to [-1, 1].

        Deterministic mapping based on value type and known ranges.
        Never modifies the original snapshot.
        """
        if snapshot.status != FeatureStatus.COMPUTED:
            return 0.0
        if snapshot.value is None:
            return 0.0

        value = snapshot.value

        # Boolean normalization
        if isinstance(value, bool):
            return _BOOL_POSITIVE if value else _BOOL_NEGATIVE

        # Float normalization
        if isinstance(value, float):
            return self._normalize_float(value, snapshot)

        # Integer normalization
        if isinstance(value, int):
            return self._normalize_int(value, snapshot)

        # String normalization — categorical encoding
        if isinstance(value, str):
            return self._normalize_string(value, snapshot)

        # List normalization — length-based
        if isinstance(value, list):
            return self._normalize_list(value, snapshot)

        # Dict normalization — key count based
        if isinstance(value, dict):
            return self._normalize_dict(value, snapshot)

        return 0.0

    def compute_feature_weight(
        self,
        snapshot: FeatureSnapshot,
        all_snapshots: dict[str, FeatureSnapshot] | None = None,
    ) -> float:
        """Compute the weight of a feature based on its category and status.

        Deterministic: same inputs always produce the same weight.
        """
        category_weight = self._category_weights.get(snapshot.category, 0.1)

        # Adjust weight based on feature status
        if snapshot.status != FeatureStatus.COMPUTED:
            return 0.0

        # Boost weight if feature has evidence references
        evidence_boost = min(len(snapshot.evidence_references) * 0.02, 0.1)

        weight = category_weight + evidence_boost
        return round(min(weight, 1.0), 4)

    def compute_contribution(
        self,
        snapshot: FeatureSnapshot,
        all_snapshots: dict[str, FeatureSnapshot] | None = None,
    ) -> Contribution:
        """Compute a single feature's contribution to the decision.

        Returns a Contribution with normalized value, weight, and
        computed contribution preserving evidence provenance.
        """
        normalized = self.normalize_feature(snapshot)
        weight = self.compute_feature_weight(snapshot, all_snapshots)
        computed = round(normalized * weight, 6)

        contribution_type = self._classify_contribution(normalized)

        evidence_refs = [
            {
                "source_type": e.source_type,
                "source_id": e.source_id,
                "source_field": e.source_field,
                "confidence": e.confidence,
            }
            for e in snapshot.evidence_references
        ]

        human_explanation = self._build_feature_explanation(
            snapshot, normalized, contribution_type,
        )

        return Contribution(
            feature_id=snapshot.feature_id,
            feature_name=snapshot.feature_name,
            category=snapshot.category.value,
            contribution_type=contribution_type,
            raw_value=snapshot.value,
            normalized_value=round(normalized, 4),
            weight=weight,
            computed_contribution=computed,
            evidence_references=evidence_refs,
            human_explanation=human_explanation,
            supporting_evidence=self._build_evidence_summary(snapshot),
            provenance={
                "feature_id": snapshot.feature_id,
                "computation_version": snapshot.computation_version,
                "snapshot_id": snapshot.snapshot_id,
                "category": snapshot.category.value,
            },
        )

    def compute_all_contributions(
        self,
        feature_set: CompanyFeatureSet,
    ) -> list[Contribution]:
        """Compute contributions for all features in a company's feature set.

        Returns contributions in deterministic order (sorted by feature_id).
        """
        all_snapshots = dict(feature_set.features)
        contributions: list[Contribution] = []

        for feature_id in sorted(all_snapshots.keys()):
            snapshot = all_snapshots[feature_id]
            contribution = self.compute_contribution(snapshot, all_snapshots)
            contributions.append(contribution)

        return contributions

    def classify_contributions(
        self,
        contributions: list[Contribution],
    ) -> dict[str, list[Contribution]]:
        """Classify contributions into positive, negative, neutral, confidence.

        Returns a dict with keys: positive, negative, neutral, confidence.
        Each list is sorted by absolute computed_contribution (descending).
        """
        result: dict[str, list[Contribution]] = {
            "positive": [],
            "negative": [],
            "neutral": [],
            "confidence": [],
        }

        for c in contributions:
            if c.contribution_type == ContributionType.POSITIVE:
                result["positive"].append(c)
            elif c.contribution_type == ContributionType.NEGATIVE:
                result["negative"].append(c)
            elif c.contribution_type == ContributionType.CONFIDENCE:
                result["confidence"].append(c)
            else:
                result["neutral"].append(c)

        # Sort each list by absolute computed contribution (descending)
        for key in result:
            result[key] = sorted(
                result[key],
                key=lambda x: abs(x.computed_contribution),
                reverse=True,
            )

        return result

    def get_feature_provenance(
        self,
        feature_set: CompanyFeatureSet,
    ) -> list[dict[str, Any]]:
        """Extract provenance chain for all features in a company's set."""
        provenance: list[dict[str, Any]] = []

        for feature_id in sorted(feature_set.features.keys()):
            snapshot = feature_set.features[feature_id]
            provenance.append({
                "feature_id": snapshot.feature_id,
                "feature_name": snapshot.feature_name,
                "category": snapshot.category.value,
                "status": snapshot.status.value,
                "computation_version": snapshot.computation_version,
                "computed_at": snapshot.computed_at.isoformat(),
                "provenance": snapshot.provenance,
                "evidence_references": [
                    {
                        "source_type": e.source_type,
                        "source_id": e.source_id,
                        "source_field": e.source_field,
                        "confidence": e.confidence,
                    }
                    for e in snapshot.evidence_references
                ],
                "dependencies_resolved": snapshot.dependencies_resolved,
                "error_message": snapshot.error_message,
            })

        return provenance

    def get_all_evidence_references(
        self,
        feature_set: CompanyFeatureSet,
    ) -> list[dict[str, Any]]:
        """Collect all unique evidence references from a feature set."""
        seen: set[tuple[str, str, str]] = set()
        refs: list[dict[str, Any]] = []

        for feature_id in sorted(feature_set.features.keys()):
            snapshot = feature_set.features[feature_id]
            for e in snapshot.evidence_references:
                key = (e.source_type, e.source_id, e.source_field)
                if key not in seen:
                    seen.add(key)
                    refs.append({
                        "source_type": e.source_type,
                        "source_id": e.source_id,
                        "source_field": e.source_field,
                        "confidence": e.confidence,
                        "feature_id": snapshot.feature_id,
                    })

        return refs

    # ------------------------------------------------------------------
    # Internal normalization helpers
    # ------------------------------------------------------------------

    def _normalize_float(self, value: float, snapshot: FeatureSnapshot) -> float:
        """Normalize a float value to [-1, 1].

        Uses feature-specific ranges when available, otherwise
        applies sigmoid-like normalization centered at 0.
        """
        fid = snapshot.feature_id

        # Domain-specific normalization for known features
        ranges = _get_feature_ranges()
        if fid in ranges:
            low, high, center = ranges[fid]
            if high == low:
                return 0.0
            normalized = (value - center) / (high - low)
            return round(max(-1.0, min(1.0, normalized)), 4)

        # Default: normalize assuming [0, 1] range is already normalized
        if _FLOAT_MIN <= value <= _FLOAT_MAX:
            return round(2.0 * (value - 0.5), 4)
        # Out of [0,1] range — use tanh-like compression
        return round(math.tanh(value) if value != 0 else 0.0, 4)

    def _normalize_int(self, value: int, snapshot: FeatureSnapshot) -> float:
        """Normalize an integer value to [-1, 1]."""
        fid = snapshot.feature_id

        ranges = _get_feature_ranges()
        if fid in ranges:
            low, high, center = ranges[fid]
            if high == low:
                return 0.0
            normalized = (value - center) / (high - low)
            return round(max(-1.0, min(1.0, normalized)), 4)

        # Count-like features: saturate at reasonable bounds
        if value >= 100:
            return 1.0
        if value <= -100:
            return -1.0
        return round(value / 100.0, 4)

    def _normalize_string(self, value: str, snapshot: FeatureSnapshot) -> float:
        """Normalize a string value to [-1, 1].

        Categorical encoding based on known positive/negative indicators.
        """
        value_lower = value.lower().strip()

        # Positive indicators
        positive_terms = {
            "strong", "high", "excellent", "good", "growing", "positive",
            "successful", "profitable", "leading", "innovative", "scalable",
            "saas", "b2b", "enterprise", "expansion", "series_a", "series_b",
            "series_c", "growth",
        }
        negative_terms = {
            "weak", "low", "poor", "declining", "negative", "failing",
            "unprofitable", "lagging", "outdated", "niche", "decline",
            "seed", "pre_seed", "idea",
        }

        if value_lower in positive_terms:
            return 0.5
        if value_lower in negative_terms:
            return -0.3

        # Neutral strings
        return 0.0

    def _normalize_list(self, value: list[Any], snapshot: FeatureSnapshot) -> float:
        """Normalize a list value to [-1, 1] based on length."""
        length = len(value)
        if length == 0:
            return -0.2
        if length >= 10:
            return 0.8
        return round(length / 10.0, 4)

    def _normalize_dict(self, value: dict[str, Any], snapshot: FeatureSnapshot) -> float:
        """Normalize a dict value to [-1, 1] based on key count."""
        count = len(value)
        if count == 0:
            return -0.1
        if count >= 10:
            return 0.7
        return round(count / 10.0, 4)

    def _classify_contribution(self, normalized: float) -> ContributionType:
        """Classify a normalized value as positive, negative, neutral, or confidence."""
        if normalized >= _POSITIVE_THRESHOLD:
            return ContributionType.POSITIVE
        if normalized <= _NEGATIVE_THRESHOLD:
            return ContributionType.NEGATIVE
        return ContributionType.NEUTRAL

    def _build_feature_explanation(
        self,
        snapshot: FeatureSnapshot,
        normalized: float,
        contribution_type: ContributionType,
    ) -> str:
        """Build a human-readable explanation for a feature contribution."""
        direction = "positive" if normalized > 0 else "negative" if normalized < 0 else "neutral"
        magnitude = abs(normalized)

        if magnitude >= 0.7:
            strength = "strong"
        elif magnitude >= 0.3:
            strength = "moderate"
        else:
            strength = "weak"

        return (
            f"{snapshot.feature_name}: {strength} {direction} contribution "
            f"(normalized: {normalized:.2f}, weight: {self.compute_feature_weight(snapshot):.2f})"
        )

    def _build_evidence_summary(self, snapshot: FeatureSnapshot) -> str:
        """Build a summary of evidence supporting this feature."""
        if not snapshot.evidence_references:
            return f"Feature '{snapshot.feature_name}' computed from pipeline data"

        source_types = {e.source_type for e in snapshot.evidence_references}
        count = len(snapshot.evidence_references)

        return (
            f"Supported by {count} evidence reference(s) "
            f"from source types: {', '.join(sorted(source_types))}"
        )


def _get_feature_ranges() -> dict[str, tuple[float, float, float]]:
    """Feature-specific normalization ranges: (low, high, center).

    Returns ranges for features where domain-specific normalization
    is more appropriate than default normalization.
    """
    return {
        # Funding features
        "total_funding": (0.0, 1_000_000_000.0, 50_000_000.0),
        "funding_round_count": (0, 10, 3),
        "average_round_size": (0.0, 500_000_000.0, 10_000_000.0),
        "investor_count": (0, 30, 10),
        "funding_recency": (0.0, 1_095.0, 365.0),
        # Growth features
        "funding_velocity": (0.0, 2.0, 0.5),
        "hiring_velocity": (0.0, 3.0, 1.0),
        "milestone_frequency": (0.0, 4.0, 1.5),
        "growth_consistency": (0.0, 2.0, 0.5),
        "momentum_score": (0.0, 10.0, 3.0),
        # Founder features
        "founder_count": (0, 8, 2),
        "repeat_founder_indicator": (0.0, 1.0, 0.5),
        "founder_change_count": (0, 5, 1),
        # Company features
        "company_age": (0.0, 20.0, 3.0),
        # Knowledge graph features
        "graph_degree": (0.0, 50.0, 10.0),
        "graph_density": (0.0, 1.0, 0.3),
        "connected_component_size": (0.0, 500.0, 50.0),
        "ecosystem_connections": (0.0, 200.0, 20.0),
        # Signals features
        "signal_frequency": (0.0, 20.0, 6.0),
        "signal_diversity": (0.0, 1.0, 0.4),
        "signal_recency": (0.0, 1.0, 0.5),
        "activity_score": (0.0, 20.0, 6.0),
        # Benchmark features
        "benchmark_similarity": (0.0, 1.0, 0.5),
        "historical_success_rate": (0.0, 1.0, 0.5),
        "sector_accuracy_reference": (0.0, 1.0, 0.5),
    }
