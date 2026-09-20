"""Contribution Engine — computes feature contributions to investment decisions.

For every investment decision, categorizes contributions into:
  - positive contributors (features that support investment)
  - negative contributors (features that reduce investment attractiveness)
  - neutral contributors (features with no directional impact)
  - confidence contributors (features affecting confidence only)

Every contribution includes:
  - feature (id, name, category)
  - weight (deterministic feature weight)
  - computed contribution (normalized value * weight)
  - supporting evidence (evidence references with provenance)
  - human explanation (deterministic prose)

Key principles:
  - Fully deterministic: same features always produce the same contributions
  - Only consumes Feature Store snapshots; never recomputes raw features
  - No ML, no LLMs, no probabilistic black boxes
"""

from __future__ import annotations

from typing import Any

from predictron_engine.decision.feature_engine import DecisionFeatureEngine
from predictron_engine.decision.intelligence_models import (
    Contribution,
    ContributionType,
)
from predictron_engine.feature_store.models import CompanyFeatureSet

# ---------------------------------------------------------------------------
# Contribution computation constants.
# ---------------------------------------------------------------------------

# Thresholds used to reclassify contributions when evidence quality is low.
_LOW_EVIDENCE_RESOLUTION = 0.05

# Confidence contributors are features that indicate data availability
# rather than directional investment appeal.
_CONFIDENCE_FEATURE_PREFIXES = (
    "data_",
    "coverage_",
    "completeness_",
    "evidence_",
)

# Categories that typically reflect data quality rather than direction.
_CONFIDENCE_CATEGORIES = frozenset({"signals", "benchmark"})


class ContributionEngine:
    """Computes weighted contributions of features to an investment decision.

    Responsibilities:
      - Compute contribution for every feature
      - Classify contributions into positive/negative/neutral/confidence
      - Rank contributors by absolute computed contribution
      - Compute overall contribution aggregates
      - Generate deterministic human explanations per contribution
    """

    def __init__(
        self,
        feature_engine: DecisionFeatureEngine | None = None,
    ) -> None:
        self._feature_engine = feature_engine or DecisionFeatureEngine()

    @property
    def feature_engine(self) -> DecisionFeatureEngine:
        return self._feature_engine

    def compute_contributions(
        self,
        feature_set: CompanyFeatureSet,
    ) -> list[Contribution]:
        """Compute all contributions for a company's feature set.

        Returns contributions in deterministic order (by feature_id).
        """
        return self._feature_engine.compute_all_contributions(feature_set)

    def classify(
        self,
        contributions: list[Contribution],
    ) -> dict[str, list[Contribution]]:
        """Classify contributions into positive/negative/neutral/confidence.

        Classification applies the following deterministic rules:
          - confidence contributions: features that signal data availability
            are reclassified as confidence contributors when their
            computed contribution magnitude is below the resolution floor.
          - positive: normalized value above threshold
          - negative: normalized value below threshold
          - neutral: otherwise
        """
        classified: dict[str, list[Contribution]] = {
            "positive": [],
            "negative": [],
            "neutral": [],
            "confidence": [],
        }

        for c in contributions:
            final_type = self._reclassify_confidence(c)
            if final_type != c.contribution_type:
                c = c.model_copy(update={"contribution_type": final_type})
            bucket = {
                ContributionType.POSITIVE: "positive",
                ContributionType.NEGATIVE: "negative",
                ContributionType.NEUTRAL: "neutral",
                ContributionType.CONFIDENCE: "confidence",
            }[final_type]
            classified[bucket].append(c)

        # Sort each bucket by |computed_contribution| descending (deterministic).
        for key, items in classified.items():
            classified[key] = sorted(
                items,
                key=lambda x: abs(x.computed_contribution),
                reverse=True,
            )

        return classified

    def get_positive_contributors(
        self,
        feature_set: CompanyFeatureSet,
    ) -> list[Contribution]:
        """Return the top positive contributors for a company."""
        classified = self.classify(self.compute_contributions(feature_set))
        return classified["positive"]

    def get_negative_contributors(
        self,
        feature_set: CompanyFeatureSet,
    ) -> list[Contribution]:
        """Return the top negative contributors for a company."""
        classified = self.classify(self.compute_contributions(feature_set))
        return classified["negative"]

    def get_neutral_contributors(
        self,
        feature_set: CompanyFeatureSet,
    ) -> list[Contribution]:
        """Return the neutral contributors for a company."""
        classified = self.classify(self.compute_contributions(feature_set))
        return classified["neutral"]

    def get_confidence_contributors(
        self,
        feature_set: CompanyFeatureSet,
    ) -> list[Contribution]:
        """Return the confidence contributors for a company."""
        classified = self.classify(self.compute_contributions(feature_set))
        return classified["confidence"]

    def compute_contribution_score(
        self,
        feature_set: CompanyFeatureSet,
    ) -> float:
        """Compute the overall contribution score (0-100).

        Weighted sum of positive contributions minus weighted sum of
        negative contributions, scaled to [0, 100].

        Formula:
          net = sum(positive.computed_contribution)
                - sum(|negative.computed_contribution|)
          score = clip(50 + net * 50, 0, 100)
        """
        classified = self.classify(self.compute_contributions(feature_set))

        positive_total = sum(c.computed_contribution for c in classified["positive"])
        negative_total = sum(
            abs(c.computed_contribution) for c in classified["negative"]
        )

        net = positive_total - negative_total
        score = 50.0 + net * 50.0
        return round(max(0.0, min(100.0, score)), 4)

    def compute_positive_ratio(self, feature_set: CompanyFeatureSet) -> float:
        """Fraction of directional contributions that are positive."""
        classified = self.classify(self.compute_contributions(feature_set))
        positive_count = len(classified["positive"])
        negative_count = len(classified["negative"])
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        return round(positive_count / total, 4)

    def compute_dimension_scores(
        self,
        feature_set: CompanyFeatureSet,
    ) -> dict[str, float]:
        """Compute per-category dimension scores (0-100).

        Each category's dimension score is the weighted sum of its
        contributions scaled to [0, 100].
        """
        contributions = self.compute_contributions(feature_set)
        by_category: dict[str, list[Contribution]] = {}

        for c in contributions:
            by_category.setdefault(c.category, []).append(c)

        scores: dict[str, float] = {}
        for category, items in sorted(by_category.items()):
            positive_total = sum(
                c.computed_contribution for c in items
                if c.contribution_type == ContributionType.POSITIVE
            )
            negative_total = sum(
                abs(c.computed_contribution) for c in items
                if c.contribution_type == ContributionType.NEGATIVE
            )
            net = positive_total - negative_total
            score = 50.0 + net * 50.0
            scores[category] = round(max(0.0, min(100.0, score)), 4)

        return scores

    def compute_top_strengths(
        self,
        feature_set: CompanyFeatureSet,
        top_n: int = 5,
    ) -> list[Contribution]:
        """Return the top positive contributors (strengths)."""
        positives = self.get_positive_contributors(feature_set)
        return positives[:top_n]

    def compute_top_weaknesses(
        self,
        feature_set: CompanyFeatureSet,
        top_n: int = 5,
    ) -> list[Contribution]:
        """Return the top negative contributors (weaknesses)."""
        negatives = self.get_negative_contributors(feature_set)
        return negatives[:top_n]

    def _reclassify_confidence(self, contribution: Contribution) -> ContributionType:
        """Reclassify a low-magnitude data-quality feature as confidence.

        Deterministic rule: features whose category is signals/benchmark
        or whose id starts with confidence prefixes, with computed
        contribution magnitude below the resolution floor, are reclassified
        as confidence contributors.
        """
        is_confidence_like = (
            contribution.category in _CONFIDENCE_CATEGORIES
            or contribution.feature_id.startswith(_CONFIDENCE_FEATURE_PREFIXES)
        )
        if is_confidence_like and (
            abs(contribution.computed_contribution) < _LOW_EVIDENCE_RESOLUTION
        ):
            return ContributionType.CONFIDENCE
        return contribution.contribution_type

    def to_dict(
        self,
        contributions: list[Contribution],
    ) -> list[dict[str, Any]]:
        """Serialize contributions to deterministic dicts."""
        return [c.to_dict() for c in contributions]
