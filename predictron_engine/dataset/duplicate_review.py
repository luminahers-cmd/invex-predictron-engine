"""Duplicate review queue (Project E2).

Classifies resolution results into ``automatic_merge``, ``manual_review``,
and ``unrelated`` and produces a review report listing every candidate a
human should inspect before any merge is applied.

- ``automatic_merge`` — record groups whose best-edge confidence is at or
  above the identity threshold.  These have already been clustered into a
  single :class:`CompanyIdentity`.
- ``manual_review`` — record pairs whose confidence is between the review
  floor and the identity threshold.  They are surfaced here and are
  *never* merged automatically.
- ``unrelated`` — pairs a downstream caller explicitly splits.  By default
  no pair is considered unrelated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from predictron_engine.dataset.entity_resolution import (
    IDENTITY_BUCKET_THRESHOLD,
    REVIEW_BUCKET_THRESHOLD,
    MatchResult,
    ResolutionReport,
    ResolvedCluster,
)

# Confidence floor below which a pair is treated as unrelated (not listed
# for review).  Mirrors the resolver's review threshold.
REVIEW_FLOOR: float = REVIEW_BUCKET_THRESHOLD


@dataclass
class ReviewItem:
    """A single manual-review candidate pair."""

    match: MatchResult

    @property
    def record_ids(self) -> tuple[str, str]:
        return self.match.pair

    @property
    def confidence(self) -> float:
        return self.match.confidence

    @property
    def match_reason(self) -> str:
        return self.match.match_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_ids": list(self.match.pair),
            "confidence": round(self.match.confidence, 4),
            "match_reason": self.match.match_reason,
            "matched_fields": list(self.match.matched_fields),
            "evidence": self.match.evidence.to_dict(),
        }


@dataclass
class DuplicateReviewReport:
    """Classification summary + manual-review candidates."""

    automatic: list[ResolvedCluster] = field(default_factory=list)
    manual_review: list[ReviewItem] = field(default_factory=list)
    unrelated: list[ResolvedCluster] = field(default_factory=list)

    @property
    def automatic_count(self) -> int:
        return len(self.automatic)

    @property
    def manual_review_count(self) -> int:
        return len(self.manual_review)

    @property
    def unrelated_count(self) -> int:
        return len(self.unrelated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_type": "duplicate_review_report",
            "automatic": [cluster.to_dict() for cluster in self.automatic],
            "manual_review": [item.to_dict() for item in self.manual_review],
            "unrelated": [cluster.to_dict() for cluster in self.unrelated],
            "counts": {
                "automatic": self.automatic_count,
                "manual_review": self.manual_review_count,
                "unrelated": self.unrelated_count,
            },
        }


def classify_clusters(
    clusters: list[ResolvedCluster],
) -> DuplicateReviewReport:
    """Classify a list of resolved clusters into queue buckets.

    ``automatic_merge`` clusters carry a merge_decision of
    ``automatic_merge`` and at least two members.  Clusters explicitly
    marked ``unrelated`` land in the unrelated bucket.
    """
    automatic: list[ResolvedCluster] = []
    unrelated: list[ResolvedCluster] = []
    for cluster in clusters:
        if len(cluster.record_ids) < 2:
            continue
        if cluster.merge_decision == "unrelated":
            unrelated.append(cluster)
        elif cluster.merge_decision == "automatic_merge":
            automatic.append(cluster)

    automatic.sort(key=lambda cluster: sorted(cluster.record_ids))
    unrelated.sort(key=lambda cluster: sorted(cluster.record_ids))
    return DuplicateReviewReport(automatic=automatic, unrelated=unrelated)


def build_review_report(
    report: ResolutionReport,
    *,
    review_floor: float = REVIEW_FLOOR,
    identity_threshold: float = IDENTITY_BUCKET_THRESHOLD,
) -> DuplicateReviewReport:
    """Build a duplicate-review report from a resolution report.

    Automatic buckets come from the report's clusters; manual-review
    candidates come from the resolution report's review pairs.
    """
    automatic = [
        cluster
        for cluster in report.clusters
        if len(cluster.record_ids) >= 2
        and cluster.merge_decision == "automatic_merge"
        and cluster.confidence >= identity_threshold
    ]
    manual = [
        ReviewItem(match=match)
        for match in report.review_pairs
        if review_floor <= match.confidence < identity_threshold
    ]
    automatic.sort(key=lambda cluster: sorted(cluster.record_ids))
    manual.sort(key=lambda item: item.record_ids)
    return DuplicateReviewReport(automatic=automatic, manual_review=manual)
