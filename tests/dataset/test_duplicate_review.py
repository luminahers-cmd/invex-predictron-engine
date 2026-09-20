"""Tests for duplicate review queue."""

from __future__ import annotations

from predictron_engine.dataset.duplicate_review import (
    REVIEW_FLOOR,
    DuplicateReviewReport,
    ReviewItem,
    build_review_report,
    classify_clusters,
)
from predictron_engine.dataset.entity_resolution import (
    IDENTITY_BUCKET_THRESHOLD,
    REVIEW_BUCKET_THRESHOLD,
    MatchEvidence,
    MatchResult,
    ResolutionReport,
    ResolvedCluster,
)


def _cluster(
    record_ids: list[str],
    confidence: float = 0.95,
    merge_decision: str = "automatic_merge",
) -> ResolvedCluster:
    return ResolvedCluster(
        record_ids=record_ids,
        confidence=confidence,
        merge_decision=merge_decision,
    )


def _match(
    pair: tuple[str, str],
    confidence: float,
    reason: str = "fuzzy_review",
) -> MatchResult:
    return MatchResult(
        pair=pair,
        confidence=confidence,
        match_reason=reason,
        matched_fields=["name"],
        evidence=MatchEvidence(pair=pair),
    )


# ---------------------------------------------------------------------------
# classify_clusters
# ---------------------------------------------------------------------------

class TestClassifyClusters:
    def test_automatic_clusters(self) -> None:
        clusters = [
            _cluster(["a", "b"], confidence=0.95, merge_decision="automatic_merge"),
            _cluster(["c", "d"], confidence=0.92, merge_decision="automatic_merge"),
        ]
        report = classify_clusters(clusters)
        assert report.automatic_count == 2
        assert report.manual_review_count == 0
        assert report.unrelated_count == 0

    def test_singletons_excluded(self) -> None:
        clusters = [
            _cluster(["a"], confidence=1.0, merge_decision="singleton"),
        ]
        report = classify_clusters(clusters)
        assert report.automatic_count == 0

    def test_unrelated_clusters(self) -> None:
        clusters = [
            _cluster(["a", "b"], merge_decision="unrelated"),
        ]
        report = classify_clusters(clusters)
        assert report.unrelated_count == 1

    def test_sorted_output(self) -> None:
        clusters = [
            _cluster(["z", "a"], merge_decision="automatic_merge"),
            _cluster(["m", "b"], merge_decision="automatic_merge"),
        ]
        report = classify_clusters(clusters)
        # Clusters are ordered by their sorted record IDs: ("a","z") before
        # ("b","m"). Internal record_ids order is preserved as given.
        assert report.automatic[0].record_ids == ["z", "a"]
        assert report.automatic[1].record_ids == ["m", "b"]

    def test_empty_clusters(self) -> None:
        report = classify_clusters([])
        assert report.automatic_count == 0
        assert report.manual_review_count == 0
        assert report.unrelated_count == 0


# ---------------------------------------------------------------------------
# build_review_report
# ---------------------------------------------------------------------------

class TestBuildReviewReport:
    def test_automatic_from_clusters(self) -> None:
        resolution_report = ResolutionReport(
            record_ids=["a", "b", "c"],
            clusters=[
                _cluster(["a", "b"], confidence=0.95, merge_decision="automatic_merge"),
                _cluster(["c"], confidence=1.0, merge_decision="singleton"),
            ],
            review_pairs=[],
        )
        report = build_review_report(resolution_report)
        assert report.automatic_count == 1
        assert report.manual_review_count == 0

    def test_manual_review_from_pairs(self) -> None:
        resolution_report = ResolutionReport(
            record_ids=["a", "b"],
            clusters=[
                _cluster(["a"], confidence=1.0, merge_decision="singleton"),
                _cluster(["b"], confidence=1.0, merge_decision="singleton"),
            ],
            review_pairs=[
                _match(("a", "b"), confidence=0.78),
            ],
        )
        report = build_review_report(resolution_report)
        assert report.manual_review_count == 1
        assert report.manual_review[0].record_ids == ("a", "b")

    def test_pairs_below_floor_excluded(self) -> None:
        resolution_report = ResolutionReport(
            record_ids=["a", "b"],
            clusters=[
                _cluster(["a"], confidence=1.0, merge_decision="singleton"),
                _cluster(["b"], confidence=1.0, merge_decision="singleton"),
            ],
            review_pairs=[
                _match(("a", "b"), confidence=0.50),  # below REVIEW_FLOOR
            ],
        )
        report = build_review_report(resolution_report)
        assert report.manual_review_count == 0

    def test_pairs_at_identity_threshold_excluded_from_review(self) -> None:
        resolution_report = ResolutionReport(
            record_ids=["a", "b"],
            clusters=[],
            review_pairs=[
                _match(
                    ("a", "b"),
                    confidence=IDENTITY_BUCKET_THRESHOLD + 0.01,
                ),
            ],
        )
        report = build_review_report(resolution_report)
        # Above identity threshold should not appear in manual review.
        assert report.manual_review_count == 0

    def test_empty_report(self) -> None:
        resolution_report = ResolutionReport()
        report = build_review_report(resolution_report)
        assert report.automatic_count == 0
        assert report.manual_review_count == 0


# ---------------------------------------------------------------------------
# DuplicateReviewReport
# ---------------------------------------------------------------------------

class TestDuplicateReviewReport:
    def test_to_dict(self) -> None:
        report = DuplicateReviewReport(
            automatic=[_cluster(["a", "b"])],
            manual_review=[
                ReviewItem(match=_match(("c", "d"), confidence=0.75)),
            ],
        )
        d = report.to_dict()
        assert d["report_type"] == "duplicate_review_report"
        assert d["counts"]["automatic"] == 1
        assert d["counts"]["manual_review"] == 1
        assert d["counts"]["unrelated"] == 0

    def test_counts(self) -> None:
        report = DuplicateReviewReport(
            automatic=[_cluster(["a", "b"]), _cluster(["c", "d"])],
            manual_review=[
                ReviewItem(match=_match(("e", "f"), confidence=0.75)),
            ],
            unrelated=[_cluster(["g", "h"], merge_decision="unrelated")],
        )
        assert report.automatic_count == 2
        assert report.manual_review_count == 1
        assert report.unrelated_count == 1


# ---------------------------------------------------------------------------
# ReviewItem
# ---------------------------------------------------------------------------

class TestReviewItem:
    def test_properties(self) -> None:
        match = _match(("a", "b"), confidence=0.78, reason="fuzzy_review")
        item = ReviewItem(match=match)
        assert item.record_ids == ("a", "b")
        assert item.confidence == 0.78
        assert item.match_reason == "fuzzy_review"

    def test_to_dict(self) -> None:
        match = _match(("a", "b"), confidence=0.78)
        item = ReviewItem(match=match)
        d = item.to_dict()
        assert d["record_ids"] == ["a", "b"]
        assert d["confidence"] == 0.78


# ---------------------------------------------------------------------------
# REVIEW_FLOOR constant
# ---------------------------------------------------------------------------

class TestReviewFloor:
    def test_matches_bucket_threshold(self) -> None:
        assert REVIEW_FLOOR == REVIEW_BUCKET_THRESHOLD
