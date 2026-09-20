"""Deterministic entity resolution (Project E2).

Transforms a collection of :class:`DatasetRecord`\\ s into a set of
canonical :class:`CompanyIdentity`\\ s using a multi-stage, explainable,
machine-learning-free matching pipeline.

Pipeline
--------

1. **Blocking** (:meth:`ResolutionIndex.build`) — records are bucketed by
   strong identifiers, exact canonical domain, exact canonical name, and
   shared name tokens.  Only records sharing a bucket are compared, so
   worst-case work is bounded well below O(n^2).

2. **Multi-stage matching** (:meth:`ResolutionIndex.match_pair`) — records
   in the same bucket are scored with progressively stronger evidence:

   Stage 1 — exact keys: SEC CIK, Companies House / registration number,
   other registry identifiers, canonical domain, canonical name.
   Stage 2 — normalized core-name equality (after suffix stripping and
   Unicode normalization).
   Stage 3 — fuzzy composite similarity (Jaro-Winkler, token overlap,
   normalized Levenshtein) on canonical core names.
   Stage 4 — composite profile evidence (name + domain + country +
   industry + city/state).

3. **Confidence** — every candidate pair yields a 0..1 confidence, a
   human-readable ``match_reason``, the matched fields, and supporting
   evidence.  Pairs below ``min_confidence`` are never merged.

4. **Clustering** — accepted pair edges are unioned into clusters; each
   cluster becomes one :class:`CompanyIdentity` built with the lossless
   merge rules in :mod:`identity`.

5. **Classification** — each cluster is labeled ``automatic_merge``,
   ``manual_review``, or ``unrelated``.

No ML, no heuristics that can flip with numeric precision: the whole
pipeline is deterministic given the same inputs.

Design note — scalability
-------------------------
Blocking is the mechanism that keeps resolution tractable at 100k–1M
records.  Records that share no identifier/domain/exact-name/token bucket
are never compared, so pathological O(n^2) behaviour is avoided.  The
fuzzy stage runs only *within* buckets, not across the whole dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from predictron_engine.dataset.company_name import (
    canonical_name_key,
    core_name,
    name_tokens,
)
from predictron_engine.dataset.enrichment import extract_domain
from predictron_engine.dataset.fuzzy import name_similarity
from predictron_engine.dataset.identity import (
    CompanyIdentity,
    CompanyIdentityBuilder,
    extract_identifiers,
)
from predictron_engine.dataset.models import DatasetRecord

# Confidence thresholds.  These are public module constants so callers can
# tune classification and auto-merge behaviour without editing the logic.
IDENTITY_BUCKET_THRESHOLD: float = 0.86
REVIEW_BUCKET_THRESHOLD: float = 0.72

# Minimum confidence for a pair to be accepted as a merge edge at all.
MIN_MATCH_CONFIDENCE: float = 0.70


@dataclass
class MatchEvidence:
    """Supporting evidence for a candidate match between two records."""

    pair: tuple[str, str]
    identifier_match: bool = False
    identifier_kind: str | None = None
    domain_match: bool = False
    domain: str | None = None
    exact_name: bool = False
    canonical_name: str | None = None
    fuzzy_name_score: float = 0.0
    country_match: bool = False
    industry_match: bool = False
    city_match: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchResult:
    """Result of matching one candidate pair."""

    pair: tuple[str, str]
    confidence: float
    match_reason: str
    matched_fields: list[str] = field(default_factory=list)
    evidence: MatchEvidence = field(default_factory=lambda: MatchEvidence(("", "")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": list(self.pair),
            "confidence": self.confidence,
            "match_reason": self.match_reason,
            "matched_fields": list(self.matched_fields),
            "evidence": self.evidence.to_dict(),
        }


@dataclass
class BlockingIndex:
    """Pre-computed buckets that reduce the candidate comparison space."""

    identifier_to_ids: dict[str, list[str]] = field(default_factory=dict)
    domain_to_ids: dict[str, list[str]] = field(default_factory=dict)
    name_to_ids: dict[str, list[str]] = field(default_factory=dict)
    token_to_ids: dict[str, list[str]] = field(default_factory=dict)


class ResolutionIndex:
    """Builds blocking buckets and runs pair matching for a record set."""

    def __init__(self) -> None:
        self._records: dict[str, DatasetRecord] = {}
        self._index: BlockingIndex = BlockingIndex()

    def build(self, records: list[DatasetRecord]) -> BlockingIndex:
        self._records = {record.record_id: record for record in records}
        index = BlockingIndex()
        for record in records:
            self._add_record_to_index(record, index)
        self._index = index
        return index

    def _add_record_to_index(
        self, record: DatasetRecord, index: BlockingIndex
    ) -> None:
        for kind, value in extract_identifiers(record).items():
            key = f"{kind}:{value}"
            index.identifier_to_ids.setdefault(key, []).append(record.record_id)
        domain = _profile_domain(record)
        if domain:
            index.domain_to_ids.setdefault(domain, []).append(record.record_id)
        name_key = canonical_name_key(record.startup_name or _profile_legal_name(record))
        if name_key:
            index.name_to_ids.setdefault(name_key, []).append(record.record_id)
        token_yielded: set[str] = set()
        for token in name_tokens(record.startup_name or _profile_legal_name(record)):
            if token in token_yielded:
                continue
            token_yielded.add(token)
            index.token_to_ids.setdefault(token, []).append(record.record_id)

    def candidate_pairs(self) -> set[tuple[str, str]]:
        """Return the set of record pairs sharing at least one bucket.

        Pairs are canonicalized so each unordered pair appears once.
        This is the candidate edge set that gets scored.
        """
        pairs: set[tuple[str, str]] = set()
        buckets = list(self._index.identifier_to_ids.values())
        buckets.extend(self._index.domain_to_ids.values())
        buckets.extend(self._index.name_to_ids.values())
        buckets.extend(self._index.token_to_ids.values())
        for ids in buckets:
            uniq = sorted(set(ids))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    pair = (uniq[i], uniq[j])
                    if pair[0] <= pair[1]:
                        pairs.add(pair)
                    else:
                        pairs.add((pair[1], pair[0]))
        return pairs

    def match_pair(self, id_a: str, id_b: str) -> MatchResult:
        """Score the match between two record IDs deterministically."""
        record_a = self._records[id_a]
        record_b = self._records[id_b]
        return score_pair(record_a, record_b)


# ---- Pair scoring (the heart of multi-stage matching) ----

def score_pair(record_a: DatasetRecord, record_b: DatasetRecord) -> MatchResult:
    """Score a single record pair through all four matching stages.

    Returns a :class:`MatchResult` with confidence, match reason, matched
    fields, and supporting evidence.  Never raises on malformed input —
    missing data simply contributes nothing.
    """
    sorted_ids = sorted([record_a.record_id, record_b.record_id])
    pair: tuple[str, str] = (sorted_ids[0], sorted_ids[1])
    if pair[0] == pair[1]:
        return MatchResult(
            pair=pair,
            confidence=1.0,
            match_reason="same_record",
            matched_fields=["record_id"],
            evidence=MatchEvidence(pair=pair),
        )

    evidence = MatchEvidence(pair=pair)
    matched_fields: list[str] = []

    # Stage 1: exact strong identifiers.
    identifiers_a = extract_identifiers(record_a)
    identifiers_b = extract_identifiers(record_b)
    shared_kinds = sorted(set(identifiers_a) & set(identifiers_b))
    for kind in shared_kinds:
        if identifiers_a[kind] == identifiers_b[kind] and identifiers_a[kind]:
            evidence.identifier_match = True
            evidence.identifier_kind = kind
            matched_fields.append(f"identifier:{kind}")
            break

    if evidence.identifier_match:
        return MatchResult(
            pair=pair,
            confidence=_exact_identifier_confidence(evidence.identifier_kind),
            match_reason=f"exact_{evidence.identifier_kind}",
            matched_fields=matched_fields,
            evidence=evidence,
        )

    # Stage 1b: exact canonical domain.
    domain_a = _record_domain(record_a)
    domain_b = _record_domain(record_b)
    if domain_a and domain_b and domain_a == domain_b:
        evidence.domain_match = True
        evidence.domain = domain_a
        matched_fields.append("domain")
        return MatchResult(
            pair=pair,
            confidence=0.98,
            match_reason="exact_domain",
            matched_fields=matched_fields,
            evidence=evidence,
        )

    # Stage 2: exact normalized core-name equality (suffix-stripped,
    # Unicode-normalized, punctuation-free).
    core_a = core_name(record_a.startup_name or _profile_legal_name(record_a))
    core_b = core_name(record_b.startup_name or _profile_legal_name(record_b))
    country_a = _country(record_a)
    country_b = _country(record_b)
    country_conflict = bool(country_a and country_b and country_a != country_b)
    if core_a and core_b and core_a == core_b:
        evidence.exact_name = True
        evidence.canonical_name = core_a
        matched_fields = ["name"]
        if country_conflict:
            # Identical names registered in different countries are a
            # classic false positive; never auto-merge conflicting
            # jurisdiction data.
            return MatchResult(
                pair=pair,
                confidence=0.68,
                match_reason="exact_name_country_conflict",
                matched_fields=matched_fields,
                evidence=evidence,
            )
        confidence = 0.92
        if country_a and country_b and country_a == country_b:
            confidence += 0.03
            matched_fields.append("country")
            evidence.country_match = True
        if _shares_industry(record_a, record_b):
            confidence += 0.02
            matched_fields.append("industry")
            evidence.industry_match = True
        return MatchResult(
            pair=pair,
            confidence=round(min(confidence, 0.97), 4),
            match_reason="exact_name",
            matched_fields=matched_fields,
            evidence=evidence,
        )

    # Stage 3: fuzzy composite name similarity.
    fuzzy = name_similarity(
        core_a or record_a.startup_name, core_b or record_b.startup_name
    )
    evidence.fuzzy_name_score = round(fuzzy, 6)
    fuzzy_fields = []

    # Stage 4: composite profile evidence (name + domain + country +
    # industry + city/state).  Each contributes to the final confidence.
    if country_a and country_b and country_a == country_b:
        evidence.country_match = True
        fuzzy_fields.append("country")
    if _shares_industry(record_a, record_b):
        evidence.industry_match = True
        fuzzy_fields.append("industry")
    if _shares_city(record_a, record_b):
        evidence.city_match = True
        fuzzy_fields.append("city")
    if domain_a and domain_b and _domains_overlap(domain_a, domain_b):
        fuzzy_fields.append("domain")

    if fuzzy >= IDENTITY_BUCKET_THRESHOLD:
        matched_fields = ["name"] + fuzzy_fields
        reason = "fuzzy_name"
        if evidence.country_match:
            reason = "fuzzy_name_country"
        if evidence.country_match and evidence.industry_match:
            reason = "fuzzy_name_country_industry"
        confidence = _composite_fuzzy_confidence(fuzzy, fuzzy_fields)
        if country_conflict:
            confidence = min(confidence, REVIEW_BUCKET_THRESHOLD - 0.01)
        return MatchResult(
            pair=pair,
            confidence=confidence,
            match_reason=reason,
            matched_fields=matched_fields,
            evidence=evidence,
        )

    if fuzzy >= REVIEW_BUCKET_THRESHOLD:
        matched_fields = ["name"] + fuzzy_fields
        confidence = _composite_fuzzy_confidence(fuzzy, fuzzy_fields)
        return MatchResult(
            pair=pair,
            confidence=confidence,
            match_reason="fuzzy_review",
            matched_fields=matched_fields,
            evidence=evidence,
        )

    return MatchResult(
        pair=pair,
        confidence=0.0,
        match_reason="below_threshold",
        matched_fields=[],
        evidence=evidence,
    )


# ---- Consensus / clustering / reporting ----

@dataclass
class ResolvedCluster:
    """A cluster of records resolved to one identity."""

    record_ids: list[str]
    confidence: float
    merge_decision: str
    matched_fields: list[str] = field(default_factory=list)
    match_reason: str = "seed"
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_ids": list(sorted(self.record_ids)),
            "confidence": round(self.confidence, 4),
            "merge_decision": self.merge_decision,
            "matched_fields": list(self.matched_fields),
            "match_reason": self.match_reason,
            "evidence": self.evidence,
        }


@dataclass
class ResolutionReport:
    """Aggregate report for a resolution pass."""

    record_ids: list[str] = field(default_factory=list)
    clusters: list[ResolvedCluster] = field(default_factory=list)
    review_pairs: list[MatchResult] = field(default_factory=list)
    pair_candidates: int = 0
    pair_confidences: list[float] = field(default_factory=list)
    identity_count: int = 0
    merged_count: int = 0
    automatic_count: int = 0
    manual_review_count: int = 0
    unrelated_count: int = 0

    def confidence_distribution(self) -> dict[str, int]:
        """Summarize pair-confidences into fractional buckets."""
        distribution: dict[str, int] = {}
        for confidence in self.pair_confidences:
            bucket = (
                f"{round(int(confidence * 10) / 10, 1)}-"
                f"{round((int(confidence * 10) + 1) / 10, 1)}"
            )
            distribution[bucket] = distribution.get(bucket, 0) + 1
        return dict(sorted(distribution.items()))

    def merge_statistics(self) -> dict[str, Any]:
        """Return summarized merge statistics."""
        sizes = sorted(len(cluster.record_ids) for cluster in self.clusters)
        return {
            "clusters": len(self.clusters),
            "identity_count": self.identity_count,
            "merged_record_count": self.merged_count,
            "automatic_merges": self.automatic_count,
            "manual_review_pairs": self.manual_review_count,
            "unrelated_pairs": self.unrelated_count,
            "largest_cluster_size": sizes[-1] if sizes else 0,
            "average_cluster_size": (
                round(len(self.record_ids) / self.identity_count, 4)
                if self.identity_count
                else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_type": "entity_resolution_report",
            "record_count": len(self.record_ids),
            "identity_count": self.identity_count,
            "merged_record_count": self.merged_count,
            "duplicate_clusters": [c.to_dict() for c in self.clusters],
            "manual_review_pairs": [m.to_dict() for m in self.review_pairs],
            "confidence_distribution": self.confidence_distribution(),
            "merge_statistics": self.merge_statistics(),
        }


@dataclass
class ResolutionResult:
    """Outcome of resolving a record collection."""

    identities: list[CompanyIdentity]
    report: ResolutionReport
    matches: list[MatchResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "identities": [identity.to_dict() for identity in self.identities],
            "report": self.report.to_dict(),
        }


class EntityResolver:
    """Resolves a collection of records into canonical identities.

    The resolver is deterministic: given the same records in any order,
    it produces the same identities and the same report.
    """

    def __init__(
        self,
        *,
        min_confidence: float = REVIEW_BUCKET_THRESHOLD,
        identity_threshold: float = IDENTITY_BUCKET_THRESHOLD,
    ) -> None:
        if not (0.0 <= min_confidence <= 1.0):
            msg = "min_confidence must be in [0, 1]"
            raise ValueError(msg)
        if not (0.0 <= identity_threshold <= 1.0):
            msg = "identity_threshold must be in [0, 1]"
            raise ValueError(msg)
        if min_confidence > identity_threshold:
            msg = "min_confidence must be <= identity_threshold"
            raise ValueError(msg)
        self._min_confidence = min_confidence
        self._identity_threshold = identity_threshold

    def resolve(self, records: list[DatasetRecord]) -> ResolutionResult:
        """Resolve a collection of records into identities + report.

        Pairs at or above ``identity_threshold`` become automatic-merge
        edges and are clustered into a single identity.  Pairs between
        ``min_confidence`` and the identity threshold are surfaced as
        manual-review candidates and are *never* merged automatically.
        Pairs below ``min_confidence`` are unrelated.
        """
        index = ResolutionIndex()
        index.build(records)

        pairs = index.candidate_pairs()
        matches: list[MatchResult] = []
        auto_edges: list[MatchResult] = []
        review_edges: list[MatchResult] = []
        for pair in sorted(pairs):
            result = index.match_pair(*pair)
            matches.append(result)
            if result.confidence >= self._identity_threshold:
                auto_edges.append(result)
            elif result.confidence >= self._min_confidence:
                review_edges.append(result)

        record_id_set = {record.record_id for record in records}
        cluster_ids = _union_clusters([result.pair for result in auto_edges])

        identities: list[CompanyIdentity] = []
        clusters: list[ResolvedCluster] = []
        assigned: set[str] = set()

        # Groups of >= 2 are automatic merges; single-member groups are
        # singleton identities.
        cluster_groups = sorted(cluster_ids, key=lambda members: sorted(members))
        if not cluster_groups and not records:
            cluster_groups = []
        for members_raw in cluster_groups:
            members = sorted(members_raw)
            if len(members) < 2:
                continue

            edge_confidences = [
                result.confidence
                for result in auto_edges
                if result.pair[0] in members and result.pair[1] in members
            ]
            confidence = max(edge_confidences) if edge_confidences else 0.0
            match_reason = ""
            matched_fields: list[str] = []
            evidence: list[dict[str, Any]] = []
            for result in auto_edges:
                if result.pair[0] in members and result.pair[1] in members:
                    if result.confidence:
                        match_reason = result.match_reason
                        matched_fields = list(result.matched_fields)
                        evidence.append(result.evidence.to_dict())
            evidence = evidence[:3]

            decision = "automatic_merge"
            identity = CompanyIdentityBuilder().build(
                [index._records[rid] for rid in members],
                merge_decision=decision,
            )
            identities.append(identity)
            assigned.update(members)
            clusters.append(
                ResolvedCluster(
                    record_ids=members,
                    confidence=confidence,
                    merge_decision=decision,
                    matched_fields=matched_fields,
                    match_reason=match_reason,
                    evidence=evidence,
                )
            )

        # Every remaining record is its own (unmerged) identity.
        for record in sorted(records, key=lambda record: record.record_id):
            if record.record_id in assigned:
                continue
            assigned.add(record.record_id)
            identity = CompanyIdentityBuilder().build(
                [record], merge_decision="singleton"
            )
            identities.append(identity)
            clusters.append(
                ResolvedCluster(
                    record_ids=[record.record_id],
                    confidence=1.0,
                    merge_decision="singleton",
                )
            )

        identities.sort(key=lambda identity: identity.canonical_name)
        clusters.sort(key=lambda cluster: sorted(cluster.record_ids))

        review_edges.sort(key=lambda result: result.pair)
        report = ResolutionReport(
            record_ids=sorted(record_id_set),
            clusters=clusters,
            review_pairs=review_edges,
            pair_candidates=len(pairs),
            pair_confidences=[result.confidence for result in matches],
            identity_count=len(identities),
            merged_count=sum(
                1 for cluster in clusters if len(cluster.record_ids) > 1
            ),
            automatic_count=sum(
                1
                for cluster in clusters
                if cluster.merge_decision == "automatic_merge"
            ),
            manual_review_count=len(review_edges),
            unrelated_count=len(matches)
            - len(auto_edges)
            - len(review_edges),
        )
        return ResolutionResult(identities=identities, report=report, matches=matches)


# ---- Internal helpers ----

def _profile_domain(record: DatasetRecord) -> str:
    if record.profile.domain:
        return record.profile.domain
    return extract_domain(record.website) or ""


def _record_domain(record: DatasetRecord) -> str:
    return _profile_domain(record)


def _profile_legal_name(record: DatasetRecord) -> str:
    return record.profile.legal_name or ""


def _country(record: DatasetRecord) -> str:
    return record.profile.country_code or ""


def _shares_industry(record_a: DatasetRecord, record_b: DatasetRecord) -> bool:
    industries_a = set(record_a.profile.industries)
    industries_b = set(record_b.profile.industries)
    return bool(industries_a & industries_b)


def _shares_city(record_a: DatasetRecord, record_b: DatasetRecord) -> bool:
    if (
        record_a.profile.city
        and record_b.profile.city
        and record_a.profile.city.lower() == record_b.profile.city.lower()
    ):
        return True
    if (
        record_a.profile.region
        and record_b.profile.region
        and record_a.profile.region.lower() == record_b.profile.region.lower()
    ):
        return True
    return False


def _domains_overlap(domain_a: str, domain_b: str) -> bool:
    """True when one domain's tokens overlap the other's."""
    tokens_a = set(domain_a.replace(".", " ").split())
    tokens_b = set(domain_b.replace(".", " ").split())
    return bool(tokens_a & tokens_b) and domain_a != domain_b


def _exact_identifier_confidence(kind: str | None) -> float:
    if kind in ("sec_cik", "company_number", "registration_number"):
        return 0.99
    return 0.98


def _composite_fuzzy_confidence(
    fuzzy: float, supporting_fields: list[str]
) -> float:
    """Blend fuzzy name similarity with supporting profile evidence."""
    base = fuzzy * 0.95
    boost = min(len(supporting_fields) * 0.02, 0.10)
    return round(min(base + boost, 0.9999), 4)


def _union_clusters(edges: list[tuple[str, str]]) -> list[set[str]]:
    """Union pairs of record IDs into connected clusters."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(node_a: str, node_b: str) -> None:
        root_a = find(node_a)
        root_b = find(node_b)
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    for node_a, node_b in edges:
        parent.setdefault(node_a, node_a)
        parent.setdefault(node_b, node_b)
    for node_a, node_b in edges:
        union(node_a, node_b)

    groups: dict[str, set[str]] = {}
    for node in parent:
        root = find(node)
        groups.setdefault(root, set()).add(node)
    return list(groups.values())
