"""Decision Trace — deterministic reasoning graph for investment decisions.

Generates a directed acyclic graph that traces the complete reasoning
path from raw feature to final decision:

    Raw Feature
        ↓
    Normalized Feature
        ↓
    Rule
        ↓
    Intermediate Score
        ↓
    Dimension Score
        ↓
    Overall Score
        ↓
    Decision

Every node is fully traceable with evidence references and
deterministic computation metadata.

Key principles:
  - Fully deterministic: same feature set always produces the same trace
  - No ML, no LLMs, no randomness
  - Every node carries evidence provenance
  - Graph is a DAG with explicit edges
"""

from __future__ import annotations

from typing import Any

from predictron_engine.decision.feature_engine import DecisionFeatureEngine
from predictron_engine.decision.intelligence_models import (
    Contribution,
    DecisionTrace,
    DecisionVerdict,
    TraceNode,
    TraceNodeType,
)
from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    FeatureStatus,
)

# ---------------------------------------------------------------------------
# Trace constants — named, deterministic.
# ---------------------------------------------------------------------------

# Verdict thresholds derived from the overall score (0-100).
_VERDICT_THRESHOLDS: list[tuple[float, DecisionVerdict]] = [
    (75.0, DecisionVerdict.STRONG_INVEST),
    (60.0, DecisionVerdict.INVEST),
    (45.0, DecisionVerdict.WATCH),
    (30.0, DecisionVerdict.INVESTIGATE_FURTHER),
    (0.0, DecisionVerdict.PASS),
]

# Rule templates by node type.
_RULE_NORMALIZE = "normalize_feature({feature_id})"
_RULE_WEIGHT = "weight({category}) * (1 + evidence_boost)"
_RULE_INTERMEDIATE = "sum of contribution nodes for {feature_id}"
_RULE_DIMENSION = "50 + (positive_net - negative_net) * 50"
_RULE_OVERALL = "weighted aggregate of dimension scores"
_RULE_DECISION = "map overall_score to verdict via deterministic thresholds"


class ContributionArgumentError(ValueError):
    """Raised when the contribution engine cannot produce a trace."""


class DecisionTraceEngine:
    """Builds deterministic reasoning graphs for investment decisions.

    Responsibilities:
      - Construct trace nodes for every stage of reasoning
      - Wire edges between nodes (DAG)
      - Compute intermediate, dimension, and overall scores
      - Classify final decision verdict
      - Preserve evidence on every node
    """

    def __init__(
        self,
        feature_engine: DecisionFeatureEngine | None = None,
        verdict_thresholds: list[tuple[float, DecisionVerdict]] | None = None,
    ) -> None:
        self._feature_engine = feature_engine or DecisionFeatureEngine()
        self._verdict_thresholds = verdict_thresholds or _VERDICT_THRESHOLDS

    def build_trace(
        self,
        feature_set: CompanyFeatureSet,
        contributions: list[Contribution] | None = None,
        *,
        confidence: float | None = None,
    ) -> DecisionTrace:
        """Build the complete reasoning trace for a company.

        Args:
            feature_set: Feature Store snapshot to trace.
            contributions: Optional pre-computed contributions (if omitted,
                they are computed deterministically from the feature set).
            confidence: Optional overall confidence (0-1).

        Returns:
            A fully wired DecisionTrace graph.
        """
        company_id = feature_set.company_id

        if contributions is None:
            contributions = self._feature_engine.compute_all_contributions(feature_set)

        # Build node sets by stage.
        raw_nodes = self._build_raw_nodes(feature_set)
        normalized_nodes = self._build_normalized_nodes(feature_set, contributions)
        rule_nodes = self._build_rule_nodes(contributions)
        intermediate_nodes = self._build_intermediate_nodes(contributions)
        dimension_nodes = self._build_dimension_nodes(contributions, intermediate_nodes)
        overall_node = self._build_overall_node(dimension_nodes)
        decision_node = self._build_decision_node(overall_node)

        nodes = (
            raw_nodes
            + normalized_nodes
            + rule_nodes
            + intermediate_nodes
            + dimension_nodes
            + [overall_node, decision_node]
        )

        # Wire edges.
        edges = self._wire_edges(
            raw_nodes,
            normalized_nodes,
            rule_nodes,
            intermediate_nodes,
            dimension_nodes,
            overall_node,
            decision_node,
        )

        overall_score = float(overall_node.value)
        verdict = self.classify_verdict(overall_score)
        resolved_confidence = confidence if confidence is not None else 0.0

        return DecisionTrace(
            company_id=company_id,
            nodes=nodes,
            edges=edges,
            overall_score=round(overall_score, 4),
            verdict=verdict,
            confidence=round(max(0.0, min(1.0, resolved_confidence)), 4),
        )

    def classify_verdict(self, overall_score: float) -> DecisionVerdict:
        """Map an overall score (0-100) to a deterministic verdict."""
        for threshold, verdict in self._verdict_thresholds:
            if overall_score >= threshold:
                return verdict
        return DecisionVerdict.PASS

    # ------------------------------------------------------------------
    # Node builders
    # ------------------------------------------------------------------

    def _build_raw_nodes(self, feature_set: CompanyFeatureSet) -> list[TraceNode]:
        """Raw feature nodes from the Feature Store snapshot."""
        nodes: list[TraceNode] = []
        for feature_id in sorted(feature_set.features.keys()):
            snapshot = feature_set.features[feature_id]
            evidence_refs = self._evidence_dicts(snapshot.evidence_references)
            nodes.append(TraceNode(
                node_type=TraceNodeType.RAW_FEATURE,
                node_id=f"raw:{feature_id}",
                feature_id=feature_id,
                label=f"Raw feature: {snapshot.feature_name}",
                value=snapshot.value,
                value_type="raw",
                computation_metadata={
                    "status": snapshot.status.value,
                    "computation_version": snapshot.computation_version,
                    "snapshot_id": snapshot.snapshot_id,
                },
                evidence_references=evidence_refs,
            ))
        return nodes

    def _build_normalized_nodes(
        self,
        feature_set: CompanyFeatureSet,
        contributions: list[Contribution],
    ) -> list[TraceNode]:
        """Normalized feature nodes from contributions."""
        nodes: list[TraceNode] = []
        contrib_by_id = {c.feature_id: c for c in contributions}
        for feature_id in sorted(feature_set.features.keys()):
            snapshot = feature_set.features[feature_id]
            contrib = contrib_by_id.get(feature_id)
            if contrib is not None:
                value = contrib.normalized_value
            elif snapshot.status == FeatureStatus.COMPUTED:
                value = self._feature_engine.normalize_feature(snapshot)
            else:
                value = 0.0
            evidence_refs = self._evidence_dicts(snapshot.evidence_references)
            nodes.append(TraceNode(
                node_type=TraceNodeType.NORMALIZED_FEATURE,
                node_id=f"normalized:{feature_id}",
                feature_id=feature_id,
                label=f"Normalized: {snapshot.feature_name}",
                value=round(float(value), 6),
                value_type="float",
                rule=_RULE_NORMALIZE.format(feature_id=feature_id),
                evidence_references=evidence_refs,
                computation_metadata={
                    "normalization": "feature-specific range",
                },
            ))
        return nodes

    def _build_rule_nodes(self, contributions: list[Contribution]) -> list[TraceNode]:
        """Rule nodes describing how contributions are computed."""
        nodes: list[TraceNode] = []
        for c in sorted(contributions, key=lambda x: x.feature_id):
            nodes.append(TraceNode(
                node_type=TraceNodeType.RULE,
                node_id=f"rule:{c.feature_id}",
                feature_id=c.feature_id,
                label=f"Rule: contribution of {c.feature_name}",
                value=c.computed_contribution,
                value_type="float",
                rule=f"{_RULE_NORMALIZE.format(feature_id=c.feature_id)}; "
                     f"{_RULE_WEIGHT.format(category=c.category)}",
                computation_metadata={
                    "weight": c.weight,
                    "contribution_type": c.contribution_type.value,
                },
                evidence_references=c.evidence_references,
            ))
        return nodes

    def _build_intermediate_nodes(
        self,
        contributions: list[Contribution],
    ) -> list[TraceNode]:
        """Intermediate score nodes per feature."""
        nodes: list[TraceNode] = []
        for c in sorted(contributions, key=lambda x: x.feature_id):
            nodes.append(TraceNode(
                node_type=TraceNodeType.INTERMEDIATE_SCORE,
                node_id=f"intermediate:{c.feature_id}",
                feature_id=c.feature_id,
                label=f"Intermediate: {c.feature_name}",
                value=c.computed_contribution,
                value_type="float",
                rule=_RULE_INTERMEDIATE.format(feature_id=c.feature_id),
                computation_metadata={
                    "normalized_value": c.normalized_value,
                    "weight": c.weight,
                },
                evidence_references=c.evidence_references,
            ))
        return nodes

    def _build_dimension_nodes(
        self,
        contributions: list[Contribution],
        intermediate_nodes: list[TraceNode],
    ) -> list[TraceNode]:
        """Dimension score nodes per category (0-100)."""
        by_category: dict[str, list[Contribution]] = {}
        for c in contributions:
            by_category.setdefault(c.category, []).append(c)

        nodes: list[TraceNode] = []
        for category in sorted(by_category.keys()):
            items = by_category[category]
            positive_net = sum(
                c.computed_contribution for c in items
                if c.computed_contribution > 0
            )
            negative_net = sum(
                abs(c.computed_contribution) for c in items
                if c.computed_contribution < 0
            )
            net = positive_net - negative_net
            dimension_score = round(max(0.0, min(100.0, 50.0 + net * 50.0)), 4)

            nodes.append(TraceNode(
                node_type=TraceNodeType.DIMENSION_SCORE,
                node_id=f"dimension:{category}",
                label=f"Dimension score: {category}",
                value=dimension_score,
                value_type="float",
                rule=_RULE_DIMENSION,
                computation_metadata={
                    "category": category,
                    "positive_net": round(positive_net, 6),
                    "negative_net": round(negative_net, 6),
                    "feature_count": len(items),
                },
            ))
        return nodes

    def _build_overall_node(
        self,
        dimension_nodes: list[TraceNode],
    ) -> TraceNode:
        """Overall score node — weighted aggregate of dimension scores."""
        if not dimension_nodes:
            return TraceNode(
                node_type=TraceNodeType.OVERALL_SCORE,
                node_id="overall",
                label="Overall score",
                value=0.0,
                value_type="float",
                rule=_RULE_OVERALL,
                computation_metadata={"dimension_count": 0},
            )

        weights = self._overall_dimension_weights(dimension_nodes)
        total = sum(
            float(dn.value) * weights[i]
            for i, dn in enumerate(dimension_nodes)
        )
        # Normalize weights in case they don't sum exactly to 1.
        weight_sum = sum(weights)
        if weight_sum > 0:
            total = total / weight_sum

        return TraceNode(
            node_type=TraceNodeType.OVERALL_SCORE,
            node_id="overall",
            label="Overall score",
            value=round(max(0.0, min(100.0, total)), 4),
            value_type="float",
            rule=_RULE_OVERALL,
            computation_metadata={
                "dimension_count": len(dimension_nodes),
                "dimension_scores": {
                    dn.label.split(": ")[1]: round(float(dn.value), 4)
                    for dn in dimension_nodes
                },
            },
        )

    def _build_decision_node(self, overall_node: TraceNode) -> TraceNode:
        """Final decision node."""
        score = float(overall_node.value)
        verdict = self.classify_verdict(score)
        return TraceNode(
            node_type=TraceNodeType.DECISION,
            node_id="decision",
            label=f"Decision: {verdict.value}",
            value=verdict.value,
            value_type="str",
            rule=_RULE_DECISION,
            computation_metadata={
                "overall_score": score,
                "verdict": verdict.value,
            },
        )

    # ------------------------------------------------------------------
    # Edge wiring
    # ------------------------------------------------------------------

    def _wire_edges(
        self,
        raw_nodes: list[TraceNode],
        normalized_nodes: list[TraceNode],
        rule_nodes: list[TraceNode],
        intermediate_nodes: list[TraceNode],
        dimension_nodes: list[TraceNode],
        overall_node: TraceNode,
        decision_node: TraceNode,
    ) -> list[tuple[str, str]]:
        """Wire edges in topological order: raw -> normalized -> rule
        -> intermediate -> dimension -> overall -> decision."""
        edges: list[tuple[str, str]] = []

        raw_by_id = {n.feature_id: n.node_id for n in raw_nodes if n.feature_id}
        norm_by_id = {n.feature_id: n.node_id for n in normalized_nodes if n.feature_id}
        rule_by_id = {n.feature_id: n.node_id for n in rule_nodes if n.feature_id}
        inter_by_id = {n.feature_id: n.node_id for n in intermediate_nodes if n.feature_id}

        # raw -> normalized
        for fid in sorted(raw_by_id):
            if fid in norm_by_id:
                edges.append((raw_by_id[fid], norm_by_id[fid]))

        # normalized -> rule
        for fid in sorted(norm_by_id):
            if fid in rule_by_id:
                edges.append((norm_by_id[fid], rule_by_id[fid]))

        # rule -> intermediate
        for fid in sorted(rule_by_id):
            if fid in inter_by_id:
                edges.append((rule_by_id[fid], inter_by_id[fid]))

        # intermediate -> dimension (grouped by category)
        inter_by_category: dict[str, list[str]] = {}
        for n in intermediate_nodes:
            if n.feature_id:
                inter_by_category.setdefault("_", []).append(n.node_id)

        dim_node_ids = [n.node_id for n in dimension_nodes]
        for inter_id in inter_by_category.get("_", []):
            if dim_node_ids:
                edges.append((inter_id, dim_node_ids[0]))

        # dimension -> overall
        for dim_id in dim_node_ids:
            edges.append((dim_id, overall_node.node_id))

        # overall -> decision
        edges.append((overall_node.node_id, decision_node.node_id))

        return edges

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _overall_dimension_weights(
        dimension_nodes: list[TraceNode],
    ) -> list[float]:
        """Equal weights per dimension (deterministic)."""
        if not dimension_nodes:
            return []
        return [1.0 / len(dimension_nodes)] * len(dimension_nodes)

    @staticmethod
    def _evidence_dicts(evidence: Any) -> list[dict[str, Any]]:
        """Convert evidence references to primitives."""
        refs: list[dict[str, Any]] = []
        for e in evidence:
            refs.append({
                "source_type": getattr(e, "source_type", ""),
                "source_id": getattr(e, "source_id", ""),
                "source_field": getattr(e, "source_field", ""),
                "confidence": getattr(e, "confidence", 1.0),
            })
        return refs

    def trace_to_dict(self, trace: DecisionTrace) -> dict[str, Any]:
        """Deterministic serialization of a trace."""
        return trace.to_dict()
