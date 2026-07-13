"""Traceability models and TraceGraph for the validation framework.

Provides structured trace objects that link every pipeline output back
to its source inputs, enabling full auditability from recommendation
to original startup input.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    """Pipeline stages for traceability."""

    NORMALIZE = "normalize"
    COLLECT = "collect"
    EXTRACT = "extract"
    EVIDENCE = "evidence"
    REASON = "reason"
    EVALUATE = "evaluate"
    SCORE = "score"
    RECOMMEND = "recommend"
    CONFIDENCE = "confidence"
    BUILD_REPORT = "build_report"


class TraceNode(BaseModel):
    """A single node in the trace graph representing one pipeline artifact."""

    artifact_id: str = Field(
        ..., description="Unique identifier for this artifact"
    )
    stage: PipelineStage = Field(
        ..., description="Pipeline stage that produced this artifact"
    )
    artifact_type: str = Field(
        ..., description="Type of artifact (e.g. 'Recommendation', 'Observation')"
    )
    label: str = Field(
        default="", description="Human-readable label for this artifact"
    )
    input_ids: list[str] = Field(
        default_factory=list,
        description="IDs of artifacts that were inputs to producing this artifact",
    )
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Additional metadata about this artifact",
    )


class TracePath(BaseModel):
    """A single trace path from a downstream artifact back to its source."""

    source_id: str = Field(..., description="Starting artifact ID")
    source_type: str = Field(..., description="Type of source artifact")
    target_id: str = Field(..., description="Ending artifact ID")
    target_type: str = Field(..., description="Type of target artifact")
    path: list[str] = Field(
        default_factory=list,
        description="Ordered list of artifact IDs along the path",
    )
    path_types: list[str] = Field(
        default_factory=list,
        description="Artifact types along the path",
    )


class TraceGraph(BaseModel):
    """Complete trace graph for a single pipeline execution.

    Maps every artifact produced by the pipeline to its inputs,
    enabling full traceability from any output back to the original
    startup input.
    """

    nodes: dict[str, TraceNode] = Field(
        default_factory=dict,
        description="All trace nodes keyed by artifact ID",
    )
    edges: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Adjacency list: artifact_id -> list of input artifact_ids",
    )
    reverse_edges: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Reverse adjacency: artifact_id -> list of dependent artifact_ids",
    )


def _make_id(stage: str, artifact_type: str, index: int) -> str:
    """Generate a deterministic artifact ID."""
    return f"{stage}:{artifact_type}:{index}"


def _content_id(content: str) -> str:
    """Generate a short hash-based ID from content."""
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"hash:{h}"


class TraceGraphBuilder:
    """Builds a TraceGraph from a Report and its pipeline artifacts.

    Analyzes the Report object and reconstructs the trace relationships
    between all pipeline artifacts by examining their cross-references.
    """

    def build(
        self,
        startup_name: str,
        features: Any,
        evidence: list[Any],
        observations: list[Any],
        assessments: list[Any],
        scores: list[Any],
        recommendations: list[Any],
        confidence: list[Any],
    ) -> TraceGraph:
        """Build a complete TraceGraph from pipeline outputs."""
        graph = TraceGraph()
        evidence_map: dict[str, str] = {}

        # Node 0: Startup input
        startup_id = "input:startup:0"
        graph.nodes[startup_id] = TraceNode(
            artifact_id=startup_id,
            stage=PipelineStage.NORMALIZE,
            artifact_type="Startup",
            label=startup_name,
        )

        # Node 1: Extracted Features
        features_id = "extract:features:0"
        graph.nodes[features_id] = TraceNode(
            artifact_id=features_id,
            stage=PipelineStage.EXTRACT,
            artifact_type="ExtractedFeatures",
            label=f"Features for {startup_name}",
            input_ids=[startup_id],
        )
        graph.edges[features_id] = [startup_id]
        graph.reverse_edges.setdefault(startup_id, []).append(features_id)

        # Evidence nodes
        for i, item in enumerate(evidence):
            eid = f"evidence:item:{i}"
            evidence_map[item.statement] = eid
            graph.nodes[eid] = TraceNode(
                artifact_id=eid,
                stage=PipelineStage.EVIDENCE,
                artifact_type="EvidenceItem",
                label=item.statement[:80],
                input_ids=[features_id],
                metadata={"domain": item.domain, "category": item.category},
            )
            graph.edges[eid] = [features_id]
            graph.reverse_edges.setdefault(features_id, []).append(eid)

        # Observation nodes
        obs_id_map: dict[int, str] = {}
        for i, obs in enumerate(observations):
            oid = f"reason:observation:{i}"
            obs_id_map[id(obs)] = oid
            obs_input_ids: list[str] = [features_id]
            for ref in obs.evidence:
                if ref in evidence_map:
                    obs_input_ids.append(evidence_map[ref])
            graph.nodes[oid] = TraceNode(
                artifact_id=oid,
                stage=PipelineStage.REASON,
                artifact_type="Observation",
                label=obs.statement[:80],
                input_ids=obs_input_ids,
                metadata={
                    "dimension": obs.dimension,
                    "source_rule": obs.source_rule,
                },
            )
            graph.edges[oid] = obs_input_ids
            for parent_id in obs_input_ids:
                graph.reverse_edges.setdefault(parent_id, []).append(oid)

        # Dimension Assessment nodes
        assess_id_map: dict[str, str] = {}
        for i, assess in enumerate(assessments):
            aid = f"evaluate:assessment:{i}"
            assess_id_map[assess.dimension] = aid
            assess_input_ids: list[str] = []
            for j, obs in enumerate(observations):
                if obs.dimension == assess.dimension:
                    assess_input_ids.append(
                        f"reason:observation:{j}"
                    )
            for j, item in enumerate(evidence):
                if item.domain == assess.dimension:
                    assess_input_ids.append(
                        f"evidence:item:{j}"
                    )
            graph.nodes[aid] = TraceNode(
                artifact_id=aid,
                stage=PipelineStage.EVALUATE,
                artifact_type="DimensionAssessment",
                label=f"Assessment: {assess.dimension}",
                input_ids=assess_input_ids,
                metadata={"dimension": assess.dimension},
            )
            graph.edges[aid] = assess_input_ids
            for parent_id in assess_input_ids:
                graph.reverse_edges.setdefault(parent_id, []).append(aid)

        # Score nodes
        score_id_map: dict[str, str] = {}
        for i, score in enumerate(scores):
            sid = f"score:result:{i}"
            score_id_map[score.dimension] = sid
            score_input_ids: list[str] = [features_id]
            for j, obs in enumerate(observations):
                if obs.dimension == score.dimension:
                    score_input_ids.append(
                        f"reason:observation:{j}"
                    )
            graph.nodes[sid] = TraceNode(
                artifact_id=sid,
                stage=PipelineStage.SCORE,
                artifact_type="ScoreResult",
                label=f"Score: {score.dimension} = {score.score}",
                input_ids=score_input_ids,
                metadata={
                    "dimension": score.dimension,
                    "score": score.score,
                },
            )
            graph.edges[sid] = score_input_ids
            for parent_id in score_input_ids:
                graph.reverse_edges.setdefault(parent_id, []).append(sid)

        # Recommendation nodes
        for i, rec in enumerate(recommendations):
            rid = f"recommend:rec:{i}"
            rec_input_ids: list[str] = []
            for obs in rec.supporting_observations:
                key = id(obs)
                if key in obs_id_map:
                    rec_input_ids.append(obs_id_map[key])
            for assess in rec.supporting_assessments:
                if assess.dimension in assess_id_map:
                    rec_input_ids.append(assess_id_map[assess.dimension])
            graph.nodes[rid] = TraceNode(
                artifact_id=rid,
                stage=PipelineStage.RECOMMEND,
                artifact_type="Recommendation",
                label=rec.title or rec.action[:80],
                input_ids=rec_input_ids,
                metadata={
                    "category": rec.category,
                    "priority": rec.priority,
                },
            )
            graph.edges[rid] = rec_input_ids
            for parent_id in rec_input_ids:
                graph.reverse_edges.setdefault(parent_id, []).append(rid)

        # Confidence nodes
        for i, conf in enumerate(confidence):
            cid = f"confidence:assess:{i}"
            conf_input_ids: list[str] = [features_id]
            if conf.dimension in score_id_map:
                conf_input_ids.append(score_id_map[conf.dimension])
            graph.nodes[cid] = TraceNode(
                artifact_id=cid,
                stage=PipelineStage.CONFIDENCE,
                artifact_type="ConfidenceAssessment",
                label=f"Confidence: {conf.dimension} = {conf.confidence}",
                input_ids=conf_input_ids,
                metadata={"dimension": conf.dimension},
            )
            graph.edges[cid] = conf_input_ids
            for parent_id in conf_input_ids:
                graph.reverse_edges.setdefault(parent_id, []).append(cid)

        return graph

    def get_trace_paths(
        self, graph: TraceGraph, target_id: str
    ) -> list[TracePath]:
        """Get all trace paths from a target artifact back to its sources."""
        paths: list[TracePath] = []
        if target_id not in graph.nodes:
            return paths

        target_node = graph.nodes[target_id]
        visited: set[str] = set()

        def _dfs(current_id: str, current_path: list[str], current_types: list[str]) -> None:
            if current_id in visited:
                return
            visited.add(current_id)
            current_path.append(current_id)
            if current_id in graph.nodes:
                current_types.append(graph.nodes[current_id].artifact_type)

            parents = graph.edges.get(current_id, [])
            if not parents:
                paths.append(
                    TracePath(
                        source_id=current_id,
                        source_type=graph.nodes.get(current_id, TraceNode(
                            artifact_id=current_id,
                            stage=PipelineStage.NORMALIZE,
                            artifact_type="Unknown",
                        )).artifact_type,
                        target_id=target_id,
                        target_type=target_node.artifact_type,
                        path=list(current_path),
                        path_types=list(current_types),
                    )
                )
            else:
                for parent_id in parents:
                    _dfs(parent_id, current_path, current_types)

            current_path.pop()
            if current_types:
                current_types.pop()
            visited.discard(current_id)

        _dfs(target_id, [], [])
        return paths

    def get_all_downstream(
        self, graph: TraceGraph, source_id: str
    ) -> list[str]:
        """Get all artifact IDs downstream of a given source."""
        result: list[str] = []
        visited: set[str] = set()
        queue = [source_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for dependent in graph.reverse_edges.get(current, []):
                if dependent not in visited:
                    result.append(dependent)
                    queue.append(dependent)
        return result
