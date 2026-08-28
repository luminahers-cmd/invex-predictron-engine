"""Reasoning — explainable observation generation from features and evidence.

Sprint 6A adds the evidence-aware reasoning layer:

* :class:`ReasoningContext` — structured, pre-indexed context passed to
  context-aware rules.
* Cross-feature consistency detectors (:mod:`consistency`).
* Deterministic reasoning confidence (:mod:`confidence`).
* Per-rule diagnostics (:mod:`diagnostics`).
* Evidence-backed observation enrichment (:mod:`evidence_backed`).
* Provenance traceability chain resolution (:mod:`traceability`).

Sprint 8 adds adaptive reasoning:

* Adaptive reasoning budget (:mod:`adaptive_budget`).
* Progressive evidence evaluation (:mod:`progressive_evidence`).
* Contradiction graph (:mod:`contradiction_graph`).
* Reasoning trace for explainability (:mod:`trace`).
"""

from predictron_engine.reasoning.adaptive_budget import (
    BudgetReport,
    ReasoningBudget,
    compute_reasoning_budget,
    should_skip_rule,
)
from predictron_engine.reasoning.composite import CompositeReasoner
from predictron_engine.reasoning.confidence import (
    ConfidenceBreakdown,
    compute_context_confidence,
    compute_reasoning_confidence,
    compute_reasoning_confidence_breakdown,
)
from predictron_engine.reasoning.consistency import (
    ConsistencyReport,
    ContradictionFinding,
    ReinforcementFinding,
    UnsupportedFinding,
    build_consistency_report,
    detect_contradictory_features,
    detect_missing_evidence,
    detect_reinforcing_features,
    detect_unsupported_conclusions,
)
from predictron_engine.reasoning.context import ReasoningContext
from predictron_engine.reasoning.contradiction_graph import (
    ContradictionEdge,
    ContradictionGraph,
    DominantConflict,
    build_contradiction_graph,
)
from predictron_engine.reasoning.diagnostics import RuleDiagnostic
from predictron_engine.reasoning.evidence_backed import (
    enrich_observation,
    match_evidence_for_observation,
)
from predictron_engine.reasoning.progressive_evidence import (
    EvidenceCheckpoint,
    ProgressiveEvaluation,
    evaluate_evidence_progressively,
)
from predictron_engine.reasoning.reasoning_engine import DefaultReasoningEngine
from predictron_engine.reasoning.trace import (
    ConfidenceEvolution,
    ReasoningTrace,
    TraceEntry,
    build_reasoning_trace,
)
from predictron_engine.reasoning.traceability import (
    ProvenanceChainEntry,
    resolve_provenance_chain,
    unresolved_document_ids,
    verify_traceability,
)

__all__ = [
    "BudgetReport",
    "CompositeReasoner",
    "ConfidenceBreakdown",
    "ConfidenceEvolution",
    "ConsistencyReport",
    "ContradictionEdge",
    "ContradictionFinding",
    "ContradictionGraph",
    "DefaultReasoningEngine",
    "DominantConflict",
    "EvidenceCheckpoint",
    "ProgressiveEvaluation",
    "ProvenanceChainEntry",
    "ReasoningBudget",
    "ReasoningContext",
    "ReasoningTrace",
    "ReinforcementFinding",
    "RuleDiagnostic",
    "TraceEntry",
    "UnsupportedFinding",
    "build_contradiction_graph",
    "build_consistency_report",
    "build_reasoning_trace",
    "compute_context_confidence",
    "compute_reasoning_budget",
    "compute_reasoning_confidence",
    "compute_reasoning_confidence_breakdown",
    "detect_contradictory_features",
    "detect_missing_evidence",
    "detect_reinforcing_features",
    "detect_unsupported_conclusions",
    "enrich_observation",
    "evaluate_evidence_progressively",
    "match_evidence_for_observation",
    "resolve_provenance_chain",
    "should_skip_rule",
    "unresolved_document_ids",
    "verify_traceability",
]
