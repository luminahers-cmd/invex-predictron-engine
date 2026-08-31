"""Reasoning Engine — generates explainable observations from features and evidence.

The reasoning layer transforms structured facts (ExtractedFeatures) and
domain evidence (EvidenceItem) into explainable observations. Each
observation links a conclusion to the specific evidence that supports it.

Key principles:
  - All observations must be explainable (evidence chain required)
  - No scoring or recommendation — just observations
  - Rules are injectable and independently testable
  - Default rules produce domain-informed observations from evidence

Sprint 6A additions (fully backward compatible):
  - An optional EvidenceBundle can be supplied to ``reason``; when
    present it feeds the structured ReasoningContext handed to
    context-aware rules.
  - Per-rule diagnostics are exposed via ``last_diagnostics`` and
    ``reason_with_diagnostics`` without changing existing signatures.

Sprint 8 additions (fully backward compatible):
  - ``reason_adaptive`` uses an adaptive reasoning budget to skip
    unnecessary computation on straightforward cases.
  - ``last_budget`` exposes the budget allocation from the most
    recent adaptive reasoning pass.
  - ``last_trace`` exposes the reasoning trace for explainability.

Extensibility:
  - Add new ReasoningRule implementations
  - Inject rule sets via the constructor
  - Rules can be enabled/disabled per analysis context
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from predictron_engine.reasoning.composite import CompositeReasoner
from predictron_engine.reasoning.contradiction_graph import (
    ContradictionGraph,
    build_contradiction_graph,
)
from predictron_engine.reasoning.diagnostics import RuleDiagnostic
from predictron_engine.reasoning.rules import DEFAULT_RULES

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.evidence.models import EvidenceBundle
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation, ScoreResult
    from predictron_engine.reasoning.adaptive_budget import ReasoningBudget
    from predictron_engine.reasoning.composite import ReasoningRule
    from predictron_engine.reasoning.consistency import ConsistencyReport
    from predictron_engine.reasoning.trace import ReasoningTrace

logger = logging.getLogger(__name__)


class DefaultReasoningEngine:
    """Standard implementation of the ReasoningEngine protocol.

    Wraps a CompositeReasoner with the default set of reasoning rules.
    Custom rules can be injected to replace or extend the defaults.
    """

    def __init__(
        self,
        rules: list[ReasoningRule] | None = None,
    ) -> None:
        effective_rules = list(rules) if rules is not None else list(DEFAULT_RULES)
        self._reasoner = CompositeReasoner(effective_rules)

    def reason(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem] | None = None,
        bundle: EvidenceBundle | None = None,
    ) -> list[Observation]:
        """Evaluate all rules and return combined observations.

        After reasoning completes, a contradiction graph is deterministically
        built from the observations so downstream stages (confidence,
        calibration, synthesis) can reuse richer conflict data without
        recomputing observations.
        """
        if evidence is None:
            evidence = []
        observations = self._reasoner.reason(features, evidence, bundle)
        self._last_contradiction_graph = build_contradiction_graph(observations)
        return observations

    def reason_with_diagnostics(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem] | None = None,
        bundle: EvidenceBundle | None = None,
    ) -> tuple[list[Observation], list[RuleDiagnostic]]:
        """Evaluate all rules and return observations plus rule diagnostics."""
        if evidence is None:
            evidence = []
        return self._reasoner.reason_with_diagnostics(features, evidence, bundle)

    @property
    def last_diagnostics(self) -> list[RuleDiagnostic]:
        """Diagnostics from the most recent reasoning pass."""
        return self._reasoner.last_diagnostics

    @property
    def last_consistency(self) -> ConsistencyReport | None:
        """Consistency report from the most recent reasoning pass."""
        return self._reasoner.last_consistency

    def reason_adaptive(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem] | None = None,
        bundle: EvidenceBundle | None = None,
    ) -> tuple[list[Observation], ReasoningBudget]:
        """Evaluate rules with adaptive budget allocation.

        Computes a reasoning budget based on data completeness and
        evidence quality, then runs only the rules necessary for
        the estimated complexity. Returns observations plus the
        budget allocation used.

        The existing ``reason()`` method is unchanged and always runs
        all rules. This method optimizes for cases where full reasoning
        is unnecessary.

        Returns
        -------
        tuple of (observations, budget)
        """
        if evidence is None:
            evidence = []

        from predictron_engine.reasoning.adaptive_budget import (
            compute_reasoning_budget,
        )

        budget = compute_reasoning_budget(features, evidence)
        self._last_budget = budget

        if budget.budget_fraction >= 1.0:
            observations = self._reasoner.reason(features, evidence, bundle)
            return observations, budget

        observations = self._reasoner.reason_with_budget(
            features, evidence, bundle, budget,
        )
        return observations, budget

    def reason_with_trace(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem] | None = None,
        bundle: EvidenceBundle | None = None,
        scores: list[ScoreResult] | None = None,
    ) -> tuple[list[Observation], ReasoningTrace]:
        """Run reasoning and produce a full reasoning trace.

        Combines adaptive reasoning with trace generation for
        complete explainability.

        Returns
        -------
        tuple of (observations, trace)
        """
        if evidence is None:
            evidence = []

        observations = self._reasoner.reason(features, evidence, bundle)

        # Build and store the contradiction graph from the observations
        # so it is available through last_contradiction_graph() for
        # downstream stages even when only reason() was called.
        graph = build_contradiction_graph(observations)
        self._last_contradiction_graph = graph

        from predictron_engine.reasoning.trace import build_reasoning_trace

        trace = build_reasoning_trace(
            observations, evidence, scores or [],
            contradiction_graph=graph,
        )
        self._last_trace = trace

        return observations, trace

    @property
    def last_budget(self) -> ReasoningBudget | None:
        """Budget from the most recent adaptive reasoning pass."""
        return getattr(self, "_last_budget", None)

    @property
    def last_trace(self) -> ReasoningTrace | None:
        """Trace from the most recent reasoning-with-trace pass."""
        return getattr(self, "_last_trace", None)

    @property
    def last_contradiction_graph(self) -> ContradictionGraph | None:
        """Contradiction graph from the most recent reasoning pass.

        Built deterministically from the observations produced by
        ``reason()``, ``reason_with_trace()``, or
        ``reason_adaptive()``.  Available for downstream consumers
        (confidence, calibration, synthesis) without recomputation.
        """
        return getattr(self, "_last_contradiction_graph", None)
