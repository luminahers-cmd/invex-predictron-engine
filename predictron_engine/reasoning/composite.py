"""CompositeReasoner — orchestrates multiple reasoning rules.

The composite reasoner is the central orchestrator of the reasoning layer.
It accepts a list of independent reasoning rules via dependency injection
and evaluates each one against the extracted features and evidence.

Sprint 6A additions (fully backward compatible):

* A :class:`~predictron_engine.reasoning.context.ReasoningContext` is
  built once per reasoning pass and passed to every rule that implements
  ``evaluate_context(context)``.  Rules that only implement the legacy
  ``evaluate(features, evidence)`` signature keep working unchanged.
* Every observation returned by any rule is deterministically enriched
  with evidence-backed metadata (trust score, provenance document ids,
  agreement ratio, conflict count) via
  :func:`~predictron_engine.reasoning.evidence_backed.enrich_observation`.
* Per-rule :class:`~predictron_engine.reasoning.diagnostics.RuleDiagnostic`
  records are collected and exposed through ``last_diagnostics`` and
  ``reason_with_diagnostics`` without changing existing return types.

Key principles:
  - Each rule is independent and stateless
  - Rules are evaluated in insertion order
  - A failing rule is logged and skipped without affecting others
  - The composite has no reasoning logic of its own
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from predictron_engine.reasoning.consistency import (
    build_consistency_report,
    detect_missing_evidence,
)
from predictron_engine.reasoning.context import ReasoningContext
from predictron_engine.reasoning.diagnostics import (
    RuleDiagnostic,
    RuleTimer,
    build_rule_diagnostic,
)
from predictron_engine.reasoning.evidence_backed import enrich_observation

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.evidence.models import EvidenceBundle
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation
    from predictron_engine.reasoning.consistency import ConsistencyReport
    from predictron_engine.reasoning.reasoning_engine import ReasoningRule

logger = logging.getLogger(__name__)


class CompositeReasoner:
    """Orchestrates multiple reasoning rules into a single observation list.

    Accepts a list of ReasoningRule instances via dependency injection
    and evaluates each one against the provided features and evidence.
    Individual rule failures are caught and logged without disrupting
    the remaining rules.

    Parameters
    ----------
    rules:
        Ordered list of reasoning rules to evaluate.
    """

    def __init__(self, rules: list[ReasoningRule]) -> None:
        self._rules = list(rules)
        self._last_diagnostics: list[RuleDiagnostic] = []
        self._last_consistency: ConsistencyReport | None = None

    def reason(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
        bundle: EvidenceBundle | None = None,
    ) -> list[Observation]:
        """Evaluate all rules and return the combined observations.

        Each rule receives the full features and evidence context.
        Rules implementing ``evaluate_context`` additionally receive the
        structured :class:`ReasoningContext`.  Rules that fail are
        logged and skipped.
        """
        observations, _ = self._run(features, evidence, bundle)
        return observations

    def reason_with_diagnostics(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
        bundle: EvidenceBundle | None = None,
    ) -> tuple[list[Observation], list[RuleDiagnostic]]:
        """Evaluate all rules and return observations plus rule diagnostics."""
        return self._run(features, evidence, bundle)

    def _run(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
        bundle: EvidenceBundle | None,
    ) -> tuple[list[Observation], list[RuleDiagnostic]]:
        """Shared execution path for reason() and reason_with_diagnostics()."""
        logger.info("CompositeReasoner: running %d rules", len(self._rules))

        context = ReasoningContext(features, evidence, bundle)
        documents = context.documents
        missing_domains = detect_missing_evidence(features, evidence)

        observations: list[Observation] = []
        diagnostics: list[RuleDiagnostic] = []

        for rule in self._rules:
            timer = RuleTimer()
            try:
                result = self._invoke_rule(rule, context)
            except Exception:
                logger.warning("Rule %s failed, skipping", type(rule).__name__)
                continue

            enriched = [
                enrich_observation(obs, evidence, documents) for obs in result
            ]
            observations.extend(enriched)

            diagnostics.append(
                build_rule_diagnostic(
                    rule_name=self._rule_name(rule),
                    evidence_examined=len(evidence),
                    observations=enriched,
                    duration_ms=timer.elapsed_ms(),
                    missing_evidence=missing_domains,
                )
            )

        self._last_diagnostics = diagnostics
        self._last_consistency = build_consistency_report(
            features, evidence, observations
        )

        logger.info(
            "CompositeReasoner: generated %d observations", len(observations)
        )
        return observations, diagnostics

    @staticmethod
    def _invoke_rule(
        rule: ReasoningRule,
        context: ReasoningContext,
    ) -> list[Observation]:
        """Dispatch to context-aware evaluation when supported.

        Rules implementing ``evaluate_context(context)`` receive the
        structured ReasoningContext; all other rules receive the legacy
        ``(features, evidence)`` arguments unchanged.
        """
        evaluate_context = getattr(rule, "evaluate_context", None)
        if callable(evaluate_context):
            return evaluate_context(context)
        return rule.evaluate(context.features, context.evidence_items)  # type: ignore[union-attr]

    @staticmethod
    def _rule_name(rule: ReasoningRule) -> str:
        """Best-effort rule name: ``name`` property, else class name."""
        name = getattr(rule, "name", None)
        if isinstance(name, str) and name:
            return name
        return type(rule).__name__

    @property
    def last_diagnostics(self) -> list[RuleDiagnostic]:
        """Diagnostics from the most recent reasoning pass."""
        return list(self._last_diagnostics)

    @property
    def last_consistency(self) -> ConsistencyReport | None:
        """Consistency report from the most recent reasoning pass."""
        return self._last_consistency

    @property
    def rule_count(self) -> int:
        """Return the number of registered rules."""
        return len(self._rules)
