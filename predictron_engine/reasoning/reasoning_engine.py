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

Extensibility:
  - Add new ReasoningRule implementations
  - Inject rule sets via the constructor
  - Rules can be enabled/disabled per analysis context
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from predictron_engine.reasoning.composite import CompositeReasoner
from predictron_engine.reasoning.diagnostics import RuleDiagnostic
from predictron_engine.reasoning.rules import DEFAULT_RULES

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.evidence.models import EvidenceBundle
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation

logger = logging.getLogger(__name__)


class DefaultReasoningEngine:
    """Standard implementation of the ReasoningEngine protocol.

    Wraps a CompositeReasoner with the default set of reasoning rules.
    Custom rules can be injected to replace or extend the defaults.
    """

    def __init__(
        self,
        rules: list | None = None,
    ) -> None:
        effective_rules = list(rules) if rules is not None else list(DEFAULT_RULES)
        self._reasoner = CompositeReasoner(effective_rules)

    def reason(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem] | None = None,
        bundle: EvidenceBundle | None = None,
    ) -> list[Observation]:
        """Evaluate all rules and return combined observations."""
        if evidence is None:
            evidence = []
        return self._reasoner.reason(features, evidence, bundle)

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
    def last_consistency(self):
        """Consistency report from the most recent reasoning pass."""
        return self._reasoner.last_consistency
