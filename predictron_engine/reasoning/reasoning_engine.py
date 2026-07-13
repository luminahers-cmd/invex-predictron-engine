"""Reasoning Engine — generates explainable observations from features and evidence.

The reasoning layer transforms structured facts (ExtractedFeatures) and
domain evidence (EvidenceItem) into explainable observations. Each
observation links a conclusion to the specific evidence that supports it.

Key principles:
  - All observations must be explainable (evidence chain required)
  - No scoring or recommendation — just observations
  - Rules are injectable and independently testable
  - Default rules produce domain-informed observations from evidence

Extensibility:
  - Add new ReasoningRule implementations
  - Inject rule sets via the constructor
  - Rules can be enabled/disabled per analysis context
"""

from __future__ import annotations

import logging

from predictron_engine.reasoning.composite import CompositeReasoner
from predictron_engine.reasoning.rules import DEFAULT_RULES

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

    def reason(self, features, evidence=None):
        """Evaluate all rules and return combined observations."""
        if evidence is None:
            evidence = []
        return self._reasoner.reason(features, evidence)
