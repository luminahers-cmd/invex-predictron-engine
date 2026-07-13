"""CompositeReasoner — orchestrates multiple reasoning rules.

The composite reasoner is the central orchestrator of the reasoning layer.
It accepts a list of independent reasoning rules via dependency injection
and evaluates each one against the extracted features and evidence.

Key principles:
  - Each rule is independent and stateless
  - Rules are evaluated in insertion order
  - A failing rule is logged and skipped without affecting others
  - The composite has no reasoning logic of its own
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation
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

    def reason(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        """Evaluate all rules and return the combined observations.

        Each rule receives the full features and evidence context.
        Rules that fail are logged and skipped.
        """
        logger.info("CompositeReasoner: running %d rules", len(self._rules))

        observations: list[Observation] = []
        for rule in self._rules:
            try:
                result = rule.evaluate(features, evidence)
                observations.extend(result)
            except Exception:
                logger.warning(
                    "Rule %s failed, skipping",
                    type(rule).__name__,
                )

        logger.info("CompositeReasoner: generated %d observations", len(observations))
        return observations

    @property
    def rule_count(self) -> int:
        """Return the number of registered rules."""
        return len(self._rules)
