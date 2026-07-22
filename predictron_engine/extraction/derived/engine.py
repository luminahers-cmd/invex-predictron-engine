"""Deterministic Inference Engine — derives structured metrics from extracted features.

The DerivedMetricsEngine applies a registry of deterministic inference rules
to an ExtractedFeatures object, producing additional derived metrics that
are not explicitly stated but can be logically computed.

This engine is the single point of responsibility for all metric derivation.
No inference logic is embedded in individual extractors or downstream components.

Design principles:
  - Fully deterministic: same inputs always produce same outputs
  - Traceable: every derived value is recorded with full provenance
  - Conservative: never guesses — only derives when sufficient evidence exists
  - Independent: does not modify the input ExtractedFeatures
  - Auditable: produces a log of all derivations attempted and completed

Usage:
    engine = DerivedMetricsEngine()
    features, logs = engine.derive(features)
"""

from __future__ import annotations

import logging

from predictron_engine.extraction.derived.models import DerivedMetricLog
from predictron_engine.extraction.derived.rules import (
    ALL_INFERENCE_RULES,
)
from predictron_engine.models.extracted_features import ExtractedFeatures

logger = logging.getLogger(__name__)


class DerivedMetricsEngine:
    """Applies deterministic inference rules to produce derived metrics.

    The engine runs each registered inference rule against the extracted
    features. Rules that produce results have their derived values applied
    to a copy of the features, and the full derivation log is returned
    alongside.

    The engine never modifies the original ExtractedFeatures — it operates
    on a copy and returns the enriched version.
    """

    def __init__(
        self,
        rules: list | None = None,
    ) -> None:
        """Initialize with an optional custom rule set.

        Args:
            rules: List of (name, rule_fn, metric_name) tuples.
                   If None, uses ALL_INFERENCE_RULES.
        """
        self._rules = rules if rules is not None else list(ALL_INFERENCE_RULES)

    def derive(
        self, features: ExtractedFeatures
    ) -> tuple[ExtractedFeatures, list[DerivedMetricLog]]:
        """Apply all inference rules and return enriched features with derivation log.

        Args:
            features: The extracted features from the domain extractors.

        Returns:
            A tuple of (enriched_features, derivation_log). The enriched
            features contain any derived metric values that were computed.
            The derivation log contains full provenance for each derivation.
        """
        logger.info("DerivedMetricsEngine: running %d inference rules", len(self._rules))

        # Work on a mutable copy
        enriched_dict = features.model_dump()
        derivation_log: list[DerivedMetricLog] = []

        # Track which fields have been derived (to avoid cascading derivations
        # that depend on other derived values in the same pass)
        derived_fields: set[str] = set()

        for rule_name, rule_fn, metric_name in self._rules:
            try:
                logs = rule_fn(features)
            except Exception:
                logger.exception("DerivedMetricsEngine: rule %s raised", rule_name)
                continue

            for log_entry in logs:
                derivation_log.append(log_entry)
                # Map derived metric name to the corresponding ExtractedFeatures field
                target_field = _METRIC_TO_FIELD.get(log_entry.metric_name)
                if target_field and target_field not in derived_fields:
                    enriched_dict[target_field] = log_entry.derived_value
                    derived_fields.add(target_field)
                    logger.debug(
                        "DerivedMetricsEngine: %s -> %s = %s",
                        log_entry.metric_name,
                        target_field,
                        log_entry.derived_value,
                    )

        enriched_features = ExtractedFeatures(**enriched_dict)
        enriched_features.derived_metrics_log = [
            log.model_dump() for log in derivation_log
        ]

        logger.info(
            "DerivedMetricsEngine: produced %d derivations for %d fields",
            len(derivation_log),
            len(derived_fields),
        )

        return enriched_features, derivation_log


# ---------------------------------------------------------------------------
# Mapping from derived metric names to ExtractedFeatures fields
# ---------------------------------------------------------------------------

_METRIC_TO_FIELD: dict[str, str] = {
    "arr_from_mrr": "arr_usd",
    "mrr_from_arr": "mrr_usd",
    "revenue_per_employee": "revenue_per_employee_usd",
    "runway_from_cash_and_burn": "runway_months",
    "funding_efficiency": "funding_efficiency_ratio",
    "burn_multiple": "burn_multiple",
    "ltv_cac_ratio": "ltv_cac_ratio",
    "acv_per_customer": "acv_per_customer_usd",
}
