"""Evidence models — structured contextual knowledge about startup domains.

Evidence items are objective, domain-specific facts retrieved from the
knowledge base. They are NOT conclusions about the startup — they are
observations about the domain the startup operates in.

For example, if a startup is classified as "healthtech", the evidence
layer retrieves domain knowledge like "long regulatory approval cycles"
and "high switching costs". These are facts about the healthtech domain,
not judgments about the startup itself.
"""

from predictron_engine.models.report import EvidenceItem  # noqa: F401

__all__ = ["EvidenceItem", "EvidenceSet"]


class EvidenceSet:
    """Aggregated evidence from all providers for a single analysis run.

    Collects every evidence item produced by every provider and provides
    aggregate metrics about evidence coverage.
    """

    def __init__(
        self,
        items: list[EvidenceItem] | None = None,
        provider_count: int = 0,
        feature_coverage: float = 0.0,
    ) -> None:
        self.items: list[EvidenceItem] = items or []
        self.provider_count = provider_count
        self.feature_coverage = feature_coverage
