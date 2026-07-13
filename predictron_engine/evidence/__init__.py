"""Evidence module — contextual domain knowledge for startup analysis.

The evidence layer transforms extracted facts into structured evidence by
querying domain knowledge providers. It sits between extraction and reasoning
in the pipeline:

  extract → evidence → reason

Evidence items are objective, domain-specific facts retrieved from the
knowledge base. They are NOT conclusions about the startup — they are
observations about the domain the startup operates in.

Providers:
  - IndustryEvidenceProvider — industry-specific market knowledge
  - BusinessModelEvidenceProvider — business model characteristics
  - StageEvidenceProvider — funding stage expectations
  - TechnologyEvidenceProvider — technology stack context
  - GeographyEvidenceProvider — geographic market knowledge
"""

from predictron_engine.evidence.evidence_engine import DefaultEvidenceEngine
from predictron_engine.evidence.evidence_models import EvidenceItem, EvidenceSet

__all__ = ["DefaultEvidenceEngine", "EvidenceItem", "EvidenceSet"]
