"""Evidence providers — one per domain knowledge source.

Each provider retrieves objective, contextual facts from the knowledge
base based on the startup's extracted features. Providers do NOT perform
reasoning or make judgments — they gather relevant domain context.
"""

from predictron_engine.evidence.providers.business_model_provider import (
    BusinessModelEvidenceProvider,
)
from predictron_engine.evidence.providers.competition_provider import (
    CompetitionEvidenceProvider,
)
from predictron_engine.evidence.providers.geography_provider import (
    GeographyEvidenceProvider,
)
from predictron_engine.evidence.providers.industry_provider import (
    IndustryEvidenceProvider,
)
from predictron_engine.evidence.providers.stage_provider import StageEvidenceProvider
from predictron_engine.evidence.providers.technology_provider import (
    TechnologyEvidenceProvider,
)

__all__ = [
    "BusinessModelEvidenceProvider",
    "CompetitionEvidenceProvider",
    "GeographyEvidenceProvider",
    "IndustryEvidenceProvider",
    "StageEvidenceProvider",
    "TechnologyEvidenceProvider",
]
