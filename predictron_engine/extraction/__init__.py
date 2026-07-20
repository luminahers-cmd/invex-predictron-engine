"""Feature extraction — structured fact derivation from collected data.

v0.3: Composition-based architecture with quantitative parsing layer.

The extraction module transforms CollectedData into ExtractedFeatures
using a composite of single-responsibility extractors. Each extractor
produces only objective, structured facts — no reasoning, scoring,
or predictions.

Architecture:
  - BaseExtractor: shared text processing utilities
  - 10 domain extractors: one per extraction concern
  - CompositeExtractor: orchestrates all extractors, merges results
  - DefaultFeatureExtractor: legacy-compatible wrapper (delegates to CompositeExtractor)
  - quantitative: reusable numeric parsing shared across all extractors

Future NLP/LLM extractors can replace any individual extractor without
modifying the composite or any other extractor.
"""

from predictron_engine.extraction.composite import CompositeExtractor
from predictron_engine.extraction.extractor import DefaultFeatureExtractor
from predictron_engine.extraction.feature_models import NlpService

__all__ = ["CompositeExtractor", "DefaultFeatureExtractor", "NlpService"]
