"""Feature extraction — structured fact derivation from collected data.

v0.4: Composition-based architecture with quantitative parsing layer
and deterministic inference engine (Sprint 17).

The extraction module transforms CollectedData into ExtractedFeatures
using a composite of single-responsibility extractors. Each extractor
produces only objective, structured facts — no reasoning, scoring,
or predictions.

After extraction, the DerivedMetricsEngine applies deterministic
inference rules to compute additional metrics from extracted values
(e.g. ARR from MRR, revenue per employee, funding efficiency).

Architecture:
  - BaseExtractor: shared text processing utilities
  - 10 domain extractors: one per extraction concern
  - CompositeExtractor: orchestrates all extractors, merges results
  - DerivedMetricsEngine: deterministic inference of derived metrics
  - DefaultFeatureExtractor: legacy-compatible wrapper (delegates to CompositeExtractor)
  - quantitative: reusable numeric parsing shared across all extractors

Future NLP/LLM extractors can replace any individual extractor without
modifying the composite or any other extractor.
"""

from predictron_engine.extraction.composite import CompositeExtractor
from predictron_engine.extraction.derived import DerivedMetricsEngine, DerivedMetricLog
from predictron_engine.extraction.extractor import DefaultFeatureExtractor
from predictron_engine.extraction.feature_models import NlpService

__all__ = [
    "CompositeExtractor",
    "DefaultFeatureExtractor",
    "DerivedMetricsEngine",
    "DerivedMetricLog",
    "NlpService",
]
