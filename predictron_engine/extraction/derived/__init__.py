"""Deterministic Inference Layer — derived metrics from extracted features.

v1.0: Sprint 17 — Deterministic extraction intelligence.

This module provides a centralized, deterministic inference engine that
derives structured metrics not explicitly stated in input text but
computable from extracted features using arithmetic relationships.

Architecture:
  - DerivedMetricLog: audit record for each derivation
  - rules: individual inference functions (pure, stateless)
  - DerivedMetricsEngine: orchestrates rules, produces enriched features

Design principles:
  - Fully deterministic and reproducible
  - No LLMs, no embeddings, no non-deterministic components
  - Every derived value has full provenance and traceability
  - Conservative: never guesses — only derives with sufficient evidence
  - Single point of responsibility for all metric inference
"""

from predictron_engine.extraction.derived.engine import DerivedMetricsEngine
from predictron_engine.extraction.derived.models import DerivedMetricLog

__all__ = ["DerivedMetricsEngine", "DerivedMetricLog"]
