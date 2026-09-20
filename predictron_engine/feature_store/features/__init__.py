"""Feature category modules."""

from __future__ import annotations

from predictron_engine.feature_store.features.benchmark import BENCHMARK_FEATURES
from predictron_engine.feature_store.features.company import COMPANY_FEATURES
from predictron_engine.feature_store.features.founder import FOUNDER_FEATURES
from predictron_engine.feature_store.features.funding import FUNDING_FEATURES
from predictron_engine.feature_store.features.growth import GROWTH_FEATURES
from predictron_engine.feature_store.features.knowledge_graph import KNOWLEDGE_GRAPH_FEATURES
from predictron_engine.feature_store.features.signals import SIGNALS_FEATURES

ALL_FEATURES = (
    COMPANY_FEATURES
    + GROWTH_FEATURES
    + FOUNDER_FEATURES
    + FUNDING_FEATURES
    + KNOWLEDGE_GRAPH_FEATURES
    + SIGNALS_FEATURES
    + BENCHMARK_FEATURES
)

__all__ = [
    "ALL_FEATURES",
    "COMPANY_FEATURES",
    "GROWTH_FEATURES",
    "FOUNDER_FEATURES",
    "FUNDING_FEATURES",
    "KNOWLEDGE_GRAPH_FEATURES",
    "SIGNALS_FEATURES",
    "BENCHMARK_FEATURES",
]
