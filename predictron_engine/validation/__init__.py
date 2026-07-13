"""Validation & Explainability Framework for the Predictron Engine.

Provides structured traceability, explanations, and validation for
every pipeline output. Never modifies pipeline outputs — only reads
and reports findings.

Usage:
    from predictron_engine.validation import ValidationEngine

    engine = ValidationEngine()
    debug_report = engine.validate(
        startup, features, evidence, observations,
        assessments, scores, recommendations, confidence,
    )
"""

from predictron_engine.validation.debug_report import DebugReport
from predictron_engine.validation.explainability import (
    ExplanationBuilder,
    StageExplanation,
)
from predictron_engine.validation.trace import (
    TraceGraph,
    TraceGraphBuilder,
    TraceNode,
    TracePath,
)
from predictron_engine.validation.validation_engine import ValidationEngine
from predictron_engine.validation.validators.pipeline_validator import (
    ValidationFinding,
)

__all__ = [
    "DebugReport",
    "ExplanationBuilder",
    "StageExplanation",
    "TraceGraph",
    "TraceGraphBuilder",
    "TraceNode",
    "TracePath",
    "ValidationEngine",
    "ValidationFinding",
]
