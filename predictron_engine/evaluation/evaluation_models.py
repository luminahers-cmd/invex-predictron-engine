"""Models for the evaluation framework.

This module re-exports the evaluation models from models/report.py
where they are canonically defined alongside other report models.

Re-exports:
- DimensionAssessment: Explainable assessment for a single analysis dimension
- EvaluationResult: Composite result containing all dimension assessments
"""

from predictron_engine.models.report import DimensionAssessment, EvaluationResult

__all__ = ["DimensionAssessment", "EvaluationResult"]
