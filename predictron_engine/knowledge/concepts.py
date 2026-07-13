"""Venture capital concepts and terminology definitions.

This module serves as the canonical glossary for domain concepts used
throughout the Predictron Engine. It provides:

  - Named enums for common VC concepts
  - Semantic groupings for analysis dimensions
  - Display labels and descriptions for report generation

By centralizing domain vocabulary here, all pipeline modules share a
consistent language. Changing terminology requires updating only this
module rather than scattering string literals across the codebase.
"""

from enum import Enum


class AnalysisDimension(str, Enum):
    """Core dimensions evaluated during startup analysis.

    Each dimension maps to a scoring dimension, a set of reasoning rules,
    and a confidence assessment. The engine evaluates all dimensions and
    aggregates them into an overall score.
    """

    MARKET_OPPORTUNITY = "market_opportunity"
    PRODUCT_STRENGTH = "product_strength"
    FOUNDER_QUALITY = "founder_quality"
    TRACTION_SIGNALS = "traction_signals"
    BUSINESS_MODEL_VIABILITY = "business_model_viability"
    COMPETITIVE_POSITION = "competitive_position"
    TEAM_EXECUTION = "team_execution"


class RecommendationCategory(str, Enum):
    """Categories for investor recommendations."""

    DUE_DILIGENCE = "due_diligence"
    RISK_MITIGATION = "risk_mitigation"
    OPPORTUNITY = "opportunity"
    FOLLOW_UP = "follow_up"
    PORTFOLIO_FIT = "portfolio_fit"


class Priority(str, Enum):
    """Priority levels for recommendations."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Mapping from dimension enum to human-readable labels
DIMENSION_LABELS: dict[AnalysisDimension, str] = {
    AnalysisDimension.MARKET_OPPORTUNITY: "Market Opportunity",
    AnalysisDimension.PRODUCT_STRENGTH: "Product Strength",
    AnalysisDimension.FOUNDER_QUALITY: "Founder Quality",
    AnalysisDimension.TRACTION_SIGNALS: "Traction Signals",
    AnalysisDimension.BUSINESS_MODEL_VIABILITY: "Business Model Viability",
    AnalysisDimension.COMPETITIVE_POSITION: "Competitive Position",
    AnalysisDimension.TEAM_EXECUTION: "Team & Execution",
}

# Default dimensions used when no custom scorer set is provided
DEFAULT_DIMENSIONS: list[AnalysisDimension] = [
    AnalysisDimension.MARKET_OPPORTUNITY,
    AnalysisDimension.PRODUCT_STRENGTH,
    AnalysisDimension.FOUNDER_QUALITY,
    AnalysisDimension.TRACTION_SIGNALS,
    AnalysisDimension.BUSINESS_MODEL_VIABILITY,
    AnalysisDimension.COMPETITIVE_POSITION,
    AnalysisDimension.TEAM_EXECUTION,
]
