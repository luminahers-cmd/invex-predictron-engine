"""Reasoning rules — independent, single-responsibility observation generators.

Each rule evaluates extracted features and evidence to produce structured
observations. Rules are independent, stateless, and testable in isolation.
"""

from predictron_engine.reasoning.rules.business_model_context import (
    BusinessModelContextRule,
)
from predictron_engine.reasoning.rules.data_quality import DataQualityRule
from predictron_engine.reasoning.rules.market_context import MarketContextRule
from predictron_engine.reasoning.rules.risk_indicator import RiskIndicatorRule
from predictron_engine.reasoning.rules.stage_expectation import (
    StageExpectationRule,
)
from predictron_engine.reasoning.rules.team_assessment import TeamAssessmentRule
from predictron_engine.reasoning.rules.technology_context import (
    TechnologyContextRule,
)

__all__ = [
    "BusinessModelContextRule",
    "DataQualityRule",
    "DEFAULT_RULES",
    "MarketContextRule",
    "RiskIndicatorRule",
    "StageExpectationRule",
    "TeamAssessmentRule",
    "TechnologyContextRule",
]

DEFAULT_RULES = [
    MarketContextRule(),
    BusinessModelContextRule(),
    StageExpectationRule(),
    TechnologyContextRule(),
    TeamAssessmentRule(),
    DataQualityRule(),
    RiskIndicatorRule(),
]
