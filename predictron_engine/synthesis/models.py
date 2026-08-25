"""Sprint 6C synthesis model re-exports.

The canonical synthesis models (SynthesisSeverity, TradeOff,
AlternativeScenario, RiskItem, OpportunityItem, DecisionSynthesis) live
in ``predictron_engine.models.report`` next to the other canonical
output models. This module re-exports them so the synthesis package
offers a single ergonomic import surface:

    from predictron_engine.synthesis.models import DecisionSynthesis
"""

from predictron_engine.models.report import (
    AlternativeScenario,
    DecisionSynthesis,
    OpportunityItem,
    RiskItem,
    SynthesisSeverity,
    TradeOff,
    severity_order,
)

__all__ = [
    "AlternativeScenario",
    "DecisionSynthesis",
    "OpportunityItem",
    "RiskItem",
    "SynthesisSeverity",
    "TradeOff",
    "severity_order",
]
