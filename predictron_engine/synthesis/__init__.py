"""Sprint 6C — Decision Synthesis Engine.

Aggregates already-produced pipeline outputs (never recomputes them)
into a single deterministic DecisionSynthesis attached to Report:

    from predictron_engine.synthesis.engine import DecisionSynthesisEngine

    synthesis = DecisionSynthesisEngine().synthesize(...)
    report.decision_synthesis = synthesis
"""

from predictron_engine.synthesis.engine import DecisionSynthesisEngine
from predictron_engine.synthesis.models import (
    AlternativeScenario,
    DecisionSynthesis,
    OpportunityItem,
    RiskItem,
    SynthesisSeverity,
    TradeOff,
)
from predictron_engine.synthesis.opportunities import aggregate_opportunities
from predictron_engine.synthesis.priorities import prioritize_recommendations
from predictron_engine.synthesis.risks import aggregate_risks
from predictron_engine.synthesis.scenarios import generate_scenarios
from predictron_engine.synthesis.summary import build_executive_summary
from predictron_engine.synthesis.tradeoffs import analyze_trade_offs

__all__ = [
    "AlternativeScenario",
    "DecisionSynthesis",
    "DecisionSynthesisEngine",
    "OpportunityItem",
    "RiskItem",
    "SynthesisSeverity",
    "TradeOff",
    "aggregate_opportunities",
    "aggregate_risks",
    "analyze_trade_offs",
    "build_executive_summary",
    "generate_scenarios",
    "prioritize_recommendations",
]
