"""Autonomous Research Intelligence — Research Planner (Phase 8, Sprint 1).

The deterministic planning layer that decides, for a given company:

* What information is missing,
* What research should be performed,
* In what order that research should occur.

This sprint is **planning only**: no source discovery, no scraping, no
search APIs, no LLMs, and no evidence collection.  The planner is pure —
the same input always produces the same plan.
"""

from predictron_engine.research.exceptions import (
    DependencyCycleError,
    InvalidPlannerInputError,
    ResearchError,
    RuleRegistrationError,
    UnknownTopicError,
)
from predictron_engine.research.models import (
    PLAN_SCHEMA_VERSION,
    EvidenceStatus,
    KnowledgeGap,
    PlannerInput,
    ResearchPlan,
    ResearchPriority,
    ResearchTask,
    ResearchTopic,
)
from predictron_engine.research.planner import (
    ResearchPlanner,
    recommend_topics,
)
from predictron_engine.research.priorities import (
    DEPENDENCY_WEIGHT,
    FRESHNESS_WEIGHT,
    IMPORTANCE_WEIGHT,
    MISSING_EVIDENCE_WEIGHT,
    PREDICTION_IMPACT_WEIGHT,
    SOURCE_AVAILABILITY_WEIGHT,
    compute_priority_score,
    evidence_missing_factor,
    horizon_freshness_scale,
    priority_from_score,
    source_availability_factor,
)
from predictron_engine.research.rules import (
    DEFAULT_RULES,
    CoverageRule,
    MissingFoundersRule,
    MissingFundingRule,
    MissingMarketRule,
    MissingPricingRule,
    MissingTechnologyRule,
    ResearchRule,
    coverage_status,
    evaluate_rules,
    register_rule,
    registered_rules,
    research_rules,
)
from predictron_engine.research.topics import (
    RESEARCH_TOPICS,
    SOURCE_CATEGORIES,
    SOURCE_CATEGORY_DESCRIPTIONS,
    TOPIC_REGISTRY,
    all_topic_ids,
    dependency_edges,
    get_topic,
    has_topic,
    topic_index,
)

__all__ = [
    "CoverageRule",
    "DEFAULT_RULES",
    "DEPENDENCY_WEIGHT",
    "DependencyCycleError",
    "EvidenceStatus",
    "FRESHNESS_WEIGHT",
    "IMPORTANCE_WEIGHT",
    "InvalidPlannerInputError",
    "KnowledgeGap",
    "MISSING_EVIDENCE_WEIGHT",
    "MissingFoundersRule",
    "MissingFundingRule",
    "MissingMarketRule",
    "MissingPricingRule",
    "MissingTechnologyRule",
    "PLAN_SCHEMA_VERSION",
    "PREDICTION_IMPACT_WEIGHT",
    "PlannerInput",
    "RESEARCH_TOPICS",
    "RuleRegistrationError",
    "SOURCE_AVAILABILITY_WEIGHT",
    "SOURCE_CATEGORIES",
    "SOURCE_CATEGORY_DESCRIPTIONS",
    "TOPIC_REGISTRY",
    "ResearchError",
    "ResearchPlan",
    "ResearchPlanner",
    "ResearchPriority",
    "ResearchRule",
    "ResearchTask",
    "ResearchTopic",
    "UnknownTopicError",
    "all_topic_ids",
    "compute_priority_score",
    "coverage_status",
    "dependency_edges",
    "evaluate_rules",
    "evidence_missing_factor",
    "get_topic",
    "has_topic",
    "horizon_freshness_scale",
    "priority_from_score",
    "recommend_topics",
    "register_rule",
    "registered_rules",
    "research_rules",
    "source_availability_factor",
    "topic_index",
]
