"""Scoring Engine — assigns numerical scores to analysis dimensions.

The scoring layer produces a ScoreResult for each analysis dimension.
It consumes features and observations as input and produces numerical
scores (0-100) with rationales.

Key principles:
  - No hardcoded scoring formulas
  - Scores are produced by injectable DimensionScorer components
  - Each scorer operates independently on a single dimension
  - Each scorer is deterministic and fully explainable
  - Risk signals penalize relevant dimensions
  - Data completeness acts as a confidence dampener

Extensibility:
  - Implement DimensionScorer for each dimension
  - Inject scorer sets for different analysis contexts
  - Scorer priority and weighting can be configured externally
"""

import logging
from typing import Protocol, runtime_checkable

from predictron_engine.knowledge.concepts import DIMENSION_LABELS, AnalysisDimension
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation, ScoreResult

logger = logging.getLogger(__name__)


@runtime_checkable
class DimensionScorer(Protocol):
    """Protocol for individual dimension scorers.

    Each scorer evaluates features and observations for one analysis
    dimension and produces a ScoreResult with a score and rationale.
    """

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        """Score the relevant dimension based on features and observations."""
        ...


class PlaceholderDimensionScorer:
    """Default scorer that returns a neutral placeholder score.

    This scorer uses basic feature heuristics to produce a minimal
    score. It exists as a template — real scorers will replace it.
    """

    def __init__(self, dimension: AnalysisDimension) -> None:
        self._dimension = dimension

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        relevant_observations = [
            o for o in observations if o.dimension == self._dimension.value
        ]

        base_score = 50.0

        if features.has_pitch_deck:
            base_score += 5.0
        if features.founder_profile_count > 0:
            base_score += 3.0
        if features.description_length > 200:
            base_score += 2.0

        score = max(0.0, min(100.0, base_score))

        evidence = [obs.statement for obs in relevant_observations]

        label = DIMENSION_LABELS.get(self._dimension, self._dimension.value)
        rationale = f"Placeholder scoring for {label} dimension."

        return ScoreResult(
            dimension=self._dimension.value,
            score=score,
            rationale=rationale,
            evidence=evidence,
        )


class CompetitionScorer:
    """Scores the COMPETITIVE_POSITION dimension based on competition features.

    Uses weighted contributions from market concentration, competitive density,
    moat indicators, switching costs, network effects, and barriers to entry.
    All scoring logic is deterministic and traceable.
    """

    def __init__(self) -> None:
        self._dimension = AnalysisDimension.COMPETITIVE_POSITION

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        relevant_obs = [
            o for o in observations if o.dimension == self._dimension.value
        ]

        base_score = 50.0
        adjustments: list[tuple[str, float]] = []

        base_score, adj_conc = self._score_concentration(base_score, features)
        if adj_conc != 0:
            adjustments.append(("market_concentration", adj_conc))

        base_score, adj_den = self._score_density(base_score, features)
        if adj_den != 0:
            adjustments.append(("competitive_density", adj_den))

        base_score, adj_moat = self._score_moats(base_score, features)
        if adj_moat != 0:
            adjustments.append(("moat_indicators", adj_moat))

        base_score, adj_sw = self._score_switching(base_score, features)
        if adj_sw != 0:
            adjustments.append(("switching_costs", adj_sw))

        base_score, adj_net = self._score_network(base_score, features)
        if adj_net != 0:
            adjustments.append(("network_effects", adj_net))

        base_score, adj_bar = self._score_barriers(base_score, features)
        if adj_bar != 0:
            adjustments.append(("barriers_to_entry", adj_bar))

        base_score, adj_oss = self._score_open_source(base_score, features)
        if adj_oss != 0:
            adjustments.append(("open_source_competition", adj_oss))

        base_score, adj_diff = self._score_differentiation(base_score, features)
        if adj_diff != 0:
            adjustments.append(("differentiation", adj_diff))

        base_score, adj_obs = self._score_observations(base_score, relevant_obs)
        if adj_obs != 0:
            adjustments.append(("observations", adj_obs))

        final_score = max(0.0, min(100.0, base_score))
        rationale = (
            f"Competitive position scored from {len(adjustments)} signal dimensions. "
            f"Base 50, final {final_score:.1f}."
        )

        return ScoreResult(
            dimension=self._dimension.value,
            score=final_score,
            rationale=rationale,
            evidence=[o.statement for o in relevant_obs],
        )

    @staticmethod
    def _score_concentration(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.market_concentration:
            return score, 0.0
        key = features.market_concentration.lower()
        mapping = {
            "fragmented": 5.0,
            "moderately_concentrated": 0.0,
            "concentrated": -8.0,
            "dominated": -15.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_density(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.competitive_density:
            return score, 0.0
        key = features.competitive_density.lower()
        mapping = {
            "sparse": 5.0,
            "moderate": 0.0,
            "dense": -6.0,
            "hyper_competitive": -12.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_moats(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.competitive_moat_indicators:
            return score, 0.0
        count = len(features.competitive_moat_indicators)
        delta = min(15.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_switching(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.switching_cost_signals:
            return score, 0.0
        count = len(features.switching_cost_signals)
        delta = min(12.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_network(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.network_effect_competition:
            return score, 0.0
        key = features.network_effect_competition.lower()
        mapping = {
            "strong_network_effects": 12.0,
            "moderate_network_effects": 5.0,
            "no_network_effects": -3.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_barriers(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.barriers_to_entry:
            return score, 0.0
        count = len(features.barriers_to_entry)
        delta = min(10.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_open_source(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.open_source_competition:
            return score, 0.0
        count = len(features.open_source_competition)
        delta = min(10.0, count * 2.5)
        return score - delta, -delta

    @staticmethod
    def _score_differentiation(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.differentiation_signals:
            return score, 0.0
        count = len(features.differentiation_signals)
        delta = min(10.0, count * 2.5)
        return score + delta, delta

    @staticmethod
    def _score_observations(
        score: float, observations: list[Observation]
    ) -> tuple[float, float]:
        if not observations:
            return score, 0.0
        total_impact = sum(o.importance for o in observations if hasattr(o, "importance"))
        avg_importance = total_impact / len(observations)
        delta = avg_importance * 10.0
        return score + delta, delta


class MarketOpportunityScorer:
    """Scores the MARKET_OPPORTUNITY dimension based on market intelligence features.

    Evaluates market maturity, industry confidence, enterprise orientation,
    market signals, market characteristics, and geography. Incorporates
    market-related risk signals as penalties and data completeness as a
    confidence dampener. All scoring logic is deterministic and traceable.
    """

    def __init__(self) -> None:
        self._dimension = AnalysisDimension.MARKET_OPPORTUNITY

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        relevant_obs = [
            o for o in observations if o.dimension == self._dimension.value
        ]

        base_score = 50.0
        adjustments: list[tuple[str, float]] = []

        base_score, adj = self._score_maturity(base_score, features)
        if adj != 0:
            adjustments.append(("market_maturity", adj))

        base_score, adj = self._score_orientation(base_score, features)
        if adj != 0:
            adjustments.append(("enterprise_orientation", adj))

        base_score, adj = self._score_industry_confidence(base_score, features)
        if adj != 0:
            adjustments.append(("industry_confidence", adj))

        base_score, adj = self._score_market_signals(base_score, features)
        if adj != 0:
            adjustments.append(("market_signals", adj))

        base_score, adj = self._score_market_characteristics(base_score, features)
        if adj != 0:
            adjustments.append(("market_characteristics", adj))

        base_score, adj = self._score_geography(base_score, features)
        if adj != 0:
            adjustments.append(("geography", adj))

        base_score, adj = self._score_market_risk(base_score, features)
        if adj != 0:
            adjustments.append(("market_risk", adj))

        base_score, adj = self._score_observations(base_score, relevant_obs)
        if adj != 0:
            adjustments.append(("observations", adj))

        final_score = self._apply_data_dampener(base_score, features)

        rationale = (
            f"Market opportunity scored from {len(adjustments)} signal dimensions. "
            f"Base 50, final {final_score:.1f}."
        )

        return ScoreResult(
            dimension=self._dimension.value,
            score=final_score,
            rationale=rationale,
            evidence=[o.statement for o in relevant_obs],
        )

    @staticmethod
    def _score_maturity(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.market_maturity:
            return score, 0.0
        key = features.market_maturity.lower()
        mapping = {
            "emerging": 10.0,
            "growth": 5.0,
            "mature": -5.0,
            "saturated": -15.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_orientation(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.enterprise_orientation:
            return score, 0.0
        key = features.enterprise_orientation.lower()
        mapping = {
            "enterprise": 3.0,
            "hybrid": 1.0,
            "consumer": 0.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_industry_confidence(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if features.industry_confidence <= 0.0:
            return score, 0.0
        delta = (features.industry_confidence - 0.5) * 12.0
        return score + delta, delta

    @staticmethod
    def _score_market_signals(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        """Score based on market signal count and/or structured market size."""
        delta = 0.0
        if features.market_signals:
            count = len(features.market_signals)
            delta = min(8.0, count * 1.5)

        if features.market_size_usd is not None:
            market = features.market_size_usd
            if market >= 1_000_000_000_000:
                market_delta = 8.0
            elif market >= 100_000_000_000:
                market_delta = 6.0
            elif market >= 10_000_000_000:
                market_delta = 4.0
            else:
                market_delta = 1.0
            delta = max(delta, market_delta)

        return score + delta, delta

    @staticmethod
    def _score_market_characteristics(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.market_characteristics:
            return score, 0.0
        count = len(features.market_characteristics)
        delta = min(6.0, count * 1.5)
        return score + delta, delta

    @staticmethod
    def _score_geography(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.geography:
            return score, 0.0
        key = features.geography.lower().replace(" ", "_")
        strong_markets = {
            "north_america", "europe", "asia_pacific",
        }
        if key in strong_markets:
            delta = 3.0
            return score + delta, delta
        return score, 0.0

    @staticmethod
    def _score_market_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.market_risk:
            return score, 0.0
        count = len(features.market_risk)
        delta = min(-3.0, count * -2.0)
        return score + delta, delta

    @staticmethod
    def _score_observations(
        score: float, observations: list[Observation]
    ) -> tuple[float, float]:
        if not observations:
            return score, 0.0
        total_impact = sum(
            o.importance for o in observations if hasattr(o, "importance")
        )
        avg_importance = total_impact / len(observations)
        delta = avg_importance * 8.0
        return score + delta, delta

    @staticmethod
    def _apply_data_dampener(
        score: float, features: ExtractedFeatures
    ) -> float:
        dampening = 1.0 - features.data_completeness
        deviation = score - 50.0
        dampened = 50.0 + deviation * (1.0 - dampening * 0.3)
        return max(0.0, min(100.0, dampened))


class FounderQualityScorer:
    """Scores the FOUNDER_QUALITY dimension based on founder intelligence features.

    Evaluates founder team type, profile count, domain expertise, serial
    founder indicators, leadership roles, advisors, market fit, engineering
    strength, and risk signals. Incorporates founder risk signals as penalties
    and data completeness as a dampener. All scoring is deterministic.
    """

    def __init__(self) -> None:
        self._dimension = AnalysisDimension.FOUNDER_QUALITY

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        relevant_obs = [
            o for o in observations if o.dimension == self._dimension.value
        ]

        base_score = 50.0
        adjustments: list[tuple[str, float]] = []

        base_score, adj = self._score_team_type(base_score, features)
        if adj != 0:
            adjustments.append(("founder_team_type", adj))

        base_score, adj = self._score_profile_count(base_score, features)
        if adj != 0:
            adjustments.append(("founder_profiles", adj))

        base_score, adj = self._score_domain_expertise(base_score, features)
        if adj != 0:
            adjustments.append(("domain_expertise", adj))

        base_score, adj = self._score_serial_founder(base_score, features)
        if adj != 0:
            adjustments.append(("serial_founder", adj))

        base_score, adj = self._score_leadership(base_score, features)
        if adj != 0:
            adjustments.append(("leadership_roles", adj))

        base_score, adj = self._score_advisors(base_score, features)
        if adj != 0:
            adjustments.append(("advisor_mentions", adj))

        base_score, adj = self._score_market_fit(base_score, features)
        if adj != 0:
            adjustments.append(("founder_market_fit", adj))

        base_score, adj = self._score_engineering_strength(base_score, features)
        if adj != 0:
            adjustments.append(("engineering_strength", adj))

        base_score, adj = self._score_founder_risk(base_score, features)
        if adj != 0:
            adjustments.append(("founder_risk", adj))

        base_score, adj = self._score_observations(base_score, relevant_obs)
        if adj != 0:
            adjustments.append(("observations", adj))

        final_score = self._apply_data_dampener(base_score, features)

        rationale = (
            f"Founder quality scored from {len(adjustments)} signal dimensions. "
            f"Base 50, final {final_score:.1f}."
        )

        return ScoreResult(
            dimension=self._dimension.value,
            score=final_score,
            rationale=rationale,
            evidence=[o.statement for o in relevant_obs],
        )

    @staticmethod
    def _score_team_type(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.founder_team_type:
            return score, 0.0
        key = features.founder_team_type.lower()
        mapping = {
            "technical": 8.0,
            "mixed": 4.0,
            "business": 2.0,
            "unknown": -2.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_profile_count(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        count = features.founder_profile_count
        if count == 0:
            delta = -10.0
        elif count == 1:
            delta = -2.0
        elif count == 2:
            delta = 5.0
        else:
            delta = 8.0
        return score + delta, delta

    @staticmethod
    def _score_domain_expertise(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.domain_expertise_signals:
            return score, 0.0
        count = len(features.domain_expertise_signals)
        delta = min(12.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_serial_founder(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.serial_founder_indicators:
            return score, 0.0
        count = len(features.serial_founder_indicators)
        delta = min(12.0, count * 4.0)
        return score + delta, delta

    @staticmethod
    def _score_leadership(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.leadership_roles:
            return score, 0.0
        count = len(features.leadership_roles)
        delta = min(5.0, count * 1.5)
        return score + delta, delta

    @staticmethod
    def _score_advisors(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.advisor_mentions:
            return score, 0.0
        count = len(features.advisor_mentions)
        delta = min(6.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_market_fit(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.founder_market_fit_signals:
            return score, 0.0
        count = len(features.founder_market_fit_signals)
        delta = min(9.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_engineering_strength(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.engineering_strength:
            return score, 0.0
        key = features.engineering_strength.lower()
        mapping = {
            "strong": 6.0,
            "moderate": 2.0,
            "weak": -5.0,
            "unknown": 0.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_founder_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.founder_risk:
            return score, 0.0
        count = len(features.founder_risk)
        delta = min(-3.0, count * -3.0)
        return score + delta, delta

    @staticmethod
    def _score_observations(
        score: float, observations: list[Observation]
    ) -> tuple[float, float]:
        if not observations:
            return score, 0.0
        total_impact = sum(
            o.importance for o in observations if hasattr(o, "importance")
        )
        avg_importance = total_impact / len(observations)
        delta = avg_importance * 8.0
        return score + delta, delta

    @staticmethod
    def _apply_data_dampener(
        score: float, features: ExtractedFeatures
    ) -> float:
        dampening = 1.0 - features.data_completeness
        deviation = score - 50.0
        dampened = 50.0 + deviation * (1.0 - dampening * 0.3)
        return max(0.0, min(100.0, dampened))


class ProductStrengthScorer:
    """Scores the PRODUCT_STRENGTH dimension based on product and technology features.

    Evaluates product category/type, AI orientation, capabilities, features,
    integrations, technical complexity, innovation, defensibility, scalability,
    differentiation, and technology confidence. Incorporates product and
    technology risk signals as penalties. All scoring is deterministic.
    """

    def __init__(self) -> None:
        self._dimension = AnalysisDimension.PRODUCT_STRENGTH

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        relevant_obs = [
            o for o in observations if o.dimension == self._dimension.value
        ]

        base_score = 50.0
        adjustments: list[tuple[str, float]] = []

        base_score, adj = self._score_product_type(base_score, features)
        if adj != 0:
            adjustments.append(("product_type", adj))

        base_score, adj = self._score_ai_orientation(base_score, features)
        if adj != 0:
            adjustments.append(("ai_orientation", adj))

        base_score, adj = self._score_capabilities(base_score, features)
        if adj != 0:
            adjustments.append(("primary_capabilities", adj))

        base_score, adj = self._score_features(base_score, features)
        if adj != 0:
            adjustments.append(("feature_signals", adj))

        base_score, adj = self._score_integrations(base_score, features)
        if adj != 0:
            adjustments.append(("integration_ecosystem", adj))

        base_score, adj = self._score_technical_complexity(base_score, features)
        if adj != 0:
            adjustments.append(("technical_complexity", adj))

        base_score, adj = self._score_innovation(base_score, features)
        if adj != 0:
            adjustments.append(("innovation_signals", adj))

        base_score, adj = self._score_defensibility(base_score, features)
        if adj != 0:
            adjustments.append(("defensibility_signals", adj))

        base_score, adj = self._score_scalability(base_score, features)
        if adj != 0:
            adjustments.append(("scalability_indicators", adj))

        base_score, adj = self._score_differentiation(base_score, features)
        if adj != 0:
            adjustments.append(("differentiation_signals", adj))

        base_score, adj = self._score_product_confidence(base_score, features)
        if adj != 0:
            adjustments.append(("product_confidence", adj))

        base_score, adj = self._score_tech_confidence(base_score, features)
        if adj != 0:
            adjustments.append(("technology_confidence", adj))

        base_score, adj = self._score_product_risk(base_score, features)
        if adj != 0:
            adjustments.append(("product_risk", adj))

        base_score, adj = self._score_technology_risk(base_score, features)
        if adj != 0:
            adjustments.append(("technology_risk", adj))

        base_score, adj = self._score_observations(base_score, relevant_obs)
        if adj != 0:
            adjustments.append(("observations", adj))

        final_score = self._apply_data_dampener(base_score, features)

        rationale = (
            f"Product strength scored from {len(adjustments)} signal dimensions. "
            f"Base 50, final {final_score:.1f}."
        )

        return ScoreResult(
            dimension=self._dimension.value,
            score=final_score,
            rationale=rationale,
            evidence=[o.statement for o in relevant_obs],
        )

    @staticmethod
    def _score_product_type(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.product_type:
            return score, 0.0
        key = features.product_type.lower()
        mapping = {
            "platform": 5.0,
            "application": 3.0,
            "tool": 2.0,
            "api": 4.0,
            "infrastructure": 6.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_ai_orientation(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.ai_orientation:
            return score, 0.0
        key = features.ai_orientation.lower()
        mapping = {
            "ai_native": 8.0,
            "ai_enabled": 4.0,
            "non_ai": 0.0,
            "unknown": -2.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_capabilities(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.primary_capabilities:
            return score, 0.0
        count = len(features.primary_capabilities)
        delta = min(10.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_features(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.feature_signals:
            return score, 0.0
        count = len(features.feature_signals)
        delta = min(8.0, count * 1.5)
        return score + delta, delta

    @staticmethod
    def _score_integrations(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.integration_ecosystem:
            return score, 0.0
        count = len(features.integration_ecosystem)
        delta = min(5.0, count * 1.5)
        return score + delta, delta

    @staticmethod
    def _score_technical_complexity(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.technical_complexity:
            return score, 0.0
        key = features.technical_complexity.lower()
        mapping = {
            "high": 6.0,
            "moderate": 3.0,
            "low": -2.0,
            "unknown": 0.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_innovation(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.innovation_signals:
            return score, 0.0
        count = len(features.innovation_signals)
        delta = min(8.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_defensibility(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.defensibility_signals:
            return score, 0.0
        count = len(features.defensibility_signals)
        delta = min(8.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_scalability(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.scalability_indicators:
            return score, 0.0
        count = len(features.scalability_indicators)
        delta = min(6.0, count * 1.5)
        return score + delta, delta

    @staticmethod
    def _score_differentiation(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.differentiation_signals:
            return score, 0.0
        count = len(features.differentiation_signals)
        delta = min(8.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_product_confidence(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if features.product_confidence <= 0.0:
            return score, 0.0
        delta = (features.product_confidence - 0.5) * 8.0
        return score + delta, delta

    @staticmethod
    def _score_tech_confidence(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if features.technology_confidence <= 0.0:
            return score, 0.0
        delta = (features.technology_confidence - 0.5) * 8.0
        return score + delta, delta

    @staticmethod
    def _score_product_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.product_risk:
            return score, 0.0
        count = len(features.product_risk)
        delta = min(-3.0, count * -2.0)
        return score + delta, delta

    @staticmethod
    def _score_technology_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.technology_risk:
            return score, 0.0
        count = len(features.technology_risk)
        delta = min(-3.0, count * -2.0)
        return score + delta, delta

    @staticmethod
    def _score_observations(
        score: float, observations: list[Observation]
    ) -> tuple[float, float]:
        if not observations:
            return score, 0.0
        total_impact = sum(
            o.importance for o in observations if hasattr(o, "importance")
        )
        avg_importance = total_impact / len(observations)
        delta = avg_importance * 8.0
        return score + delta, delta

    @staticmethod
    def _apply_data_dampener(
        score: float, features: ExtractedFeatures
    ) -> float:
        dampening = 1.0 - features.data_completeness
        deviation = score - 50.0
        dampened = 50.0 + deviation * (1.0 - dampening * 0.3)
        return max(0.0, min(100.0, dampened))


class BusinessModelScorer:
    """Scores the BUSINESS_MODEL_VIABILITY dimension based on business model features.

    Evaluates revenue model, pricing, monetization, customer acquisition,
    sales motion, distribution, recurring revenue, model maturity, unit
    economics, network effects, platform characteristics, switching costs,
    and business model risk. All scoring is deterministic.
    """

    def __init__(self) -> None:
        self._dimension = AnalysisDimension.BUSINESS_MODEL_VIABILITY

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        relevant_obs = [
            o for o in observations if o.dimension == self._dimension.value
        ]

        base_score = 50.0
        adjustments: list[tuple[str, float]] = []

        base_score, adj = self._score_revenue_model(base_score, features)
        if adj != 0:
            adjustments.append(("revenue_model", adj))

        base_score, adj = self._score_pricing_model(base_score, features)
        if adj != 0:
            adjustments.append(("pricing_model", adj))

        base_score, adj = self._score_acquisition(base_score, features)
        if adj != 0:
            adjustments.append(("customer_acquisition", adj))

        base_score, adj = self._score_distribution(base_score, features)
        if adj != 0:
            adjustments.append(("distribution_model", adj))

        base_score, adj = self._score_recurring_revenue(base_score, features)
        if adj != 0:
            adjustments.append(("recurring_revenue", adj))

        base_score, adj = self._score_model_maturity(base_score, features)
        if adj != 0:
            adjustments.append(("business_model_maturity", adj))

        base_score, adj = self._score_unit_economics(base_score, features)
        if adj != 0:
            adjustments.append(("unit_economics", adj))

        base_score, adj = self._score_network_effects(base_score, features)
        if adj != 0:
            adjustments.append(("network_effects", adj))

        base_score, adj = self._score_platform(base_score, features)
        if adj != 0:
            adjustments.append(("platform_characteristics", adj))

        base_score, adj = self._score_switching(base_score, features)
        if adj != 0:
            adjustments.append(("switching_costs", adj))

        base_score, adj = self._score_model_confidence(base_score, features)
        if adj != 0:
            adjustments.append(("business_model_confidence", adj))

        base_score, adj = self._score_business_model_risk(base_score, features)
        if adj != 0:
            adjustments.append(("business_model_risk", adj))

        base_score, adj = self._score_observations(base_score, relevant_obs)
        if adj != 0:
            adjustments.append(("observations", adj))

        final_score = self._apply_data_dampener(base_score, features)

        rationale = (
            f"Business model viability scored from {len(adjustments)} signal dimensions. "
            f"Base 50, final {final_score:.1f}."
        )

        return ScoreResult(
            dimension=self._dimension.value,
            score=final_score,
            rationale=rationale,
            evidence=[o.statement for o in relevant_obs],
        )

    @staticmethod
    def _score_revenue_model(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.revenue_model:
            return score, 0.0
        key = features.revenue_model.lower()
        mapping = {
            "subscription": 8.0,
            "licensing": 6.0,
            "usage_based": 5.0,
            "transaction_fee": 4.0,
            "commission": 3.0,
            "freemium": 2.0,
            "advertising": -2.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_pricing_model(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.pricing_model:
            return score, 0.0
        key = features.pricing_model.lower()
        mapping = {
            "tiered": 6.0,
            "enterprise_contract": 5.0,
            "per_seat": 4.0,
            "usage_based": 4.0,
            "annual_contract": 3.0,
            "per_study": 4.0,
            "freemium": 2.0,
            "flat_rate": 2.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_acquisition(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.customer_acquisition_model:
            return score, 0.0
        key = features.customer_acquisition_model.lower()
        mapping = {
            "product_led": 6.0,
            "hybrid": 4.0,
            "sales_led": 3.0,
            "marketplace_organic": 5.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_distribution(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.distribution_model:
            return score, 0.0
        key = features.distribution_model.lower()
        mapping = {
            "direct": 4.0,
            "api": 3.0,
            "partner": 2.0,
            "marketplace": 3.0,
            "app_store": 2.0,
            "open_source": 1.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_recurring_revenue(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.recurring_revenue_signal:
            return score, 0.0
        key = features.recurring_revenue_signal.lower()
        mapping = {
            "recurring": 8.0,
            "mixed": 3.0,
            "transactional": -3.0,
            "unknown": 0.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_model_maturity(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.business_model_maturity:
            return score, 0.0
        key = features.business_model_maturity.lower()
        mapping = {
            "mature": 8.0,
            "growth": 5.0,
            "early": 2.0,
            "nascent": -2.0,
            "evolving": 3.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_unit_economics(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.unit_economics_indicators:
            return score, 0.0
        count = len(features.unit_economics_indicators)
        delta = min(8.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_network_effects(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.network_effects_signals:
            return score, 0.0
        count = len(features.network_effects_signals)
        delta = min(8.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_platform(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.platform_characteristics:
            return score, 0.0
        count = len(features.platform_characteristics)
        delta = min(6.0, count * 1.5)
        return score + delta, delta

    @staticmethod
    def _score_switching(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.switching_cost_indicators:
            return score, 0.0
        count = len(features.switching_cost_indicators)
        delta = min(6.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_model_confidence(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if features.business_model_confidence <= 0.0:
            return score, 0.0
        delta = (features.business_model_confidence - 0.5) * 10.0
        return score + delta, delta

    @staticmethod
    def _score_business_model_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.business_model_risk:
            return score, 0.0
        count = len(features.business_model_risk)
        delta = min(-3.0, count * -2.0)
        return score + delta, delta

    @staticmethod
    def _score_observations(
        score: float, observations: list[Observation]
    ) -> tuple[float, float]:
        if not observations:
            return score, 0.0
        total_impact = sum(
            o.importance for o in observations if hasattr(o, "importance")
        )
        avg_importance = total_impact / len(observations)
        delta = avg_importance * 8.0
        return score + delta, delta

    @staticmethod
    def _apply_data_dampener(
        score: float, features: ExtractedFeatures
    ) -> float:
        dampening = 1.0 - features.data_completeness
        deviation = score - 50.0
        dampened = 50.0 + deviation * (1.0 - dampening * 0.3)
        return max(0.0, min(100.0, dampened))


class TractionScorer:
    """Scores the TRACTION_SIGNALS dimension based on traction intelligence features.

    Evaluates revenue signals, funding signals, investor quality, customer
    metrics, user engagement, growth, retention, expansion, milestones,
    partnerships, awards, traction risk, AND structured quantitative metrics
    (ARR, growth rate, NRR, churn, customer count, runway, burn rate).

    Structured metrics augment signal-list scoring. When both a structured
    metric and a signal list are available, the structured metric provides
    additional deterministic score adjustments. All scoring is deterministic
    and traceable.
    """

    # --- Structured metric scoring thresholds ---
    _ARR_PRE_SCALE: float = 1_000_000.0
    _ARR_MEANINGFUL: float = 10_000_000.0
    _ARR_STRONG: float = 50_000_000.0
    _GROWTH_SLOW: float = 20.0
    _GROWTH_MODERATE: float = 50.0
    _GROWTH_STRONG: float = 100.0
    _NRR_CONTRACTION: float = 90.0
    _NRR_STRONG: float = 110.0
    _NRR_EXCEPTIONAL: float = 130.0
    _CHURN_EXCELLENT: float = 2.0
    _CHURN_ACCEPTABLE: float = 5.0
    _CHURN_CONCERNING: float = 10.0
    _RUNWAY_CRITICAL: int = 6
    _RUNWAY_ELEVATED: int = 12
    _RUNWAY_COMFORTABLE: int = 24
    _BURN_HIGH: float = 1_000_000.0
    _BURN_VERY_HIGH: float = 5_000_000.0
    _CUSTOMERS_EARLY: int = 50
    _CUSTOMERS_GROWING: int = 200
    _CUSTOMERS_SCALING: int = 1000

    def __init__(self) -> None:
        self._dimension = AnalysisDimension.TRACTION_SIGNALS

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        relevant_obs = [
            o for o in observations if o.dimension == self._dimension.value
        ]

        base_score = 50.0
        adjustments: list[tuple[str, float]] = []

        base_score, adj = self._score_has_revenue(base_score, features)
        if adj != 0:
            adjustments.append(("has_revenue", adj))

        base_score, adj = self._score_funding_signals(base_score, features)
        if adj != 0:
            adjustments.append(("funding_amount", adj))

        base_score, adj = self._score_investor_signals(base_score, features)
        if adj != 0:
            adjustments.append(("investor_signals", adj))

        base_score, adj = self._score_grants(base_score, features)
        if adj != 0:
            adjustments.append(("grants_accelerator", adj))

        base_score, adj = self._score_revenue_signals(base_score, features)
        if adj != 0:
            adjustments.append(("revenue_amount", adj))

        base_score, adj = self._score_arr_mrr(base_score, features)
        if adj != 0:
            adjustments.append(("arr_mrr", adj))

        base_score, adj = self._score_customers(base_score, features)
        if adj != 0:
            adjustments.append(("customer_count", adj))

        base_score, adj = self._score_active_users(base_score, features)
        if adj != 0:
            adjustments.append(("active_users", adj))

        base_score, adj = self._score_enterprise_customers(base_score, features)
        if adj != 0:
            adjustments.append(("enterprise_customers", adj))

        base_score, adj = self._score_paying_customers(base_score, features)
        if adj != 0:
            adjustments.append(("paying_customers", adj))

        base_score, adj = self._score_retention(base_score, features)
        if adj != 0:
            adjustments.append(("retention", adj))

        base_score, adj = self._score_growth(base_score, features)
        if adj != 0:
            adjustments.append(("growth", adj))

        base_score, adj = self._score_expansion(base_score, features)
        if adj != 0:
            adjustments.append(("expansion", adj))

        base_score, adj = self._score_milestones(base_score, features)
        if adj != 0:
            adjustments.append(("milestones", adj))

        base_score, adj = self._score_partnerships(base_score, features)
        if adj != 0:
            adjustments.append(("partnerships", adj))

        base_score, adj = self._score_engagement(base_score, features)
        if adj != 0:
            adjustments.append(("engagement", adj))

        base_score, adj = self._score_traction_risk(base_score, features)
        if adj != 0:
            adjustments.append(("traction_risk", adj))

        base_score, adj = self._score_structured_arr(base_score, features)
        if adj != 0:
            adjustments.append(("structured_arr", adj))

        base_score, adj = self._score_structured_growth(base_score, features)
        if adj != 0:
            adjustments.append(("structured_growth", adj))

        base_score, adj = self._score_structured_nrr(base_score, features)
        if adj != 0:
            adjustments.append(("structured_nrr", adj))

        base_score, adj = self._score_structured_churn(base_score, features)
        if adj != 0:
            adjustments.append(("structured_churn", adj))

        base_score, adj = self._score_structured_customer_count(base_score, features)
        if adj != 0:
            adjustments.append(("structured_customer_count", adj))

        base_score, adj = self._score_structured_runway(base_score, features)
        if adj != 0:
            adjustments.append(("structured_runway", adj))

        base_score, adj = self._score_structured_burn_rate(base_score, features)
        if adj != 0:
            adjustments.append(("structured_burn_rate", adj))

        base_score, adj = self._score_observations(base_score, relevant_obs)
        if adj != 0:
            adjustments.append(("observations", adj))

        final_score = self._apply_data_dampener(base_score, features)

        rationale = (
            f"Traction signals scored from {len(adjustments)} signal dimensions. "
            f"Base 50, final {final_score:.1f}."
        )

        return ScoreResult(
            dimension=self._dimension.value,
            score=final_score,
            rationale=rationale,
            evidence=[o.statement for o in relevant_obs],
        )

    # ------------------------------------------------------------------
    # Qualitative signal scoring (existing)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_has_revenue(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if features.has_revenue is True:
            delta = 8.0
        elif features.has_revenue is False:
            delta = -5.0
        else:
            delta = -3.0
        return score + delta, delta

    @staticmethod
    def _score_funding_signals(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.funding_amount_signals:
            return score, 0.0
        count = len(features.funding_amount_signals)
        delta = min(12.0, count * 4.0)
        return score + delta, delta

    @staticmethod
    def _score_investor_signals(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.investor_signals:
            return score, 0.0
        count = len(features.investor_signals)
        delta = min(9.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_grants(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.grants_accelerator_signals:
            return score, 0.0
        count = len(features.grants_accelerator_signals)
        delta = min(6.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_revenue_signals(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.revenue_amount_signals:
            return score, 0.0
        count = len(features.revenue_amount_signals)
        delta = min(9.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_arr_mrr(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.arr_mrr_signals:
            return score, 0.0
        count = len(features.arr_mrr_signals)
        delta = min(10.0, count * 4.0)
        return score + delta, delta

    @staticmethod
    def _score_customers(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.customer_count_signals:
            return score, 0.0
        count = len(features.customer_count_signals)
        delta = min(9.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_active_users(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.active_user_signals:
            return score, 0.0
        count = len(features.active_user_signals)
        delta = min(6.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_enterprise_customers(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.enterprise_customer_signals:
            return score, 0.0
        count = len(features.enterprise_customer_signals)
        delta = min(9.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_paying_customers(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.paying_customer_signals:
            return score, 0.0
        count = len(features.paying_customer_signals)
        delta = min(6.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_retention(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.retention_signals:
            return score, 0.0
        count = len(features.retention_signals)
        delta = min(9.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_growth(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.growth_signals:
            return score, 0.0
        count = len(features.growth_signals)
        delta = min(9.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_expansion(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.expansion_signals:
            return score, 0.0
        count = len(features.expansion_signals)
        delta = min(6.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_milestones(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.milestone_signals:
            return score, 0.0
        count = len(features.milestone_signals)
        delta = min(6.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_partnerships(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.partnership_signals:
            return score, 0.0
        count = len(features.partnership_signals)
        delta = min(5.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_engagement(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.engagement_signals:
            return score, 0.0
        count = len(features.engagement_signals)
        delta = min(5.0, count * 1.5)
        return score + delta, delta

    @staticmethod
    def _score_traction_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.traction_risk:
            return score, 0.0
        count = len(features.traction_risk)
        delta = min(-5.0, count * -3.0)
        return score + delta, delta

    @staticmethod
    def _score_observations(
        score: float, observations: list[Observation]
    ) -> tuple[float, float]:
        if not observations:
            return score, 0.0
        total_impact = sum(
            o.importance for o in observations if hasattr(o, "importance")
        )
        avg_importance = total_impact / len(observations)
        delta = avg_importance * 10.0
        return score + delta, delta

    @staticmethod
    def _apply_data_dampener(
        score: float, features: ExtractedFeatures
    ) -> float:
        dampening = 1.0 - features.data_completeness
        deviation = score - 50.0
        dampened = 50.0 + deviation * (1.0 - dampening * 0.3)
        return max(0.0, min(100.0, dampened))

    # ------------------------------------------------------------------
    # Structured quantitative scoring (Sprint 15)
    # ------------------------------------------------------------------

    def _score_structured_arr(
        self, score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        """Score based on actual ARR value rather than signal mentions."""
        if features.arr_usd is None:
            return score, 0.0
        arr = features.arr_usd
        if arr >= self._ARR_STRONG:
            delta = 14.0
        elif arr >= self._ARR_MEANINGFUL:
            delta = 10.0
        elif arr >= self._ARR_PRE_SCALE:
            delta = 6.0
        else:
            delta = 2.0
        return score + delta, delta

    def _score_structured_growth(
        self, score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        """Score based on actual growth rate percentage."""
        if features.growth_rate_pct is None:
            return score, 0.0
        growth = features.growth_rate_pct
        if growth >= self._GROWTH_STRONG:
            delta = 12.0
        elif growth >= self._GROWTH_MODERATE:
            delta = 8.0
        elif growth >= self._GROWTH_SLOW:
            delta = 4.0
        else:
            delta = 0.0
        return score + delta, delta

    def _score_structured_nrr(
        self, score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        """Score based on net revenue retention percentage."""
        if features.nrr_pct is None:
            return score, 0.0
        nrr = features.nrr_pct
        if nrr >= self._NRR_EXCEPTIONAL:
            delta = 12.0
        elif nrr >= self._NRR_STRONG:
            delta = 9.0
        elif nrr >= 100.0:
            delta = 5.0
        elif nrr >= self._NRR_CONTRACTION:
            delta = -2.0
        else:
            delta = -6.0
        return score + delta, delta

    def _score_structured_churn(
        self, score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        """Score based on churn rate percentage (lower is better)."""
        if features.churn_rate_pct is None:
            return score, 0.0
        churn = features.churn_rate_pct
        if churn <= self._CHURN_EXCELLENT:
            delta = 10.0
        elif churn <= self._CHURN_ACCEPTABLE:
            delta = 5.0
        elif churn <= self._CHURN_CONCERNING:
            delta = -3.0
        else:
            delta = -8.0
        return score + delta, delta

    def _score_structured_customer_count(
        self, score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        """Score based on actual customer count."""
        if features.customer_count is None:
            return score, 0.0
        count = features.customer_count
        if count >= self._CUSTOMERS_SCALING:
            delta = 10.0
        elif count >= self._CUSTOMERS_GROWING:
            delta = 7.0
        elif count >= self._CUSTOMERS_EARLY:
            delta = 4.0
        elif count >= 10:
            delta = 2.0
        else:
            delta = 0.0
        return score + delta, delta

    def _score_structured_runway(
        self, score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        """Score based on runway in months (longer is better)."""
        if features.runway_months is None:
            return score, 0.0
        runway = features.runway_months
        if runway >= self._RUNWAY_COMFORTABLE:
            delta = 6.0
        elif runway >= self._RUNWAY_ELEVATED:
            delta = 2.0
        elif runway >= self._RUNWAY_CRITICAL:
            delta = -4.0
        else:
            delta = -8.0
        return score + delta, delta

    def _score_structured_burn_rate(
        self, score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        """Score based on burn rate (lower is generally better)."""
        if features.burn_rate_usd is None:
            return score, 0.0
        burn = features.burn_rate_usd
        if burn >= self._BURN_VERY_HIGH:
            delta = -6.0
        elif burn >= self._BURN_HIGH:
            delta = -2.0
        else:
            delta = 3.0
        return score + delta, delta


class TeamExecutionScorer:
    """Scores the TEAM_EXECUTION dimension based on team and execution features.

    Evaluates team size, hiring signals, execution signals, engineering
    strength, engineering maturity, infrastructure maturity, open source
    involvement, developer tooling, recency, data completeness, and
    operational/hiring/funding/scaling/security/compliance risk. This
    dimension also serves as the proxy for data quality assessment.
    All scoring is deterministic.
    """

    def __init__(self) -> None:
        self._dimension = AnalysisDimension.TEAM_EXECUTION

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        relevant_obs = [
            o for o in observations if o.dimension == self._dimension.value
        ]

        base_score = 50.0
        adjustments: list[tuple[str, float]] = []

        base_score, adj = self._score_team_size(base_score, features)
        if adj != 0:
            adjustments.append(("team_size", adj))

        base_score, adj = self._score_hiring_signals(base_score, features)
        if adj != 0:
            adjustments.append(("hiring_signals", adj))

        base_score, adj = self._score_hiring_growth(base_score, features)
        if adj != 0:
            adjustments.append(("hiring_growth", adj))

        base_score, adj = self._score_execution_signals(base_score, features)
        if adj != 0:
            adjustments.append(("execution_signals", adj))

        base_score, adj = self._score_engineering_strength(base_score, features)
        if adj != 0:
            adjustments.append(("engineering_strength", adj))

        base_score, adj = self._score_engineering_maturity(base_score, features)
        if adj != 0:
            adjustments.append(("engineering_maturity", adj))

        base_score, adj = self._score_infrastructure_maturity(
            base_score, features
        )
        if adj != 0:
            adjustments.append(("infrastructure_maturity", adj))

        base_score, adj = self._score_open_source(base_score, features)
        if adj != 0:
            adjustments.append(("open_source", adj))

        base_score, adj = self._score_dev_tooling(base_score, features)
        if adj != 0:
            adjustments.append(("developer_tooling", adj))

        base_score, adj = self._score_recency(base_score, features)
        if adj != 0:
            adjustments.append(("founding_recency", adj))

        base_score, adj = self._score_data_completeness(base_score, features)
        if adj != 0:
            adjustments.append(("data_completeness", adj))

        base_score, adj = self._score_operational_risk(base_score, features)
        if adj != 0:
            adjustments.append(("operational_risk", adj))

        base_score, adj = self._score_hiring_risk(base_score, features)
        if adj != 0:
            adjustments.append(("hiring_risk", adj))

        base_score, adj = self._score_funding_risk(base_score, features)
        if adj != 0:
            adjustments.append(("funding_risk", adj))

        base_score, adj = self._score_scaling_risk(base_score, features)
        if adj != 0:
            adjustments.append(("scaling_risk", adj))

        base_score, adj = self._score_security_risk(base_score, features)
        if adj != 0:
            adjustments.append(("security_risk", adj))

        base_score, adj = self._score_compliance_risk(base_score, features)
        if adj != 0:
            adjustments.append(("compliance_risk", adj))

        base_score, adj = self._score_observations(base_score, relevant_obs)
        if adj != 0:
            adjustments.append(("observations", adj))

        final_score = max(0.0, min(100.0, base_score))

        rationale = (
            f"Team and execution scored from {len(adjustments)} signal dimensions. "
            f"Base 50, final {final_score:.1f}."
        )

        return ScoreResult(
            dimension=self._dimension.value,
            score=final_score,
            rationale=rationale,
            evidence=[o.statement for o in relevant_obs],
        )

    def _score_team_size(
        self, score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        """Score based on numeric team size, falling back to text indicator."""
        if features.team_size_numeric is not None:
            size = features.team_size_numeric
            if size >= 1000:
                delta = 7.0
            elif size >= 201:
                delta = 6.0
            elif size >= 51:
                delta = 5.0
            elif size >= 11:
                delta = 3.0
            else:
                delta = 0.0
            return score + delta, delta

        if not features.team_size_indicator:
            return score, 0.0
        key = features.team_size_indicator.lower().replace(" ", "")
        mapping = {
            "1-10": 0.0,
            "11-50": 3.0,
            "51-200": 5.0,
            "201-1000": 6.0,
            "1000+": 7.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_hiring_signals(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.hiring_signals:
            return score, 0.0
        count = len(features.hiring_signals)
        delta = min(6.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_hiring_growth(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.hiring_growth_signals:
            return score, 0.0
        count = len(features.hiring_growth_signals)
        delta = min(9.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_execution_signals(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.execution_signals:
            return score, 0.0
        count = len(features.execution_signals)
        delta = min(10.0, count * 2.0)
        return score + delta, delta

    @staticmethod
    def _score_engineering_strength(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.engineering_strength:
            return score, 0.0
        key = features.engineering_strength.lower()
        mapping = {
            "strong": 8.0,
            "moderate": 3.0,
            "weak": -5.0,
            "unknown": 0.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_engineering_maturity(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.engineering_maturity:
            return score, 0.0
        key = features.engineering_maturity.lower()
        mapping = {
            "sophisticated": 8.0,
            "established": 5.0,
            "developing": 2.0,
            "nascent": -3.0,
            "unknown": 0.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_infrastructure_maturity(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.infrastructure_maturity:
            return score, 0.0
        key = features.infrastructure_maturity.lower()
        mapping = {
            "enterprise_grade": 6.0,
            "mature": 4.0,
            "developing": 2.0,
            "early": -2.0,
            "unknown": 0.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_open_source(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.open_source_signals:
            return score, 0.0
        count = len(features.open_source_signals)
        delta = min(5.0, count * 1.5)
        return score + delta, delta

    @staticmethod
    def _score_dev_tooling(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.developer_tooling_signals:
            return score, 0.0
        count = len(features.developer_tooling_signals)
        delta = min(5.0, count * 1.5)
        return score + delta, delta

    @staticmethod
    def _score_recency(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.founded_year:
            return score, 0.0
        current_year = 2026
        age = current_year - features.founded_year
        if age < 3:
            delta = 3.0
        elif age <= 7:
            delta = 1.0
        else:
            delta = -2.0
        return score + delta, delta

    @staticmethod
    def _score_data_completeness(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        dc = features.data_completeness
        delta = (dc - 0.5) * 16.0
        return score + delta, delta

    @staticmethod
    def _score_operational_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.operational_risk:
            return score, 0.0
        count = len(features.operational_risk)
        delta = min(-3.0, count * -2.0)
        return score + delta, delta

    @staticmethod
    def _score_hiring_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.hiring_risk:
            return score, 0.0
        count = len(features.hiring_risk)
        delta = min(-3.0, count * -1.5)
        return score + delta, delta

    @staticmethod
    def _score_funding_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.funding_risk:
            return score, 0.0
        count = len(features.funding_risk)
        delta = min(-5.0, count * -3.0)
        return score + delta, delta

    @staticmethod
    def _score_scaling_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.scaling_risk:
            return score, 0.0
        count = len(features.scaling_risk)
        delta = min(-3.0, count * -2.0)
        return score + delta, delta

    @staticmethod
    def _score_security_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.security_risk:
            return score, 0.0
        count = len(features.security_risk)
        delta = min(-3.0, count * -2.0)
        return score + delta, delta

    @staticmethod
    def _score_compliance_risk(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.compliance_risk:
            return score, 0.0
        count = len(features.compliance_risk)
        delta = min(-3.0, count * -2.0)
        return score + delta, delta

    @staticmethod
    def _score_observations(
        score: float, observations: list[Observation]
    ) -> tuple[float, float]:
        if not observations:
            return score, 0.0
        total_impact = sum(
            o.importance for o in observations if hasattr(o, "importance")
        )
        avg_importance = total_impact / len(observations)
        delta = avg_importance * 8.0
        return score + delta, delta


def _build_default_scorers() -> list[DimensionScorer]:
    """Create scorers for each default dimension."""
    scorers: list[DimensionScorer] = []
    for dim in AnalysisDimension:
        if dim == AnalysisDimension.MARKET_OPPORTUNITY:
            scorers.append(MarketOpportunityScorer())
        elif dim == AnalysisDimension.FOUNDER_QUALITY:
            scorers.append(FounderQualityScorer())
        elif dim == AnalysisDimension.PRODUCT_STRENGTH:
            scorers.append(ProductStrengthScorer())
        elif dim == AnalysisDimension.BUSINESS_MODEL_VIABILITY:
            scorers.append(BusinessModelScorer())
        elif dim == AnalysisDimension.TRACTION_SIGNALS:
            scorers.append(TractionScorer())
        elif dim == AnalysisDimension.COMPETITIVE_POSITION:
            scorers.append(CompetitionScorer())
        elif dim == AnalysisDimension.TEAM_EXECUTION:
            scorers.append(TeamExecutionScorer())
    return scorers


class DefaultScoringEngine:
    """Standard implementation of the ScoringEngine protocol.

    Delegates scoring to a set of injectable DimensionScorer components.
    When no scorers are provided, uses placeholder scorers for each
    default analysis dimension.
    """

    def __init__(self, scorers: list[DimensionScorer] | None = None) -> None:
        self._scorers = scorers if scorers is not None else _build_default_scorers()

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> list[ScoreResult]:
        """Score all dimensions and return results."""
        logger.info("Running scoring engine with %d scorers", len(self._scorers))

        obs_by_dim: dict[str, list[Observation]] = {}
        for obs in observations:
            obs_by_dim.setdefault(obs.dimension, []).append(obs)

        results: list[ScoreResult] = []
        for scorer in self._scorers:
            try:
                dim = getattr(scorer, "_dimension", None)
                if dim is not None:
                    dim_observations = obs_by_dim.get(dim.value, [])
                    results.append(scorer.score(features, dim_observations))
                else:
                    results.append(scorer.score(features, observations))
            except Exception:
                logger.warning(
                    "Scorer %s failed, skipping", type(scorer).__name__
                )

        logger.info("Produced %d dimension scores", len(results))
        return results
