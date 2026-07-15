"""Competition assessment reasoning rule.

Produces observations about competitive landscape quality, moat strength,
and market entry difficulty based on extracted competition features
and competition evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.reasoning.rules.base import evidence_ref, feature_ref

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation

DIMENSION = "competitive_position"

MOAT_SIGNALS = {
    "proprietary_data": ("high", 0.7),
    "patent_portfolio": ("high", 0.65),
    "strong_network_effects": ("high", 0.7),
    "high_switching_costs": ("medium", 0.6),
    "brand_recognition": ("medium", 0.55),
    "economies_of_scale": ("medium", 0.5),
    "regulatory_advantage": ("high", 0.6),
    "exclusive_partnerships": ("medium", 0.5),
    "proprietary_technology": ("high", 0.65),
    "data_network_effects": ("high", 0.7),
    "ecosystem_lock_in": ("medium", 0.6),
    "talent_concentration": ("low", 0.4),
}


class CompetitionAssessmentRule:
    """Produces observations about the competitive landscape.

    Evaluates competitive features and evidence to produce structured
    observations about moat strength, market concentration, switching
    costs, and competitive entry difficulty.
    """

    @property
    def name(self) -> str:
        return "competition_assessment"

    def evaluate(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        observations: list[Obs] = []
        observations.extend(self._observe_moat_strength(features, evidence))
        observations.extend(self._observe_concentration(features))
        observations.extend(self._observe_switching_costs(features, evidence))
        observations.extend(self._observe_entry_difficulty(features, evidence))
        observations.extend(self._observe_open_source_risk(features))
        return observations

    def _observe_moat_strength(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if not features.competitive_moat_indicators:
            return []

        scored: list[tuple[str, float]] = []
        for signal in features.competitive_moat_indicators:
            key = signal.lower().strip()
            if key in MOAT_SIGNALS:
                _, confidence = MOAT_SIGNALS[key]
                scored.append((signal, confidence))

        if not scored:
            return [
                Obs(
                    dimension=DIMENSION,
                    category="moat_analysis",
                    statement=(
                        "Competitive moat indicators detected but none matched "
                        "known strong moat patterns."
                    ),
                    evidence=[
                        feature_ref(
                            "competitive_moat_indicators",
                            features.competitive_moat_indicators,
                        )
                    ],
                    confidence=0.4,
                    importance=0.5,
                    source_rule="CompetitionAssessmentRule",
                )
            ]

        avg_conf = sum(c for _, c in scored) / len(scored)
        high_count = sum(1 for _, c in scored if c >= 0.65)
        signal_names = [s for s, _ in scored]

        if high_count >= 2:
            impact = 0.8
            statement = (
                f"Strong competitive moat detected with {high_count} high-quality "
                f"indicators ({', '.join(signal_names[:3])}). "
                f"This suggests significant defensibility."
            )
        else:
            impact = 0.4
            statement = (
                f"Moderate competitive moat with {len(scored)} indicators "
                f"({', '.join(signal_names[:3])}). "
                f"Moat strength is present but not dominant."
            )

        evidence_refs = [
            evidence_ref(e) for e in evidence
            if e.domain == "competition" and e.category == "moat"
        ]

        return [
            Obs(
                dimension=DIMENSION,
                category="moat_analysis",
                statement=statement,
                evidence=[feature_ref("competitive_moat_indicators", signal_names)] + evidence_refs,
                confidence=min(avg_conf, 1.0),
                importance=impact,
                source_rule="CompetitionAssessmentRule",
            )
        ]

    def _observe_concentration(
        self,
        features: ExtractedFeatures,
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if not features.market_concentration:
            return []

        concentration = features.market_concentration.lower()

        impact_map = {
            "fragmented": (
                0.5,
                "Fragmented market suggests opportunity "
                "for differentiation.",
            ),
            "moderately_concentrated": (
                0.3,
                "Moderately concentrated market with viable "
                "positioning opportunities.",
            ),
            "concentrated": (
                -0.4,
                "Concentrated market poses significant "
                "competitive headwinds.",
            ),
            "dominated": (
                -0.7,
                "Market dominated by incumbents; entry "
                "requires significant differentiation.",
            ),
        }

        if concentration not in impact_map:
            return []

        impact, desc = impact_map[concentration]
        density = f" ({features.competitive_density})" if features.competitive_density else ""

        return [
            Obs(
                dimension=DIMENSION,
                category="market_concentration",
                statement=f"{desc}{density}",
                evidence=[
                    feature_ref("market_concentration", concentration),
                ],
                confidence=0.75,
                importance=abs(impact),
                source_rule="CompetitionAssessmentRule",
            )
        ]

    def _observe_switching_costs(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if not features.switching_cost_signals:
            return []

        count = len(features.switching_cost_signals)
        if count >= 3:
            impact = 0.7
            statement = (
                f"Strong switching cost signals ({count} detected) create "
                f"meaningful barriers for competitors and protect market position."
            )
        elif count >= 1:
            impact = 0.4
            statement = (
                f"Moderate switching cost signals ({count} detected) provide "
                f"some competitive protection."
            )
        else:
            return []

        evidence_refs = [
            evidence_ref(e) for e in evidence
            if e.domain == "competition" and e.category == "switching_costs"
        ]

        return [
            Obs(
                dimension=DIMENSION,
                category="switching_costs",
                statement=statement,
                evidence=[
                    feature_ref("switching_cost_signals", features.switching_cost_signals),
                ] + evidence_refs,
                confidence=0.7,
                importance=impact,
                source_rule="CompetitionAssessmentRule",
            )
        ]

    def _observe_entry_difficulty(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if not features.barriers_to_entry:
            return []

        count = len(features.barriers_to_entry)
        if count >= 3:
            impact = 0.6
            statement = (
                f"Multiple barriers to entry detected ({count}). "
                f"The market presents significant challenges for new entrants."
            )
        else:
            impact = 0.3
            statement = (
                f"Some barriers to entry detected ({count}). "
                f"Entry is possible but requires strategic positioning."
            )

        evidence_refs = [
            evidence_ref(e) for e in evidence
            if e.domain == "competition" and e.category == "barriers_to_entry"
        ]

        return [
            Obs(
                dimension=DIMENSION,
                category="entry_difficulty",
                statement=statement,
                evidence=[
                    feature_ref("barriers_to_entry", features.barriers_to_entry),
                ] + evidence_refs,
                confidence=0.65,
                importance=impact,
                source_rule="CompetitionAssessmentRule",
            )
        ]

    def _observe_open_source_risk(
        self,
        features: ExtractedFeatures,
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        if not features.open_source_competition:
            return []

        count = len(features.open_source_competition)

        return [
            Obs(
                dimension=DIMENSION,
                category="open_source_risk",
                statement=(
                    f"Open-source competition detected ({count} signals). "
                    f"Open-source alternatives may compress margins and "
                    f"reduce defensibility in the product category."
                ),
                evidence=[
                    feature_ref("open_source_competition", features.open_source_competition),
                ],
                confidence=0.6,
                importance=0.5,
                source_rule="CompetitionAssessmentRule",
            )
        ]
