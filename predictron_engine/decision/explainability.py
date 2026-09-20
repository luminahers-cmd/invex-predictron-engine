"""Explainability Layer — human-readable explanations of investment decisions.

Produces transparent explanations such as:

    Investment recommendation driven by:
      Strong funding velocity
      Consistent hiring growth
      High ecosystem connectivity

    − Weak founder experience
    − Low market momentum

Every explanation references actual evidence and traces back to
specific features and rules.  No fabricated reasoning — every statement
is derived deterministically from Feature Store snapshots and
contribution computations.

Key principles:
  - Fully deterministic: same features always produce the same explanation
  - Every claim traces to real evidence
  - No ML, no LLMs, no probabilistic black boxes
  - Structured prose with evidence references
"""

from __future__ import annotations

from typing import Any

from predictron_engine.decision.contribution import ContributionEngine
from predictron_engine.decision.intelligence_models import (
    Contribution,
    DecisionVerdict,
    Explanation,
)
from predictron_engine.feature_store.models import CompanyFeatureSet

# ---------------------------------------------------------------------------
# Explanation constants — named thresholds for prose generation.
# ---------------------------------------------------------------------------

_STRONG_CONTRIBUTION = 0.05
_WEAK_CONTRIBUTION = 0.01
_DEFAULT_TOP_N = 3
_MAX_EXPLANATION_FACTORS = 5

_STRENGTH_PREFIX = "Investment recommendation driven by:"
_WEAKNESS_PREFIX = "− Constrained by:"
_NEUTRAL_PREFIX = "Informational factors:"

_VERDICT_HEADLINES: dict[DecisionVerdict, str] = {
    DecisionVerdict.STRONG_INVEST: "Strong investment opportunity",
    DecisionVerdict.INVEST: "Investment opportunity",
    DecisionVerdict.WATCH: "Watchlist candidate",
    DecisionVerdict.INVESTIGATE_FURTHER: "Requires further investigation",
    DecisionVerdict.PASS: "Pass — not currently attractive",
}


class ExplainabilityEngine:
    """Generates deterministic, evidence-referencing explanations.

    Responsibilities:
      - Build structured explanations from contributions
      - Ensure every claim references actual evidence
      - Provide both summary and full prose forms
      - Never fabricate reasoning
    """

    def __init__(
        self,
        contribution_engine: ContributionEngine | None = None,
    ) -> None:
        self._contribution_engine = contribution_engine or ContributionEngine()

    @property
    def contribution_engine(self) -> ContributionEngine:
        return self._contribution_engine

    def build_explanation(
        self,
        feature_set: CompanyFeatureSet,
        contributions: list[Contribution] | None = None,
        *,
        verdict: DecisionVerdict | None = None,
        recommendation_text: str | None = None,
    ) -> Explanation:
        """Build a complete explanation for a company's decision.

        Args:
            feature_set: Feature Store snapshot.
            contributions: Optional pre-computed contributions (computed
                deterministically if omitted).
            verdict: Optional verdict (defaults to strong_invest headline
                only when headline is needed).
            recommendation_text: Optional custom recommendation text.

        Returns:
            A fully populated Explanation with evidence references.
        """
        if contributions is None:
            contributions = self._contribution_engine.compute_contributions(feature_set)

        company_id = feature_set.company_id
        classified = self._contribution_engine.classify(contributions)

        positives = classified["positive"]
        negatives = classified["negative"]
        neutrals = classified["neutral"]
        confidences = classified["confidence"]

        resolved_verdict = verdict if verdict is not None else DecisionVerdict.WATCH

        strengths = self._build_factor_statements(positives, top_n=_DEFAULT_TOP_N)
        weaknesses = self._build_factor_statements(negatives, top_n=_DEFAULT_TOP_N)
        neutral_factors = self._build_neutral_statements(neutrals, top_n=2)
        confidence_factors = self._build_confidence_statements(confidences, top_n=2)

        evidence_refs = self._collect_evidence_references(contributions)
        evidence_summary = self._build_evidence_summary(contributions, evidence_refs)

        headline = _VERDICT_HEADLINES[resolved_verdict]
        full_explanation = self._build_full_explanation(
            strengths,
            weaknesses,
            neutral_factors,
            confidence_factors,
            evidence_summary,
        )

        recommendation = recommendation_text or self._build_recommendation(
            resolved_verdict,
            positives,
            negatives,
        )

        return Explanation(
            company_id=company_id,
            headline=headline,
            strengths=strengths,
            weaknesses=weaknesses,
            neutral_factors=neutral_factors,
            confidence_factors=confidence_factors,
            evidence_summary=evidence_summary,
            full_explanation=full_explanation,
            recommendation=recommendation,
            supporting_contributions=contributions,
            evidence_references=evidence_refs,
            metadata={
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "neutral_count": len(neutrals),
                "confidence_count": len(confidences),
                "verdict": resolved_verdict.value,
            },
        )

    def format_strengths(
        self,
        strengths: list[str],
    ) -> str:
        """Format strength statements into prose (with prefix)."""
        if not strengths:
            return ""
        lines = [_STRENGTH_PREFIX]
        lines.extend(f"  {s}" for s in strengths)
        return "\n".join(lines)

    def format_weaknesses(
        self,
        weaknesses: list[str],
    ) -> str:
        """Format weakness statements into prose (with prefix)."""
        if not weaknesses:
            return ""
        lines = [_WEAKNESS_PREFIX]
        lines.extend(f"  − {s}" for s in weaknesses)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_factor_statements(
        self,
        contributions: list[Contribution],
        top_n: int,
    ) -> list[str]:
        """Build prose statements for directional contributions.

        Only contributions above the strong threshold produce statements.
        Statements reference the feature name and its normalized value.
        """
        statements: list[str] = []
        for c in contributions[:top_n]:
            if abs(c.computed_contribution) < _STRONG_CONTRIBUTION:
                continue
            statements.append(
                f"{c.feature_name} "
                f"({self._direction_word(c.computed_contribution)}, "
                f"weight {c.weight:.2f})"
            )
        return statements

    def _build_neutral_statements(
        self,
        contributions: list[Contribution],
        top_n: int,
    ) -> list[str]:
        """Build statements for neutral contributions."""
        statements: list[str] = []
        for c in contributions[:top_n]:
            statements.append(
                f"{c.feature_name} is neutral "
                f"(normalized {c.normalized_value:+.2f})"
            )
        return statements

    def _build_confidence_statements(
        self,
        contributions: list[Contribution],
        top_n: int,
    ) -> list[str]:
        """Build statements for confidence contributors.

        Confidence contributions reflect data availability and
        reliability rather than directional investment appeal.
        """
        statements: list[str] = []
        for c in contributions[:top_n]:
            statements.append(
                f"Confidence signal: {c.feature_name} "
                f"(raw {c.raw_value!r})"
            )
        return statements

    def _collect_evidence_references(
        self,
        contributions: list[Contribution],
    ) -> list[dict[str, Any]]:
        """Collect unique evidence references across all contributions."""
        seen: set[tuple[str, str, str]] = set()
        refs: list[dict[str, Any]] = []
        for c in contributions:
            for ref in c.evidence_references:
                key = (
                    ref.get("source_type", ""),
                    ref.get("source_id", ""),
                    ref.get("source_field", ""),
                )
                if key not in seen:
                    seen.add(key)
                    refs.append({
                        "source_type": ref.get("source_type", ""),
                        "source_id": ref.get("source_id", ""),
                        "source_field": ref.get("source_field", ""),
                        "confidence": ref.get("confidence", 1.0),
                        "feature_id": c.feature_id,
                    })
        return refs

    def _build_evidence_summary(
        self,
        contributions: list[Contribution],
        evidence_refs: list[dict[str, Any]],
    ) -> str:
        """Build a summary of evidence quality and provenance."""
        if not contributions:
            return "No contributions computed; no evidence available."

        with_evidence = sum(1 for c in contributions if c.evidence_references)
        source_types = sorted({r["source_type"] for r in evidence_refs})

        if not evidence_refs:
            return (
                f"{len(contributions)} feature contributions computed "
                f"from pipeline data; no external evidence references."
            )

        return (
            f"{with_evidence} of {len(contributions)} feature contributions "
            f"cite {len(evidence_refs)} evidence reference(s) "
            f"from source types: {', '.join(source_types)}."
        )

    def _build_full_explanation(
        self,
        strengths: list[str],
        weaknesses: list[str],
        neutral_factors: list[str],
        confidence_factors: list[str],
        evidence_summary: str,
    ) -> str:
        """Build the complete prose explanation."""
        parts: list[str] = []

        if strengths:
            parts.append(self.format_strengths(strengths))
        if weaknesses:
            parts.append(self.format_weaknesses(weaknesses))
        if neutral_factors:
            parts.append(_NEUTRAL_PREFIX)
            parts.extend(f"  {s}" for s in neutral_factors)
        if confidence_factors:
            parts.append("Confidence:")

        if evidence_summary:
            parts.append(evidence_summary)

        return "\n".join(parts)

    def _build_recommendation(
        self,
        verdict: DecisionVerdict,
        positives: list[Contribution],
        negatives: list[Contribution],
    ) -> str:
        """Build a deterministic recommendation statement."""
        positive_names = [f"'{c.feature_name}'" for c in positives[:2]]
        negative_names = [f"'{c.feature_name}'" for c in negatives[:2]]

        if verdict in (DecisionVerdict.STRONG_INVEST, DecisionVerdict.INVEST):
            base = (
                "Proceed with investment: driven by "
                f"{', '.join(positive_names) or 'existing positives'}."
            )
        elif verdict == DecisionVerdict.WATCH:
            base = "Monitor and re-evaluate as new evidence emerges."
        elif verdict == DecisionVerdict.INVESTIGATE_FURTHER:
            base = "Conduct further investigation before any investment decision."
        else:
            base = "Do not invest at this time."

        if negative_names and verdict not in (DecisionVerdict.PASS,):
            base += f" Address weaknesses in {', '.join(negative_names)}."

        return base

    @staticmethod
    def _direction_word(value: float) -> str:
        """Describe the direction of a contribution."""
        if value > 0:
            return "strong support"
        if value < 0:
            return "weakness"
        return "neutral"
