"""Cross-feature consistency utilities for the reasoning layer.

Sprint 6A provides reusable, deterministic detectors for reasoning-time
consistency analysis:

* :func:`detect_contradictory_features` — feature pairs that cannot
  both hold at the same time.
* :func:`detect_reinforcing_features` — feature pairs that mutually
  corroborate each other.
* :func:`detect_unsupported_conclusions` — observations not backed by
  any evidence item, citation, or provenance reference.
* :func:`detect_missing_evidence` — populated feature domains with no
  corresponding evidence items.

Every function is pure: identical inputs always produce identical
outputs.  No ML, no LLMs, no I/O.

Diagnostics are aggregated into a :class:`ConsistencyReport` which is
exposed to reasoning rules through ``ReasoningContext.consistency``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import EvidenceItem, Observation


# ────────────────────────────────────────────────────────────────────
# Findings
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContradictionFinding:
    """Two feature values that cannot both hold simultaneously."""

    field_a: str
    value_a: object
    field_b: str
    value_b: object
    reason: str


@dataclass(frozen=True)
class ReinforcementFinding:
    """Two feature values that mutually corroborate each other."""

    field_a: str
    value_a: object
    field_b: str
    value_b: object
    reason: str


@dataclass(frozen=True)
class UnsupportedFinding:
    """An observation with no resolvable evidentiary support."""

    source_rule: str
    statement: str
    reason: str


# ────────────────────────────────────────────────────────────────────
# Report
# ────────────────────────────────────────────────────────────────────


@dataclass
class ConsistencyReport:
    """Aggregated cross-feature consistency diagnostics.

    Exposed to reasoning rules via ``ReasoningContext.consistency`` so
    rules can consult consistency findings without re-running detectors.
    """

    contradictions: list[ContradictionFinding] = field(default_factory=list)
    reinforcements: list[ReinforcementFinding] = field(default_factory=list)
    unsupported_observations: list[UnsupportedFinding] = field(
        default_factory=list
    )
    missing_evidence_domains: list[str] = field(default_factory=list)

    @property
    def contradiction_count(self) -> int:
        return len(self.contradictions)

    @property
    def reinforcement_count(self) -> int:
        return len(self.reinforcements)

    @property
    def unsupported_count(self) -> int:
        return len(self.unsupported_observations)

    @property
    def missing_evidence_count(self) -> int:
        return len(self.missing_evidence_domains)


# ────────────────────────────────────────────────────────────────────
# Contradiction rules (deterministic, table-driven)
# ────────────────────────────────────────────────────────────────────

_PRE_REVENUE_STAGES = {"pre_seed", "idea", "pre-seed", "preseed"}


def _get(features: ExtractedFeatures, name: str) -> object:
    return getattr(features, name, None)


def _populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list | dict | set):
        return len(value) > 0
    return True


def detect_contradictory_features(
    features: ExtractedFeatures,
) -> list[ContradictionFinding]:
    """Detect feature pairs whose values cannot both hold.

    Deterministic checks over scalar feature values.  Missing (None or
    empty) fields never produce contradictions.
    """
    findings: list[ContradictionFinding] = []

    # Revenue claimed at a pre-revenue stage.
    stage = _get(features, "funding_stage")
    has_revenue = _get(features, "has_revenue")
    if (
        isinstance(stage, str)
        and stage.lower() in _PRE_REVENUE_STAGES
        and has_revenue is True
    ):
        findings.append(
            ContradictionFinding(
                field_a="funding_stage",
                value_a=stage,
                field_b="has_revenue",
                value_b=has_revenue,
                reason=(
                    f"Revenue reported while funding stage is '{stage}', "
                    "which is typically pre-revenue"
                ),
            )
        )

    # Customer type vs enterprise orientation mismatch.
    customer_type = _get(features, "customer_type")
    orientation = _get(features, "enterprise_orientation")
    if (
        isinstance(customer_type, str)
        and isinstance(orientation, str)
        and customer_type.lower() == "b2b"
        and orientation.lower() == "consumer"
    ):
        findings.append(
            ContradictionFinding(
                field_a="customer_type",
                value_a=customer_type,
                field_b="enterprise_orientation",
                value_b=orientation,
                reason="B2B customer type conflicts with consumer orientation",
            )
        )
    if (
        isinstance(customer_type, str)
        and isinstance(orientation, str)
        and customer_type.lower() == "b2c"
        and orientation.lower() == "enterprise"
    ):
        findings.append(
            ContradictionFinding(
                field_a="customer_type",
                value_a=customer_type,
                field_b="enterprise_orientation",
                value_b=orientation,
                reason="B2C customer type conflicts with enterprise orientation",
            )
        )

    # B2C customer type vs enterprise sales motion.
    sales_motion = _get(features, "sales_motion")
    if (
        isinstance(customer_type, str)
        and isinstance(sales_motion, str)
        and customer_type.lower() == "b2c"
        and sales_motion.lower() == "enterprise_sales"
    ):
        findings.append(
            ContradictionFinding(
                field_a="customer_type",
                value_a=customer_type,
                field_b="sales_motion",
                value_b=sales_motion,
                reason=(
                    "B2C customer type conflicts with an enterprise "
                    "sales motion"
                ),
            )
        )

    # On-premise deployment vs SaaS delivery model.
    deployment = _get(features, "deployment_model")
    saas_model = _get(features, "saas_model")
    if (
        isinstance(deployment, str)
        and isinstance(saas_model, str)
        and deployment.lower() == "on_premise"
        and saas_model.lower() == "saas"
    ):
        findings.append(
            ContradictionFinding(
                field_a="deployment_model",
                value_a=deployment,
                field_b="saas_model",
                value_b=saas_model,
                reason=(
                    "On-premise deployment conflicts with SaaS delivery model"
                ),
            )
        )

    # Transactional revenue signal vs subscription-only pricing.
    recurrence = _get(features, "recurring_revenue_signal")
    pricing = _get(features, "pricing_model")
    if (
        isinstance(recurrence, str)
        and isinstance(pricing, str)
        and recurrence.lower() == "transactional"
        and pricing.lower() == "annual_contract"
    ):
        findings.append(
            ContradictionFinding(
                field_a="recurring_revenue_signal",
                value_a=recurrence,
                field_b="pricing_model",
                value_b=pricing,
                reason=(
                    "Transactional revenue signal conflicts with "
                    "annual-contract pricing"
                ),
            )
        )

    return findings


# ────────────────────────────────────────────────────────────────────
# Reinforcement rules (deterministic, table-driven)
# ────────────────────────────────────────────────────────────────────


def detect_reinforcing_features(
    features: ExtractedFeatures,
) -> list[ReinforcementFinding]:
    """Detect feature pairs that mutually reinforce each other.

    Two features reinforce each other when both are populated and each
    independently supports the same underlying conclusion.
    """
    findings: list[ReinforcementFinding] = []

    # Revenue + recurring revenue pattern.
    if _get(features, "has_revenue") is True and _get(
        features, "recurring_revenue_signal"
    ) == "recurring":
        findings.append(
            ReinforcementFinding(
                field_a="has_revenue",
                value_a=True,
                field_b="recurring_revenue_signal",
                value_b="recurring",
                reason="Revenue presence corroborated by recurring pattern",
            )
        )

    # B2B customer type + enterprise sales motion.
    if _get(features, "customer_type") == "b2b" and _get(
        features, "sales_motion"
    ) == "enterprise_sales":
        findings.append(
            ReinforcementFinding(
                field_a="customer_type",
                value_a="b2b",
                field_b="sales_motion",
                value_b="enterprise_sales",
                reason="B2B focus corroborated by enterprise sales motion",
            )
        )

    # Technology stack + classified technology domain.
    stack = _get(features, "technology_stack")
    domain = _get(features, "primary_technology_domain")
    if _populated(stack) and _populated(domain):
        findings.append(
            ReinforcementFinding(
                field_a="technology_stack",
                value_a=len(stack),  # type: ignore[arg-type]
                field_b="primary_technology_domain",
                value_b=domain,
                reason=(
                    "Identified technologies corroborated by a classified "
                    "technology domain"
                ),
            )
        )

    # Multiple founders + detected leadership roles.
    founders = _get(features, "founder_profile_count")
    roles = _get(features, "leadership_roles")
    if isinstance(founders, int) and founders >= 2 and _populated(roles):
        findings.append(
            ReinforcementFinding(
                field_a="founder_profile_count",
                value_a=founders,
                field_b="leadership_roles",
                value_b=len(roles),  # type: ignore[arg-type]
                reason=(
                    "Multiple founder profiles corroborated by detected "
                    "leadership roles"
                ),
            )
        )

    # Network effects + marketplace dynamics.
    if _populated(_get(features, "network_effects_signals")) and _populated(
        _get(features, "marketplace_dynamics")
    ):
        findings.append(
            ReinforcementFinding(
                field_a="network_effects_signals",
                value_a=len(  # type: ignore[arg-type]
                    features.network_effects_signals
                ),
                field_b="marketplace_dynamics",
                value_b=len(features.marketplace_dynamics),
                reason=(
                    "Network effect signals corroborated by marketplace "
                    "dynamics"
                ),
            )
        )

    # Defensibility signals + moat indicators.
    if _populated(_get(features, "defensibility_signals")) and _populated(
        _get(features, "competitive_moat_indicators")
    ):
        findings.append(
            ReinforcementFinding(
                field_a="defensibility_signals",
                value_a=len(features.defensibility_signals),
                field_b="competitive_moat_indicators",
                value_b=len(features.competitive_moat_indicators),
                reason=(
                    "Defensibility signals corroborated by competitive "
                    "moat indicators"
                ),
            )
        )

    return findings


# ────────────────────────────────────────────────────────────────────
# Unsupported conclusions & missing evidence
# ────────────────────────────────────────────────────────────────────


def _ref_matches_item(ref: str, item: EvidenceItem) -> bool:
    """Check whether an ``evidence:`` reference string resolves to item."""
    if not ref.startswith("evidence:"):
        return False
    body = ref[len("evidence:") :]
    slash_idx = body.find("/")
    colon_idx = body.find(": ")
    if slash_idx == -1 or colon_idx == -1:
        return False
    ref_domain = body[:slash_idx]
    ref_category = body[slash_idx + 1 : colon_idx]
    ref_statement = body[colon_idx + 2 :]
    return (
        item.domain == ref_domain
        and item.category == ref_category
        and ref_statement in item.statement
    )


def observation_is_supported(
    observation: Observation,
    evidence_items: list[EvidenceItem],
) -> bool:
    """Return True when the observation has resolvable support.

    Support means at least one of:
      * structured citations are attached,
      * provenance document ids are recorded,
      * an ``evidence:`` reference resolves to a real evidence item.
    """
    if observation.citations:
        return True
    if observation.provenance_document_ids:
        return True
    return any(
        _ref_matches_item(ref, item)
        for ref in observation.evidence
        for item in evidence_items
    )


def detect_unsupported_conclusions(
    observations: list[Observation],
    evidence_items: list[EvidenceItem],
) -> list[UnsupportedFinding]:
    """Find observations without any resolvable evidentiary support."""
    findings: list[UnsupportedFinding] = []
    for obs in observations:
        if not observation_is_supported(obs, evidence_items):
            findings.append(
                UnsupportedFinding(
                    source_rule=obs.source_rule,
                    statement=obs.statement,
                    reason=(
                        "No citation, provenance id, or resolvable "
                        "evidence reference supports this observation"
                    ),
                )
            )
    return findings


_FEATURE_EVIDENCE_DOMAINS: list[tuple[str, str]] = [
    ("industry", "industry"),
    ("business_model", "business_model"),
    ("funding_stage", "funding_stage"),
    ("technology_stack", "technology"),
    ("geography", "geography"),
    ("headquarters_region", "geography"),
    ("market_concentration", "competition"),
    ("competitive_density", "competition"),
]


def detect_missing_evidence(
    features: ExtractedFeatures,
    evidence_items: list[EvidenceItem],
) -> list[str]:
    """Return evidence domains for populated features with no evidence.

    A domain is "missing" when at least one mapped feature field is
    populated but the evidence pool contains zero items for the domain.
    Results are ordered by the mapping table and deduplicated.
    """
    covered = {item.domain for item in evidence_items}
    missing: list[str] = []
    for field_name, domain in _FEATURE_EVIDENCE_DOMAINS:
        if domain in covered or domain in missing:
            continue
        if _populated(_get(features, field_name)):
            missing.append(domain)
    return missing


# ────────────────────────────────────────────────────────────────────
# Aggregation
# ────────────────────────────────────────────────────────────────────


def build_consistency_report(
    features: ExtractedFeatures,
    evidence_items: list[EvidenceItem],
    observations: list[Observation] | None = None,
) -> ConsistencyReport:
    """Build a full consistency report from features, evidence, observations.

    Pure function — identical inputs always produce identical reports.
    ``observations`` defaults to empty (no unsupported-conclusion scan).
    """
    return ConsistencyReport(
        contradictions=detect_contradictory_features(features),
        reinforcements=detect_reinforcing_features(features),
        unsupported_observations=detect_unsupported_conclusions(
            observations or [], evidence_items
        ),
        missing_evidence_domains=detect_missing_evidence(
            features, evidence_items
        ),
    )
