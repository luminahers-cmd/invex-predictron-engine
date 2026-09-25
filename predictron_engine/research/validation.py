"""Deterministic Evidence Validation Engine — Phase 8 Sprint 4.

:class:`EvidenceValidationEngine` decides whether collected evidence is
trustworthy before downstream systems consume it.  It is the validation
layer's orchestrator:

* accepts an :class:`EvidenceCollection`,
* validates every :class:`Evidence` object through the registered
  :class:`~predictron_engine.research.validators.EvidenceValidator`
  instances (resolved via the
  :class:`~predictron_engine.research.validator_registry.ValidatorRegistry`,
  never concrete classes),
* aggregates per-validator :class:`ValidationResult` objects,
* assigns deterministic confidence scores,
* detects conflicts (contradictions, duplicates, missing references,
  incomplete metadata, invalid timestamps, schema violations),
* fingerprints every evidence item deterministically,
* produces a :class:`ValidationSummary`.

The engine is **pure and deterministic**: it performs no collection, no
web requests, no browser automation, no LLM reasoning, and no knowledge
graph updates.  Identical input always produces an identical
:class:`ValidationSummary`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from urllib.parse import urlsplit

from predictron_engine.research.exceptions import InvalidEvidenceError
from predictron_engine.research.models import Evidence, EvidenceCollection
from predictron_engine.research.validation_models import (
    VALIDATION_SCHEMA_VERSION,
    EvidenceConfidence,
    EvidenceConflict,
    EvidenceFingerprint,
    EvidenceValidation,
    TopicValidation,
    ValidationCounts,
    ValidationIssue,
    ValidationMetadata,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
    ValidationSummary,
    collection_fingerprint,
    compute_fingerprint,
    content_digest,
)
from predictron_engine.research.validation_rules import (
    DEFAULT_RULES,
    ValidationContext,
    ValidationRule,
    ValidationRuleSet,
)
from predictron_engine.research.validator_registry import (
    VALIDATOR_REGISTRY,
    ValidatorRegistry,
)
from predictron_engine.research.validators import EvidenceValidator

ENGINE_VERSION = "1.0.0"

# ----------------------------------------------------------------------
# Deterministic confidence scoring configuration
# ----------------------------------------------------------------------

#: Weight of each scored factor in the deterministic confidence formula.
FACTOR_WEIGHTS: dict[str, float] = {
    "completeness": 0.25,
    "metadata": 0.15,
    "reference": 0.20,
    "freshness": 0.15,
    "consistency": 0.15,
    "stated": 0.10,
}

#: Which validator results feed each scored factor.
FACTOR_VALIDATOR_IDS: dict[str, tuple[str, ...]] = {
    "completeness": ("completeness_validator",),
    "metadata": ("source_metadata_validator",),
    "reference": ("reference_validator",),
    "freshness": ("freshness_validator",),
    "consistency": ("schema_validator", "consistency_validator"),
}

#: Deterministic multiplicative penalties (no randomness involved).
DUPLICATE_PENALTY = 0.9
CONFLICT_PENALTY = 0.85
HISTORY_BLEND_CURRENT = 0.9
HISTORY_BLEND_PREVIOUS = 0.1


def compute_confidence_score(
    evidence: Evidence,
    factor_scores: Mapping[str, float],
    *,
    duplicate_count: int = 1,
    conflict_count: int = 0,
    previous_score: float | None = None,
) -> float:
    """Compute the deterministic confidence score for ``evidence``.

    Parameters
    ----------
    evidence:
        The evidence being scored (its stated confidence is the fallback
        and the ``"stated"`` factor).
    factor_scores:
        Deterministic scores in ``[0, 1]`` keyed by factor name.
    duplicate_count:
        How many copies of this evidence exist (``1`` normally).
    conflict_count:
        How many conflicts involve this evidence.
    previous_score:
        Optional historical score in ``[0, 1]`` used as the validation
        history signal (blended deterministically).

    Returns
    -------
    A reproducible score in ``[0, 1]``.  The same inputs always produce
    the same output.
    """
    available = [factor for factor in FACTOR_WEIGHTS if factor in factor_scores]
    if available:
        denominator = sum(FACTOR_WEIGHTS[factor] for factor in available)
        numerator = sum(
            FACTOR_WEIGHTS[factor] * factor_scores[factor]
            for factor in available
        )
        base = numerator / denominator if denominator else evidence.confidence
    else:
        base = evidence.confidence
    if duplicate_count > 1:
        base *= DUPLICATE_PENALTY ** (duplicate_count - 1)
    if conflict_count > 0:
        base *= CONFLICT_PENALTY ** conflict_count
    if previous_score is not None:
        prior = min(1.0, max(0.0, previous_score))
        base = HISTORY_BLEND_CURRENT * base + HISTORY_BLEND_PREVIOUS * prior
    return min(1.0, max(0.0, base))


class EvidenceValidationEngine:
    """Deterministic validator of :class:`EvidenceCollection` objects.

    Parameters
    ----------
    registry:
        Optional :class:`ValidatorRegistry` used to resolve validators.
        When ``None`` the shared registry is used.
    rules:
        Optional default :class:`ValidationRule` instances.  When ``None``
        the built-in :data:`DEFAULT_RULES` are used.
    """

    def __init__(
        self,
        registry: ValidatorRegistry | None = None,
        rules: Sequence[ValidationRule] | None = None,
    ) -> None:
        self._registry = registry if registry is not None else VALIDATOR_REGISTRY
        self._default_rules = tuple(rules) if rules is not None else DEFAULT_RULES

    @property
    def registry(self) -> ValidatorRegistry:
        """Return the registry this engine resolves validators through."""
        return self._registry

    @property
    def default_rules(self) -> tuple[ValidationRule, ...]:
        """Return the default rules used when a run supplies none."""
        return self._default_rules

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        collection: EvidenceCollection,
        *,
        registry: ValidatorRegistry | None = None,
        rules: Sequence[ValidationRule] | None = None,
        evidence_timestamps: Mapping[str, str] | None = None,
        reference_time: str | None = None,
        previous_validations: Mapping[str, float] | None = None,
    ) -> ValidationSummary:
        """Validate ``collection`` and return a deterministic summary.

        Parameters
        ----------
        collection:
            The :class:`EvidenceCollection` to validate.
        registry:
            Optional registry overriding the engine's default for this run.
        rules:
            Optional rules overriding the engine's defaults for this run.
        evidence_timestamps:
            Optional deterministic mapping from evidence id to an ISO-8601
            timestamp used by freshness validation.
        reference_time:
            Optional deterministic ISO-8601 "now" for freshness validation.
        previous_validations:
            Optional mapping from evidence id to a previous confidence
            score in ``[0, 1]`` used as the validation history factor.

        Raises
        ------
        InvalidEvidenceError:
            When ``collection`` is not an :class:`EvidenceCollection`.
        """
        if not isinstance(collection, EvidenceCollection):
            raise InvalidEvidenceError(
                "validate expects an EvidenceCollection instance"
            )
        resolved_registry = registry if registry is not None else self._registry
        selected_rules = tuple(rules) if rules is not None else self._default_rules
        rule_set = ValidationRuleSet(selected_rules)
        validators = resolved_registry.list_validators()
        items = collection.items()

        fingerprints = {
            item.evidence_id: compute_fingerprint(item) for item in items
        }
        timestamps = dict(evidence_timestamps or {})
        duplicate_counts = self._duplicate_counts(items, rule_set)
        conflicts, conflicts_by_evidence = self._detect_conflicts(
            items,
            rule_set=rule_set,
            timestamps=timestamps,
            duplicate_counts=duplicate_counts,
        )
        return self._summarize(
            collection=collection,
            items=items,
            validators=validators,
            rule_set=rule_set,
            fingerprints=fingerprints,
            timestamps=timestamps,
            previous_scores=dict(previous_validations or {}),
            reference_time=reference_time,
            duplicate_counts=duplicate_counts,
            conflicts=conflicts,
            conflicts_by_evidence=conflicts_by_evidence,
        )

    # ------------------------------------------------------------------
    # Fingerprinting
    # ------------------------------------------------------------------

    def fingerprint(self, evidence: Evidence) -> EvidenceFingerprint:
        """Return the deterministic fingerprint of one ``Evidence``."""
        return compute_fingerprint(evidence)

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _duplicate_counts(
        items: tuple[Evidence, ...],
        rule_set: ValidationRuleSet,
    ) -> dict[str, int]:
        """Return how many copies exist per evidence id.

        Duplicate detection is disabled when the ``duplicate_detection``
        rule is disabled, in which case every item counts as unique.
        """
        enabled = rule_set.get_bool_parameter(
            "duplicate_detection", "enabled", True
        )
        if not enabled:
            return {item.evidence_id: 1 for item in items}
        counts: dict[str, int] = {}
        for item in items:
            counts[item.evidence_id] = counts.get(item.evidence_id, 0) + 1
        return counts

    def _detect_conflicts(
        self,
        items: tuple[Evidence, ...],
        *,
        rule_set: ValidationRuleSet,
        timestamps: Mapping[str, str],
        duplicate_counts: Mapping[str, int],
    ) -> tuple[
        tuple[EvidenceConflict, ...],
        dict[str, tuple[EvidenceConflict, ...]],
    ]:
        """Detect deterministic conflicts across the collection.

        Produces :class:`EvidenceConflict` records for contradictory
        values, duplicate evidence, missing/invalid references, incomplete
        metadata, invalid timestamps, and schema violations.  The returned
        pair is ``(all_conflicts, conflicts_by_evidence)``.
        """
        conflicts: list[EvidenceConflict] = []
        by_evidence: dict[str, list[EvidenceConflict]] = {}

        def _add(conflict: EvidenceConflict) -> None:
            conflicts.append(conflict)
            by_evidence.setdefault(conflict.evidence_a, []).append(conflict)
            if conflict.evidence_b:
                by_evidence.setdefault(conflict.evidence_b, []).append(conflict)

        topics: dict[str, list[Evidence]] = {}
        for item in items:
            topics.setdefault(item.topic_id, []).append(item)

        for topic_id in sorted(topics):
            topic_items = sorted(topics[topic_id], key=lambda i: i.evidence_id)
            for left in range(len(topic_items)):
                for right in range(left + 1, len(topic_items)):
                    a = topic_items[left]
                    b = topic_items[right]
                    if a.evidence_id == b.evidence_id:
                        continue
                    if a.category == b.category and a.claim != b.claim:
                        _add(
                            EvidenceConflict(
                                conflict_type="conflicting_values",
                                evidence_a=a.evidence_id,
                                evidence_b=b.evidence_id,
                                description=(
                                    f"Evidence for category {a.category!r} "
                                    "carries conflicting claims."
                                ),
                                severity=ValidationSeverity.WARNING,
                                field="claim",
                                value_a=a.claim,
                                value_b=b.claim,
                            )
                        )

        for item in items:
            conflict_type, description, severity, field = self._diagnose(
                item, timestamps
            )
            if conflict_type is not None:
                _add(
                    EvidenceConflict(
                        conflict_type=conflict_type,
                        evidence_a=item.evidence_id,
                        description=description,
                        severity=severity,
                        field=field,
                    )
                )

        duplicate_error = rule_set.get_bool_parameter(
            "duplicate_detection", "treat_as_error", False
        )
        for evidence_id, count in sorted(duplicate_counts.items()):
            if count <= 1:
                continue
            severity = (
                ValidationSeverity.ERROR
                if duplicate_error
                else ValidationSeverity.WARNING
            )
            _add(
                EvidenceConflict(
                    conflict_type="duplicate_evidence",
                    evidence_a=evidence_id,
                    description=f"Evidence {evidence_id!r} appears {count} times.",
                    severity=severity,
                    field="evidence_id",
                )
            )
        by_final: dict[str, tuple[EvidenceConflict, ...]] = {
            evidence_id: tuple(grouped)
            for evidence_id, grouped in by_evidence.items()
        }
        return tuple(conflicts), by_final

    @staticmethod
    def _diagnose(
        item: Evidence,
        timestamps: Mapping[str, str],
    ) -> tuple[str | None, str, ValidationSeverity, str]:
        """Classify one evidence item into a self-diagnostic conflict.

        Returns ``(None, ...)`` when the item carries no such diagnostic.
        """
        reference = item.reference
        if not reference.url.strip():
            return (
                "missing_reference",
                "Evidence references no source URL.",
                ValidationSeverity.WARNING,
                "reference.url",
            )
        if not _valid_web_url(reference.url):
            return (
                "invalid_reference",
                "Evidence reference URL is not a well-formed http(s) URL.",
                ValidationSeverity.ERROR,
                "reference.url",
            )
        if not reference.description.strip():
            return (
                "incomplete_metadata",
                "Evidence reference carries no description metadata.",
                ValidationSeverity.WARNING,
                "reference.description",
            )
        timestamp = timestamps.get(item.evidence_id)
        if timestamp is not None and timestamp.strip():
            try:
                _parse_timestamp(timestamp)
            except ValueError:
                return (
                    "invalid_timestamp",
                    "Evidence timestamp is not a valid ISO-8601 value.",
                    ValidationSeverity.ERROR,
                    "timestamp",
                )
        if not _schema_valid(item):
            return (
                "schema_violation",
                "Evidence violates the expected schema shape.",
                ValidationSeverity.ERROR,
                "evidence_id",
            )
        return (None, "", ValidationSeverity.INFO, "")

    def _summarize(
        self,
        *,
        collection: EvidenceCollection,
        items: tuple[Evidence, ...],
        validators: tuple[EvidenceValidator, ...],
        rule_set: ValidationRuleSet,
        fingerprints: Mapping[str, EvidenceFingerprint],
        timestamps: Mapping[str, str],
        previous_scores: Mapping[str, float],
        reference_time: str | None,
        duplicate_counts: Mapping[str, int],
        conflicts: tuple[EvidenceConflict, ...],
        conflicts_by_evidence: Mapping[str, tuple[EvidenceConflict, ...]],
    ) -> ValidationSummary:
        context = ValidationContext(
            rules=rule_set,
            timestamp_by_evidence=dict(timestamps),
            reference_time=reference_time,
            previous_scores=dict(previous_scores),
            duplicate_count_by_evidence=dict(duplicate_counts),
            conflicts_by_evidence=dict(conflicts_by_evidence),
            topics_in_collection=collection.topics_covered(),
            evidence_count_by_topic=self._counts_by_topic(items),
            topic_order=collection.topic_order,
        )
        validations: list[EvidenceValidation] = []
        for item in items:
            validations.append(
                self._validate_item(
                    item,
                    validators=validators,
                    context=context,
                    fingerprint=fingerprints[item.evidence_id],
                    duplicate_count=duplicate_counts.get(item.evidence_id, 1),
                    conflicts=conflicts_by_evidence.get(item.evidence_id, ()),
                    previous_score=previous_scores.get(item.evidence_id),
                    rule_set=rule_set,
                )
            )
        covered_topics = collection.topics_covered()
        finalized = tuple(validations)
        summary_status = _summary_status(finalized)
        by_topic = _by_topic(finalized, covered_topics)
        validator_ids = self._registry_ids(validators)
        rule_ids = rule_set.rule_ids()
        metadata = ValidationMetadata(
            schema_version=VALIDATION_SCHEMA_VERSION,
            engine_version=ENGINE_VERSION,
            collection_fingerprint=collection_fingerprint(
                [item.evidence_id for item in items]
            ),
            validator_ids=validator_ids,
            rule_ids=rule_ids,
            deterministic=True,
        )
        validation_id = _validation_id(
            collection=collection,
            status=summary_status,
            validator_ids=validator_ids,
            rule_ids=rule_ids,
        )
        return ValidationSummary(
            schema_version=VALIDATION_SCHEMA_VERSION,
            validation_id=validation_id,
            collection_fingerprint=metadata.collection_fingerprint,
            status=summary_status,
            metadata=metadata,
            validations=finalized,
            conflicts=conflicts,
            counts=_build_counts(
                finalized,
                conflicts=conflicts,
                duplicate_counts=duplicate_counts,
                covered_topics=covered_topics,
            ),
            covered_topics=covered_topics,
            by_topic=by_topic,
        )

    def _validate_item(
        self,
        item: Evidence,
        *,
        validators: tuple[EvidenceValidator, ...],
        context: ValidationContext,
        fingerprint: EvidenceFingerprint,
        duplicate_count: int,
        conflicts: tuple[EvidenceConflict, ...],
        previous_score: float | None,
        rule_set: ValidationRuleSet,
    ) -> EvidenceValidation:
        results: list[ValidationResult] = []
        for validator in validators:
            if not validator.supports(item.topic_id):
                continue
            results.append(validator.validate(item, context=context))
        issues = [issue for result in results for issue in result.issues]
        conflict_count = sum(
            1
            for conflict in conflicts
            if conflict.conflict_type != "duplicate_evidence"
        )
        factor_scores = _factor_scores(item, results)
        score = compute_confidence_score(
            item,
            factor_scores,
            duplicate_count=duplicate_count,
            conflict_count=conflict_count,
            previous_score=previous_score,
        )
        confidence = EvidenceConfidence.from_score(score)
        if _band_below(confidence, _minimum_band(rule_set)):
            issues.append(
                ValidationIssue(
                    code="confidence_below_minimum",
                    message=(
                        f"Computed confidence band {confidence.value!r} is "
                        f"below the required {_minimum_band(rule_set).value!r}."
                    ),
                    severity=ValidationSeverity.WARNING,
                    evidence_id=item.evidence_id,
                    field="confidence",
                )
            )
        return EvidenceValidation(
            evidence_id=item.evidence_id,
            task_id=item.task_id,
            topic_id=item.topic_id,
            fingerprint=fingerprint,
            status=_item_status(issues),
            confidence_score=score,
            confidence=confidence,
            validator_results=tuple(results),
            issues=tuple(issues),
            conflicts=conflicts,
            duplicate_count=duplicate_count,
        )

    @staticmethod
    def _counts_by_topic(items: tuple[Evidence, ...]) -> dict[str, int]:
        """Return evidence counts keyed by topic id."""
        counts: dict[str, int] = {}
        for item in items:
            counts[item.topic_id] = counts.get(item.topic_id, 0) + 1
        return counts

    @staticmethod
    def _registry_ids(validators: tuple[EvidenceValidator, ...]) -> tuple[str, ...]:
        """Return validator ids in deterministic registration order."""
        return tuple(
            validator.metadata().validator_id for validator in validators
        )


def _summary_status(validations: tuple[EvidenceValidation, ...]) -> ValidationStatus:
    """Derive the overall deterministic status from per-item statuses."""
    if not validations:
        return ValidationStatus.SKIPPED
    if any(
        validation.status is ValidationStatus.FAILED for validation in validations
    ):
        return ValidationStatus.FAILED
    if any(
        validation.status is ValidationStatus.WARNED for validation in validations
    ):
        return ValidationStatus.WARNED
    return ValidationStatus.PASSED


def _item_status(issues: Sequence[ValidationIssue]) -> ValidationStatus:
    """Derive a deterministic per-item status from its issues."""
    if any(issue.is_failure() for issue in issues):
        return ValidationStatus.FAILED
    if any(issue.severity.is_notice() for issue in issues):
        return ValidationStatus.WARNED
    return ValidationStatus.PASSED


def _factor_scores(
    evidence: Evidence,
    results: Sequence[ValidationResult],
) -> dict[str, float]:
    """Return deterministic per-factor scores from validator results."""
    by_id = {result.validator_id: result.score for result in results}
    factors: dict[str, float] = {}
    for factor, validator_ids in FACTOR_VALIDATOR_IDS.items():
        scores = [
            by_id[validator_id]
            for validator_id in validator_ids
            if validator_id in by_id
        ]
        if scores:
            factors[factor] = sum(scores) / len(scores)
    factors["stated"] = evidence.confidence
    return factors


def _minimum_band(rule_set: ValidationRuleSet) -> EvidenceConfidence:
    """Return the required minimum confidence band from the rules."""
    rule = rule_set.get("confidence_threshold")
    if rule is None:
        return EvidenceConfidence.LOW
    raw = rule.parameters().get("min_band", EvidenceConfidence.LOW.value)
    if isinstance(raw, EvidenceConfidence):
        return raw
    try:
        return EvidenceConfidence(str(raw))
    except ValueError:
        return EvidenceConfidence.LOW


def _band_below(band: EvidenceConfidence, minimum: EvidenceConfidence) -> bool:
    """Return whether ``band`` ranks below ``minimum``.

    The enum is declared best-to-worst (``HIGH``, ``MEDIUM``, ``LOW``,
    ``UNKNOWN``), so a higher index means a weaker band.
    """
    order = tuple(EvidenceConfidence)
    return order.index(band) > order.index(minimum)


def _build_counts(
    validations: tuple[EvidenceValidation, ...],
    *,
    conflicts: tuple[EvidenceConflict, ...],
    duplicate_counts: Mapping[str, int],
    covered_topics: tuple[str, ...],
) -> ValidationCounts:
    """Compute deterministic aggregate counters for the summary."""
    passed = sum(1 for v in validations if v.status is ValidationStatus.PASSED)
    warned = sum(1 for v in validations if v.status is ValidationStatus.WARNED)
    failed = sum(1 for v in validations if v.status is ValidationStatus.FAILED)
    issues = sum(len(v.issues) for v in validations)
    duplicates = sum(
        count - 1 for count in duplicate_counts.values() if count > 1
    )
    return ValidationCounts(
        total=len(validations),
        passed=passed,
        warned=warned,
        failed=failed,
        skipped=1 if not validations else 0,
        issues=issues,
        conflicts=len(conflicts),
        duplicates=duplicates,
        topics_covered=len(covered_topics),
    )


def _by_topic(
    validations: tuple[EvidenceValidation, ...],
    covered_topics: tuple[str, ...],
) -> tuple[TopicValidation, ...]:
    """Return a deterministic per-topic breakdown."""
    breakdown: list[TopicValidation] = []
    for topic_id in covered_topics:
        group = [v for v in validations if v.topic_id == topic_id]
        if not group:
            continue
        breakdown.append(
            TopicValidation(
                topic_id=topic_id,
                evidence_count=len(group),
                passed=sum(
                    1 for v in group if v.status is ValidationStatus.PASSED
                ),
                warned=sum(
                    1 for v in group if v.status is ValidationStatus.WARNED
                ),
                failed=sum(
                    1 for v in group if v.status is ValidationStatus.FAILED
                ),
                mean_confidence=sum(v.confidence_score for v in group)
                / len(group),
            )
        )
    return tuple(breakdown)


def _validation_id(
    *,
    collection: EvidenceCollection,
    status: ValidationStatus,
    validator_ids: tuple[str, ...],
    rule_ids: tuple[str, ...],
) -> str:
    """Return a stable content-hash identifier for the validation run."""
    payload = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "collection": collection_fingerprint(
            [item.evidence_id for item in collection.items()]
        ),
        "status": status.value,
        "validator_ids": list(validator_ids),
        "rule_ids": list(rule_ids),
    }
    return f"validation_{content_digest(payload)}"


def _schema_valid(item: Evidence) -> bool:
    """Return whether an item matches the expected schema shape."""
    if not item.task_id.strip() or not item.topic_id.strip():
        return False
    if not item.collector_id.strip() or not item.category.strip():
        return False
    if not item.claim.strip():
        return False
    return 0.0 <= item.confidence <= 1.0


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp (accepting a trailing ``Z``)."""
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def _valid_web_url(url: str) -> bool:
    """Return whether ``url`` is a well-formed http(s) URL."""
    if not url.strip():
        return True
    parts = urlsplit(url.strip())
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


__all__ = [
    "CONFLICT_PENALTY",
    "DUPLICATE_PENALTY",
    "ENGINE_VERSION",
    "EvidenceValidationEngine",
    "FACTOR_VALIDATOR_IDS",
    "FACTOR_WEIGHTS",
    "HISTORY_BLEND_CURRENT",
    "HISTORY_BLEND_PREVIOUS",
    "compute_confidence_score",
]
