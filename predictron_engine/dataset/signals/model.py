"""Company signal models (Project E4).

A :class:`CompanySignal` is an immutable, dated observation about a
company.  Signals are the atomic unit of temporal intelligence: every
signal carries its own source, timestamp, provenance, confidence, and an
evidence reference so each observation can be traced back to where it
came from and when it was recorded.

Guarantees
----------
- Frozen dataclasses: signals cannot be mutated once created.
- Deterministic identity: ``signal_id`` is a SHA-256 digest of the
  canonical signal payload, so identical observations (same company,
  type, timestamp, source, provenance, confidence, and metadata) always
  produce the same id — which powers duplicate prevention.
- Grounded timestamps: signals must carry a tz-aware timestamp; naive
  datetimes are rejected at construction and coerced to UTC only when
  deserializing foreign JSON.
- No fabrication: the model stores only what a caller/import provides.
  Nothing here synthesizes data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any


class SignalType(str, Enum):
    """Canonical vocabulary of company signal types.

    The sixteen signal types below cover the observable events Project E4
    tracks.  Values are stable strings (not display names) so persisted
    timelines remain readable and comparable across versions.
    """

    FUNDING_ROUND = "funding_round"
    HIRING_GROWTH = "hiring_growth"
    LAYOFFS = "layoffs"
    ACQUISITION = "acquisition"
    IPO = "ipo"
    BANKRUPTCY = "bankruptcy"
    SHUTDOWN = "shutdown"
    FOUNDER_CHANGE = "founder_change"
    INVESTOR_ADDED = "investor_added"
    VALUATION_UPDATE = "valuation_update"
    REVENUE_MILESTONE = "revenue_milestone"
    ARR_MILESTONE = "arr_milestone"
    EMPLOYEE_MILESTONE = "employee_milestone"
    PRODUCT_LAUNCH = "product_launch"
    PARTNERSHIP = "partnership"
    REGULATORY_APPROVAL = "regulatory_approval"


# The canonical set, in definition order.  Exposed so callers can assert
# the vocabulary without importing the enum members individually.
SIGNAL_TYPES: tuple[SignalType, ...] = tuple(SignalType)

# Signal types that are only ever produced by external import (never by
# the deterministic builder, which cannot ground them in stored data).
# Note: LAYOFFS is deliberately NOT here — the builder emits grounded
# ``layoffs`` signals from dated headcount decreases.
EXTERNAL_ONLY_SIGNAL_TYPES: frozenset[SignalType] = frozenset(
    {
        SignalType.FOUNDER_CHANGE,
        SignalType.PRODUCT_LAUNCH,
        SignalType.PARTNERSHIP,
        SignalType.REGULATORY_APPROVAL,
        SignalType.REVENUE_MILESTONE,
    }
)


def _utc(dt: datetime) -> datetime:
    """Require a tz-aware datetime and normalize it to UTC."""
    if dt.tzinfo is None:
        raise ValueError("signal timestamps must be tz-aware (use UTC)")
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class EvidenceReference:
    """A citation pointing at the evidence behind a signal.

    ``reference`` carries the URL or citation string; ``description``
    gives human context; ``retrieved_at`` records when the evidence was
    consulted (tz-aware when present).
    """

    reference: str = ""
    description: str = ""
    retrieved_at: datetime | None = None

    def to_dict(self) -> dict[str, object]:
        retrieved: str | None = None
        if self.retrieved_at is not None:
            retrieved = _utc(self.retrieved_at).isoformat()
        return {
            "reference": self.reference,
            "description": self.description,
            "retrieved_at": retrieved,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvidenceReference:
        retrieved: datetime | None = None
        raw = data.get("retrieved_at")
        if isinstance(raw, str) and raw:
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError:
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                retrieved = parsed.astimezone(UTC)
        return cls(
            reference=str(data.get("reference", "") or ""),
            description=str(data.get("description", "") or ""),
            retrieved_at=retrieved,
        )


def signal_canonical_dict(signal: CompanySignal) -> dict[str, object]:
    """Return the canonical payload used for identity and serialization."""
    return {
        "company_id": signal.company_id,
        "signal_type": signal.signal_type.value,
        "timestamp": _utc(signal.timestamp).isoformat(),
        "source": signal.source,
        "provenance": signal.provenance,
        "confidence": signal.confidence,
        "evidence": signal.evidence.to_dict(),
        "metadata": dict(signal.metadata),
    }


def compute_signal_id(signal: CompanySignal) -> str:
    """Compute the deterministic SHA-256 identity of a signal.

    Deterministic means: identical payloads always hash to the same id
    regardless of process, machine, or insertion order of ``metadata``
    keys (keys are canonicalized with ``sort_keys``).
    """
    canonical = json.dumps(
        signal_canonical_dict(signal),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def signal_to_dict(signal: CompanySignal) -> dict[str, object]:
    """Serialize a signal to a JSON-ready dictionary."""
    return {**signal_canonical_dict(signal), "signal_id": signal.signal_id}


def signal_from_dict(data: Mapping[str, object]) -> CompanySignal:
    """Deserialize a signal from a dictionary (from ``signal_to_dict``).

    The stored ``signal_id`` is ignored and recomputed from the canonical
    payload so the id always matches the content actually carried.
    """
    ts_raw = data.get("timestamp")
    if not isinstance(ts_raw, str) or not ts_raw:
        raise ValueError("signal requires an ISO-8601 'timestamp' string")
    try:
        timestamp = datetime.fromisoformat(ts_raw)
    except ValueError as exc:
        raise ValueError(f"invalid signal timestamp: {ts_raw!r}") from exc

    type_raw = data.get("signal_type")
    if not isinstance(type_raw, str) or not type_raw:
        raise ValueError("signal requires a 'signal_type' string")
    try:
        signal_type = SignalType(type_raw)
    except ValueError as exc:
        raise ValueError(f"unknown signal type: {type_raw!r}") from exc

    evidence_raw = data.get("evidence")
    evidence: EvidenceReference
    if isinstance(evidence_raw, dict):
        evidence = EvidenceReference.from_dict(evidence_raw)
    else:
        evidence = EvidenceReference()

    metadata_raw = data.get("metadata")
    metadata: Mapping[str, object]
    if isinstance(metadata_raw, dict):
        metadata = dict(metadata_raw)
    else:
        metadata = {}

    confidence_raw = data.get("confidence", 1.0)
    if isinstance(confidence_raw, bool):
        raise ValueError("signal confidence must be numeric")
    if isinstance(confidence_raw, int | float):
        confidence = float(confidence_raw)
    elif isinstance(confidence_raw, str):
        try:
            confidence = float(confidence_raw)
        except ValueError as exc:
            raise ValueError("signal confidence must be numeric") from exc
    else:
        raise ValueError("signal confidence must be numeric")

    return CompanySignal(
        company_id=str(data.get("company_id", "") or ""),
        signal_type=signal_type,
        timestamp=timestamp,
        source=str(data.get("source", "") or ""),
        provenance=str(data.get("provenance", "manual") or "manual"),
        confidence=confidence,
        evidence=evidence,
        metadata=metadata,
    )


@dataclass(frozen=True)
class CompanySignal:
    """An immutable, dated observation about a company.

    Parameters
    ----------
    company_id :
        Canonical resolved company identifier (e.g. the graph's
        ``company:`` node id) or a stable per-record id.
    signal_type :
        One of the sixteen :class:`SignalType` values.
    timestamp :
        TZ-aware observation datetime; normalized to UTC.
    source :
        Origin of the observation (e.g. ``crunchbase``, ``news``).
    provenance :
        How the observation was obtained (e.g. ``imported:json``,
        ``derived:outcome``, ``manual``).
    confidence :
        Confidence in the observation, 0.0–1.0.
    evidence :
        Reference point for the source material.
    metadata :
        Type-specific payload (round amounts, headcount deltas, ...).
    """

    company_id: str
    signal_type: SignalType
    timestamp: datetime
    source: str
    provenance: str = "manual"
    confidence: float = 1.0
    evidence: EvidenceReference = field(default_factory=EvidenceReference)
    metadata: Mapping[str, object] = field(default_factory=dict)
    signal_id: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if not isinstance(self.signal_type, SignalType):
            object.__setattr__(self, "signal_type", SignalType(str(self.signal_type)))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0.0, 1.0]")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "timestamp", _utc(self.timestamp))
        object.__setattr__(self, "signal_id", compute_signal_id(self))

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary (includes ``signal_id``)."""
        return signal_to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CompanySignal:
        """Deserialize from a dictionary (see :func:`signal_from_dict`)."""
        return signal_from_dict(data)

    def metadata_value(self, key: str, default: Any = None) -> Any:
        """Typed lookup into the immutable metadata mapping."""
        return self.metadata.get(key, default)

    def with_metadata(self, **updates: object) -> CompanySignal:
        """Return a copy with ``updates`` merged into ``metadata``."""
        merged = dict(self.metadata)
        merged.update(updates)
        return CompanySignal(
            company_id=self.company_id,
            signal_type=self.signal_type,
            timestamp=self.timestamp,
            source=self.source,
            provenance=self.provenance,
            confidence=self.confidence,
            evidence=self.evidence,
            metadata=merged,
        )

    def as_provenance(self) -> str:
        """Human-readable one-line summary of the observation."""
        return (
            f"{self.signal_type.value}@{self.timestamp.date().isoformat()} "
            f"[{self.source} / {self.provenance}]"
        )
