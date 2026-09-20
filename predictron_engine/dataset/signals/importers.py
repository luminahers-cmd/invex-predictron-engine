"""Signal importers (Project E4).

Imports load :class:`CompanySignal` objects from user-provided JSON so
the dataset can carry signal types the deterministic builder refuses to
synthesize (founder changes, product launches, partnerships, regulatory
approvals, revenue milestones — and any other observation a source can
ground).  Each import record is validated; invalid records are rejected
and reported, never silently dropped.

Records may refer to a company by any of ``company_id``, ``record_id``,
``startup_name``, or ``website``.  A ``company_map`` resolves those keys
to canonical company ids before the signal is constructed.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from predictron_engine.dataset.signals.model import CompanySignal, signal_from_dict

_COMPANY_KEYS = ("company_id", "record_id", "startup_name", "website")


@dataclass
class SignalImportReport:
    """Outcome of a signal import pass."""

    requested: int = 0
    imported: int = 0
    rejected: int = 0
    companies: list[str] = field(default_factory=list)
    per_type: dict[str, int] = field(default_factory=dict)
    validation_errors: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "imported": self.imported,
            "rejected": self.rejected,
            "companies": list(self.companies),
            "per_type": dict(sorted(self.per_type.items())),
            "validation_errors": self.validation_errors,
        }


def _resolve_company_id(
    data: Mapping[str, Any],
    company_map: Mapping[str, str] | None,
) -> str:
    """Resolve a company id from the identity keys of an import record.

    Precedence: explicit ``company_id``, then ``record_id``, then
    ``startup_name``, then ``website`` — each looked up in ``company_map``
    when present.
    """
    if company_map is not None:
        for key in _COMPANY_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value:
                resolved = company_map.get(value)
                if resolved:
                    return resolved
    explicit = data.get("company_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    name = data.get("startup_name")
    if isinstance(name, str) and name:
        return name
    website = data.get("website")
    if isinstance(website, str) and website:
        return website
    return ""


def parse_signal_record(
    data: Mapping[str, Any],
    *,
    company_map: Mapping[str, str] | None = None,
) -> CompanySignal:
    """Normalize one import record into a :class:`CompanySignal`.

    The record may carry identity keys (``company_id``/``record_id``/
    ``startup_name``/``website``).  After resolution a plain signal dict
    is handed to :func:`signal_from_dict`, which validates the payload
    and recomputes the deterministic ``signal_id``.
    """
    company_id = _resolve_company_id(data, company_map)
    payload: dict[str, Any] = {
        key: data[key]
        for key in (
            "signal_type",
            "timestamp",
            "source",
            "provenance",
            "confidence",
            "evidence",
            "metadata",
        )
        if key in data
    }
    payload["company_id"] = company_id
    return signal_from_dict(payload)


def import_signals_from_json(
    path: str | Path,
    *,
    company_map: Mapping[str, str] | None = None,
) -> tuple[list[CompanySignal], SignalImportReport]:
    """Import signals from a JSON file (array of record objects).

    Returns the accepted signals and a :class:`SignalImportReport`.  An
    empty ``company_id`` after resolution is an import error.
    """
    file_path = Path(path)
    report = SignalImportReport()
    if not file_path.exists():
        report.validation_errors.append(
            {"index": None, "error": f"file not found: {file_path}"}
        )
        return [], report

    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        report.validation_errors.append(
            {"index": None, "error": "expected a JSON array of signal objects"}
        )
        return [], report

    report.requested = len(payload)
    accepted: list[CompanySignal] = []
    counters: Counter[str] = Counter()
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            report.rejected += 1
            report.validation_errors.append(
                {"index": index, "error": "record is not an object"}
            )
            continue
        try:
            signal = parse_signal_record(record, company_map=company_map)
        except ValueError as exc:
            report.rejected += 1
            report.validation_errors.append(
                {"index": index, "error": str(exc)}
            )
            continue
        if not signal.company_id:
            report.rejected += 1
            report.validation_errors.append(
                {"index": index, "error": "record has no resolvable company"}
            )
            continue
        accepted.append(signal)
        counters[signal.signal_type.value] += 1

    report.imported = len(accepted)
    report.per_type = dict(sorted(counters.items()))
    report.companies = sorted({signal.company_id for signal in accepted})
    return accepted, report
