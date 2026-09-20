"""Deterministic JSON persistence for signal timelines (Project E4).

Timelines are stored one-per-company under the store's ``signals/``
directory as ``{encoded_company_id}.json``.  Company ids may contain
characters that are illegal in filenames (``:``, ``+``, ``/`` on
Windows), so ids are URL-encoded for the filename and decoded back on
read.

Serialization is deterministic: keys are sorted, ``signal_id`` values
are recomputed on read, and the on-disk bytes for identical timelines
are identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote, unquote

from predictron_engine.dataset.signals.model import CompanySignal, signal_from_dict
from predictron_engine.dataset.signals.timeline import CompanyTimeline

SIGNALS_DIR_NAME = "signals"
SIGNALS_SCHEMA_VERSION = "1.0.0"


def _signals_dir(root: Path) -> Path:
    return root / SIGNALS_DIR_NAME


def _encode_company_id(company_id: str) -> str:
    if not company_id:
        raise ValueError("company_id must not be empty")
    return quote(company_id, safe="")


def _decode_company_id(encoded: str) -> str:
    return unquote(encoded)


def timeline_to_dict(timeline: CompanyTimeline) -> dict[str, object]:
    """Serialize a timeline to a JSON-ready dictionary."""
    return {
        "schema_version": SIGNALS_SCHEMA_VERSION,
        "company_id": timeline.company_id,
        "signal_count": timeline.signal_count,
        "signals": [signal_to_dict(s) for s in timeline.signals],
    }


def signal_to_dict(signal: CompanySignal) -> dict[str, object]:
    """Serialize a single signal (cheap helper for callers)."""
    return signal.to_dict()


def timeline_from_dict(data: dict[str, object]) -> CompanyTimeline:
    """Deserialize a timeline from a dictionary produced by
    :func:`timeline_to_dict`."""
    company_id = str(data.get("company_id", "") or "")
    raw_signals = data.get("signals", [])
    if not isinstance(raw_signals, list):
        raise ValueError("timeline 'signals' must be a list")
    signals: list[CompanySignal] = []
    for raw in raw_signals:
        if not isinstance(raw, dict):
            raise ValueError("timeline contains a non-object signal")
        signals.append(signal_from_dict(raw))
    return CompanyTimeline.build(company_id, signals)


def write_timeline(root: Path, timeline: CompanyTimeline) -> Path:
    """Persist a timeline under ``root/signals/``; returns the path."""
    directory = _signals_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_encode_company_id(timeline.company_id)}.json"
    payload = timeline_to_dict(timeline)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return path


def read_timeline(root: Path, company_id: str) -> CompanyTimeline | None:
    """Load a timeline from ``root/signals/``, or None when absent."""
    directory = _signals_dir(root)
    path = directory / f"{_encode_company_id(company_id)}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("stored signal timeline is not an object")
    return timeline_from_dict(data)


def list_signal_company_ids(root: Path) -> list[str]:
    """Return sorted company ids that have a stored timeline."""
    directory = _signals_dir(root)
    if not directory.exists():
        return []
    ids: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            company_id = _decode_company_id(path.stem)
        except ValueError:
            continue
        if company_id:
            ids.append(company_id)
    return ids
