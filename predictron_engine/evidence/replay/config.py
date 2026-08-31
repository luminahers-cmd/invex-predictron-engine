"""Configuration for deterministic offline evidence replay.

Replay is opt-in and gated entirely by environment variables so that,
when disabled, production behavior is byte-identical to today — no new
providers are registered, no datasets are loaded, and no code path
changes.

Reading follows the same convention used by the evidence collection
layer (:func:`predictron_engine.evidence.orchestrator._search_enabled_from_env`):
values are read directly from the environment, never via Pydantic
``Settings``.
"""

from __future__ import annotations

import os

_REPLAY_ENABLED = "EVIDENCE_REPLAY_ENABLED"
_REPLAY_DATASET = "EVIDENCE_REPLAY_DATASET"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def replay_enabled_from_env() -> bool:
    """Return True when offline evidence replay is explicitly enabled."""
    return os.environ.get(_REPLAY_ENABLED, "").strip().lower() in _TRUE_VALUES


def replay_dataset_from_env() -> str | None:
    """Return the configured dataset name, or None when unset/blank.

    The dataset name is resolved relative to the committed corpus root
    (``EVIDENCE_REPLAY_DATASET=<name>`` maps to ``datasets/<name>.json``).
    """
    value = os.environ.get(_REPLAY_DATASET, "").strip()
    return value or None
