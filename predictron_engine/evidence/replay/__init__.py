"""Deterministic offline evidence replay (Sprint P7).

Provides everything needed to replay a recorded evidence corpus so the
engine can be regression-tested offline — without network access and
without changing production behavior.

Public API
----------
* :func:`build_replay_provider` — construct a :class:`ReplayEvidenceProvider`
  from the environment; returns ``None`` when replay is disabled.
* :class:`ReplayEvidenceProvider` — the replay provider itself (satisfies
  the :class:`~predictron_engine.evidence.provider_contracts.EvidenceProvider`
  protocol).
* :func:`~predictron_engine.evidence.replay.dataset.dump_corpus` /
  :class:`~predictron_engine.evidence.replay.dataset` helpers — corpus IO.

When ``EVIDENCE_REPLAY_ENABLED`` is falsy, :func:`build_replay_provider`
returns ``None`` so no replay provider is ever registered and production
behavior is byte-identical to today.
"""

from __future__ import annotations

from predictron_engine.evidence.provider_contracts import EvidenceProvider
from predictron_engine.evidence.replay.config import (
    replay_dataset_from_env,
    replay_enabled_from_env,
)
from predictron_engine.evidence.replay.dataset import dataset_root
from predictron_engine.evidence.replay.provider import ReplayEvidenceProvider

__all__ = [
    "ReplayEvidenceProvider",
    "build_replay_provider",
    "dataset_root",
    "replay_dataset_from_env",
    "replay_enabled_from_env",
]


def build_replay_provider() -> EvidenceProvider | None:
    """Build a replay provider from the environment, or None when disabled.

    Returns ``None`` unless ``EVIDENCE_REPLAY_ENABLED`` is truthy and a
    ``EVIDENCE_REPLAY_DATASET`` is configured.  This is the single
    registration point used by the orchestrator so replay never changes
    behavior unless explicitly enabled.
    """
    if not replay_enabled_from_env():
        return None
    dataset = replay_dataset_from_env()
    if dataset is None:
        return None
    return ReplayEvidenceProvider(dataset)
