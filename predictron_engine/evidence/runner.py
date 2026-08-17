"""Synchronous bridge for running async evidence collection from the engine.

PredictronEngine.analyze is synchronous and, when called from the web
service, executes inside a worker thread (``asyncio.to_thread``). The
EvidenceOrchestrator is fully async, so collection is bridged by running
the coroutine on a dedicated thread via ``asyncio.run``. A module-level
lock serializes collection across concurrent analyses so the
orchestrator's shared HTTP client is never used concurrently.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Protocol

from predictron_engine.evidence.models import EvidenceBundle

_LOCK = threading.Lock()


class EvidenceOrchestratorProtocol(Protocol):
    """Async evidence orchestrator accepted by the runner bridge."""

    async def collect(self, startup_name: str, website: str) -> EvidenceBundle: ...


def collect_evidence_sync(
    orchestrator: EvidenceOrchestratorProtocol,
    startup_name: str,
    website: str,
) -> EvidenceBundle:
    """Collect website evidence synchronously using a dedicated worker thread.

    Serializes collection with a module-level lock so the orchestrator's
    dependencies are never used from two threads at once. Never raises
    for collection failures.
    """
    if not website:
        return EvidenceBundle.empty(startup_name)

    with _LOCK:
        return asyncio.run(orchestrator.collect(startup_name, website))
