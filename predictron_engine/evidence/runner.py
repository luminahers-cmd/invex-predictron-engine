"""Synchronous bridge for running async evidence collection from the engine.

PredictronEngine.analyze is synchronous and, when called from the web
service, executes inside a worker thread (``asyncio.to_thread``). The
EvidenceOrchestrator is fully async, so collection is bridged by running
the coroutine on a dedicated thread via ``asyncio.run``.

Concurrency safety
------------------
Two collection modes are supported:

1. **Factory mode (production default).** The engine supplies a callable
   that returns a *fresh* :class:`EvidenceOrchestrator` (and therefore a
   fresh set of providers and HTTP clients) for every analysis. Each call
   runs on its own isolated event loop, so concurrent analyses never share
   the mutable, single-event-loop HTTP client state that an orchestrator
   holds. No lock is required, which means concurrent analyses no longer
   serialize on evidence collection.

2. **Injected-instance mode.** A caller may supply an explicit orchestrator
   instance (e.g. an offline replay provider or a test double). Because
   such an instance may hold a shared, single-event-loop HTTP client,
   access is serialized with an internal lock to preserve the previous
   safety guarantee for custom collectors that are shared across threads.

Determinism and offline replay are preserved: the default provider
sequence, the deterministic merge order, and the offline replay provider
are all unchanged.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Protocol, cast

from predictron_engine.evidence.models import EvidenceBundle


class EvidenceOrchestratorProtocol(Protocol):
    """Async evidence orchestrator accepted by the runner bridge."""

    async def collect(self, startup_name: str, website: str) -> EvidenceBundle: ...


class EvidenceCollectorFactory(Protocol):
    """Callable that produces a fresh orchestrator per analysis."""

    def __call__(self) -> EvidenceOrchestratorProtocol: ...


# Guards explicitly injected singleton orchestrators whose shared HTTP
# client is bound to a single event loop and must never be used from more
# than one thread at a time. Not used by the production factory path.
_SINGLETON_LOCK = threading.Lock()


def collect_evidence_sync(
    collector: EvidenceOrchestratorProtocol | EvidenceCollectorFactory,
    startup_name: str,
    website: str,
) -> EvidenceBundle:
    """Collect website evidence synchronously using a dedicated worker thread.

    When ``collector`` is a factory (the production default) a fresh
    orchestrator is built per call and run on its own isolated event loop,
    so concurrent analyses do not serialize. When ``collector`` is a shared
    orchestrator instance, access is serialized to protect its single-loop
    HTTP client. Never raises for collection failures.
    """
    if not website:
        return EvidenceBundle.empty(startup_name)

    is_instance = callable(getattr(collector, "collect", None))
    if is_instance:
        orchestrator = cast("EvidenceOrchestratorProtocol", collector)
        with _SINGLETON_LOCK:
            return asyncio.run(orchestrator.collect(startup_name, website))

    factory = cast("EvidenceCollectorFactory", collector)
    orchestrator = factory()
    return asyncio.run(_collect_and_close(orchestrator, startup_name, website))


async def _collect_and_close(
    orchestrator: EvidenceOrchestratorProtocol,
    startup_name: str,
    website: str,
) -> EvidenceBundle:
    """Run a collection, then release any orchestrator-owned resources.

    The fresh orchestrator created by a factory owns its providers and HTTP
    clients, so it is closed (best-effort) once the collection finishes to
    avoid leaking connections across the short-lived event loop.
    """
    try:
        return await orchestrator.collect(startup_name, website)
    finally:
        close = getattr(orchestrator, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001 - cleanup must never mask results
                pass
