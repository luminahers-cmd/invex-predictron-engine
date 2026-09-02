"""Tests for the synchronous evidence runner bridge.

Covers the two collection modes: the production factory path (fresh
orchestrator per call, no serialization) and the injected-instance path
(shared orchestrator serialized for single-loop HTTP client safety).
"""

from __future__ import annotations

import asyncio
import threading
import time

from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.evidence.runner import (
    EvidenceCollectorFactory,
    collect_evidence_sync,
)


class _SlowOrchestrator:
    """Orchestrator that records concurrency and completes after a delay."""

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self._active = 0
        self._max_active = 0
        self._deadline: float | None = None

    async def collect(self, startup_name: str, website: str) -> EvidenceBundle:
        self._active += 1
        self._max_active = max(self._max_active, self._active)
        try:
            await asyncio.sleep(self._delay)
            return EvidenceBundle.empty(startup_name)
        finally:
            self._active -= 1

    @property
    def max_active(self) -> int:
        return self._max_active


def _factory_for(instance: _SlowOrchestrator) -> EvidenceCollectorFactory:
    """Return a factory yielding fresh instances of the same class/behavior."""

    def _factory() -> _SlowOrchestrator:
        return _SlowOrchestrator(instance._delay)

    return _factory


def _run_two_collections(collector: object, delay: float) -> float:
    """Run two collections concurrently and return the wall-clock time."""
    started = time.monotonic()

    def _work() -> None:
        collect_evidence_sync(collector, "Startup", "https://example.com")  # type: ignore[arg-type]

    threads = [threading.Thread(target=_work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.monotonic() - started


class TestFactoryMode:
    """The production default must NOT serialize concurrent collections."""

    def test_concurrent_collections_do_not_serialize(self) -> None:
        delay = 0.4
        observer = _SlowOrchestrator(delay)
        factory = _factory_for(observer)
        elapsed = _run_two_collections(factory, delay)

        # Two 400ms collections running in parallel finish in well under 800ms.
        assert elapsed < delay * 1.8

    def test_factory_builds_a_fresh_orchestrator_per_call(self) -> None:
        built: list[_SlowOrchestrator] = []

        def _factory() -> _SlowOrchestrator:
            instance = _SlowOrchestrator(1.0)
            built.append(instance)
            return instance

        collect_evidence_sync(_factory, "Startup", "https://example.com")
        collect_evidence_sync(_factory, "Startup", "https://example.com")
        assert len(built) == 2
        assert built[0] is not built[1]


class TestInjectedInstanceMode:
    """A shared injected orchestrator must retain the serialization guarantee."""

    def test_shared_instance_is_serialized(self) -> None:
        delay = 0.2
        instance = _SlowOrchestrator(delay)
        elapsed = _run_two_collections(instance, delay)

        # Serialized access: two 200ms collections take at least 400ms.
        assert elapsed >= delay * 1.8

    def test_shared_instance_never_used_concurrently(self) -> None:
        instance = _SlowOrchestrator(0.2)
        _run_two_collections(instance, 0.2)
        assert instance.max_active == 1


class TestEmptyWebsite:
    """An empty website must return an empty bundle without touching the collector."""

    def test_empty_website_short_circuits(self) -> None:
        touched = False

        def _factory() -> _SlowOrchestrator:
            nonlocal touched
            touched = True
            return _SlowOrchestrator(0.0)

        bundle = collect_evidence_sync(_factory, "Startup", "")
        assert bundle.total_pages == 0
        assert not touched
