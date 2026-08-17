"""No-op evidence orchestrator — returns an empty bundle without I/O.

Injected into the engine in tests and offline use so that analysis runs
never perform network requests while still exercising the full evidence
collection pipeline wiring.
"""

from __future__ import annotations

from predictron_engine.evidence.models import EvidenceBundle


class NoOpEvidenceOrchestrator:
    """Orchestrator that always returns an empty :class:`EvidenceBundle`."""

    def __init__(self) -> None:
        self._calls = 0

    @property
    def calls(self) -> int:
        """Number of times ``collect`` has been invoked."""
        return self._calls

    async def collect(self, startup_name: str, website: str) -> EvidenceBundle:
        """Return an empty evidence bundle without performing any I/O."""
        self._calls += 1
        return EvidenceBundle.empty(startup_name)
