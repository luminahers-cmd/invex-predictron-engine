"""Live evidence sources for research grounding.

This module adapts the (async) evidence subsystem to the synchronous
:class:`~predictron_engine.research.grounding.DocumentSource` protocol.
It lives **outside** ``predictron_engine.research`` so the research
package stays pure: nothing under ``predictron_engine/research/`` performs
I/O, and this is where the live side effects live.

The adapter uses the reviewed synchronous bridge
(:func:`predictron_engine.evidence.runner.collect_evidence_sync`) — it
never introduces its own ``asyncio.run``.
"""

from __future__ import annotations

from predictron_engine.evidence.models import EvidenceDocument
from predictron_engine.evidence.runner import (
    EvidenceCollectorFactory,
    EvidenceOrchestratorProtocol,
    collect_evidence_sync,
)


class SyncEvidenceSource:
    """Live :class:`DocumentSource` over the evidence orchestrator.

    Accepts either a shared orchestrator instance or a factory that
    produces a fresh orchestrator per collection (the production default).
    Documents come back fully enriched, so ``metadata.trust_score`` is
    populated by the evidence subsystem — never recomputed here.
    """

    def __init__(
        self,
        orchestrator: EvidenceOrchestratorProtocol | EvidenceCollectorFactory,
    ) -> None:
        if not callable(orchestrator) and not callable(
            getattr(orchestrator, "collect", None)
        ):
            raise TypeError(
                "orchestrator must be an async collector or a factory"
            )
        self._orchestrator = orchestrator

    @property
    def orchestrator(
        self,
    ) -> EvidenceOrchestratorProtocol | EvidenceCollectorFactory:
        """Return the wrapped orchestrator or factory."""
        return self._orchestrator

    def documents(
        self,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[EvidenceDocument, ...]:
        """Collect and return the documents for ``company_name``.

        An empty ``website`` yields no documents (the bridge returns an
        empty bundle), never a fabricated URL.

        Provider failures are isolated by
        :class:`~predictron_engine.evidence.orchestrator.EvidenceOrchestrator`
        (a failing provider is recorded in the bundle's diagnostics and
        skipped), so the documents returned here are exactly what the
        collection produced.  This adapter performs no failure handling of
        its own: the production best-effort guarantee — a failed collection
        never aborts an analysis — lives at the integration seam,
        :meth:`ResearchAnalysisIntegration.augment
        <predictron_engine.integrations.research_analysis.ResearchAnalysisIntegration.augment>`.
        """
        bundle = collect_evidence_sync(
            self._orchestrator, company_name, website
        )
        return tuple(bundle.documents)


__all__ = ["SyncEvidenceSource"]
