"""One-off generator for the committed offline evidence corpus (Sprint P7).

Builds a representative EvidenceBundle by running the real collection
pipeline through httpx.MockTransport (no network), normalizes the
wall-clock fields to deterministic constants, and writes the resulting
corpus to benchmarks/offline_evidence/sample_evidence.json.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx

from predictron_engine.evidence.fetcher import FetcherSettings, HttpPageFetcher
from predictron_engine.evidence.models import EvidenceBundle, EvidenceDocument
from predictron_engine.evidence.orchestrator import EvidenceOrchestrator
from predictron_engine.evidence.provider_contracts import CollectContext
from predictron_engine.evidence.replay.dataset import dump_corpus
from predictron_engine.evidence.website_provider import (
    WebsiteEvidenceProvider,
    WebsiteProviderSettings,
)

OUT = Path(__file__).resolve().parent / "offline_evidence" / "sample_evidence.json"
SITE = "https://example.com/"


def _build_provider() -> WebsiteEvidenceProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        return httpx.Response(
            200,
            text=(
                f"<html><head><title>{path} page</title></head>"
                "<body><main>"
                f"<h1>{path}</h1>"
                "<p>ExampleCorp builds data platform software for enterprises.</p>"
                "<p>Built on Python with FastAPI, deployed on AWS, uses PostgreSQL.</p>"
                f"</main></body></html>"
            ),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    fetcher = HttpPageFetcher(
        settings=FetcherSettings(retry_backoff_ms=0),
        transport=httpx.MockTransport(handler),
    )
    return WebsiteEvidenceProvider(
        fetcher=fetcher, settings=WebsiteProviderSettings(max_concurrency=8)
    )


def _collector() -> EvidenceOrchestrator:
    return EvidenceOrchestrator(providers=[_build_provider()])


def _normalise(bundle: EvidenceBundle) -> EvidenceBundle:
    fixed = datetime(2026, 1, 1, tzinfo=UTC)

    def normalise_doc(doc: EvidenceDocument) -> EvidenceDocument:
        if doc.metadata is None:
            return doc.model_copy(update={"fetched_at": fixed})
        provenance = [
            p.model_copy(update={"fetched_at": fixed}) for p in doc.metadata.provenance
        ]
        metadata = doc.metadata.model_copy(update={"provenance": provenance})
        return doc.model_copy(update={"fetched_at": fixed, "metadata": metadata})

    documents = [normalise_doc(d) for d in bundle.documents]
    sources = [s.model_copy(update={"fetched_at": fixed}) for s in bundle.sources]
    return bundle.model_copy(
        update={
            "collected_at": fixed,
            "documents": documents,
            "sources": sources,
        }
    )


async def main() -> None:
    provider = _build_provider()
    raw_result = await provider.collect(
        CollectContext(startup_name="ExampleCorp", website=SITE)
    )
    bundle = await _collector().collect("ExampleCorp", SITE)
    normalised = _normalise(bundle)
    raw_documents = [
        d.model_copy(update={"fetched_at": datetime(2026, 1, 1, tzinfo=UTC)})
        for d in raw_result.documents
    ]
    dump_corpus(
        normalised,
        name="sample_evidence",
        path=OUT,
        raw_documents=raw_documents,
    )
    print(f"wrote {OUT}: {normalised.total_pages} documents, "
          f"{len(normalised.sources)} sources, website={normalised.website}")


if __name__ == "__main__":
    asyncio.run(main())
