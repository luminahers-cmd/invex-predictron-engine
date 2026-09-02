"""Generate deterministic offline evidence corpora for every benchmark case.

Each corpus is produced by running the REAL collection pipeline (via
:class:`predictron_engine.evidence.website_provider.WebsiteEvidenceProvider`
through :class:`predictron_engine.evidence.orchestrator.EvidenceOrchestrator`)
against an :class:`httpx.MockTransport` (no network).

The mock site renders each company's own benchmark ``description`` verbatim
as its website copy.  The description is the authoritative source material
for each fictitious benchmark company — no metrics or wording are invented:
the web pages state exactly what the case already states in its description.

This follows the same architecture as ``gen_offline_corpus.py`` and produces
the corpus files :func:`benchmarks.benchmark_runner.load_all_case_evidence_bundles`
consumes, so ``--offline-replay`` exercises a real, reproducible collection
for every case.

Usage:
    python -m benchmarks.gen_case_corpora          # write all case corpora
    python -m benchmarks.gen_case_corpora --case unit_economics funding_efficiency
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx

from benchmarks.startup_cases.cases import BENCHMARK_CASES, get_all_case_ids
from predictron_engine.evidence.fetcher import FetcherSettings, HttpPageFetcher
from predictron_engine.evidence.models import EvidenceBundle, EvidenceDocument
from predictron_engine.evidence.orchestrator import EvidenceOrchestrator
from predictron_engine.evidence.provider_contracts import CollectContext
from predictron_engine.evidence.replay.dataset import dump_corpus
from predictron_engine.evidence.website_provider import (
    WebsiteEvidenceProvider,
    WebsiteProviderSettings,
)

OUT = Path(__file__).resolve().parent / "offline_evidence"
_FIXED = datetime(2026, 1, 1, tzinfo=UTC)
# Matches the existing committed sample corpus (Sprint P7).
RETRY_BACKOFF_MS = 0


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_provider(description: str, startup_name: str) -> WebsiteEvidenceProvider:
    """Build a real WebsiteEvidenceProvider served by a deterministic mock."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path or ""
        name = path.strip("/").replace("/", " ") or "home"
        title = _escape(startup_name)
        page_name = _escape(name)
        page_desc = _escape(description)
        body = (
            f"<html><head><title>{title} — {page_name}</title></head>"
            "<body><main>"
            f"<h1>{title}</h1>"
            f"<p>{page_desc}</p>"
            "</main></body></html>"
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/html; charset=utf-8"},
        )

    fetcher = HttpPageFetcher(
        settings=FetcherSettings(retry_backoff_ms=RETRY_BACKOFF_MS),
        transport=httpx.MockTransport(handler),
    )
    return WebsiteEvidenceProvider(
        fetcher=fetcher, settings=WebsiteProviderSettings(max_concurrency=8)
    )


def _normalise(
    bundle: EvidenceBundle, raw_documents: list[EvidenceDocument]
) -> tuple[EvidenceBundle, list[EvidenceDocument]]:
    """Return (bundle, raw_documents) with every wall-clock timing and
    timestamp normalised to fixed values for byte-level reproducibility.

    The elapsed-time fields (``duration_ms``, ``response_time_ms``,
    ``processing_duration_ms``) are pure measurement artifacts on the
    mock transport; zeroing them does not alter any evidence content and
    makes regeneration byte-identical.
    """
    fixed = _FIXED

    def normalise_doc(doc: EvidenceDocument) -> EvidenceDocument:
        if doc.metadata is None:
            return doc.model_copy(
                update={"fetched_at": fixed, "response_time_ms": 0}
            )
        provenance = [
            p.model_copy(update={"fetched_at": fixed})
            for p in doc.metadata.provenance
        ]
        metadata = doc.metadata.model_copy(update={"provenance": provenance})
        return doc.model_copy(
            update={
                "fetched_at": fixed,
                "response_time_ms": 0,
                "metadata": metadata,
            }
        )

    documents = [normalise_doc(d) for d in bundle.documents]
    sources = [
        s.model_copy(update={"fetched_at": fixed, "response_time_ms": 0})
        for s in bundle.sources
    ]
    providers = [
        p.model_copy(update={"duration_ms": 0}) for p in bundle.providers
    ]
    intelligence = None
    if bundle.intelligence is not None:
        intelligence = bundle.intelligence.model_copy(
            update={"processing_duration_ms": 0}
        )
    bundle = bundle.model_copy(
        update={
            "collected_at": fixed,
            "duration_ms": 0,
            "documents": documents,
            "sources": sources,
            "providers": providers,
            "intelligence": intelligence,
        }
    )
    raw_documents = [normalise_doc(d) for d in raw_documents]
    return bundle, raw_documents


async def generate_case(case_id: str) -> tuple[Path, EvidenceBundle]:
    case = next(c for c in BENCHMARK_CASES if c["id"] == case_id)
    request = case["request"]
    provider = build_provider(request["description"], request["startup_name"])
    raw_result = await provider.collect(
        CollectContext(
            startup_name=request["startup_name"], website=request.get("website")
        )
    )
    orchestrator = EvidenceOrchestrator(providers=[provider])
    bundle = await orchestrator.collect(
        request["startup_name"], request.get("website") or ""
    )
    normalised, raw_documents = _normalise(bundle, raw_result.documents)
    out = OUT / f"{case_id}.json"
    dump_corpus(
        normalised,
        name=case_id,
        path=out,
        raw_documents=raw_documents,
    )
    return out, normalised


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", nargs="*", default=None,
        help="Case IDs to generate (default: all).",
    )
    args = parser.parse_args()
    ids = args.case if args.case else get_all_case_ids()
    for case_id in ids:
        out, bundle = await generate_case(case_id)
        print(
            f"wrote {out.relative_to(OUT)}: {bundle.total_pages} documents, "
            f"{len(bundle.sources)} sources, website={bundle.website}"
        )


if __name__ == "__main__":
    asyncio.run(main())
