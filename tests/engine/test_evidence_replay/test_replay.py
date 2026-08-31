"""Regression tests for deterministic offline evidence replay (Sprint P7).

These tests prove that replaying a recorded evidence corpus reproduces
the exact evidence bundle and, through the full pipeline, identical
reports, confidence, recommendations, and scores — all without network
access and without changing production behavior when replay is disabled.

Coverage
--------
* Corpus round-trip produces a field-for-field identical EvidenceBundle.
* Deterministic document/source ordering across loads.
* Committed corpus (``sample_evidence.json``) loads and rebuilds.
* ReplayEvidenceProvider satisfies the EvidenceProvider contract, is
  gateable, emits metadata-free documents, and performs no network I/O.
* Replay through a real EvidenceOrchestrator reproduces the deterministic
  bundle fields.
* Full-engine report equivalence between a live (MockTransport) run whose
  bundle is preserved verbatim and a replay run, on scores, confidence,
  recommendations, decision, and evidence metadata.
* Disabled replay leaves default provider selection byte-identical.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from pydantic import HttpUrl

from app.schemas.analysis import StartupAnalysisRequest
from predictron_engine.engine import PredictronEngine
from predictron_engine.evidence.fetcher import FetcherSettings, HttpPageFetcher
from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.evidence.orchestrator import (
    EvidenceOrchestrator,
    default_collection_providers,
)
from predictron_engine.evidence.provider_contracts import (
    EvidenceProvider,
    ProviderResult,
)
from predictron_engine.evidence.replay import (
    ReplayEvidenceProvider,
    build_replay_provider,
)
from predictron_engine.evidence.replay.dataset import (
    dump_corpus,
    rebuild_bundle,
    serialize_bundle,
)
from predictron_engine.evidence.website_provider import (
    WebsiteEvidenceProvider,
    WebsiteProviderSettings,
)

SITE = "https://example.com/"
_FIXED = datetime(2026, 1, 1, tzinfo=UTC)


def _ok_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    return httpx.Response(
        200,
        text=(
            "<html><head><title>ExampleCorp</title></head><body><main>"
            f"<h1>{path} page</h1>"
            "<p>ExampleCorp builds data platform software for enterprises.</p>"
            "<p>Built on Python with FastAPI, deployed on AWS, uses PostgreSQL.</p>"
            "</main></body></html>"
        ),
        headers={"content-type": "text/html; charset=utf-8"},
    )


def _live_collector(handler=None) -> EvidenceOrchestrator:
    handler = handler or _ok_handler
    fetcher = HttpPageFetcher(
        settings=FetcherSettings(retry_backoff_ms=0),
        transport=httpx.MockTransport(handler),
    )
    provider = WebsiteEvidenceProvider(
        fetcher=fetcher, settings=WebsiteProviderSettings(max_concurrency=8)
    )
    return EvidenceOrchestrator(providers=[provider])


def _live_bundle(startup: str = "ExampleCorp") -> EvidenceBundle:
    """Collect a live bundle through MockTransport (no network)."""
    return _collect_sync(_live_collector(), startup)


def _collect_sync(orchestrator: EvidenceOrchestrator, startup: str) -> EvidenceBundle:
    import asyncio

    return asyncio.run(orchestrator.collect(startup, SITE))


def _normalise_clock(bundle: EvidenceBundle) -> EvidenceBundle:
    """Pin all wall-clock fields so two bundles compare field-for-field.

    Only wall-clock/timing fields are touched; content, ordering, ids,
    metadata, trust, provenance are preserved verbatim.
    """

    def pin_doc(doc):
        update = {
            "fetched_at": _FIXED,
            # response_time_ms is a wall-clock per-fetch latency measured
            # during HTTP round-trip; it differs between the live pre-enrichment
            # record and the live final document. Pin it like the other timing
            # fields so the replay comparison is timing-independent.
            "response_time_ms": 0,
        }
        if doc.metadata is not None:
            update["metadata"] = doc.metadata.model_copy(
                update={
                    "provenance": [
                        p.model_copy(update={"fetched_at": _FIXED})
                        for p in doc.metadata.provenance
                    ]
                }
            )
        return doc.model_copy(update=update)

    intelligence = None
    if bundle.intelligence is not None:
        intelligence = bundle.intelligence.model_copy(
            update={"processing_duration_ms": 0}
        )
    providers = [p.model_copy(update={"duration_ms": 0}) for p in bundle.providers]
    return bundle.model_copy(
        update={
            "collected_at": _FIXED,
            "duration_ms": 0,
            "documents": [pin_doc(d) for d in bundle.documents],
            "sources": [s.model_copy(update={"fetched_at": _FIXED}) for s in bundle.sources],
            "intelligence": intelligence,
            "providers": providers,
        }
    )


class _FixedBundleCollector:
    """Orchestrator-protocol collector returning a fixed bundle verbatim.

    Represents "the live run that produced the corpus": the corpus IS the
    preserved bundle, so this collector returns it untouched.
    """

    def __init__(self, bundle: EvidenceBundle) -> None:
        self._bundle = bundle

    async def collect(self, startup_name: str, website: str) -> EvidenceBundle:
        return self._bundle


def _request(name: str = "ExampleCorp") -> StartupAnalysisRequest:
    return StartupAnalysisRequest(
        startup_name=name,
        website=HttpUrl(SITE),
        description=(
            "ExampleCorp builds data platform software for modern businesses."
        ),
    )


# ── Phase 2/5: corpus round-trip ─────────────────────────────────────────


def test_corpus_round_trip_is_byte_identical(tmp_path) -> None:
    live = _normalise_clock(_live_bundle())
    path = tmp_path / "corpus.json"
    dump_corpus(live, name="t", path=path)

    doc = serialize_bundle(live, name="t")
    rebuilt = rebuild_bundle(doc)

    assert rebuilt == live
    assert rebuilt.model_dump() == live.model_dump()


def test_corpus_deterministic_ordering(tmp_path) -> None:
    live = _normalise_clock(_live_bundle())
    path = tmp_path / "corpus.json"
    dump_corpus(live, name="t", path=path)

    rebuilt = rebuild_bundle(serialize_bundle(live, name="t"))
    assert [d.original_url for d in rebuilt.documents] == [
        d.original_url for d in live.documents
    ]
    assert [s.original_url for s in rebuilt.sources] == [
        s.original_url for s in live.sources
    ]


def test_corpus_serialization_is_stable_across_writes(tmp_path) -> None:
    live = _normalise_clock(_live_bundle())
    p1 = tmp_path / "c1.json"
    p2 = tmp_path / "c2.json"
    dump_corpus(live, name="t", path=p1)
    dump_corpus(live, name="t", path=p2)
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


def test_committed_dataset_loads_and_rebuilds(committed_dataset) -> None:
    from pathlib import Path

    from predictron_engine.evidence.replay.dataset import load_corpus

    document = load_corpus(Path(committed_dataset))
    rebuilt = rebuild_bundle(document)
    assert rebuilt.startup_name == "ExampleCorp"
    assert rebuilt.total_pages == 8
    assert len(rebuilt.sources) == 8
    assert rebuilt.website == HttpUrl(SITE)
    # deterministic document ids (derived from URL, all unique)
    from predictron_engine.evidence.models import make_document_id

    ids = [d.id for d in rebuilt.documents]
    assert len(set(ids)) == len(ids)
    for doc in rebuilt.documents:
        assert doc.id == make_document_id(str(doc.url))


# ── Phase 3: provider contract & behavior ────────────────────────────────


@pytest.mark.asyncio
async def test_replay_provider_satisfies_evidence_provider(committed_dataset) -> None:
    provider: EvidenceProvider = ReplayEvidenceProvider(committed_dataset)
    assert isinstance(provider, EvidenceProvider)  # structural protocol check
    assert provider.name == "replay"


@pytest.mark.asyncio
async def test_provider_collect_returns_result_without_network(tmp_path) -> None:
    live = _normalise_clock(await _live_collector().collect("ExampleCorp", SITE))
    path = tmp_path / "corpus.json"
    dump_corpus(live, name="t", path=path)

    provider = ReplayEvidenceProvider(path)
    from predictron_engine.evidence.provider_contracts import CollectContext

    result: ProviderResult = await provider.collect(
        CollectContext(startup_name="ExampleCorp", website=SITE)
    )
    assert isinstance(result, ProviderResult)
    assert result.success is True
    assert result.website == HttpUrl(SITE)
    assert result.attempted_pages == 8
    # documents emitted metadata-free so orchestrator re-enriches deterministically
    assert all(d.metadata is None for d in result.documents)
    # documents and sources preserved
    assert [d.original_url for d in result.documents] == [
        d.original_url for d in live.documents
    ]
    assert [s.original_url for s in result.sources] == [
        s.original_url for s in live.sources
    ]


@pytest.mark.asyncio
async def test_provider_can_collect_gates_on_dataset(tmp_path) -> None:
    live = _normalise_clock(await _live_collector().collect("ExampleCorp", SITE))
    path = tmp_path / "corpus.json"
    dump_corpus(live, name="t", path=path)
    provider = ReplayEvidenceProvider(path)

    from predictron_engine.evidence.provider_contracts import CollectContext

    assert provider.can_collect(CollectContext(startup_name="ExampleCorp", website=SITE))
    assert not provider.can_collect(
        CollectContext(startup_name="DifferentCo", website=SITE)
    )


# ── Phase 3: replay through the real orchestrator ─────────────────────────


@pytest.mark.asyncio
async def test_replay_via_orchestrator_reproduces_bundle(tmp_path) -> None:
    live = _normalise_clock(await _live_collector().collect("ExampleCorp", SITE))
    path = tmp_path / "corpus.json"
    dump_corpus(live, name="t", path=path)

    replay_orchestrator = EvidenceOrchestrator(
        providers=[ReplayEvidenceProvider(path)]
    )
    replayed = _normalise_clock(
        await replay_orchestrator.collect("ExampleCorp", SITE)
    )

    # deterministic fields must be identical
    assert replayed.documents == live.documents
    assert replayed.sources == live.sources
    assert replayed.website == live.website
    assert replayed.attempted_pages == live.attempted_pages
    assert replayed.intelligence == live.intelligence
    assert replayed.trust_summary == live.trust_summary


@pytest.mark.asyncio
async def test_replay_preserves_dedup_diagnostics(tmp_path) -> None:
    # All pages identical -> live enrichment dedupes 8 -> 1.  Recording the
    # pre-enrichment raw documents must let replay recompute the same
    # documents_input / duplicates_removed counts.
    def identical_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "<html><head><title>Same</title></head><body><main>"
                "<h1>Same</h1><p>Identical content on every page.</p>"
                "</main></body></html>"
            ),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    fetcher = HttpPageFetcher(
        settings=FetcherSettings(retry_backoff_ms=0),
        transport=httpx.MockTransport(identical_handler),
    )
    provider = WebsiteEvidenceProvider(
        fetcher=fetcher, settings=WebsiteProviderSettings(max_concurrency=8)
    )
    from predictron_engine.evidence.provider_contracts import CollectContext

    raw_result = await provider.collect(
        CollectContext(startup_name="ExampleCorp", website=SITE)
    )
    live_orch = EvidenceOrchestrator(providers=[provider])
    live = _normalise_clock(await live_orch.collect("ExampleCorp", SITE))
    assert live.total_pages == 1  # dedup collapsed 8 -> 1
    assert live.intelligence is not None
    assert live.intelligence.documents_input == 8
    assert live.intelligence.duplicates_removed == 7

    path = tmp_path / "dedup_corpus.json"
    dump_corpus(
        live,
        name="dedup",
        path=path,
        raw_documents=[
            d.model_copy(update={"fetched_at": _FIXED})
            for d in raw_result.documents
        ],
    )

    replayed = _normalise_clock(
        await EvidenceOrchestrator(
            providers=[ReplayEvidenceProvider(path)]
        ).collect("ExampleCorp", SITE)
    )
    assert replayed.documents == live.documents
    assert replayed.intelligence == live.intelligence
    assert replayed.trust_summary == live.trust_summary


# ── Phase 4: configuration ───────────────────────────────────────────────


def test_replay_disabled_leaves_providers_byte_identical(monkeypatch) -> None:
    monkeypatch.delenv("EVIDENCE_REPLAY_ENABLED", raising=False)
    monkeypatch.delenv("EVIDENCE_REPLAY_DATASET", raising=False)
    assert build_replay_provider() is None
    providers = default_collection_providers()
    # website provider is first; no replay provider present
    assert any(p.name == "website" for p in providers)
    assert not any(getattr(p, "name", "") == "replay" for p in providers)


def test_replay_disabled_with_env_unset(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EVIDENCE_REPLAY_ENABLED", "false")
    monkeypatch.setenv("EVIDENCE_REPLAY_DATASET", "sample_evidence")
    assert build_replay_provider() is None


def test_replay_enabled_without_dataset_returns_none(monkeypatch) -> None:
    monkeypatch.setenv("EVIDENCE_REPLAY_ENABLED", "true")
    monkeypatch.delenv("EVIDENCE_REPLAY_DATASET", raising=False)
    assert build_replay_provider() is None


def test_replay_enabled_builds_provider(monkeypatch, committed_dataset) -> None:
    monkeypatch.setenv("EVIDENCE_REPLAY_ENABLED", "true")
    monkeypatch.setenv("EVIDENCE_REPLAY_DATASET", committed_dataset)
    provider = build_replay_provider()
    assert provider is not None
    assert isinstance(provider, ReplayEvidenceProvider)


def test_replay_enabled_replaces_live_providers(monkeypatch, committed_dataset) -> None:
    monkeypatch.setenv("EVIDENCE_REPLAY_ENABLED", "true")
    monkeypatch.setenv("EVIDENCE_REPLAY_DATASET", committed_dataset)
    providers = default_collection_providers()
    assert len(providers) == 1
    assert providers[0].name == "replay"
    assert not any(p.name == "website" for p in providers)


# ── Phase 5: full-pipeline regression ─────────────────────────────────────


def test_replay_pipeline_matches_reference_report(tmp_path) -> None:
    live = _normalise_clock(_live_bundle())
    path = tmp_path / "corpus.json"
    dump_corpus(live, name="t", path=path)

    # Reference: the engine fed the exact preserved live bundle.
    reference = PredictronEngine(
        evidence_collector=_FixedBundleCollector(live)
    ).analyze(_request())

    # Replay: the engine through a real orchestrator re-enriching the corpus.
    replay = PredictronEngine(
        evidence_collector=EvidenceOrchestrator(
            providers=[ReplayEvidenceProvider(path)]
        )
    ).analyze(_request())

    _assert_reports_equivalent(reference, replay)


_CLOCK_TIMING_KEYS = frozenset(
    {"processing_time_ms", "collection_time_ms", "duration_ms", "response_time_ms"}
)


def _assert_reports_equivalent(a, b) -> None:
    """Assert reports match on every semantic field, excluding wall-clock."""
    a_dict = a.model_dump()
    b_dict = b.model_dump()
    _strip_wallclock(a_dict)
    _strip_wallclock(b_dict)
    assert a_dict == b_dict


def _strip_wallclock(node) -> None:
    """Normalize wall-clock values in a report dict in place.

    Any ``datetime`` is pinned to a constant and wall-clock duration keys
    are removed, so two runs differ only in genuine timing.
    """
    if isinstance(node, dict):
        for key in list(node.keys()):
            value = node[key]
            if isinstance(value, datetime):
                node[key] = _WALLCLOCK_FIXED
            elif isinstance(value, list):
                _strip_wallclock(value)
            elif isinstance(value, dict):
                _strip_wallclock(value)
            if key in _CLOCK_TIMING_KEYS:
                node.pop(key, None)
    elif isinstance(node, list):
        for item in node:
            _strip_wallclock(item)


_WALLCLOCK_FIXED = datetime(2026, 1, 1, tzinfo=UTC)


def test_replay_report_scores_confidence_recommendations(tmp_path) -> None:
    live = _normalise_clock(_live_bundle())
    path = tmp_path / "corpus.json"
    dump_corpus(live, name="t", path=path)
    reference = PredictronEngine(
        evidence_collector=_FixedBundleCollector(live)
    ).analyze(_request())
    replay = PredictronEngine(
        evidence_collector=EvidenceOrchestrator(
            providers=[ReplayEvidenceProvider(path)]
        )
    ).analyze(_request())

    assert reference.scores == replay.scores
    assert reference.confidence == replay.confidence
    assert reference.recommendations == replay.recommendations
    assert reference.investment_decision == replay.investment_decision
    assert reference.evidence == replay.evidence


def test_replay_repeated_runs_are_deterministic(tmp_path) -> None:
    live = _normalise_clock(_live_bundle())
    path = tmp_path / "corpus.json"
    dump_corpus(live, name="t", path=path)
    build = lambda: PredictronEngine(  # noqa: E731
        evidence_collector=EvidenceOrchestrator(
            providers=[ReplayEvidenceProvider(path)]
        )
    ).analyze(_request())
    r1 = build()
    r2 = build()
    _assert_reports_equivalent(r1, r2)
