"""Integration tests wiring the evidence collection layer into the pipeline.

These tests exercise the PredictronEngine with a real EvidenceOrchestrator
backed by httpx MockTransport (no network), and assert that collection
statistics flow into report metadata and that extracted features can be
enriched with website content.
"""

from __future__ import annotations

import httpx
from pydantic import HttpUrl

from app.schemas.analysis import StartupAnalysisRequest
from predictron_engine.engine import PredictronEngine
from predictron_engine.evidence.exceptions import InvalidWebsiteError
from predictron_engine.evidence.fetcher import FetcherSettings, HttpPageFetcher
from predictron_engine.evidence.orchestrator import EvidenceOrchestrator
from predictron_engine.evidence.website_provider import (
    WebsiteEvidenceProvider,
    WebsiteProviderSettings,
)
from predictron_engine.extraction.composite import CompositeExtractor
from predictron_engine.extraction.feature_models import (
    call_extractor_with_evidence,
    extractor_accepts_evidence,
)
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Report
from predictron_engine.models.startup import Startup

SITE = "https://example.com/"


def _collector(handler) -> EvidenceOrchestrator:
    """Build an EvidenceOrchestrator that routes fetches through MockTransport."""
    fetcher = HttpPageFetcher(
        settings=FetcherSettings(retry_backoff_ms=0),
        transport=httpx.MockTransport(handler),
    )
    provider = WebsiteEvidenceProvider(
        fetcher=fetcher, settings=WebsiteProviderSettings(max_concurrency=8)
    )
    return EvidenceOrchestrator(providers=[provider])


def _ok_page(path: str) -> httpx.Response:
    return httpx.Response(
        200,
        text=(
            f"<html><head><title>{path} page</title></head>"
            f"<body><main><h1>{path}</h1><p>Content for {path}.</p></main></body></html>"
        ),
        headers={"content-type": "text/html; charset=utf-8"},
    )


def _request(
    *,
    name: str = "EvidenceCo",
    website: str | None = SITE,
    description: str = (
        "EvidenceCo builds software products for modern businesses."
    ),
) -> StartupAnalysisRequest:
    return StartupAnalysisRequest(
        startup_name=name,
        website=HttpUrl(website) if website else None,
        description=description,
    )


class TestEvidenceMetadataInReport:
    def test_metadata_populated_when_all_pages_succeed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _ok_page(request.url.path)

        engine = PredictronEngine(evidence_collector=_collector(handler))
        report = engine.analyze(_request())

        meta = report.analysis_metadata.evidence_collection
        assert meta is not None
        assert meta.website == SITE
        assert meta.pages_discovered == 8
        assert meta.pages_fetched == 8
        assert meta.successful_sources == 8
        assert meta.failed_sources == 0
        assert meta.collection_time_ms >= 0

    def test_metadata_records_partial_failures(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path in ("/products", "/services"):
                return httpx.Response(404)
            return _ok_page(request.url.path)

        engine = PredictronEngine(evidence_collector=_collector(handler))
        report = engine.analyze(_request())

        meta = report.analysis_metadata.evidence_collection
        assert meta is not None
        assert meta.pages_discovered == 8
        assert meta.pages_fetched == 6
        assert meta.successful_sources == 6
        assert meta.failed_sources == 2
        assert meta.successful_sources + meta.failed_sources == meta.pages_discovered

    def test_metadata_zeroed_when_website_omitted(self, engine) -> None:
        report = engine.analyze(_request(website=None))

        meta = report.analysis_metadata.evidence_collection
        assert meta is not None
        assert meta.website is None
        assert meta.pages_discovered == 0
        assert meta.pages_fetched == 0
        assert meta.successful_sources == 0
        assert meta.failed_sources == 0
        assert meta.collection_time_ms == 0

    def test_analysis_succeeds_when_collection_raises(self, engine) -> None:
        class ExplodingCollector:
            async def collect(self, startup_name: str, website: str):
                raise InvalidWebsiteError("boom")

        engine = PredictronEngine(evidence_collector=ExplodingCollector())
        report = engine.analyze(_request())

        assert isinstance(report, Report)
        meta = report.analysis_metadata.evidence_collection
        assert meta is not None
        assert meta.pages_discovered == 0
        assert meta.pages_fetched == 0


class TestEvidenceEnrichesExtraction:
    def test_market_classification_uses_website_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text=(
                    "<html><body><main>"
                    "<h1>Cybersecurity platform</h1>"
                    "<p>We offer threat detection, zero trust, and SOC 2 compliance.</p>"
                    "</main></body></html>"
                ),
            )

        engine = PredictronEngine(evidence_collector=_collector(handler))
        report = engine.analyze(
            _request(description="EvidenceCo makes software for business teams.")
        )

        assert report.features.industry == "cybersecurity"

    def test_technology_signals_use_website_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text=(
                    "<html><body><main>"
                    "<p>Built on Python with FastAPI, deployed on AWS, uses PostgreSQL.</p>"
                    "</main></body></html>"
                ),
            )

        engine = PredictronEngine(evidence_collector=_collector(handler))
        report = engine.analyze(
            _request(description="EvidenceCo ships business software.")
        )

        assert "python" in report.features.technology_stack
        assert "aws" in report.features.technology_stack


class TestBackwardCompatibility:
    def test_legacy_extractor_without_evidence_support(self) -> None:
        class LegacyExtractor:
            def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
                assert not hasattr(self, "_saw_evidence")
                return ExtractedFeatures(industry="legacy")

        assert extractor_accepts_evidence(LegacyExtractor()) is False
        composite = CompositeExtractor(extractors=[LegacyExtractor()])
        startup = Startup(name="Legacy", website="", description="Legacy business software.")
        data = CollectedData(
            startup_name="Legacy",
            website_domain="",
            description_tokens=[],
            description_word_count=0,
            has_website=False,
            has_pitch_deck=False,
            founder_count=0,
            url_metadata={},
            enrichment_signals={},
        )
        features = composite.extract(startup, data)
        assert features.industry == "legacy"

    def test_modern_extractor_receives_evidence(self) -> None:
        seen: list[object] = []

        class ModernExtractor:
            def extract(
                self,
                startup: Startup,
                data: CollectedData,
                evidence=None,
            ) -> ExtractedFeatures:
                seen.append(evidence)
                return ExtractedFeatures()

        assert extractor_accepts_evidence(ModernExtractor()) is True
        startup = Startup(name="Modern", website="", description="Modern business software.")
        data = CollectedData(
            startup_name="Modern",
            website_domain="",
            description_tokens=[],
            description_word_count=0,
            has_website=False,
            has_pitch_deck=False,
            founder_count=0,
            url_metadata={},
            enrichment_signals={},
        )
        result = call_extractor_with_evidence(
            ModernExtractor().extract, startup, data, evidence="evidence"
        )
        assert isinstance(result, ExtractedFeatures)
        assert seen == ["evidence"]
