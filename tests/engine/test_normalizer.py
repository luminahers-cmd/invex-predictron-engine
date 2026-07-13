from pydantic import HttpUrl

from app.schemas.analysis import StartupAnalysisRequest
from predictron_engine.ingest.normalizer import DefaultNormalizer
from predictron_engine.models.startup import Startup

"""Tests for the DefaultNormalizer."""


class TestDefaultNormalizer:
    """Unit tests for the normalization stage."""

    def test_normalize_minimal_request(self, minimal_request):
        normalizer = DefaultNormalizer()
        result = normalizer.normalize(minimal_request)

        assert isinstance(result, Startup)
        assert result.name == "TestCo"
        assert "testco.example.com" in result.website
        assert result.pitch_deck_url is None
        assert result.founder_linkedin_urls == []

    def test_normalize_full_request(self, full_request):
        normalizer = DefaultNormalizer()
        result = normalizer.normalize(full_request)

        assert result.name == "FullCo"
        assert result.pitch_deck_url is not None
        assert len(result.founder_linkedin_urls) == 2

    def test_name_is_stripped(self):
        normalizer = DefaultNormalizer()
        request = StartupAnalysisRequest(
            startup_name="  PaddedCo  ",
            website=HttpUrl("https://padded.example.com"),
            description="A padded startup name test case.",
        )
        result = normalizer.normalize(request)
        assert result.name == "PaddedCo"

    def test_raw_data_preserved(self, minimal_request):
        normalizer = DefaultNormalizer()
        result = normalizer.normalize(minimal_request)

        assert "startup_name" in result.raw_data
        assert "website" in result.raw_data
        assert "description" in result.raw_data

    def test_normalized_at_timestamp_set(self, minimal_request):
        normalizer = DefaultNormalizer()
        result = normalizer.normalize(minimal_request)

        assert result.normalized_at is not None
