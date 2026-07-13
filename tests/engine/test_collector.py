from predictron_engine.collection.collector import DefaultDataCollector
from predictron_engine.models.collected_data import CollectedData

"""Tests for the DefaultDataCollector."""


class TestDefaultDataCollector:
    """Unit tests for the collection stage."""

    def test_collect_returns_collected_data(self, sample_startup):
        collector = DefaultDataCollector()
        result = collector.collect(sample_startup)

        assert isinstance(result, CollectedData)
        assert result.startup_name == "SampleCo"

    def test_domain_extraction(self, sample_startup):
        collector = DefaultDataCollector()
        result = collector.collect(sample_startup)

        assert result.website_domain == "example.com"

    def test_word_count(self, sample_startup):
        collector = DefaultDataCollector()
        result = collector.collect(sample_startup)

        assert result.description_word_count > 0
        assert len(result.description_tokens) == result.description_word_count

    def test_pitch_deck_detected(self, full_request):
        from predictron_engine.ingest.normalizer import DefaultNormalizer

        normalizer = DefaultNormalizer()
        startup = normalizer.normalize(full_request)
        collector = DefaultDataCollector()
        result = collector.collect(startup)

        assert result.has_pitch_deck is True

    def test_founder_count(self, full_request):
        from predictron_engine.ingest.normalizer import DefaultNormalizer

        normalizer = DefaultNormalizer()
        startup = normalizer.normalize(full_request)
        collector = DefaultDataCollector()
        result = collector.collect(startup)

        assert result.founder_count == 2

    def test_url_metadata_populated(self, sample_startup):
        collector = DefaultDataCollector()
        result = collector.collect(sample_startup)

        assert "website_url" in result.url_metadata

    def test_enrichment_signals_empty_by_default(self, sample_startup):
        collector = DefaultDataCollector()
        result = collector.collect(sample_startup)

        assert result.enrichment_signals == {}
